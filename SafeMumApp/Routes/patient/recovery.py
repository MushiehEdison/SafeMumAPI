from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import json
from groq import Groq
from SafeMumApp import db
from SafeMumApp.models import (
    User, MedicalProfile, CommunityHealthWorker,
    SupportRequest, CHWCase, AIMemory,
    CheckIn, CommunityPost, CommunityReply
)
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from SafeMumApp.utils.chw_assignment import assign_chw
from SafeMumApp.Ai_Analysis.classifier import (
    classify_risk, predict_care_seeking,
    detect_isolation, get_vulnerability_category,
)
from SafeMumApp.Ai_Analysis.dataset_interpreter import get_risk_context_for_prompt

bp = Blueprint('recovery', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY PHASE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_recovery_phase(days_since_loss: int | None) -> str:
    """
    early_acute    → 0–14 days
    processing     → 15–42 days
    rebuilding     → 43–84 days
    stabilised     → 85+ days
    """
    if days_since_loss is None:
        return "early_acute"
    if days_since_loss <= 14:
        return "early_acute"
    if days_since_loss <= 42:
        return "processing"
    if days_since_loss <= 84:
        return "rebuilding"
    return "stabilised"


PHASE_LABELS = {
    "early_acute":  "Early Acute (0–2 weeks)",
    "processing":   "Processing (2–6 weeks)",
    "rebuilding":   "Rebuilding (6–12 weeks)",
    "stabilised":   "Stabilised (12+ weeks)",
}

PHASE_PROGRESS = {
    "early_acute": 15,
    "processing":  40,
    "rebuilding":  70,
    "stabilised":  100,
}

PHASE_TIPS = {
    "early_acute": [
        "Rest as much as your body needs. There is no rush to feel better.",
        "Drink water regularly, even if you do not feel thirsty.",
        "Let one trusted person know how you are feeling today.",
        "It is okay to cry. It is okay to feel nothing. Both are normal.",
        "Eat something small, even if you have no appetite.",
    ],
    "processing": [
        "Try to go outside for a short walk, even 10 minutes helps.",
        "Write down one thing you are feeling today — it does not have to make sense.",
        "Reach out to a friend or family member today, even just a message.",
        "Your grief does not follow a timeline. Be gentle with yourself.",
        "If sleep is difficult, try a simple breathing exercise before bed.",
    ],
    "rebuilding": [
        "Notice one small thing that brought you comfort this week.",
        "Consider speaking with a counsellor if you have not already.",
        "Re-establishing a gentle daily routine can help your body heal.",
        "It is okay to have good days. They do not mean you have forgotten.",
        "You are allowed to laugh, rest, and enjoy things while still grieving.",
    ],
    "stabilised": [
        "You have shown remarkable strength to reach this point.",
        "Checking in regularly — even when things feel okay — protects your progress.",
        "If you are thinking about a future pregnancy, speak with a healthcare provider.",
        "Your experience can help others. Consider sharing when you feel ready.",
        "Keep nurturing the support network around you.",
    ],
}


def _get_daily_tip(phase: str, user_id: int) -> str:
    """Return a tip rotated daily based on user_id so different users get variety."""
    tips = PHASE_TIPS.get(phase, PHASE_TIPS["processing"])
    day_of_year = datetime.utcnow().timetuple().tm_yday
    index = (day_of_year + int(user_id)) % len(tips)
    return tips[index]


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISERS
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_checkin(c) -> dict:
    return {
        "id":         c.id,
        "date":       c.created_at.isoformat() if c.created_at else None,
        "mood":       c.mood,
        "note":       c.note,
        "conclusion": c.conclusion,   
        "color":      _mood_color(c.mood),
    }


def _serialise_post(p) -> dict:
    return {
        "id":      p.id,
        "content": p.content,
        "timeAgo": _time_ago(p.created_at),
        "replies": [_serialise_reply(r) for r in (p.replies or [])],
    }


def _serialise_reply(r) -> dict:
    return {
        "id":      r.id,
        "content": r.content,
        "timeAgo": _time_ago(r.created_at),
    }


def _serialise_counsellor(chw: CommunityHealthWorker) -> dict:
    return {
        "id":        chw.id,
        "name":      chw.full_name,
        "phone":     chw.phone,
        "speciality": (chw.speciality or "").replace("_", " ").title(),
        "area":      chw.coverage_area or "Your area",
        "available": chw.is_available,
    }
def _generate_checkin_conclusion(mood, note, answers, memory,
                                  consecutive_low, isolation, vuln,
                                  recovery_phase, daily_tip):
    
    try:
        import json as _json
        dataset_context = get_risk_context_for_prompt()

        memory_ctx = ""
        if memory and memory.memory_summary:
            memory_ctx = f"\nWhat you know about her: {memory.memory_summary}"
        if memory and memory.recurring_themes:
            memory_ctx += f"\nRecurring themes: {', '.join(memory.recurring_themes)}"

        alert_ctx = ""
        if consecutive_low >= 3:
            alert_ctx = "\nFLAG: 3+ consecutive low mood check-ins."
        if isolation.get('is_isolated'):
            alert_ctx += "\nFLAG: Social isolation detected."

        answers_text = ""
        if answers:
            answers_text = "\nHer answers to today's check-in questions:\n"
            for qid, val in answers.items():
                answers_text += f"  {qid}: {val}\n"

        prompt = f"""You are a compassionate AI health companion for a woman recovering from pregnancy loss in Sub-Saharan Africa.

{dataset_context}

Her mood today: "{mood}"
Her note: "{note or 'No note provided'}"
Recovery phase: {PHASE_LABELS.get(recovery_phase, recovery_phase)}
Vulnerability level: {vuln}{memory_ctx}{alert_ctx}{answers_text}

Return ONLY a JSON object with exactly these two keys:
{{
  "response": "A warm 3-4 sentence conversational reply acknowledging her mood and note. If flagged, gently mention a counsellor. End with one gentle question or encouragement.",
  "conclusion": "A 2-3 sentence clinical-style summary of her overall condition today based on ALL her answers — physical and emotional. Use plain language. State what is going well, what needs watching, and one clear recommendation. This is what she reads as her check-in summary."
}}

Return ONLY valid JSON. No markdown."""

        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()
        try:
            result = _json.loads(raw)
        except Exception:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            result = _json.loads(match.group()) if match else {}

        return (
            result.get("response", "Thank you for sharing. Take care of yourself today."),
            result.get("conclusion", "Based on your check-in, you are being monitored. Keep checking in daily.")
        )

    except Exception as e:
        print(f"[SafeMum AI] Groq conclusion failed: {e}")
        fallbacks = {
            "red":   "Thank you for sharing. You do not have to carry this alone.",
            "gray":  "Thank you for being here. Up and down days are part of this journey.",
            "green": "It is good to hear you are doing a little better.",
        }
        color = _mood_color(mood)
        return (
            fallbacks.get(color, fallbacks["gray"]),
            "Based on your responses today, keep monitoring how you feel and check in again tomorrow."
        )

# ─────────────────────────────────────────────────────────────────────────────
# CHECK-IN
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/checkin', methods=['GET'])
@patient_required
def get_checkin_history():
    user_id = get_current_user_id()
    history = (
        CheckIn.query
        .filter_by(user_id=user_id)
        .order_by(CheckIn.created_at.desc())
        .limit(30)
        .all()
    )
    return jsonify({
        "message": "ok",
        "data":    [_serialise_checkin(c) for c in history],
    }), 200


@bp.route('/checkin', methods=['POST'])
@patient_required
def submit_checkin():
    user_id = get_current_user_id()
    body    = request.get_json(silent=True) or {}

    mood = (body.get('mood') or '').strip()
    if not mood:
        return jsonify({"error": "mood is required"}), 400

    note = (body.get('note') or '').strip() or None
    answers = body.get('answers') or {}  

    # 1. Save check-in
    checkin = CheckIn(user_id=user_id, mood=mood, note=note)
    db.session.add(checkin)
    db.session.flush()

    # 2. Count consecutive low moods
    recent = (
        CheckIn.query
        .filter_by(user_id=user_id)
        .order_by(CheckIn.created_at.desc())
        .limit(10)
        .all()
    )
    consecutive_low = 0
    for c in recent:
        if _mood_color(c.mood) == 'red':
            consecutive_low += 1
        else:
            break
    if _mood_color(mood) == 'red':
        consecutive_low += 1

    # 3. Load or create AIMemory
    memory = AIMemory.query.filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id)
        db.session.add(memory)

    # 4. Compute recovery phase from days_since_loss
    days_since_loss = memory.days_since_loss
    recovery_phase  = _compute_recovery_phase(days_since_loss)

    # 5. Run ML models
    profile_dict = {
        "Crisis_ML":   memory.consecutive_low_moods or 0,
        "Activekin":   3 if not memory.flagged_for_counsellor else 1,
        "Wealthscore": 3,
    }
    isolation_result = detect_isolation(profile_dict)
    vuln_category    = get_vulnerability_category(
        crisis_score=float(memory.consecutive_low_moods or 0),
        wealth_score=3.0,
    )

    # 6. Update AIMemory
    mood_score = 1 if _mood_color(mood) == 'red' else (3 if _mood_color(mood) == 'gray' else 5)
    memory.last_mood_score       = mood_score
    memory.consecutive_low_moods = consecutive_low
    memory.total_checkins        = (memory.total_checkins or 0) + 1
    memory.vulnerability_level   = vuln_category
    memory.recovery_phase        = recovery_phase

    flag_for_counsellor = consecutive_low >= 3 or isolation_result.get('is_isolated', False)
    memory.flagged_for_counsellor = flag_for_counsellor

    # 7. Checkin streak
    yesterday = datetime.utcnow() - timedelta(days=1)
    checked_yesterday = CheckIn.query.filter(
        CheckIn.user_id == user_id,
        CheckIn.created_at >= yesterday.replace(hour=0, minute=0, second=0),
        CheckIn.created_at < datetime.utcnow().replace(hour=0, minute=0, second=0),
    ).first()
    if checked_yesterday:
        memory.checkin_streak = (memory.checkin_streak or 0) + 1
    else:
        memory.checkin_streak = 1

    # 8. Auto-assign CHW if urgent
    chw_assigned = None
    if isolation_result.get('action') == 'assign_chw_urgent' or consecutive_low >= 3:
        user = User.query.get(user_id)
        trigger = "isolation" if isolation_result.get('is_isolated') else "mood_checkin"
        result = assign_chw(
            patient_id=user_id,
            patient_lat=user.latitude if user else None,
            patient_lng=user.longitude if user else None,
            reason=f"Auto-assigned after {consecutive_low} consecutive low mood check-ins.",
            preferred_speciality="counsellor",
            trigger=trigger,
        )
        if result:
            chw_assigned = result["chw_name"]

    # 9. Get daily tip
    daily_tip = _get_daily_tip(recovery_phase, user_id)

    # 10. Generate AI response
    ai_response, conclusion = _generate_checkin_conclusion(
        mood=mood,
        note=note,
        answers=answers,
        memory=memory,
        consecutive_low=consecutive_low,
        isolation=isolation_result,
        vuln=vuln_category,
        recovery_phase=recovery_phase,
        daily_tip=daily_tip,
    )

    checkin.conclusion = conclusion

    db.session.commit()

    return jsonify({
        "message":    "Check-in saved",
        "data":       _serialise_checkin(checkin),
        "ai_response": ai_response,
        "conclusion":  conclusion,
        "daily_tip":  daily_tip,
        "recovery": {
            "phase":        recovery_phase,
            "phase_label":  PHASE_LABELS[recovery_phase],
            "progress_pct": PHASE_PROGRESS[recovery_phase],
            "streak":       memory.checkin_streak or 1,
        },
        "flags": {
            "consecutive_low_moods":  consecutive_low,
            "flagged_for_counsellor": flag_for_counsellor,
            "isolation_detected":     isolation_result.get('is_isolated', False),
            "vulnerability":          vuln_category,
            "chw_assigned":           chw_assigned,
        },
    }), 201


def _generate_checkin_response(mood, note, memory, consecutive_low,
                                isolation, vuln, recovery_phase, daily_tip):
    try:
        # Inject dataset knowledge
        dataset_context = get_risk_context_for_prompt()

        memory_ctx = ""
        if memory.memory_summary:
            memory_ctx = f"\nWhat you know about her: {memory.memory_summary}"
        if memory.recurring_themes:
            memory_ctx += f"\nRecurring themes: {', '.join(memory.recurring_themes)}"

        alert_ctx = ""
        if consecutive_low >= 3:
            alert_ctx = "\nIMPORTANT: She has had 3 or more consecutive low mood check-ins. Be especially gentle and suggest she connect with a counsellor."
        if isolation.get('is_isolated'):
            alert_ctx += "\nIMPORTANT: Isolation detected. Gently remind her that support is available."

        phase_ctx = f"\nHer recovery phase: {PHASE_LABELS.get(recovery_phase, recovery_phase)}"

        prompt = f"""You are a warm, compassionate AI companion supporting a woman through pregnancy loss in Sub-Saharan Africa.

{dataset_context}

Her mood today: "{mood}"
Her note: "{note or 'No note provided'}"
Vulnerability level: {vuln}{phase_ctx}{memory_ctx}{alert_ctx}

Write a short (3-4 sentences), warm, human response to her check-in.
- Acknowledge exactly what she said
- Reflect her recovery phase naturally without naming it clinically
- Do not use clinical language or give medical advice
- If flagged, gently mention a counsellor is available
- End with one gentle supportive question or encouragement
- Sound human, not like a chatbot"""

        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"[SafeMum AI] Groq checkin response failed: {e}")
        fallbacks = {
            "red":   "Thank you for sharing that. You do not have to carry this alone. What you are feeling is valid, and it takes courage to show up here even on the hardest days.",
            "gray":  "Thank you for being here. Up and down days are part of this journey. I am glad you checked in today.",
            "green": "It is good to hear you are doing a little better. You have shown real strength. I am here whenever you need to talk.",
        }
        return fallbacks.get(_mood_color(mood), fallbacks["gray"])


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/progress', methods=['GET'])
@patient_required
def get_recovery_progress():
    """
    Returns the patient's full recovery progress summary.
    Used by the recovery dashboard / progress tab.
    """
    user_id = get_current_user_id()
    memory  = AIMemory.query.filter_by(user_id=user_id).first()

    if not memory:
        return jsonify({
            "message": "ok",
            "data": {
                "phase":        "early_acute",
                "phase_label":  PHASE_LABELS["early_acute"],
                "progress_pct": PHASE_PROGRESS["early_acute"],
                "streak":       0,
                "total_checkins": 0,
                "vulnerability": "unknown",
                "daily_tip":    PHASE_TIPS["early_acute"][0],
                "flagged_for_counsellor": False,
            }
        }), 200

    recovery_phase = _compute_recovery_phase(memory.days_since_loss)

    # Checkin history for mood chart
    history = (
        CheckIn.query
        .filter_by(user_id=user_id)
        .order_by(CheckIn.created_at.desc())
        .limit(14)
        .all()
    )
    mood_trend = [
        {
            "date":  c.created_at.strftime("%d %b"),
            "mood":  c.mood,
            "color": _mood_color(c.mood),
            "score": 1 if _mood_color(c.mood) == 'red' else (3 if _mood_color(c.mood) == 'gray' else 5),
        }
        for c in reversed(history)
    ]

    return jsonify({
        "message": "ok",
        "data": {
            "phase":                  recovery_phase,
            "phase_label":            PHASE_LABELS[recovery_phase],
            "progress_pct":           PHASE_PROGRESS[recovery_phase],
            "streak":                 memory.checkin_streak or 0,
            "total_checkins":         memory.total_checkins or 0,
            "vulnerability":          memory.vulnerability_level or "unknown",
            "flagged_for_counsellor": memory.flagged_for_counsellor or False,
            "days_since_loss":        memory.days_since_loss,
            "daily_tip":              _get_daily_tip(recovery_phase, user_id),
            "mood_trend":             mood_trend,
            "phases": [
                {
                    "key":      phase,
                    "label":    PHASE_LABELS[phase],
                    "progress": PHASE_PROGRESS[phase],
                    "active":   phase == recovery_phase,
                    "done":     PHASE_PROGRESS[phase] < PHASE_PROGRESS[recovery_phase],
                }
                for phase in ["early_acute", "processing", "rebuilding", "stabilised"]
            ],
        }
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# COMMUNITY POSTS
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/community', methods=['GET'])
@patient_required
def get_community_posts():
    posts = (
        CommunityPost.query
        .order_by(CommunityPost.created_at.desc())
        .limit(50)
        .all()
    )
    return jsonify({"message": "ok", "data": [_serialise_post(p) for p in posts]}), 200


@bp.route('/community', methods=['POST'])
@patient_required
def create_community_post():
    body    = request.get_json(silent=True) or {}
    content = (body.get('content') or '').strip()
    if not content:
        return jsonify({"error": "content is required"}), 400
    post = CommunityPost(content=content)
    db.session.add(post)
    db.session.commit()
    return jsonify({"message": "Post created", "data": _serialise_post(post)}), 201


@bp.route('/community/<int:post_id>/reply', methods=['POST'])
@patient_required
def reply_to_post(post_id):
    post = CommunityPost.query.get(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    body    = request.get_json(silent=True) or {}
    content = (body.get('content') or '').strip()
    if not content:
        return jsonify({"error": "content is required"}), 400
    reply = CommunityReply(post_id=post_id, content=content)
    db.session.add(reply)
    db.session.commit()
    return jsonify({"message": "Reply added", "data": _serialise_reply(reply)}), 201


# ─────────────────────────────────────────────────────────────────────────────
# COUNSELLORS
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/counsellors', methods=['GET'])
@patient_required
def get_counsellors():
    counsellors = (
        CommunityHealthWorker.query
        .filter_by(is_verified=True)
        .filter(CommunityHealthWorker.speciality.in_(['counsellor', 'nurse', 'midwife']))
        .order_by(
            CommunityHealthWorker.is_available.desc(),
            CommunityHealthWorker.full_name,
        )
        .all()
    )
    return jsonify({"message": "ok", "data": [_serialise_counsellor(c) for c in counsellors]}), 200


# ─────────────────────────────────────────────────────────────────────────────
# NGO SUPPORT REQUEST
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/support', methods=['POST'])
@patient_required
def submit_support_request():
    user_id = get_current_user_id()
    body    = request.get_json(silent=True) or {}
    r_type  = (body.get('type') or '').strip()
    desc    = (body.get('description') or '').strip()
    if not r_type:
        return jsonify({"error": "type is required"}), 400
    type_map = {
        "Counselling":   "counselling",
        "Transport":     "transport",
        "Financial Aid": "financial_aid",
    }
    req = SupportRequest(
        patient_id  = user_id,
        type        = type_map.get(r_type, r_type.lower()),
        description = desc or None,
        status      = 'pending',
    )
    db.session.add(req)
    db.session.commit()
    return jsonify({"message": "Support request submitted", "data": {"id": req.id, "type": r_type, "status": req.status}}), 201


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM CHECK-IN
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/symptom-checkin', methods=['POST'])
@patient_required
def submit_symptom_checkin():
    user_id  = get_current_user_id()
    body     = request.get_json(silent=True) or {}
    symptoms = body.get('symptoms', [])
    note     = (body.get('note') or '').strip() or None

    SYMPTOM_MAP = {
        "Heavy or unusual bleeding":          {"pds207a": 1},
        "Foul-smelling discharge":            {"pds207b": 1},
        "Fever or chills":                    {"pds207c": 1},
        "Persistent headache":                {"pds207f": 1},
        "Severe abdominal pain or cramping":  {"pds207g": 1},
        "Dizziness or fainting":              {"pds207k": 1},
        "Swollen or painful legs":            {"pds207k": 1},
        "Cold hands or feet":                 {"pds207n": 1},
        "Chest pain or difficulty breathing": {"pds207n": 1},
        "Persistent cough (new)":             {"pds207f": 1},
        "Wound pain or redness":              {"pds207b": 1},
        "Unable to sleep":                    {},
        "Not eating or no appetite":          {},
        "Feeling hopeless or empty":          {},
        "Withdrawing from people":            {},
        "Crying most of the day":             {},
        "Feeling completely numb":            {},
        "I feel physically stable today":     {},
        "I am emotionally okay today":        {},
        "No unusual bleeding or pain":        {},
        "I am eating and sleeping normally":  {},
    }

    feature_input = {f"pds207{c}": 0 for c in "abcdefghijklmn"}

    user    = User.query.get(user_id)
    profile = MedicalProfile.query.filter_by(user_id=user_id).first()
    memory  = AIMemory.query.filter_by(user_id=user_id).first()

    feature_input.update({
        "pds101":    profile.age if profile and profile.age else 25,
        "pds102":    "urban",
        "education": "secondary",
        "pds201":    0,
        "pds202":    memory.previous_losses if memory else 0,
        "pds203":    0,
        "county":    profile.region if profile and profile.region else "unknown",
    })

    for symptom in symptoms:
        for col, val in SYMPTOM_MAP.get(symptom, {}).items():
            feature_input[col] = val

    risk_result = classify_risk(feature_input)
    care_result = predict_care_seeking({
        "age":          feature_input.get("pds101", 25),
        "education":    "secondary",
        "marital":      "married",
        "employment":   "employed",
        "religion":     "christian",
        "facility_type": "public",
    })

    # Get recovery phase for context
    recovery_phase = _compute_recovery_phase(memory.days_since_loss if memory else None)

    ai_response = _generate_symptom_response(
        symptoms=symptoms,
        note=note,
        risk_result=risk_result,
        care_result=care_result,
        memory=memory,
        recovery_phase=recovery_phase,
    )

    chw_assigned = None
    if risk_result['risk_level'] == 'high' and care_result['recommendation'] == 'assign_chw':
        res = assign_chw(
            patient_id=user_id,
            patient_lat=user.latitude if user else None,
            patient_lng=user.longitude if user else None,
            reason=f"Auto-assigned after high-risk symptom check-in. Symptoms: {', '.join(symptoms[:5])}",
            preferred_speciality=None,
            trigger="symptom_checkin",
        )
        if res:
            chw_assigned = res["chw_name"]

    if memory and risk_result['risk_level'] == 'high':
        memory.vulnerability_level = 'high'
        physical = [s for s in symptoms if s in [
            "Heavy or unusual bleeding", "Fever or chills",
            "Severe abdominal pain or cramping", "Foul-smelling discharge"
        ]]
        if physical:
            themes = list(memory.recurring_themes or [])
            themes.extend(physical)
            memory.recurring_themes = list(set(themes))[:10]
        db.session.commit()

    return jsonify({
        "message":     "Symptom check-in processed",
        "risk":        risk_result,
        "care":        care_result,
        "ai_response": ai_response,
        "flags": {
            "risk_level":      risk_result['risk_level'],
            "confidence":      risk_result['confidence'],
            "recommendation":  care_result['recommendation'],
            "chw_assigned":    chw_assigned,
        }
    }), 200


def _generate_symptom_response(symptoms, note, risk_result, care_result,
                                memory, recovery_phase):
    try:
        dataset_context = get_risk_context_for_prompt()

        memory_ctx = ""
        if memory and memory.memory_summary:
            memory_ctx = f"\nContext about her: {memory.memory_summary}"

        physical = [s for s in symptoms if s not in [
            "Unable to sleep", "Not eating or no appetite",
            "Feeling hopeless or empty", "Withdrawing from people",
            "Crying most of the day", "Feeling completely numb",
            "I feel physically stable today", "I am emotionally okay today",
            "No unusual bleeding or pain", "I am eating and sleeping normally"
        ]]
        emotional = [s for s in symptoms if s in [
            "Unable to sleep", "Not eating or no appetite",
            "Feeling hopeless or empty", "Withdrawing from people",
            "Crying most of the day", "Feeling completely numb"
        ]]

        prompt = f"""You are a compassionate AI health companion for a woman recovering from pregnancy loss in Sub-Saharan Africa.

{dataset_context}

She has reported these symptoms:
Physical: {', '.join(physical) if physical else 'None'}
Emotional: {', '.join(emotional) if emotional else 'None'}
Her note: "{note or 'No note provided'}"
Recovery phase: {PHASE_LABELS.get(recovery_phase, recovery_phase)}

ML Risk Assessment: {risk_result['risk_level']} risk (confidence: {risk_result['confidence']})
Recommended action: {care_result['recommendation']}{memory_ctx}

Write a 3-4 sentence warm, clear response:
- Acknowledge what she reported
- If high risk: clearly but gently urge her to seek care TODAY
- If low risk: reassure her and give one practical recovery suggestion matching her phase
- Do NOT diagnose or use medical jargon
- Sound human and caring"""

        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"[SafeMum AI] Groq symptom response failed: {e}")
        if risk_result['risk_level'] == 'high':
            return "Some of what you are experiencing needs medical attention today. Please do not wait — visit the nearest facility or send an emergency alert now."
        return "Thank you for checking in. Keep monitoring how you feel and reach out if anything changes."


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mood_color(mood: str) -> str:
    mood_lower = (mood or '').lower()
    if any(w in mood_lower for w in ('struggling', 'hard', 'difficult', 'bad')):
        return 'red'
    if any(w in mood_lower for w in ('okay', 'better', 'good', 'well')):
        return 'green'
    return 'gray'


def _time_ago(dt) -> str:
    if not dt:
        return ""
    delta = datetime.utcnow() - dt
    s = int(delta.total_seconds())
    if s < 60:     return "Just now"
    if s < 3600:   return f"{s // 60}m ago"
    if s < 86400:  return f"{s // 3600}h ago"
    if s < 604800: return f"{s // 86400}d ago"
    # Fixed: removed %-d which crashes on Windows
    return dt.strftime("%d %b").lstrip("0")



@bp.route('/checkin/questions', methods=['GET'])
@patient_required
def get_checkin_questions():
    """
    Returns today's personalised check-in questions based on
    recovery phase and previous check-in history.
    """
    user_id = get_current_user_id()
    memory  = AIMemory.query.filter_by(user_id=user_id).first()
    
    recovery_phase = _compute_recovery_phase(
        memory.days_since_loss if memory else None
    )

    # Get last check-in for context
    last_checkin = (
        CheckIn.query
        .filter_by(user_id=user_id)
        .order_by(CheckIn.created_at.desc())
        .first()
    )

    try:
        dataset_context = get_risk_context_for_prompt()

        last_ctx = ""
        if last_checkin:
            last_ctx = f"\nHer last check-in was: mood='{last_checkin.mood}', note='{last_checkin.note or 'none'}', {_time_ago(last_checkin.created_at)} ago."

        prompt = f"""You are a compassionate AI health companion for a woman recovering from pregnancy loss in Sub-Saharan Africa.

{dataset_context}

Her recovery phase: {PHASE_LABELS.get(recovery_phase, recovery_phase)}{last_ctx}
{"Previous recurring themes: " + ", ".join(memory.recurring_themes) if memory and memory.recurring_themes else ""}

Generate a check-in for her covering BOTH physical and emotional wellbeing.
Return ONLY a JSON object in this exact format:
{{
  "greeting": "A warm 1-sentence greeting personalised to her phase",
  "physical_questions": [
    {{"id": "p1", "text": "question text", "type": "scale", "min_label": "Very painful", "max_label": "No pain"}},
    {{"id": "p2", "text": "question text", "type": "yesno"}},
    {{"id": "p3", "text": "question text", "type": "yesno"}},
    {{"id": "p4", "text": "question text", "type": "yesno"}}
  ],
  "emotional_questions": [
    {{"id": "e1", "text": "question text", "type": "scale", "min_label": "Not at all", "max_label": "Very much so"}},
    {{"id": "e2", "text": "question text", "type": "yesno"}},
    {{"id": "e3", "text": "question text", "type": "yesno"}},
    {{"id": "e4", "text": "question text", "type": "choice", "options": ["option1", "option2", "option3"]}}
  ],
  "closing": "A warm 1-sentence encouragement to answer honestly"
}}

Question types: scale (1-5), yesno (yes/no), choice (multiple options).
Make questions specific to her phase. Early acute = basic safety. Processing = grief patterns. Rebuilding = daily function. Stabilised = future outlook.
Return ONLY valid JSON."""

        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()
        try:
            questions = json.loads(raw)
        except Exception:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            questions = json.loads(match.group()) if match else {}

        return jsonify({
            "message": "ok",
            "data": {
                "questions":      questions,
                "recovery_phase": recovery_phase,
                "phase_label":    PHASE_LABELS[recovery_phase],
                "daily_tip":      _get_daily_tip(recovery_phase, int(user_id)),
            }
        }), 200

    except Exception as e:
        print(f"[SafeMum AI] Failed to generate questions: {e}")
        # Fallback static questions
        return jsonify({
            "message": "ok",
            "data": {
                "questions": {
                    "greeting": "I am glad you are here today. Let us check in together.",
                    "physical_questions": [
                        {"id": "p1", "text": "How would you rate your physical pain or discomfort today?", "type": "scale", "min_label": "Very painful", "max_label": "No pain"},
                        {"id": "p2", "text": "Have you eaten at least one meal today?", "type": "yesno"},
                        {"id": "p3", "text": "Have you had any unusual bleeding or discharge?", "type": "yesno"},
                        {"id": "p4", "text": "Did you sleep last night?", "type": "yesno"},
                    ],
                    "emotional_questions": [
                        {"id": "e1", "text": "How are you feeling emotionally right now?", "type": "scale", "min_label": "Very low", "max_label": "Doing well"},
                        {"id": "e2", "text": "Have you spoken to someone you trust today?", "type": "yesno"},
                        {"id": "e3", "text": "Are you feeling safe?", "type": "yesno"},
                        {"id": "e4", "text": "What best describes your mood today?", "type": "choice", "options": ["Sad and heavy", "Numb or empty", "Up and down", "Calm", "Better than yesterday"]},
                    ],
                    "closing": "There are no right or wrong answers. Just answer honestly.",
                },
                "recovery_phase": recovery_phase,
                "phase_label":    PHASE_LABELS.get(recovery_phase, ""),
                "daily_tip":      _get_daily_tip(recovery_phase, int(user_id)),
            }
        }), 200