import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Model used ────────────────────────────────────────────────────
GROQ_MODEL = "llama3-70b-8192"


# ── Main interpreter function ─────────────────────────────────────

def interpret_ml_output(ml_result: dict, user_context: dict) -> dict:
    prompt = _build_prompt(ml_result, user_context)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError:
        return _fallback_response(ml_result, user_context)
    except Exception as e:
        print(f"[interpreter] Groq API error: {e}")
        return _fallback_response(ml_result, user_context)


# ── Chat interpreter ──────────────────────────────────────────────

def interpret_chat_message(user_message: str, user_context: dict, conversation_history: list) -> dict:
    

    system = _chat_system_prompt(user_context)

    messages = [{"role": "system", "content": system}]
    messages.extend(conversation_history[-10:])  # last 10 messages for context window
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=800,
        )

        reply_text = response.choices[0].message.content.strip()

        # Detect if the message triggers any actions
        actions = _detect_actions_from_reply(reply_text, user_message)

        return {
            "reply": reply_text,
            "actions": actions
        }

    except Exception as e:
        print(f"[interpreter] Chat error: {e}")
        return {
            "reply": "I am here with you. Could you tell me a little more about how you are feeling right now?",
            "actions": []
        }


# ── Recovery hub interpreter ──────────────────────────────────────

def interpret_checkin(mood_score: int, mood_label: str, notes: str, user_context: dict) -> dict:
    """
    Used by the recovery hub check-in endpoint.
    Takes mood score (1-5), the label, optional notes, and user context.
    Returns a warm AI response and any follow-up actions.

    mood_score: 1=very low, 2=low, 3=okay, 4=good, 5=much better
    """

    prompt = f"""
A woman just completed her emotional check-in on SafeMum AI.

Her profile:
- Name: {user_context.get('name', 'her')}
- Days since pregnancy loss: {user_context.get('days_since_loss', 'unknown')}
- Loss type: {user_context.get('loss_type', 'pregnancy loss')}
- Previous losses: {user_context.get('previous_losses', 0)}
- Mood trend over last 5 check-ins: {user_context.get('mood_trend', 'unknown')}
- Vulnerability category: {user_context.get('vulnerability_category', 'unknown')}
- Has CHW assigned: {user_context.get('has_chw', False)}
- Cultural profile: {user_context.get('cultural_profile', 'unknown')}

Today's check-in:
- Mood score: {mood_score} out of 5
- Mood label: {mood_label}
- Notes she wrote: "{notes or 'none'}"

Respond ONLY with a JSON object with these exact keys:
{{
  "ai_response": "warm, personal response to her check-in — 2 to 3 sentences max, not clinical",
  "flag_for_counsellor": true or false,
  "assign_chw": true or false,
  "urgency": "none" or "low" or "high",
  "follow_up_message": "a short message to send her tomorrow based on today"
}}

Flag for counsellor if mood score is 1 or 2 for 3 or more consecutive check-ins.
Assign CHW if mood score is 1 and vulnerability is high and she has no CHW.
Urgency is high if mood score is 1 and notes mention hopeless, harm, or not wanting to continue.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a compassionate maternal health AI. Always respond in JSON only. No preamble, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"[interpreter] Check-in error: {e}")
        return {
            "ai_response": "Thank you for checking in today. I see you. Take it one moment at a time.",
            "flag_for_counsellor": mood_score <= 2,
            "assign_chw": False,
            "urgency": "low" if mood_score <= 2 else "none",
            "follow_up_message": "How are you feeling today?"
        }


# ── Symptom interpreter ───────────────────────────────────────────

def interpret_symptoms(selected_symptoms: list, user_context: dict, ml_risk: dict) -> dict:
    """
    Used by the symptom checklist endpoint.
    Takes selected symptoms from the home page checklist,
    user context, and the ML risk classifier output.
    Returns action guidance.

    selected_symptoms: list of symptom strings from the checklist
    ml_risk: output from classifier.classify_risk()
    """

    # Map symptoms to clinical meaning based on what we know from pds207 columns
    complication_map = {
        "Heavy bleeding": "potential hemorrhage — pds207a equivalent",
        "Fever": "potential infection or sepsis — pds207f equivalent",
        "Severe pain": "potential incomplete abortion — pds207k equivalent",
        "Chest pain": "potential pulmonary embolism",
        "Persistent cough": "potential pulmonary embolism",
        "Cold hands or feet": "potential internal bleeding or shock — pds207d equivalent",
        "Dizziness": "potential shock or hemorrhage",
        "Foul discharge": "potential infection — pds207b equivalent",
        "Wound pain": "potential surgical site infection",
    }

    clinical_flags = [
        complication_map[s] for s in selected_symptoms
        if s in complication_map
    ]

    prompt = f"""
A woman has just completed a symptom checklist on SafeMum AI.

Her profile:
- Name: {user_context.get('name', 'her')}
- Days since loss: {user_context.get('days_since_loss', 'unknown')}
- Loss type: {user_context.get('loss_type', 'pregnancy loss')}
- Previous losses: {user_context.get('previous_losses', 0)}
- County: {user_context.get('county', 'unknown')}
- Cultural profile: {user_context.get('cultural_profile', 'unknown')}
- Will seek care probability: {user_context.get('care_seeking_probability', 0.5)}
- Vulnerability: {user_context.get('vulnerability_category', 'unknown')}

Symptoms she selected: {', '.join(selected_symptoms) if selected_symptoms else 'none — she selected only positive signs'}

ML Risk Classifier output:
- Risk level: {ml_risk.get('risk_level', 'unknown')}
- Confidence: {ml_risk.get('confidence', 0)}
- Top driving features: {ml_risk.get('top_features', [])}

Clinical significance of her symptoms: {', '.join(clinical_flags) if clinical_flags else 'no high-risk symptoms'}

Respond ONLY with a JSON object with these exact keys:
{{
  "risk_level": "emergency" or "urgent" or "monitor" or "stable",
  "title": "short headline — max 10 words",
  "message": "specific clear explanation — 2 sentences — what this means for her body right now",
  "chat_message": "what the AI assistant should say to her — warm, personal, not clinical — max 3 sentences",
  "action": "emergency_alert" or "find_facility" or "talk_to_ai" or "rest_and_monitor",
  "trigger_emergency_alert": true or false,
  "assign_chw": true or false,
  "chw_reason": "why assign a CHW or null if not assigning"
}}

Be specific. Do not say 'please see a doctor'. Say what the symptom combination suggests and what she should do about it in the next hour.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical maternal health AI. Always respond in JSON only. No preamble, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"[interpreter] Symptom error: {e}")
        return _symptom_fallback(selected_symptoms)


# ── Admin insight interpreter ─────────────────────────────────────

def interpret_service_gaps(gap_data: dict) -> dict:
    """
    Used by the admin insights endpoint.
    Takes service gap cluster data and generates
    plain-language recommendations for health ministries.

    gap_data: output from classifier.get_high_need_areas()
    """

    prompt = f"""
You are generating a health ministry briefing for SafeMum AI.

Service gap analysis data:
{json.dumps(gap_data, indent=2)}

This data shows patient volumes vs facility coverage per county in Kenya.
High need score = many patients, few facilities.

Generate a plain-language briefing. Respond ONLY with JSON:
{{
  "headline": "one sentence summary of the situation",
  "top_priority_counties": ["list of top 3 county names needing urgent attention"],
  "key_finding": "the single most important finding in 2 sentences",
  "recommended_action": "one specific actionable recommendation for health authorities",
  "data_note": "one sentence on data limitations or caveats"
}}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a public health data analyst. Always respond in JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"[interpreter] Service gap error: {e}")
        return {
            "headline": "Service gap analysis unavailable",
            "top_priority_counties": [],
            "key_finding": "Unable to generate insight at this time.",
            "recommended_action": "Please review raw data directly.",
            "data_note": "System error during analysis."
        }


# ── Private helpers ───────────────────────────────────────────────

def _system_prompt() -> str:
    return """
You are SafeMum AI — a compassionate, warm maternal health assistant built for 
Sub-Saharan Africa. You help women who have experienced pregnancy loss through 
physical recovery, emotional support, and connection to care.

Your responses are:
- Warm and personal, not clinical or robotic
- Specific to the woman's situation, not generic
- Culturally aware — you adjust tone based on cultural profile
- Action-oriented — you always tell her what to do next
- Honest — you do not minimise real risk

You always respond ONLY in valid JSON. No preamble, no explanation, no markdown.
"""


def _build_prompt(ml_result: dict, user_context: dict) -> str:
    """Build the main ML interpretation prompt."""

    # Cultural tone guidance based on AKU data
    cultural_tone = {
        "rural_conservative": "Use simple, direct language. Emphasise family and community support. Avoid medical jargon.",
        "mixed_transitional": "Balance warmth with clear information. She is comfortable with health information but values personal connection.",
        "urban_educated": "She understands health language. Be direct and evidence-based while remaining warm."
    }

    tone_guidance = cultural_tone.get(
        ml_result.get("cultural_profile", ""),
        "Use warm, clear language appropriate for a woman navigating grief and health concerns."
    )

    return f"""
A woman has just interacted with SafeMum AI. Here is the full picture.

MACHINE LEARNING ASSESSMENT:
- Risk level: {ml_result.get('risk_level', 'unknown')} (confidence: {ml_result.get('confidence', 0):.0%})
- Symptoms driving risk: {', '.join(ml_result.get('top_features', [])) or 'none flagged'}
- Repeat loss risk: {ml_result.get('repeat_risk', 'unknown')} (probability: {ml_result.get('repeat_probability', 0):.0%})
- Will seek care if referred: {ml_result.get('will_seek_care', True)} (probability: {ml_result.get('care_seeking_probability', 0.5):.0%})
- Care seeking recommendation: {ml_result.get('care_seeking_recommendation', 'send_reminder')}
- Social vulnerability: {ml_result.get('vulnerability_category', 'unknown')}
- Cultural profile: {ml_result.get('cultural_profile', 'unknown')}
- Isolation detected: {ml_result.get('isolation_detected', False)}
- In high-need service gap area: {ml_result.get('service_gap_area', False)}

USER CONTEXT:
- Name: {user_context.get('name', 'her')}
- Language: {user_context.get('language', 'en')}
- Days since loss: {user_context.get('days_since_loss', 'unknown')}
- Loss type: {user_context.get('loss_type', 'pregnancy loss')}
- Previous losses: {user_context.get('previous_losses', 0)}
- Mood trend: {user_context.get('mood_trend', 'unknown')}
- Check-in streak: {user_context.get('checkin_streak', 0)} days
- Reminders missed: {user_context.get('reminders_missed', 0)}
- Last active: {user_context.get('last_active', 'unknown')}
- CHW assigned: {user_context.get('has_chw', False)}
- Primary hospital set: {user_context.get('has_primary_hospital', False)}
- Emergency alerts triggered: {user_context.get('emergency_alerts_triggered', 0)}
- Community posts: {user_context.get('community_posts', 0)}
- App opens this week: {user_context.get('app_opens_this_week', 0)}
- County: {user_context.get('county', 'unknown')}

TONE GUIDANCE FOR THIS WOMAN: {tone_guidance}

Respond ONLY with a JSON object with these exact keys:
{{
  "chat_message": "what the AI assistant says to her right now — warm, personal, specific — max 3 sentences",
  "mascot_mood": "idle" or "happy" or "concerned" or "celebrating",
  "risk_summary": "plain English summary of her risk for the recovery overview card — max 15 words",
  "physical_status": "Stable" or "Monitored" or "At Risk",
  "emotional_status": "Stable" or "Monitored" or "At Risk",
  "trigger_emergency_alert": true or false,
  "assign_chw": true or false,
  "chw_assignment_reason": "why assign CHW or null",
  "refer_to_facility": true or false,
  "referral_urgency": "none" or "routine" or "urgent" or "emergency",
  "send_notification": true or false,
  "notification_message": "SMS or push notification text — max 20 words — or null",
  "weekly_tip_category": "physical" or "emotional" or "nutrition" or "danger_signs",
  "weekly_tip_reason": "why this tip category is most relevant right now"
}}
"""


def _chat_system_prompt(user_context: dict) -> str:
    """System prompt for the AI chat assistant."""

    cultural_map = {
        "rural_conservative": "Use simple, direct language. Reference family support. Avoid medical jargon.",
        "mixed_transitional": "Balance warmth with clear health information.",
        "urban_educated": "Be direct and evidence-based while remaining compassionate."
    }

    cultural_note = cultural_map.get(
        user_context.get("cultural_profile", ""),
        "Use warm, plain language."
    )

    return f"""
You are SafeMum AI, a compassionate maternal health companion for a woman who has 
experienced pregnancy loss. You are warm, personal, and never clinical or robotic.

About this woman:
- Name: {user_context.get('name', 'her')}
- Days since her loss: {user_context.get('days_since_loss', 'unknown')}
- Loss type: {user_context.get('loss_type', 'pregnancy loss')}
- Previous losses: {user_context.get('previous_losses', 0)}
- Mood trend: {user_context.get('mood_trend', 'unknown')}
- Vulnerability: {user_context.get('vulnerability_category', 'unknown')}
- Cultural profile: {user_context.get('cultural_profile', 'unknown')}
- Language: {user_context.get('language', 'en')}

Communication guidance: {cultural_note}

Rules:
- Never diagnose. Never say "you have X condition."
- When you detect danger signs, always tell her to seek care and tell her where.
- Never dismiss symptoms. Take everything seriously.
- Keep responses under 4 sentences unless she is asking a detailed question.
- If she expresses hopelessness or suicidal thoughts, acknowledge her, validate her feelings,
  and gently direct her to call her CHW or go to the nearest facility immediately.
- You are not a replacement for medical care. You are a companion that connects her to it.
"""


def _detect_actions_from_reply(reply: str, user_message: str) -> list:
    """
    Scan the AI reply and user message for keywords that trigger actions.
    Returns a list of action strings.
    """
    actions = []
    reply_lower = reply.lower()
    message_lower = user_message.lower()

    emergency_keywords = [
        "emergency", "immediately", "right now", "call for help",
        "go to a hospital", "go to the nearest", "do not wait"
    ]
    if any(k in reply_lower for k in emergency_keywords):
        actions.append("suggest_emergency_alert")

    facility_keywords = [
        "nearest facility", "clinic", "hospital", "go to", "find a"
    ]
    if any(k in reply_lower for k in facility_keywords):
        actions.append("suggest_map")

    checkin_keywords = [
        "how are you feeling", "check in", "tell me more"
    ]
    if any(k in reply_lower for k in checkin_keywords):
        actions.append("suggest_checkin")

    return actions


def _fallback_response(ml_result: dict, user_context: dict) -> dict:
    """Returned when Groq API fails."""
    risk = ml_result.get("risk_level", "low")
    return {
        "chat_message": "I am here with you. How are you feeling right now? Tell me what has been on your mind.",
        "mascot_mood": "concerned" if risk == "high" else "idle",
        "risk_summary": "Monitoring your recovery",
        "physical_status": "Monitored" if risk == "high" else "Stable",
        "emotional_status": "Monitored",
        "trigger_emergency_alert": risk == "high" and ml_result.get("confidence", 0) > 0.9,
        "assign_chw": ml_result.get("vulnerability_category") == "high",
        "chw_assignment_reason": "High vulnerability detected" if ml_result.get("vulnerability_category") == "high" else None,
        "refer_to_facility": risk in ["high", "urgent"],
        "referral_urgency": "urgent" if risk == "high" else "none",
        "send_notification": False,
        "notification_message": None,
        "weekly_tip_category": "emotional",
        "weekly_tip_reason": "Default to emotional support during system unavailability"
    }


def _symptom_fallback(symptoms: list) -> dict:
    """Returned when symptom interpretation fails."""
    high_risk = any(s in symptoms for s in [
        "Heavy bleeding", "Chest pain", "Severe pain", "Dizziness", "Cold hands or feet"
    ])
    return {
        "risk_level": "urgent" if high_risk else "monitor",
        "title": "Please check in with a healthcare provider",
        "message": "Some of what you are experiencing needs attention. Please reach out to a health facility or your CHW today.",
        "chat_message": "I noticed some concerning symptoms in what you shared. Please do not ignore these — let me help you find the right care.",
        "action": "find_facility" if high_risk else "talk_to_ai",
        "trigger_emergency_alert": False,
        "assign_chw": True,
        "chw_reason": "Symptom check triggered CHW follow-up"
    }