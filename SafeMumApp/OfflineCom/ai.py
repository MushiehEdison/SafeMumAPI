"""
SafeMum AI — ai_assistant.py
Healia: maternal health companion for SafeMum.
Emotionally intelligent. Clinically aware. Multilingual. Channel-aware.
Learns from user history, DB profile, and clinical dataset.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# Fast model for USSD (prevents timeout), quality model for voice/chat
_USSD_MODEL  = "llama-3.1-8b-instant"
_VOICE_MODEL = GROQ_MODEL
_CHAT_MODEL  = GROQ_MODEL

CONTEXT_WINDOW       = 12
MEMORY_REBUILD_EVERY = 10
USSD_CHAR_LIMIT      = 155


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

LANG_INDICATORS = {
    "fr": ["je","tu","il","elle","nous","vous","les","une","des","et","est","suis","bonjour","merci","oui","non","avec","pour","dans","sur","pas","mais","très","bien","ça","douleur","fièvre","mal","bébé","enceinte","grossesse","bonne"],
    "pt": ["eu","você","ele","ela","nós","eles","uma","para","com","não","sim","obrigada","obrigado","olá","bom","boa","estou","está","meu","minha","bebê","grávida","gravidez","dor","febre","sangramento","saúde"],
    "sw": ["mimi","wewe","yeye","sisi","ninyi","wao","na","kwa","ya","ni","hapana","ndiyo","asante","habari","nzuri","mtoto","mimba","maumivu","homa","damu","afya","mama"],
}

def detect_language(text: str) -> str:
    if not text or not isinstance(text, str): return "en"
    words = re.findall(r"\b\w+\b", text.lower())
    if not words: return "en"
    scores = {lang: sum(1 for w in words if w in indicators) / len(words)
              for lang, indicators in LANG_INDICATORS.items()}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    return best_lang if best_score >= 0.15 else "en"


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION & INTENT DETECTION — per message, zero LLM cost
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONAL_PATTERNS = {
    "grief":    r"\b(loss|lost|baby|child|miscarriage|stillborn|gone|died|death|grieving|grief|kid|perda|bebê|perte|bébé|mtoto|hasara)\b",
    "fear":     r"\b(scared|afraid|terrified|worried|fear|panic|anxious|peur|inquiet|medo|assustada|woga|hofu)\b",
    "pain":     r"\b(pain|hurt|ache|hurts|bleeding|fever|burning|cramp|headache|dizzy|nausea|douleur|fièvre|dor|febre|maumivu|homa)\b",
    "hopeless": r"\b(hopeless|give up|can't go on|no point|end it|not worth|want to die|désespoir|abandonner|sem esperança|desistir|kukata tamaa)\b",
    "lonely":   r"\b(alone|lonely|no one|nobody|isolated|abandoned|seul|isolé|sozinha|abandonada|peke yangu)\b",
    "positive": r"\b(better|good|great|happy|okay|fine|improving|hopeful|well|grateful|mieux|bien|melhor|bem|vizuri|nzuri)\b",
    "casual":   r"\b(hi|hello|hey|how are you|bonjour|salut|olá|oi|habari|mambo|hujambo)\b",
    "question": r"\b(what|how|why|when|where|can you|tell me|explain|should|quoi|comment|pourquoi|o que|como|por que|nini|jinsi|kwa nini)\b",
    "anger":    r"\b(angry|frustrated|annoyed|upset|furious|fed up|énervé|frustré|com raiva|frustrada|hasira|uchovu)\b",
}

DANGER_SIGNS = [
    "heavy bleeding","chest pain","can't breathe","difficulty breathing",
    "severe pain","very high fever","foul discharge","unconscious",
    "saignement abondant","douleur thoracique","sangramento intenso",
    "dor no peito","damu nyingi","maumivu makali",
]

CRISIS_SIGNS = [
    "want to die","end it","no point","give up on life","harm myself",
    "suicid","mourir","je veux mourir","quero morrer","kujiua",
]

MODERATE_KEYWORDS = [
    "bleeding","fever","saignement","fièvre","sangramento","febre",
    "infection","infection","infecção","pain","douleur","dor",
    "smell","odeur","cheiro","discharge","pertes","corrimento",
]

GRIEF_KEYWORDS = [
    "grief","sad","crying","tears","mourning","deuil","triste","luto",
    "huzuni","machofu","kilio",
]

def detect_emotion(text: str) -> str:
    t = text.lower()
    for emotion, pattern in EMOTIONAL_PATTERNS.items():
        if re.search(pattern, t): return emotion
    return "neutral"

def detect_danger(text: str) -> bool:
    return any(d in text.lower() for d in DANGER_SIGNS)

def detect_crisis(text: str) -> bool:
    return any(c in text.lower() for c in CRISIS_SIGNS)

def detect_grief(text: str) -> bool:
    return any(g in text.lower() for g in GRIEF_KEYWORDS)

def risk_level(text: str) -> str:
    lowered = text.lower()
    if any(d in lowered for d in DANGER_SIGNS): return "high"
    if any(m in lowered for m in MODERATE_KEYWORDS): return "medium"
    return "low"

def message_weight(text: str) -> str:
    n = len(text.split())
    if n <= 4: return "very_short"
    if n <= 12: return "short"
    if n <= 35: return "medium"
    return "long"


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_PATTERNS = [
    r"\b(pain|hurt|ache|sore|burning|cramping|douleur|mal|brûlure|dor|maumivu)\b",
    r"\b(fever|temperature|chills|sweating|fièvre|febre|homa)\b",
    r"\b(nausea|vomit|dizzy|headache|migraine|nausée|vertige|náusea|kizunguzungu)\b",
    r"\b(cough|sneeze|sore throat|toux|gorge|tosse|kukohoa)\b",
    r"\b(tired|fatigue|exhausted|weak|fatigué|cansada|uchovu)\b",
    r"\b(sleep|insomnia|nightmare|sommeil|insomnie|usingizi)\b",
    r"\b(bleeding|discharge|spotting|saignement|perda de sangue|damu)\b",
    r"\b(breathing|shortness|chest|respiration|essoufflement|kupumua)\b",
    r"\b(swollen|rash|itchy|rouge|gonflé|coceira|uvimbe)\b",
]

CONDITION_PATTERNS = [
    r"\b(diabetes|hypertension|blood pressure|diabète|diabete|kisukari)\b",
    r"\b(malaria|typhoid|paludisme|malária|malaria)\b",
    r"\b(asthma|bronchitis|pneumonia|asthme|pneumonie|pumu)\b",
    r"\b(depression|anxiety|stress|dépression|ansiedade|msongo)\b",
    r"\b(pregnancy|pregnant|expecting|grossesse|enceinte|gravidez|grávida|mimba)\b",
    r"\b(miscarriage|stillbirth|fausse couche|aborto|kuharibika kwa mimba)\b",
    r"\b(c-section|cesarean|césarienne|cesariana|upasuaji)\b",
    r"\b(anemia|anémie|anemia)\b",
    r"\b(preeclampsia|eclampsia|pré-éclampsie|pré-eclâmpsia)\b",
]

def extract_entities(text: str):
    topics, symptoms, conditions = set(), set(), set()
    t = text.lower()
    for p in SYMPTOM_PATTERNS:
        m = re.findall(p, t)
        symptoms.update(m)
        if m: topics.add("symptoms")
    for p in CONDITION_PATTERNS:
        m = re.findall(p, t)
        conditions.update(m)
        if m: topics.add("medical_conditions")
    kw_topics = {
        "nutrition": ["diet","food","eat","nutrition","manger","nourriture","comer","chakula"],
        "physical_activity": ["exercise","sport","fitness","activité","exercício","mazoezi"],
        "mental_health": ["mental","stress","anxiety","grief","émotionnel","emocional","afya ya akili"],
        "medication": ["medicine","medication","drug","traitement","médicament","medicamento","dawa"],
        "prenatal": ["prenatal","antenatal","pregnancy","grossesse","gravidez","mimba"],
        "postnatal": ["postnatal","postpartum","après accouchement","pós-parto","baada ya kujifungua"],
    }
    for topic, keywords in kw_topics.items():
        if any(k in t for k in keywords): topics.add(topic)
    return topics, symptoms, conditions


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL DATASET + MATRIX FACTORIZATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "clinical_summaries.csv")
dataset_df         = pd.DataFrame()
symptom_embeddings = None
condition_embeddings = None
record_embeddings  = None
symptom_to_idx     = {}
condition_to_idx   = {}

def _train_mf_model(df, n_components=10):
    global symptom_embeddings, condition_embeddings, record_embeddings, symptom_to_idx, condition_to_idx
    if df.empty: return
    all_symptoms, all_conditions = set(), set()
    for _, row in df.iterrows():
        text = f"{row.get('summary_text','')} {row.get('diagnosis','')}".lower()
        _, syms, conds = extract_entities(text)
        all_symptoms.update(syms); all_conditions.update(conds)
    symptom_to_idx = {s: i for i, s in enumerate(sorted(all_symptoms))}
    condition_to_idx = {c: i for i, c in enumerate(sorted(all_conditions))}
    n_rec, n_sym, n_con = len(df), len(symptom_to_idx), len(condition_to_idx)
    RS = np.zeros((n_rec, n_sym)); RC = np.zeros((n_rec, n_con))
    for idx, row in df.iterrows():
        text = f"{row.get('summary_text','')} {row.get('diagnosis','')}".lower()
        _, syms, conds = extract_entities(text)
        for s in syms:
            if s in symptom_to_idx: RS[idx, symptom_to_idx[s]] = 1
        for c in conds:
            if c in condition_to_idx: RC[idx, condition_to_idx[c]] = 1
    M = np.hstack([RS, RC])
    try:
        U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
        U = U[:, :n_components]; S = np.diag(sigma[:n_components]); Vt = Vt[:n_components, :]
        record_embeddings = U @ S
        feat = Vt.T
        symptom_embeddings = feat[:n_sym, :]; condition_embeddings = feat[n_sym:, :]
        print(f"[healia] MF model trained — {n_rec} records, {n_components} components")
    except Exception as e:
        print(f"[healia] MF training failed: {e}")

try:
    if os.path.exists(DATASET_PATH):
        _df = pd.read_csv(DATASET_PATH, encoding="utf-8", on_bad_lines="warn")
        required = ["summary_id","patient_id","patient_age","patient_gender","diagnosis","body_temp_c","blood_pressure_systolic","heart_rate","summary_text","date_recorded"]
        missing = [c for c in required if c not in _df.columns]
        if not missing:
            dataset_df = _df.dropna(subset=required)
            _train_mf_model(dataset_df)
        else:
            print(f"[healia] Dataset missing columns: {missing}")
    else:
        print("[healia] No clinical dataset found — proceeding without it")
except Exception as e:
    print(f"[healia] Dataset load error: {e}")

def _query_dataset(symptoms: set, conditions: set, max_records=3) -> list:
    if dataset_df.empty or symptom_embeddings is None: return []
    n_sym, n_con = len(symptom_to_idx), len(condition_to_idx)
    qv = np.zeros(n_sym + n_con)
    for s in symptoms:
        if s in symptom_to_idx: qv[symptom_to_idx[s]] = 1
    for c in conditions:
        if c in condition_to_idx: qv[condition_to_idx[c] + n_sym] = 1
    try:
        feat = np.vstack([symptom_embeddings, condition_embeddings]); qe = qv @ feat
    except Exception:
        return []
    sims = []
    for idx in range(len(dataset_df)):
        re_ = record_embeddings[idx]; nq = np.linalg.norm(qe); nr = np.linalg.norm(re_)
        sim = float(np.dot(qe, re_) / (nq * nr)) if nq > 0 and nr > 0 else 0
        sims.append((idx, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    records = []
    for idx, sim in sims[:max_records]:
        if sim <= 0: continue
        row = dataset_df.iloc[idx]
        records.append({"diagnosis": row.get("diagnosis","N/A"), "summary": str(row.get("summary_text",""))[:200], "age": row.get("patient_age","N/A"), "gender": row.get("patient_gender","N/A")})
    return records

def _should_use_clinical_data(symptoms, conditions, user_input: str, depth: int) -> bool:
    t = user_input.lower()
    skip = ["hi","hello","hey","thank","bye","merci","bonjour","salut","obrigada","habari","asante"]
    if any(w in t for w in skip): return False
    medical_kw = ["what is","how to","treatment","symptoms of","cause","explain","qu'est-ce","comment","traitement","symptômes","o que é","como tratar","nini","jinsi ya","matibabu"]
    return bool(symptoms) or bool(conditions) or any(k in t for k in medical_kw)


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CONVERSATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class SessionMemory:
    def __init__(self):
        self.depth = 0; self.emotion_history = []; self.symptoms_seen = set()
        self.conditions_seen = set(); self.topics_seen = set(); self.language = "en"
    def update(self, emotion, topics, symptoms, conditions, lang):
        self.depth += 1; self.emotion_history.append(emotion)
        self.topics_seen.update(topics); self.symptoms_seen.update(symptoms)
        self.conditions_seen.update(conditions); self.language = lang
    def recent_emotions(self, n=4): return self.emotion_history[-n:]

session_store: dict = defaultdict(SessionMemory)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE-SPECIFIC LABELS
# ─────────────────────────────────────────────────────────────────────────────

_LANG_LABELS = {
    "en": {"name": "Healia", "role": "maternal health companion"},
    "fr": {"name": "Healia", "role": "accompagnatrice en santé maternelle"},
    "pt": {"name": "Healia", "role": "acompanhante de saúde materna"},
    "sw": {"name": "Healia", "role": "msaidizi wa afya ya uzazi"},
}

_PHASE_NOTES = {
    "en": {"early_acute":"She is in the first two weeks after loss. She needs presence above all.","processing":"She is living with what happened. Be real, be steady.","rebuilding":"She is finding her footing again. Acknowledge small wins.","stabilised":"She has reached a more stable place. Look forward gently."},
    "fr": {"early_acute":"Elle est dans les deux premières semaines après la perte.","processing":"Elle vit avec ce qui s'est passé.","rebuilding":"Elle reprend pied.","stabilised":"Elle est dans un endroit plus stable."},
    "pt": {"early_acute":"Ela está nas primeiras duas semanas após a perda.","processing":"Ela está vivendo com o que aconteceu.","rebuilding":"Ela está encontrando seu caminho.","stabilised":"Ela chegou a um lugar mais estável."},
    "sw": {"early_acute":"Yuko katika wiki mbili za kwanza baada ya kupoteza.","processing":"Anaishi na kilichotokea.","rebuilding":"Anaanza kupata nguvu tena.","stabilised":"Amefika mahali salama zaidi."},
}

_EMOTION_GUIDANCE = {
    "en": {"grief":"She is in grief. Sit with her.","fear":"She is scared. Name that first.","pain":"Physical symptom reported. Take it seriously.","hopeless":"She is losing hope. Full presence needed.","lonely":"She feels alone. Make her feel seen.","positive":"She is in a good place. Meet her there.","casual":"She is greeting. Keep it natural.","question":"She has a question. Answer directly.","anger":"She is frustrated. Acknowledge it.","neutral":"No strong signal. Respond naturally."},
    "fr": {"grief":"Elle est en deuil. Reste avec elle.","fear":"Elle a peur. Nomme cela.","pain":"Symptôme physique. Prends-le au sérieux.","hopeless":"Elle perd espoir. Présence totale.","lonely":"Elle se sent seule. Fais-la se sentir vue.","positive":"Elle va bien aujourd'hui. Rejoins-la.","casual":"Elle dit bonjour. Reste naturelle.","question":"Elle a une question. Réponds directement.","anger":"Elle est frustrée. Reconnais-le.","neutral":"Pas de signal fort. Réponds naturellement."},
    "pt": {"grief":"Ela está de luto. Fique com ela.","fear":"Ela está com medo. Reconheça isso.","pain":"Sintoma físico. Leve a sério.","hopeless":"Ela está perdendo esperança.","lonely":"Ela se sente sozinha.","positive":"Ela está bem hoje.","casual":"Ela está cumprimentando.","question":"Ela tem uma pergunta.","anger":"Ela está frustrada.","neutral":"Nenhum sinal forte."},
    "sw": {"grief":"Yuko katika huzuni. Kaa naye.","fear":"Anaogopa. Tambua hilo.","pain":"Dalili ya kimwili. Ichukue kwa uzito.","hopeless":"Anapoteza matumaini.","lonely":"Anahisi upweke.","positive":"Yuko mahali pazuri leo.","casual":"Anasalimia tu.","question":"Ana swali. Jibu moja kwa moja.","anger":"Amekasirika. Tambua.","neutral":"Hakuna ishara kali."},
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(memory, user_id, db_session, lang="en", emotion="neutral", is_danger=False, is_crisis=False, weight="short", sess=None, clinical_records=None, channel="chat") -> str:
    from ..models import User, CHWCase

    user = db_session.query(User).get(user_id)
    name = user.name if user else "her"

    has_chw = db_session.query(CHWCase).filter_by(patient_id=user_id, status="assigned").first() is not None

    mc = memory.to_context_dict() if memory else {}
    memory_summary = mc.get("memory_summary") or "Still getting to know her."
    recovery_phase = mc.get("recovery_phase") or "processing"
    cultural = mc.get("cultural_profile") or "mixed_transitional"
    low_moods = mc.get("consecutive_low_moods") or 0
    vulnerability = mc.get("vulnerability_level") or "medium"
    days_since = mc.get("days_since_loss", "unknown")
    loss_type = mc.get("loss_type", "pregnancy loss")
    prev_losses = mc.get("previous_losses", 0)

    all_fragments = mc.get("things_she_shared") or []
    PHYSICAL_FRAGS = {"Reported bleeding","Reported pain","Reported fever","Reported headaches","Mentioned a hospital visit","Mentioned a doctor"}
    fragments = [f for f in all_fragments if f not in PHYSICAL_FRAGS] if emotion not in ("pain",) else all_fragments

    tone_map = {
        "rural_conservative": {"en":"Traditional background. Family matters. Keep language simple and warm.","fr":"Milieu traditionnel. Famille importante.","pt":"Meio tradicional. Família importa.","sw":"Mazingira ya kimapokeo. Familia ni muhimu."},
        "mixed_transitional": {"en":"Balances tradition and modern life. Be warm and clear.","fr":"Équilibre tradition et moderne.","pt":"Equilibra tradição e moderno.","sw":"Sawazisha mila na kisasa."},
        "urban_educated": {"en":"Comfortable with health info. Be direct, informed, human.","fr":"À l'aise avec l'info santé.","pt":"Confortável com info de saúde.","sw":"Yupo tayari na taarifa za afya."},
    }
    tone_note = tone_map.get(cultural, tone_map["mixed_transitional"]).get(lang, tone_map["mixed_transitional"]["en"])

    phase_note = _PHASE_NOTES.get(lang, _PHASE_NOTES["en"]).get(recovery_phase, "")
    emotion_note = _EMOTION_GUIDANCE.get(lang, _EMOTION_GUIDANCE["en"]).get(emotion, "")
    weight_note = {"very_short":"Match her brevity — 1-2 sentences.","short":"Keep it brief.","medium":"Give it space.","long":"Be fully present."}.get(weight, "")

    depth = sess.depth if sess else 0
    depth_note = "First exchange — introduce yourself and invite her to share." if depth <= 1 else "Getting to know each other. Be warm and curious." if depth < 5 else "Real history now. Reference what she shared like a real person."

    if sess and len(sess.emotion_history) >= 3:
        recent_emos = sess.recent_emotions(3)
        if all(e in ("grief","hopeless","lonely","fear") for e in recent_emos):
            depth_note += " She has been heavy across multiple messages — be especially gentle."

    clinical_note = ""
    if clinical_records:
        summaries = "\n".join([f"- {r['diagnosis']}: {r['summary']}" for r in clinical_records])
        clinical_note = f"\n━━━ RELEVANT CLINICAL CONTEXT ━━━\n{summaries}\nUse this to inform — do not quote directly."

    alerts = ""
    if is_danger: alerts += "\n🚨 DANGER SIGNS: Tell her exactly what to do in the next 30 minutes. Direct. Do not soften."
    if is_crisis: alerts += "\n🚨 CRISIS SIGNAL: She may be in emotional crisis. Stay with her. Get her to real help."
    if low_moods >= 3: alerts += f"\n⚠️ {low_moods} consecutive low mood days. Be especially present."
    if vulnerability == "high" and not has_chw: alerts += "\n⚠️ High vulnerability, no CHW. Recommend CHW contact if in crisis."

    # Channel-specific instructions
    channel_instructions = {
        "ussd": "You are responding via USSD on a basic phone. 155 characters MAX. Plain text only. No markdown, bullets, emojis. Be warm but incredibly concise. Every character counts.",
        "voice": "You are speaking on a voice call. 2-3 short spoken sentences. Natural pauses. Numbers as words: 'zero eight hundred' not '0800'. Calm, never rushed.",
        "chat": "You are in a chat interface. 2-4 sentences. Can use 1 emoji if natural. Be warm and present.",
    }
    channel_note = channel_instructions.get(channel, channel_instructions["chat"])

    lang_instruction = {"en":"Respond in English.","fr":"Réponds en français.","pt":"Responda em português.","sw":"Jibu kwa Kiswahili."}.get(lang, "Respond in English.")

    return f"""You are Healia — a maternal health companion at SafeMum. You support women through pregnancy, pregnancy loss, and recovery.
You are not an AI assistant. You are Healia — a real companion who knows {name}.

{lang_instruction}
{channel_note}

━━━ WHO SHE IS ━━━
Name: {name} | Days since loss: {days_since} | Loss type: {loss_type} | Previous losses: {prev_losses}
Recovery phase: {recovery_phase} | Vulnerability: {vulnerability} | CHW assigned: {has_chw}

━━━ WHAT YOU KNOW ━━━
{memory_summary}

━━━ THINGS SHE HAS SHARED ━━━
{chr(10).join(f"- {f}" for f in fragments) if fragments else "- Still getting to know her"}
{alerts}
{clinical_note}

━━━ CONTEXT ━━━
{tone_note} | {phase_note} | {emotion_note} | {weight_note} | {depth_note}

Use her name at most once per conversation. Never ask more than one question per reply.
Lead with action or acknowledgement — questions after, not instead.
For physical symptoms: tell her what to do first, then one follow-up if needed.
Never make this feel like an intake form."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT (chat + voice)
# ─────────────────────────────────────────────────────────────────────────────

def chat(user_message: str, user_id: int, db_session, channel: str = "chat") -> dict:
    from ..models import Conversation, AIMemory, User, MedicalProfile, Pregnancy
    from .classifier import classify_risk
    from .context_builder import get_user_context

    _, symptoms_in_msg, _ = extract_entities(user_message)
    ml_risk = {"risk_level":"low","confidence":0.0,"top_features":[]}
    if symptoms_in_msg or detect_danger(user_message):
        try:
            user_context = get_user_context(user_id, db_session)
            symptom_dict = {s.lower().replace(" ","_"):1 for s in symptoms_in_msg}
            symptom_dict.update({"pds101":user_context.get("age",25),"pds102":user_context.get("urban_rural","Urban"),"pds201":user_context.get("previous_pregnancies",0),"pds202":user_context.get("previous_losses",0),"county":user_context.get("county","Unknown")})
            ml_risk = classify_risk(symptom_dict)
        except Exception as e:
            print(f"[healia] Classifier error: {e}")

    conversation = db_session.query(Conversation).filter_by(user_id=user_id, type="health_assistant").order_by(Conversation.created_at.desc()).first()
    if not conversation:
        conversation = Conversation(user_id=user_id, type="health_assistant", messages=[])
        db_session.add(conversation); db_session.flush()

    messages = list(conversation.messages or [])

    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id); db_session.add(memory); db_session.flush()
        _seed_memory_from_profile(memory, user_id, db_session)

    sess = session_store[user_id]
    lang = detect_language(user_message)
    emotion = detect_emotion(user_message)
    is_danger = detect_danger(user_message)
    is_crisis = detect_crisis(user_message)
    weight = message_weight(user_message)
    topics, symptoms, conditions = extract_entities(user_message)
    sess.update(emotion, topics, symptoms, conditions, lang)

    clinical_records = []
    if _should_use_clinical_data(sess.symptoms_seen, sess.conditions_seen, user_message, sess.depth):
        clinical_records = _query_dataset(sess.symptoms_seen, sess.conditions_seen)

    system = _build_system_prompt(memory=memory, user_id=user_id, db_session=db_session, lang=lang, emotion=emotion, is_danger=is_danger, is_crisis=is_crisis, weight=weight, sess=sess, clinical_records=clinical_records, channel=channel)

    if is_danger or (ml_risk.get("risk_level")=="high" and ml_risk.get("confidence",0)>0.6):
        system += f"\n\n🚨 CLINICAL OVERRIDE: Danger sign reported. ML: risk={ml_risk.get('risk_level')}, confidence={ml_risk.get('confidence',0):.0%}. Tell her exactly what to do now. One clear instruction. Then one question only if it helps her act."

    model = _USSD_MODEL if channel == "ussd" else (_VOICE_MODEL if channel == "voice" else _CHAT_MODEL)
    max_tokens = 55 if channel == "ussd" else (130 if channel == "voice" else 650)
    temperature = 0.4 if channel == "ussd" else (0.5 if channel == "voice" else 0.78)

    groq_msgs = [{"role":"system","content":system}]
    recent = messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages
    for m in recent:
        if isinstance(m, dict) and "role" in m and "content" in m:
            groq_msgs.append({"role":m["role"],"content":m["content"]})
    groq_msgs.append({"role":"user","content":user_message})

    try:
        resp = client.chat.completions.create(model=model, messages=groq_msgs, temperature=temperature, max_tokens=max_tokens)
        reply = resp.choices[0].message.content.strip()
        if channel == "ussd":
            reply = reply.replace("*","").replace("#","").replace("-","")[:USSD_CHAR_LIMIT]
    except Exception as e:
        print(f"[healia] Groq error: {e}")
        reply = _fallback_reply(lang)

    actions = _detect_actions(reply, user_message, is_danger, is_crisis)

    messages.append({"role":"user","content":user_message,"timestamp":_now()})
    messages.append({"role":"assistant","content":reply,"timestamp":_now()})
    conversation.messages = messages; conversation.updated_at = datetime.utcnow()
    _extract_memory_fragments(user_message, memory)

    memory_updated = False
    if len(messages) % MEMORY_REBUILD_EVERY == 0:
        _rebuild_memory_summary(memory, messages[-30:]); memory_updated = True

    db_session.commit()
    return {"reply":reply,"actions":actions,"memory_updated":memory_updated}


# ─────────────────────────────────────────────────────────────────────────────
# USSD
# ─────────────────────────────────────────────────────────────────────────────

def ask_ussd(user_message: str, history: list[dict], topic: str = "general") -> str:
    system = _USSD_SYSTEM_PROMPT
    if topic == "grief":
        system += "\n\nThis woman is reaching out for grief support after pregnancy loss. Be present first. Don't rush to medical advice."

    risk = risk_level(user_message)
    risk_hint = {"high":"[Risk: HIGH — urgent but proportionate]","medium":"[Risk: MEDIUM — advise clinic soon]","low":"[Risk: LOW — reassure warmly]"}[risk]

    messages = history + [{"role":"user","content":f"{risk_hint}\n{user_message}"}]

    try:
        response = client.chat.completions.create(
            model=_USSD_MODEL, max_tokens=55, temperature=0.4,
            messages=[{"role":"system","content":system}, *messages],
        )
        reply = response.choices[0].message.content.strip()
        reply = reply.replace("*","").replace("#","").replace("-","")
        return reply[:USSD_CHAR_LIMIT]
    except Exception as e:
        print(f"[healia] USSD error: {e}")
        return "Service unavailable. / Service indisponible. / Serviço indisponível."


_USSD_SYSTEM_PROMPT = """You are SafeMum, a maternal health assistant for women who experienced pregnancy loss. You respond via USSD on a basic phone.

LANGUAGE: Detect from user input. Respond in English, French, or Portuguese only.

STRICT RULES:
1. 155 characters max — hard limit.
2. Plain text only. No markdown, bullets, asterisks, emojis.
3. Assess risk proportionally — do NOT default to emergency.
4. LOW risk (grief, mild discomfort) → warm advice, suggest clinic when convenient.
5. MEDIUM risk (fever, mild bleeding, moderate pain) → advise clinic within 24hrs.
6. HIGH risk (heavy bleeding, sepsis signs, loss of consciousness, severe pain) → urgent care.
7. Never diagnose. Be warm, human, concise.
8. For grief: be empathetic first. Acknowledge the loss. Don't rush to medical advice."""


# ─────────────────────────────────────────────────────────────────────────────
# VOICE
# ─────────────────────────────────────────────────────────────────────────────

def ask_voice(user_speech: str, history: list[dict], topic: str = "general") -> str:
    system = _VOICE_SYSTEM_PROMPT
    if topic == "grief":
        system += "\n\nThis woman is calling for grief support. Be present first. Let her feel heard."

    risk = risk_level(user_speech)
    risk_hint = {"high":"[Risk: HIGH]","medium":"[Risk: MEDIUM — suggest clinic soon]","low":"[Risk: LOW — reassure warmly]"}[risk]

    messages = history + [{"role":"user","content":f"{risk_hint}\n{user_speech}"}]

    try:
        response = client.chat.completions.create(
            model=_VOICE_MODEL, max_tokens=130, temperature=0.5,
            messages=[{"role":"system","content":system}, *messages],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Voice error: {e}")
        return "I am sorry, I am having trouble right now. Please call zero eight hundred, seven two three, two five three for free support."


_VOICE_SYSTEM_PROMPT = """You are SafeMum, a compassionate maternal health assistant on a voice helpline for women who experienced pregnancy loss.

LANGUAGE: Detect from speech. Respond in English, French, or Portuguese.

RULES:
1. 2-3 short spoken sentences per response. Natural pauses.
2. Assess risk proportionally — don't escalate unless truly severe.
3. Light symptoms → warm advice + clinic suggestion. Moderate → clinic soon. Severe → urgent referral.
4. For grief: be patient, warm, empathetic. Let her feel heard first.
5. Never diagnose. Encourage professional support.
6. Speak numbers as words: "zero eight hundred" not "0800".
7. Be calm, never rushed. She may be grieving."""


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_symptoms(selected_symptoms: list, user_id: int, ml_risk: dict, db_session) -> dict:
    from ..models import AIMemory
    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    memory_ctx = memory.to_context_dict() if memory else {}

    COMPLICATION_MAP = {
        "Heavy bleeding":"potential haemorrhage","Fever":"potential infection/sepsis",
        "Severe pain":"potential incomplete abortion","Chest pain":"potential pulmonary embolism",
        "Cold hands/feet":"potential shock","Dizziness":"potential haemorrhagic shock",
        "Foul discharge":"potential pelvic infection","Wound pain":"potential surgical site infection",
        "Difficulty breathing":"potential pulmonary complication",
    }
    clinical_flags = [COMPLICATION_MAP[s] for s in selected_symptoms if s in COMPLICATION_MAP]

    prompt = f"""You are Healia from SafeMum. A woman checked her symptoms.
Profile: {json.dumps(memory_ctx, indent=2)}
Symptoms: {', '.join(selected_symptoms) if selected_symptoms else 'None'}
ML risk: level={ml_risk.get('risk_level','unknown')}, confidence={ml_risk.get('confidence',0):.0%}
Clinical significance: {', '.join(clinical_flags) if clinical_flags else 'No high-risk flags'}

JSON only:
{{"risk_level":"emergency|urgent|monitor|stable","title":"max 8 words","message":"2 sentences specific to her","reply":"Healia speaks — warm, direct, 2-3 sentences","action":"emergency_alert|open_map|talk_to_healia|rest_and_monitor","trigger_emergency_alert":bool,"assign_chw":bool,"map_action":{{"filter":"post_loss_care|emergency|nearest","reason":"one sentence"}} or null}}

Rules: emergency=heavy bleeding OR chest pain OR cold+dizziness. urgent=fever+foul discharge OR severe pain alone. monitor=single mild symptom. stable=no concerning symptoms. Never say "see a doctor" — say exactly what to do next hour."""

    try:
        r = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role":"system","content":"Clinical maternal health AI. JSON only."},{"role":"user","content":prompt}], temperature=0.3, max_tokens=600)
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Symptom error: {e}")
        high = any(s in selected_symptoms for s in ["Heavy bleeding","Chest pain","Cold hands/feet","Dizziness"])
        return {"risk_level":"urgent" if high else "monitor","title":"Please get checked today","message":"Some of what you are experiencing needs attention. Do not wait.","reply":"I noticed some symptoms that concern me. Let me help you find the nearest facility right now.","action":"open_map" if high else "talk_to_healia","trigger_emergency_alert":high,"assign_chw":True,"map_action":{"filter":"emergency","reason":"Urgent symptoms detected"} if high else None}


# ─────────────────────────────────────────────────────────────────────────────
# MOOD CHECK-IN
# ─────────────────────────────────────────────────────────────────────────────

def interpret_checkin(mood_score: int, mood_label: str, notes: str, user_id: int, db_session) -> dict:
    from ..models import AIMemory
    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id); db_session.add(memory); db_session.flush()

    memory.last_mood_score = mood_score; memory.total_checkins = (memory.total_checkins or 0) + 1
    if mood_score <= 2: memory.consecutive_low_moods = (memory.consecutive_low_moods or 0) + 1; memory.checkin_streak = 0
    else: memory.consecutive_low_moods = 0; memory.checkin_streak = (memory.checkin_streak or 0) + 1

    db_session.flush(); mc = memory.to_context_dict()

    prompt = f"""You are Healia from SafeMum. A woman completed her check-in.
Profile: {json.dumps(mc, indent=2)}
Mood: {mood_score}/5 ({mood_label}) | Note: "{notes or 'nothing written'}" | Consecutive low: {memory.consecutive_low_moods}

JSON only: {{"reply":"2-3 sentences — reference her note or history — 1 emoji only if natural","flag_for_counsellor":bool,"assign_chw":bool,"urgency":"none|low|high","follow_up_message":"what to check tomorrow"}}

flag_for_counsellor=true if consecutive_low>=3 or notes mention hopelessness/self-harm. assign_chw=true if mood_score==1 AND high vulnerability AND no CHW. urgency=high if self-harm mentioned."""

    try:
        r = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role":"system","content":"Compassionate maternal health companion. JSON only."},{"role":"user","content":prompt}], temperature=0.5, max_tokens=400)
        result = json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Check-in error: {e}")
        result = {"reply":"I see you showing up today, and that matters. Take it one moment at a time.","flag_for_counsellor":memory.consecutive_low_moods>=3,"assign_chw":False,"urgency":"low" if mood_score<=2 else "none","follow_up_message":"How are you feeling today?"}

    if result.get("flag_for_counsellor"): memory.flagged_for_counsellor = True
    db_session.commit()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE GAP BRIEFING
# ─────────────────────────────────────────────────────────────────────────────

def interpret_service_gaps(gap_data: dict) -> dict:
    prompt = f"""Health ministry briefing — SafeMum service gap analysis. Data: {json.dumps(gap_data, indent=2)}. High need_score=many patients, few facilities.
JSON only: {{"headline":"one sentence","top_priority_counties":["top 3"],"key_finding":"2 sentences","recommended_action":"one specific step","data_note":"one sentence on limitations"}}"""
    try:
        r = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role":"system","content":"Public health analyst. JSON only."},{"role":"user","content":prompt}], temperature=0.3, max_tokens=400)
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Service gap error: {e}")
        return {"headline":"Analysis unavailable","top_priority_counties":[],"key_finding":"Unable to generate insight.","recommended_action":"Review raw data.","data_note":"System error."}


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _seed_memory_from_profile(memory, user_id, db_session):
    from ..models import Pregnancy
    pregnancy = db_session.query(Pregnancy).filter_by(user_id=user_id).order_by(Pregnancy.created_at.desc()).first()
    if pregnancy:
        if pregnancy.status == "lost": memory.loss_type = "pregnancy loss"
        if pregnancy.created_at: memory.days_since_loss = (datetime.utcnow() - pregnancy.created_at).days
    days = memory.days_since_loss or 0
    if days <= 14: memory.recovery_phase = "early_acute"
    elif days <= 42: memory.recovery_phase = "processing"
    elif days <= 84: memory.recovery_phase = "rebuilding"
    else: memory.recovery_phase = "stabilised"

def _rebuild_memory_summary(memory, recent_messages: list):
    if not recent_messages: return
    convo = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_messages if isinstance(m, dict) and "role" in m and "content" in m])
    prompt = f"""Read this conversation between Healia (maternal health companion) and a woman. Write a short paragraph (max 5 sentences) capturing who she is: emotional state, what she shared, recurring concerns, how she communicates. One caring person briefing another. No clinical labels. No bullets.\n\n{convo}"""
    try:
        r = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=300)
        memory.memory_summary = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Memory rebuild error: {e}")

def _extract_memory_fragments(user_message: str, memory):
    msg = user_message.lower(); fragments = list(memory.things_she_shared or [])
    triggers = [("husband","Mentioned her husband"),("partner","Mentioned her partner"),("child","Mentioned having a child"),("son","Mentioned her son"),("daughter","Mentioned her daughter"),("mother","Mentioned her mother"),("work","Mentioned work"),("sleep","Mentioned sleep issues"),("guilt","Expressed guilt"),("afraid","Expressed fear"),("scared","Expressed being scared"),("alone","Expressed feeling alone"),("hope","Expressed hope"),("better","Said feeling better"),("hospital","Mentioned hospital"),("doctor","Mentioned doctor"),("bleeding","Reported bleeding"),("pain","Reported pain"),("fever","Reported fever"),("headache","Reported headaches"),("overthink","Mentioned overthinking"),("stress","Mentioned stress"),("faith","Mentioned faith"),("prayer","Mentioned prayer"),("pregnant","Mentioned being pregnant"),("pregnancy","Discussed pregnancy"),("nutrition","Asked about nutrition"),("exercise","Asked about exercise"),("prenatal","Asked about prenatal care")]
    for kw, note in triggers:
        if kw in msg and note not in fragments: fragments.append(note)
    memory.things_she_shared = fragments[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_actions(reply: str, user_message: str, is_danger=False, is_crisis=False) -> list:
    actions = []; rl = reply.lower()
    if is_danger or any(k in rl for k in ["go to the nearest","go now","do not wait","immediately","emergency"]):
        actions.append({"type":"suggest_emergency_alert"}); actions.append({"type":"open_map","filter":"emergency"})
    elif any(k in rl for k in ["nearest facility","find a clinic","health centre","hospital"]):
        actions.append({"type":"open_map","filter":"nearest"})
    elif any(k in rl for k in ["post-loss care","specialist","post loss"]):
        actions.append({"type":"open_map","filter":"post_loss_care"})
    if is_crisis or any(k in rl for k in ["your chw","community health worker","call your chw"]):
        actions.append({"type":"highlight_chw"})
    if any(k in rl for k in ["check in","log how you","how have you been feeling"]):
        actions.append({"type":"suggest_checkin"})
    if any(k in rl for k in ["reminder","appointment","don't forget","remember to"]):
        actions.append({"type":"suggest_reminder"})
    return actions


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_reply(lang: str) -> str:
    msgs = {"en":"I'm here with you. I had a small technical issue — could you send that again?","fr":"Je suis là avec toi. J'ai eu un petit problème technique — peux-tu renvoyer ça ?","pt":"Estou aqui com você. Tive um pequeno problema técnico — pode enviar novamente?","sw":"Niko hapa nawe. Nilikuwa na tatizo dogo la kiufundi — unaweza kutuma tena?"}
    return msgs.get(lang, msgs["en"])

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```"); raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"): raw = raw[4:]
    return raw.strip()

def _now() -> str:
    return datetime.utcnow().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — for routes to call directly
# ─────────────────────────────────────────────────────────────────────────────

def build_context_prefix(topic: str) -> str:
    contexts = {"health":"Woman is describing a health concern after pregnancy loss.","clinic":"Woman is trying to find a nearby health facility.","grief":"Woman is seeking emotional and grief support after pregnancy loss.","chw":"Woman wants to connect with a Community Health Worker."}
    return contexts.get(topic, "")

def is_emergency(text: str) -> bool:
    return detect_danger(text)