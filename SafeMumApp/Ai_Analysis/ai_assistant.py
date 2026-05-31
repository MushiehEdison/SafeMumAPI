"""
SafeMum AI — ai_assistant.py

Healia: maternal health companion for SafeMum.
Emotionally intelligent. Clinically aware. Multilingual.
Learns from the user's history, DB profile, and clinical dataset.

Languages: English, French, Portuguese, Swahili
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

CONTEXT_WINDOW       = 12
MEMORY_REBUILD_EVERY = 10


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

LANG_INDICATORS = {
    "fr": [
        "je", "tu", "il", "elle", "nous", "vous", "les", "une", "des", "et",
        "est", "suis", "bonjour", "merci", "oui", "non", "avec", "pour",
        "dans", "sur", "pas", "mais", "très", "bien", "ça", "douleur",
        "fièvre", "mal", "bébé", "enceinte", "grossesse", "bonne",
    ],
    "pt": [
        "eu", "você", "ele", "ela", "nós", "eles", "uma", "para", "com",
        "não", "sim", "obrigada", "obrigado", "olá", "bom", "boa", "estou",
        "está", "meu", "minha", "bebê", "grávida", "gravidez", "dor",
        "febre", "sangramento", "saúde",
    ],
    "sw": [
        "mimi", "wewe", "yeye", "sisi", "ninyi", "wao", "na", "kwa", "ya",
        "ni", "hapana", "ndiyo", "asante", "habari", "nzuri", "mtoto",
        "mimba", "maumivu", "homa", "damu", "afya", "mama",
    ],
}

def detect_language(text: str) -> str:
    if not text or not isinstance(text, str):
        return "en"
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "en"
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
    "heavy bleeding", "chest pain", "can't breathe", "difficulty breathing",
    "severe pain", "very high fever", "foul discharge", "unconscious",
    "saignement abondant", "douleur thoracique", "sangramento intenso",
    "dor no peito", "damu nyingi", "maumivu makali",
]

CRISIS_SIGNS = [
    "want to die", "end it", "no point", "give up on life", "harm myself",
    "suicid", "mourir", "je veux mourir", "quero morrer", "kujiua",
]

def detect_emotion(text: str) -> str:
    t = text.lower()
    for emotion, pattern in EMOTIONAL_PATTERNS.items():
        if re.search(pattern, t):
            return emotion
    return "neutral"

def detect_danger(text: str) -> bool:
    t = text.lower()
    return any(d in t for d in DANGER_SIGNS)

def detect_crisis(text: str) -> bool:
    t = text.lower()
    return any(c in t for c in CRISIS_SIGNS)

def message_weight(text: str) -> str:
    n = len(text.split())
    if n <= 4:   return "very_short"
    if n <= 12:  return "short"
    if n <= 35:  return "medium"
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
        "nutrition":        ["diet", "food", "eat", "nutrition", "manger", "nourriture", "comer", "chakula"],
        "physical_activity":["exercise", "sport", "fitness", "activité", "exercício", "mazoezi"],
        "mental_health":    ["mental", "stress", "anxiety", "grief", "émotionnel", "emocional", "afya ya akili"],
        "medication":       ["medicine", "medication", "drug", "traitement", "médicament", "medicamento", "dawa"],
        "prenatal":         ["prenatal", "antenatal", "pregnancy", "grossesse", "gravidez", "mimba"],
        "postnatal":        ["postnatal", "postpartum", "après accouchement", "pós-parto", "baada ya kujifungua"],
    }
    for topic, keywords in kw_topics.items():
        if any(k in t for k in keywords):
            topics.add(topic)
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
    global symptom_embeddings, condition_embeddings, record_embeddings
    global symptom_to_idx, condition_to_idx

    if df.empty:
        return

    all_symptoms, all_conditions = set(), set()
    for _, row in df.iterrows():
        text = f"{row.get('summary_text','')} {row.get('diagnosis','')}".lower()
        _, syms, conds = extract_entities(text)
        all_symptoms.update(syms)
        all_conditions.update(conds)

    symptom_to_idx   = {s: i for i, s in enumerate(sorted(all_symptoms))}
    condition_to_idx = {c: i for i, c in enumerate(sorted(all_conditions))}

    n_rec  = len(df)
    n_sym  = len(symptom_to_idx)
    n_con  = len(condition_to_idx)

    RS = np.zeros((n_rec, n_sym))
    RC = np.zeros((n_rec, n_con))

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
        U  = U[:, :n_components]
        S  = np.diag(sigma[:n_components])
        Vt = Vt[:n_components, :]
        record_embeddings    = U @ S
        feat                 = Vt.T
        symptom_embeddings   = feat[:n_sym, :]
        condition_embeddings = feat[n_sym:, :]
        print(f"[healia] MF model trained — {n_rec} records, {n_components} components")
    except Exception as e:
        print(f"[healia] MF training failed: {e}")


try:
    if os.path.exists(DATASET_PATH):
        _df = pd.read_csv(DATASET_PATH, encoding="utf-8", on_bad_lines="warn")
        required = [
            "summary_id","patient_id","patient_age","patient_gender",
            "diagnosis","body_temp_c","blood_pressure_systolic",
            "heart_rate","summary_text","date_recorded",
        ]
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
    if dataset_df.empty or symptom_embeddings is None:
        return []

    n_sym = len(symptom_to_idx)
    n_con = len(condition_to_idx)
    qv    = np.zeros(n_sym + n_con)

    for s in symptoms:
        if s in symptom_to_idx: qv[symptom_to_idx[s]] = 1
    for c in conditions:
        if c in condition_to_idx: qv[condition_to_idx[c] + n_sym] = 1

    try:
        feat = np.vstack([symptom_embeddings, condition_embeddings])
        qe   = qv @ feat
    except Exception:
        return []

    sims = []
    for idx in range(len(dataset_df)):
        re_ = record_embeddings[idx]
        nq  = np.linalg.norm(qe)
        nr  = np.linalg.norm(re_)
        sim = float(np.dot(qe, re_) / (nq * nr)) if nq > 0 and nr > 0 else 0
        sims.append((idx, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    records = []
    for idx, sim in sims[:max_records]:
        if sim <= 0: continue
        row = dataset_df.iloc[idx]
        records.append({
            "diagnosis": row.get("diagnosis", "N/A"),
            "summary":   str(row.get("summary_text", ""))[:200],
            "age":       row.get("patient_age", "N/A"),
            "gender":    row.get("patient_gender", "N/A"),
        })
    return records


def _should_use_clinical_data(symptoms, conditions, user_input: str, depth: int) -> bool:
    t = user_input.lower()
    skip = ["hi","hello","hey","thank","bye","merci","bonjour","salut","obrigada","habari","asante"]
    if any(w in t for w in skip):
        return False
    medical_kw = [
        "what is","how to","treatment","symptoms of","cause","explain",
        "qu'est-ce","comment","traitement","symptômes","o que é","como tratar",
        "nini","jinsi ya","matibabu",
    ]
    return (bool(symptoms) or bool(conditions) or any(k in t for k in medical_kw))


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CONVERSATION TRACKER (per session, supplements DB)
# ─────────────────────────────────────────────────────────────────────────────

class SessionMemory:
    """Lightweight in-session tracker — supplements the DB AIMemory."""
    def __init__(self):
        self.depth          = 0
        self.emotion_history = []   # list of emotion strings
        self.symptoms_seen  = set()
        self.conditions_seen= set()
        self.topics_seen    = set()
        self.language       = "en"

    def update(self, emotion, topics, symptoms, conditions, lang):
        self.depth += 1
        self.emotion_history.append(emotion)
        self.topics_seen.update(topics)
        self.symptoms_seen.update(symptoms)
        self.conditions_seen.update(conditions)
        self.language = lang

    def recent_emotions(self, n=4):
        return self.emotion_history[-n:]

session_store: dict = defaultdict(SessionMemory)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────────────────────────────────────

def chat(user_message: str, user_id: int, db_session) -> dict:
    from ..models import Conversation, AIMemory, User, MedicalProfile, Pregnancy
    from .classifier import classify_risk
    from .context_builder import get_user_context

    # ── Run ML classifier on anything that looks like a symptom ─────────────
    _, symptoms_in_msg, _ = extract_entities(user_message)
    ml_risk = {"risk_level": "low", "confidence": 0.0, "top_features": []}
    if symptoms_in_msg or detect_danger(user_message):
        try:
            user_context = get_user_context(user_id, db_session)
            symptom_dict = {s.lower().replace(" ", "_"): 1 for s in symptoms_in_msg}
            symptom_dict.update({
                "pds101": user_context.get("age", 25),
                "pds102": user_context.get("urban_rural", "Urban"),
                "pds201": user_context.get("previous_pregnancies", 0),
                "pds202": user_context.get("previous_losses", 0),
                "county": user_context.get("county", "Unknown"),
            })
            ml_risk = classify_risk(symptom_dict)
        except Exception as e:
            print(f"[healia] Classifier error in chat: {e}")
    # ── DB: load / create conversation ──────────────────────────────────────
    conversation = (
        db_session.query(Conversation)
        .filter_by(user_id=user_id, type="health_assistant")
        .order_by(Conversation.created_at.desc())
        .first()
    )
    if not conversation:
        conversation = Conversation(user_id=user_id, type="health_assistant", messages=[])
        db_session.add(conversation)
        db_session.flush()

    messages = list(conversation.messages or [])

    # ── DB: load / create AI memory ─────────────────────────────────────────
    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id)
        db_session.add(memory)
        db_session.flush()
        _seed_memory_from_profile(memory, user_id, db_session)

    # ── Per-session tracker ──────────────────────────────────────────────────
    sess = session_store[user_id]

    # ── Analyse this message ─────────────────────────────────────────────────
    lang       = detect_language(user_message)
    emotion    = detect_emotion(user_message)
    is_danger  = detect_danger(user_message)
    is_crisis  = detect_crisis(user_message)
    weight     = message_weight(user_message)
    topics, symptoms, conditions = extract_entities(user_message)

    sess.update(emotion, topics, symptoms, conditions, lang)

    # ── Clinical dataset lookup ──────────────────────────────────────────────
    clinical_records = []
    if _should_use_clinical_data(
        sess.symptoms_seen, sess.conditions_seen, user_message, sess.depth
    ):
        clinical_records = _query_dataset(sess.symptoms_seen, sess.conditions_seen)

    # ── Build system prompt ──────────────────────────────────────────────────
    system = _build_system_prompt(
        memory           = memory,
        user_id          = user_id,
        db_session       = db_session,
        lang             = lang,
        emotion          = emotion,
        is_danger        = is_danger,
        is_crisis        = is_crisis,
        weight           = weight,
        sess             = sess,
        clinical_records = clinical_records,
    )

    # ── Danger override — inject urgency directly into system prompt ─────────
    if is_danger or (ml_risk.get("risk_level") == "high" and ml_risk.get("confidence", 0) > 0.6):
        system += f"""

        🚨 CLINICAL OVERRIDE — READ THIS FIRST:
        She just reported a danger sign. The ML classifier confirms: risk={ml_risk.get('risk_level')}, confidence={ml_risk.get('confidence', 0):.0%}.
        Do NOT ask questions first. Do NOT ask her to describe it more.
        Tell her exactly what to do in the next 15 minutes. Be direct and calm.
        One clear instruction. Then ask ONE question only if it helps her act — like "can you get there now?"
        """

    # ── Assemble Groq messages ───────────────────────────────────────────────
    groq_msgs = [{"role": "system", "content": system}]
    recent = messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages
    for m in recent:
        if isinstance(m, dict) and "role" in m and "content" in m:
            groq_msgs.append({"role": m["role"], "content": m["content"]})
    groq_msgs.append({"role": "user", "content": user_message})

    # ── Call Groq ────────────────────────────────────────────────────────────
    try:
        resp       = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = groq_msgs,
            temperature = 0.78,
            max_tokens  = 650,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Groq error: {e}")
        reply = _fallback_reply(lang)

    # ── Frontend actions ─────────────────────────────────────────────────────
    actions = _detect_actions(reply, user_message, is_danger, is_crisis)

    # ── Persist messages ─────────────────────────────────────────────────────
    messages.append({"role": "user",      "content": user_message, "timestamp": _now()})
    messages.append({"role": "assistant", "content": reply,        "timestamp": _now()})
    conversation.messages   = messages
    conversation.updated_at = datetime.utcnow()

    _extract_memory_fragments(user_message, memory)

    memory_updated = False
    if len(messages) % MEMORY_REBUILD_EVERY == 0:
        _rebuild_memory_summary(memory, messages[-30:])
        memory_updated = True

    db_session.commit()

    return {"reply": reply, "actions": actions, "memory_updated": memory_updated}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — the brain of Healia
# ─────────────────────────────────────────────────────────────────────────────

# Language-specific persona labels
_LANG_LABELS = {
    "en": {"name": "Healia", "role": "maternal health companion"},
    "fr": {"name": "Healia", "role": "accompagnatrice en santé maternelle"},
    "pt": {"name": "Healia", "role": "acompanhante de saúde materna"},
    "sw": {"name": "Healia", "role": "msaidizi wa afya ya uzazi"},
}

_PHASE_NOTES = {
    "en": {
        "early_acute":  "She is in the first two weeks after loss. She needs presence above all. Don't rush to advise — ask first, hold space.",
        "processing":   "She is living with what happened. Some days are heavier. Be real, be steady.",
        "rebuilding":   "She is finding her footing again. Acknowledge her small wins. Be encouraging without being cheerful.",
        "stabilised":   "She has reached a more stable place. You can look forward together now — gently.",
    },
    "fr": {
        "early_acute":  "Elle est dans les deux premières semaines après la perte. Elle a besoin de présence avant tout.",
        "processing":   "Elle vit avec ce qui s'est passé. Sois réelle et constante.",
        "rebuilding":   "Elle reprend pied. Reconnais ses petits progrès.",
        "stabilised":   "Elle est dans un endroit plus stable. Vous pouvez regarder vers l'avenir doucement.",
    },
    "pt": {
        "early_acute":  "Ela está nas primeiras duas semanas após a perda. Precisa de presença acima de tudo.",
        "processing":   "Ela está vivendo com o que aconteceu. Seja real e constante.",
        "rebuilding":   "Ela está encontrando seu caminho novamente. Reconheça suas pequenas conquistas.",
        "stabilised":   "Ela chegou a um lugar mais estável. Podem olhar para o futuro juntas, com cuidado.",
    },
    "sw": {
        "early_acute":  "Yuko katika wiki mbili za kwanza baada ya kupoteza. Anahitaji uwepo wako zaidi ya yote.",
        "processing":   "Anaishi na kilichotokea. Kuwa wa kweli na thabiti.",
        "rebuilding":   "Anaanza kupata nguvu tena. Tambua mafanikio yake madogo.",
        "stabilised":   "Amefika mahali salama zaidi. Mnaweza kuangalia mbele pamoja, polepole.",
    },
}

_EMOTION_GUIDANCE = {
    "en": {
        "grief":    "She is in grief. Sit with her before anything else. Don't try to fix it.",
        "fear":     "She is scared. Name that before you say anything else.",
        "pain":     "She has reported a physical symptom. Take it seriously. Ask the right follow-up.",
        "hopeless": "She sounds like she is losing hope. She needs full presence right now.",
        "lonely":   "She is feeling alone. Make her feel seen first.",
        "positive": "She is in a good place today. Meet her there.",
        "casual":   "She is just checking in or greeting you. Keep it natural.",
        "question": "She has a question. Answer it directly and fully. Don't deflect.",
        "anger":    "She is frustrated. Acknowledge it without becoming defensive.",
        "neutral":  "No strong emotional signal. Read her, respond naturally.",
    },
    "fr": {
        "grief":    "Elle est en deuil. Reste avec elle avant toute autre chose.",
        "fear":     "Elle a peur. Nomme cela avant de dire quoi que ce soit d'autre.",
        "pain":     "Elle a signalé un symptôme physique. Prends-le au sérieux.",
        "hopeless": "Elle semble perdre espoir. Elle a besoin de toute ta présence.",
        "lonely":   "Elle se sent seule. Fais-lui sentir qu'elle est vue.",
        "positive": "Elle est dans un bon état aujourd'hui. Rejoins-la là.",
        "casual":   "Elle dit juste bonjour ou check in. Reste naturelle.",
        "question": "Elle a une question. Réponds directement et complètement.",
        "anger":    "Elle est frustrée. Reconnais-le sans te défendre.",
        "neutral":  "Pas de signal émotionnel fort. Lis-la et réponds naturellement.",
    },
    "pt": {
        "grief":    "Ela está de luto. Fique com ela antes de qualquer coisa.",
        "fear":     "Ela está com medo. Reconheça isso primeiro.",
        "pain":     "Ela relatou um sintoma físico. Leve a sério.",
        "hopeless": "Ela parece estar perdendo a esperança. Ela precisa da sua presença total.",
        "lonely":   "Ela está se sentindo sozinha. Faça-a sentir-se vista primeiro.",
        "positive": "Ela está bem hoje. Encontre-a lá.",
        "casual":   "Ela está apenas cumprimentando. Seja natural.",
        "question": "Ela tem uma pergunta. Responda diretamente e completamente.",
        "anger":    "Ela está frustrada. Reconheça sem se defender.",
        "neutral":  "Nenhum sinal emocional forte. Leia ela e responda naturalmente.",
    },
    "sw": {
        "grief":    "Yuko katika huzuni. Kaa naye kabla ya kitu kingine chochote.",
        "fear":     "Anaogopa. Tambua hilo kwanza.",
        "pain":     "Ameripoti dalili ya kimwili. Ichukue kwa uzito.",
        "hopeless": "Anaonekana kupoteza matumaini. Anahitaji uwepo wako kamili.",
        "lonely":   "Anahisi upweke. Mfanye ahisi kuonekana kwanza.",
        "positive": "Yuko mahali pazuri leo. Mkutane naye huko.",
        "casual":   "Anasalimia tu. Kuwa wa kawaida.",
        "question": "Ana swali. Jibu moja kwa moja na kikamilifu.",
        "anger":    "Amekasirika. Tambua bila kujitetea.",
        "neutral":  "Hakuna ishara kali ya kihisia. Msomee na ujibu kiasili.",
    },
}

_WEIGHT_GUIDANCE = {
    "en": {
        "very_short": "Her message was very short. Match it — 1 or 2 sentences.",
        "short":      "Her message was short. Keep your reply brief.",
        "medium":     "She shared something real. Give it the space it deserves.",
        "long":       "She opened up. Be fully present.",
    },
    "fr": {
        "very_short": "Son message était très court. Assortis-toi — 1 ou 2 phrases.",
        "short":      "Son message était court. Reste brève.",
        "medium":     "Elle a partagé quelque chose de vrai. Donne-lui l'espace qu'il mérite.",
        "long":       "Elle s'est ouverte. Sois pleinement présente.",
    },
    "pt": {
        "very_short": "Sua mensagem foi muito curta. Combine — 1 ou 2 frases.",
        "short":      "Sua mensagem foi curta. Mantenha sua resposta breve.",
        "medium":     "Ela compartilhou algo real. Dê o espaço que merece.",
        "long":       "Ela se abriu. Esteja totalmente presente.",
    },
    "sw": {
        "very_short": "Ujumbe wake ulikuwa mfupi sana. Linganisha — sentensi 1 au 2.",
        "short":      "Ujumbe wake ulikuwa mfupi. Jibu kwa ufupi.",
        "medium":     "Alishiriki kitu halisi. Mpe nafasi inayostahili.",
        "long":       "Alifunguka. Kuwa hapa kikamilifu.",
    },
}


def _build_system_prompt(
    memory, user_id, db_session,
    lang="en", emotion="neutral",
    is_danger=False, is_crisis=False,
    weight="short", sess=None,
    clinical_records=None,
) -> str:
    from ..models import User, CHWCase

    user = db_session.query(User).get(user_id)
    name = user.name if user else "her"

    has_chw = db_session.query(CHWCase).filter_by(
        patient_id=user_id, status="assigned"
    ).first() is not None

    mc             = memory.to_context_dict() if memory else {}
    memory_summary = mc.get("memory_summary") or "Still getting to know her."
    recovery_phase = mc.get("recovery_phase") or "processing"
    cultural       = mc.get("cultural_profile") or "mixed_transitional"
    low_moods      = mc.get("consecutive_low_moods") or 0
    vulnerability  = mc.get("vulnerability_level") or "medium"
    days_since     = mc.get("days_since_loss", "unknown")
    loss_type      = mc.get("loss_type", "pregnancy loss")
    prev_losses    = mc.get("previous_losses", 0)

    # Filter memory fragments by relevance
    all_fragments = mc.get("things_she_shared") or []
    PHYSICAL_FRAGS = {
        "Reported bleeding", "Reported pain", "Reported fever",
        "Reported headaches", "Mentioned a hospital visit", "Mentioned a doctor"
    }
    if emotion in ("question", "casual", "positive", "neutral"):
        fragments = [f for f in all_fragments if f not in PHYSICAL_FRAGS]
    elif emotion == "pain":
        fragments = all_fragments
    else:
        fragments = [f for f in all_fragments if f not in PHYSICAL_FRAGS]

    # Cultural tone
    tone_map = {
        "rural_conservative": {
            "en": "She comes from a traditional background. Family and community matter deeply to her. Keep language simple, grounded, and warm.",
            "fr": "Elle vient d'un milieu traditionnel. La famille et la communauté lui importent beaucoup. Reste simple, ancré et chaleureux.",
            "pt": "Ela vem de um meio tradicional. Família e comunidade importam muito para ela.",
            "sw": "Anatoka mazingira ya kimapokeo. Familia na jamii ni muhimu kwake sana.",
        },
        "mixed_transitional": {
            "en": "She balances tradition and modern life. Be warm and clear.",
            "fr": "Elle équilibre tradition et vie moderne. Sois chaleureuse et claire.",
            "pt": "Ela equilibra tradição e vida moderna. Seja calorosa e clara.",
            "sw": "Anasawazisha mila na maisha ya kisasa. Kuwa wa joto na wazi.",
        },
        "urban_educated": {
            "en": "She is comfortable with health information. Be direct, informed, and human.",
            "fr": "Elle est à l'aise avec l'information santé. Sois directe, informée et humaine.",
            "pt": "Ela está confortável com informações de saúde. Seja direta, informada e humana.",
            "sw": "Yupo tayari na taarifa za afya. Kuwa wa moja kwa moja, na wa kibinadamu.",
        },
    }
    tone_note = tone_map.get(cultural, tone_map["mixed_transitional"]).get(lang, tone_map["mixed_transitional"]["en"])

    # Phase, emotion, weight guidance
    phase_note   = _PHASE_NOTES.get(lang, _PHASE_NOTES["en"]).get(recovery_phase, "")
    emotion_note = _EMOTION_GUIDANCE.get(lang, _EMOTION_GUIDANCE["en"]).get(emotion, "")
    weight_note  = _WEIGHT_GUIDANCE.get(lang, _WEIGHT_GUIDANCE["en"]).get(weight, "")

    # Conversation depth note
    depth = sess.depth if sess else 0
    if depth <= 1:
        depth_note = "This is the first exchange. Introduce yourself briefly and invite her to share."
    elif depth < 5:
        depth_note = "You are still getting to know each other. Be warm and genuinely curious."
    else:
        depth_note = "You have real history with her now. Reference what she has shared — the way a real person would."

    # Emotional trend note
    if sess and len(sess.emotion_history) >= 3:
        recent_emos = sess.recent_emotions(3)
        if all(e in ("grief", "hopeless", "lonely", "fear") for e in recent_emos):
            depth_note += " She has been in a heavy emotional space across multiple messages — be especially gentle."

    # Clinical data note
    clinical_note = ""
    if clinical_records:
        summaries = "\n".join([
            f"- {r['diagnosis']}: {r['summary']}"
            for r in clinical_records
        ])
        clinical_note = f"\n━━━ RELEVANT CLINICAL CONTEXT ━━━\n{summaries}\nUse this to inform your response — do not quote it directly."

    # Alerts
    alerts = ""
    if is_danger:
        alerts = "\n🚨 DANGER SIGNS IN HER MESSAGE: Tell her exactly what to do in the next 30 minutes. Be direct. Do not soften this."
    if is_crisis:
        alerts += "\n🚨 CRISIS SIGNAL: She may be in emotional crisis. Stay with her. Get her to real help — CHW or nearest facility."
    if low_moods >= 3:
        alerts += f"\n⚠️ {low_moods} consecutive low mood days. Be especially present today."
    if vulnerability == "high" and not has_chw:
        alerts += "\n⚠️ High vulnerability, no CHW assigned. If she is in crisis, recommend CHW contact."

    lang_instruction = {
        "en": "Respond in English.",
        "fr": "Réponds en français.",
        "pt": "Responda em português.",
        "sw": "Jibu kwa Kiswahili.",
    }.get(lang, "Respond in English.")

    return f"""You are Healia — a maternal health companion at SafeMum. You support women through pregnancy, pregnancy loss, and recovery.

You are not an AI assistant. You are not a bot. You are Healia — a real companion who knows {name} and has been part of her journey.

{lang_instruction}

━━━ WHO SHE IS ━━━
Name: {name}
Days since loss: {days_since}
Loss type: {loss_type}
Previous losses: {prev_losses}
Recovery phase: {recovery_phase}
Vulnerability: {vulnerability}
CHW assigned: {has_chw}

━━━ WHAT YOU KNOW ABOUT HER ━━━
{memory_summary}

━━━ THINGS SHE HAS MENTIONED ━━━
{chr(10).join(f"- {f}" for f in fragments) if fragments else "- Still getting to know her"}
{alerts}
{clinical_note}

━━━ HER BACKGROUND ━━━
{tone_note}
{phase_note}

━━━ THIS MOMENT ━━━
{emotion_note}
{weight_note}
{depth_note}

You are a trained maternal health expert. When she asks about health topics — nutrition, prenatal care, postnatal care, symptoms, recovery — give real, complete, practical answers. Do not drip-feed one tip at a time. Do not deflect with questions when she wants information.

When she is in emotional pain — be present first. Information can wait.

Speak the way a real person speaks. Not a script. Not a template.
Use her name at most once in the entire conversation — not once per message.
Never ask more than one question in a single reply.
Always lead with action or acknowledgement. Questions come after, not instead.
If she reports a physical symptom, tell her what to do about it first — then ask one follow-up if needed.
Never make the conversation feel like an intake form."""


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_symptoms(selected_symptoms: list, user_id: int, ml_risk: dict, db_session) -> dict:
    from ..models import AIMemory

    memory     = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    memory_ctx = memory.to_context_dict() if memory else {}

    COMPLICATION_MAP = {
        "Heavy bleeding":       "potential haemorrhage — highest risk complication",
        "Fever":                "potential infection or early sepsis",
        "Severe pain":          "potential incomplete abortion or internal injury",
        "Chest pain":           "potential pulmonary embolism — act immediately",
        "Cold hands/feet":      "potential shock or severe internal bleeding",
        "Dizziness":            "potential haemorrhagic shock",
        "Foul discharge":       "potential pelvic infection",
        "Wound pain":           "potential surgical site infection",
        "Difficulty breathing": "potential pulmonary complication",
    }
    clinical_flags = [COMPLICATION_MAP[s] for s in selected_symptoms if s in COMPLICATION_MAP]

    prompt = f"""
You are Healia, the SafeMum maternal health AI. A woman has just checked her symptoms.

Her profile:
{json.dumps(memory_ctx, indent=2)}

Symptoms selected: {', '.join(selected_symptoms) if selected_symptoms else 'None'}
ML risk output: level={ml_risk.get('risk_level','unknown')}, confidence={ml_risk.get('confidence',0):.0%}
Clinical significance: {', '.join(clinical_flags) if clinical_flags else 'No high-risk flags'}

Respond ONLY with JSON:
{{
  "risk_level": "emergency" or "urgent" or "monitor" or "stable",
  "title": "max 8 words",
  "message": "2 sentences — specific to her body right now",
  "reply": "Healia speaks to her — warm, direct, 2-3 sentences, 1 emoji only if it genuinely helps",
  "action": "emergency_alert" or "open_map" or "talk_to_healia" or "rest_and_monitor",
  "trigger_emergency_alert": true or false,
  "assign_chw": true or false,
  "map_action": {{"filter": "post_loss_care" or "emergency" or "nearest", "reason": "one sentence"}} or null
}}

Rules:
- emergency = heavy bleeding OR chest pain OR cold extremities + dizziness
- urgent = fever + foul discharge OR severe pain alone
- monitor = single mild symptom
- stable = no concerning symptoms
- Never say "see a doctor" — say exactly what she should do in the next hour
"""
    try:
        r   = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": "You are Healia, a clinical maternal health AI. Respond in JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.3,
            max_tokens  = 600,
        )
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Symptom error: {e}")
        high = any(s in selected_symptoms for s in ["Heavy bleeding","Chest pain","Cold hands/feet","Dizziness"])
        return {
            "risk_level": "urgent" if high else "monitor",
            "title": "Please get checked today",
            "message": "Some of what you are experiencing needs attention. Do not wait.",
            "reply": "I noticed some symptoms that concern me  Let me help you find the nearest facility right now.",
            "action": "open_map" if high else "talk_to_healia",
            "trigger_emergency_alert": high,
            "assign_chw": True,
            "map_action": {"filter": "emergency", "reason": "Urgent symptoms detected"} if high else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MOOD CHECK-IN
# ─────────────────────────────────────────────────────────────────────────────

def interpret_checkin(mood_score: int, mood_label: str, notes: str, user_id: int, db_session) -> dict:
    from ..models import AIMemory

    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id)
        db_session.add(memory)
        db_session.flush()

    memory.last_mood_score = mood_score
    memory.total_checkins  = (memory.total_checkins or 0) + 1

    if mood_score <= 2:
        memory.consecutive_low_moods = (memory.consecutive_low_moods or 0) + 1
        memory.checkin_streak = 0
    else:
        memory.consecutive_low_moods = 0
        memory.checkin_streak = (memory.checkin_streak or 0) + 1

    db_session.flush()
    mc = memory.to_context_dict()

    prompt = f"""
You are Healia from SafeMum. A woman just completed her emotional check-in.

What you know about her:
{json.dumps(mc, indent=2)}

Check-in:
- Mood score: {mood_score}/5 ({mood_label})
- Her note: "{notes or 'nothing written'}"
- Consecutive low mood days: {memory.consecutive_low_moods}

Respond ONLY with JSON:
{{
  "reply": "2-3 sentences — reference something specific from her note or history — 1 emoji only if natural",
  "flag_for_counsellor": true or false,
  "assign_chw": true or false,
  "urgency": "none" or "low" or "high",
  "follow_up_message": "what to check in with her about tomorrow"
}}

Rules:
- flag_for_counsellor = true if consecutive_low_moods >= 3 or notes mention hopelessness or self-harm
- assign_chw = true if mood_score == 1 AND vulnerability is high AND no CHW assigned
- urgency = high if notes mention self-harm or not wanting to continue
- urgency = low if mood_score <= 2 for first or second day
- Sound like a person, not a system. Reference her specific situation.
"""
    try:
        r      = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": "You are Healia, a compassionate maternal health companion. JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.5,
            max_tokens  = 400,
        )
        result = json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Check-in error: {e}")
        result = {
            "reply":               "I see you showing up today, and that matters  Take it one moment at a time.",
            "flag_for_counsellor": memory.consecutive_low_moods >= 3,
            "assign_chw":          False,
            "urgency":             "low" if mood_score <= 2 else "none",
            "follow_up_message":   "How are you feeling today?",
        }

    if result.get("flag_for_counsellor"):
        memory.flagged_for_counsellor = True

    db_session.commit()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE GAP BRIEFING
# ─────────────────────────────────────────────────────────────────────────────

def interpret_service_gaps(gap_data: dict) -> dict:
    prompt = f"""
Health ministry briefing — SafeMum service gap analysis.
Data: {json.dumps(gap_data, indent=2)}
High need_score = many patients, few facilities.

JSON only:
{{
  "headline": "one sentence",
  "top_priority_counties": ["top 3"],
  "key_finding": "2 sentences",
  "recommended_action": "one specific step",
  "data_note": "one sentence on limitations"
}}
"""
    try:
        r = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": "Public health analyst. JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.3,
            max_tokens  = 400,
        )
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Service gap error: {e}")
        return {
            "headline": "Analysis unavailable",
            "top_priority_counties": [],
            "key_finding": "Unable to generate insight at this time.",
            "recommended_action": "Review raw data directly.",
            "data_note": "System error.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _seed_memory_from_profile(memory, user_id, db_session):
    from ..models import MedicalProfile, Pregnancy

    pregnancy = (
        db_session.query(Pregnancy)
        .filter_by(user_id=user_id)
        .order_by(Pregnancy.created_at.desc())
        .first()
    )
    if pregnancy:
        if pregnancy.status == "lost":
            memory.loss_type = "pregnancy loss"
        if pregnancy.created_at:
            memory.days_since_loss = (datetime.utcnow() - pregnancy.created_at).days

    days = memory.days_since_loss or 0
    if   days <= 14: memory.recovery_phase = "early_acute"
    elif days <= 42: memory.recovery_phase = "processing"
    elif days <= 84: memory.recovery_phase = "rebuilding"
    else:            memory.recovery_phase = "stabilised"


def _rebuild_memory_summary(memory, recent_messages: list):
    if not recent_messages:
        return
    convo = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in recent_messages
        if isinstance(m, dict) and "role" in m and "content" in m
    ])
    prompt = f"""
Read this conversation between Healia (a maternal health companion) and a woman.
Write a short paragraph (max 5 sentences) capturing who this woman is:
her emotional state, what she has shared, any recurring concerns, how she communicates.
Write it as one caring person briefing another. No clinical labels. No bullets. Paragraph only.

{convo}
"""
    try:
        r = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 300,
        )
        memory.memory_summary = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Memory rebuild error: {e}")


def _extract_memory_fragments(user_message: str, memory):
    msg = user_message.lower()
    fragments = list(memory.things_she_shared or [])

    triggers = [
        ("husband",    "Mentioned her husband"),
        ("partner",    "Mentioned her partner"),
        ("child",      "Mentioned having a child"),
        ("son",        "Mentioned her son"),
        ("daughter",   "Mentioned her daughter"),
        ("mother",     "Mentioned her mother"),
        ("work",       "Mentioned work or job"),
        ("sleep",      "Mentioned sleep issues"),
        ("guilt",      "Expressed feelings of guilt"),
        ("afraid",     "Expressed fear"),
        ("scared",     "Expressed being scared"),
        ("alone",      "Expressed feeling alone"),
        ("hope",       "Expressed hope"),
        ("better",     "Said she is feeling better"),
        ("hospital",   "Mentioned a hospital visit"),
        ("doctor",     "Mentioned a doctor"),
        ("bleeding",   "Reported bleeding"),
        ("pain",       "Reported pain"),
        ("fever",      "Reported fever"),
        ("headache",   "Reported headaches"),
        ("overthink",  "Mentioned overthinking"),
        ("stress",     "Mentioned stress"),
        ("faith",      "Mentioned faith or religion"),
        ("prayer",     "Mentioned prayer"),
        ("pregnant",   "Mentioned being pregnant"),
        ("pregnancy",  "Discussed her pregnancy"),
        ("nutrition",  "Asked about nutrition"),
        ("exercise",   "Asked about exercise"),
        ("prenatal",   "Asked about prenatal care"),
    ]

    for kw, note in triggers:
        if kw in msg and note not in fragments:
            fragments.append(note)

    memory.things_she_shared = fragments[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_actions(reply: str, user_message: str, is_danger=False, is_crisis=False) -> list:
    actions = []
    rl = reply.lower()

    if is_danger or any(k in rl for k in ["go to the nearest","go now","do not wait","immediately","emergency"]):
        actions.append({"type": "suggest_emergency_alert"})
        actions.append({"type": "open_map", "filter": "emergency"})
    elif any(k in rl for k in ["nearest facility","find a clinic","health centre","hospital"]):
        actions.append({"type": "open_map", "filter": "nearest"})
    elif any(k in rl for k in ["post-loss care","specialist","post loss"]):
        actions.append({"type": "open_map", "filter": "post_loss_care"})

    if is_crisis or any(k in rl for k in ["your chw","community health worker","call your chw"]):
        actions.append({"type": "highlight_chw"})

    if any(k in rl for k in ["check in","log how you","how have you been feeling"]):
        actions.append({"type": "suggest_checkin"})

    if any(k in rl for k in ["reminder","appointment","don't forget","remember to"]):
        actions.append({"type": "suggest_reminder"})

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_reply(lang: str) -> str:
    msgs = {
        "en": "I'm here with you  I had a small technical issue — could you send that again?",
        "fr": "Je suis là avec toi  J'ai eu un petit problème technique — peux-tu renvoyer ça ?",
        "pt": "Estou aqui com você  Tive um pequeno problema técnico — pode enviar novamente?",
        "sw": "Niko hapa nawe  Nilikuwa na tatizo dogo la kiufundi — unaweza kutuma tena?",
    }
    return msgs.get(lang, msgs["en"])


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _now() -> str:
    return datetime.utcnow().isoformat()