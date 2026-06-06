import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

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
    "fr": [
        "je","tu","il","elle","nous","vous","les","une","des","et",
        "est","suis","bonjour","merci","oui","non","avec","pour",
        "dans","sur","pas","mais","très","bien","ça","douleur",
        "fièvre","mal","bébé","enceinte","grossesse","bonne",
    ],
    "pt": [
        "eu","você","ele","ela","nós","eles","uma","para","com",
        "não","sim","obrigada","obrigado","olá","bom","boa","estou",
        "está","meu","minha","bebê","grávida","gravidez","dor",
        "febre","sangramento","saúde",
    ],
    "sw": [
        "mimi","wewe","yeye","sisi","ninyi","wao","na","kwa","ya",
        "ni","hapana","ndiyo","asante","habari","nzuri","mtoto",
        "mimba","maumivu","homa","damu","afya","mama",
    ],
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
# EMOTION & INTENT DETECTION
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
    "infection","infecção","pain","douleur","dor",
    "smell","odeur","cheiro","discharge","pertes","corrimento",
]

GRIEF_KEYWORDS = [
    "grief","sad","crying","tears","mourning","deuil","triste","luto",
    "huzuni","machofu","kilio",
]

def detect_emotion(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(what is|how do|how does|what are|can i|should i|is it normal|tell me about|explain|what happens|when should|why do|how long|how much|what to eat|what to avoid|what can i|is there|are there)\b", t):
        return "seeking_info"
    if re.search(r"\b(i have|i am having|i've been|i noticed|there is|it's been|since yesterday|since this morning|started|i keep|keeps happening|won't stop)\b", t):
        return "reporting"
    for emotion, pattern in EMOTIONAL_PATTERNS.items():
        if re.search(pattern, t):
            return emotion
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
# ENHANCED SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_sentiment_analysis(text: str) -> dict:
    result = {
        "polarity": 0, "subjectivity": 0,
        "emotion": detect_emotion(text),
        "intensity": 0, "support_needed": False,
    }
    if SENTIMENT_AVAILABLE:
        try:
            blob = TextBlob(text)
            result["polarity"]     = blob.sentiment.polarity
            result["subjectivity"] = blob.sentiment.subjectivity
        except Exception:
            pass
    text_lower = text.lower()
    if "!" in text: result["intensity"] += 2
    if text.isupper() and len(text) > 10: result["intensity"] += 2
    if re.search(r"\b(very|so|really|extremely|such)\b", text_lower): result["intensity"] += 1
    result["support_needed"] = (
        result["emotion"] in ("grief","hopeless","anger") or
        result["intensity"] >= 3 or
        result["polarity"] < -0.3
    )
    return result


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
        "nutrition":         ["diet","food","eat","nutrition","manger","nourriture","comer","chakula"],
        "physical_activity": ["exercise","sport","fitness","activité","exercício","mazoezi"],
        "mental_health":     ["mental","stress","anxiety","grief","émotionnel","emocional","afya ya akili"],
        "medication":        ["medicine","medication","drug","traitement","médicament","medicamento","dawa"],
        "prenatal":          ["prenatal","antenatal","pregnancy","grossesse","gravidez","mimba"],
        "postnatal":         ["postnatal","postpartum","après accouchement","pós-parto","baada ya kujifungua"],
    }
    for topic, keywords in kw_topics.items():
        if any(k in t for k in keywords): topics.add(topic)
    return topics, symptoms, conditions


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL DATASET + MATRIX FACTORIZATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH         = os.path.join(os.path.dirname(__file__), "clinical_summaries.csv")
dataset_df           = pd.DataFrame()
symptom_embeddings   = None
condition_embeddings = None
record_embeddings    = None
symptom_to_idx       = {}
condition_to_idx     = {}

def _train_mf_model(df, n_components=10):
    global symptom_embeddings, condition_embeddings, record_embeddings, symptom_to_idx, condition_to_idx
    if df.empty: return
    all_symptoms, all_conditions = set(), set()
    for _, row in df.iterrows():
        text = f"{row.get('summary_text','')} {row.get('diagnosis','')}".lower()
        _, syms, conds = extract_entities(text)
        all_symptoms.update(syms); all_conditions.update(conds)
    symptom_to_idx   = {s: i for i, s in enumerate(sorted(all_symptoms))}
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
        required = ["summary_id","patient_id","patient_age","patient_gender","diagnosis",
                    "body_temp_c","blood_pressure_systolic","heart_rate","summary_text","date_recorded"]
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
    except Exception: return []
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
        records.append({"diagnosis": row.get("diagnosis","N/A"), "summary": str(row.get("summary_text",""))[:200],
                        "age": row.get("patient_age","N/A"), "gender": row.get("patient_gender","N/A")})
    return records

def _should_use_clinical_data(symptoms, conditions, user_input: str, depth: int) -> bool:
    t = user_input.lower()
    skip = ["hi","hello","hey","thank","bye","merci","bonjour","salut","obrigada","habari","asante"]
    if any(w in t for w in skip): return False
    medical_kw = ["what is","how to","treatment","symptoms of","cause","explain","qu'est-ce","comment",
                  "traitement","symptômes","o que é","como tratar","nini","jinsi ya","matibabu"]
    return bool(symptoms) or bool(conditions) or any(k in t for k in medical_kw)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MEMORY  — bug fixes applied
# ─────────────────────────────────────────────────────────────────────────────

class SessionMemory:
    def __init__(self):
        self.depth               = 0
        self.emotion_history     = []
        self.symptoms_seen       = set()
        self.conditions_seen     = set()
        self.topics_seen         = set()
        self.topics_addressed    = set()   # ← was missing, caused AttributeError
        self.grief_acknowledged  = False
        self.language            = "en"
        self.last_question_asked = None
        self.user_disclosed_info = {}

    def update(self, emotion, topics, symptoms, conditions, lang):
        self.depth += 1
        self.emotion_history.append(emotion)
        self.topics_seen.update(topics)
        self.symptoms_seen.update(symptoms)
        self.conditions_seen.update(conditions)
        self.language = lang
        self.topics_addressed.update(topics)

    def recent_emotions(self, n=4):
        return self.emotion_history[-n:]

    def grief_note(self) -> str:
        if self.grief_acknowledged:
            return (
                "Grief has already been acknowledged earlier in this conversation. "
                "Do NOT re-open it unless she explicitly brings it up again."
            )
        if self.depth >= 1:
            self.grief_acknowledged = True
        return ""

    def already_answered(self, topic: str) -> bool:
        return topic in self.topics_addressed

    def should_ask(self, question_type: str) -> bool:
        if self.last_question_asked == question_type and self.depth < 5:
            return False
        self.last_question_asked = question_type
        return True

    def get_user_status(self) -> str:
        return self.user_disclosed_info.get("pregnancy_status", "unknown")

    def update_disclosed_info(self, message: str):
        msg_lower = message.lower()
        if "gave birth" in msg_lower or "delivered" in msg_lower:
            self.user_disclosed_info["pregnancy_status"] = "delivered"
            if "boy" in msg_lower:   self.user_disclosed_info["baby_gender"] = "boy"
            elif "girl" in msg_lower: self.user_disclosed_info["baby_gender"] = "girl"
        elif "pregnant" in msg_lower or "expecting" in msg_lower:
            self.user_disclosed_info["pregnancy_status"] = "pregnant"
        elif "loss" in msg_lower or "miscarriage" in msg_lower:
            self.user_disclosed_info["pregnancy_status"] = "loss"

session_store: dict = defaultdict(SessionMemory)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE-SPECIFIC LABELS & GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────

_LANG_LABELS = {
    "en": {"name": "Healia", "role": "maternal health companion"},
    "fr": {"name": "Healia", "role": "accompagnatrice en santé maternelle"},
    "pt": {"name": "Healia", "role": "acompanhante de saúde materna"},
    "sw": {"name": "Healia", "role": "msaidizi wa afya ya uzazi"},
}

_PHASE_NOTES = {
    "en": {
        "early_acute":       "She is in the first two weeks after loss. She needs presence above all.",
        "processing":        "She is living with what happened. Some days are heavier. Be real, be steady.",
        "rebuilding":        "She is finding her footing again. Acknowledge her small wins.",
        "stabilised":        "She has reached a more stable place. Look forward together, gently.",
        "active_pregnancy":  "She is currently pregnant. Do NOT reference loss or grief. Focus only on her pregnancy.",
        "postnatal":         "She has recently delivered. Support postnatal recovery. No grief framing unless she brings it up.",
        "general_support":   "Her situation is not fully known yet. Follow her lead. Do not assume loss.",
    },
    "fr": {
        "early_acute":       "Elle est dans les deux premières semaines après la perte. Présence avant tout.",
        "processing":        "Elle vit avec ce qui s'est passé. Sois réelle et constante.",
        "rebuilding":        "Elle reprend pied. Reconnais ses petits progrès.",
        "stabilised":        "Elle est dans un endroit plus stable. Regardez vers l'avenir doucement.",
        "active_pregnancy":  "Elle est actuellement enceinte. Ne mentionne pas la perte. Concentre-toi sur sa grossesse.",
        "postnatal":         "Elle vient d'accoucher. Soutien postnatal uniquement.",
        "general_support":   "Sa situation n'est pas encore connue. Suis son rythme.",
    },
    "pt": {
        "early_acute":       "Ela está nas primeiras duas semanas após a perda. Presença acima de tudo.",
        "processing":        "Ela está vivendo com o que aconteceu. Seja real e constante.",
        "rebuilding":        "Ela está encontrando seu caminho. Reconheça suas pequenas conquistas.",
        "stabilised":        "Ela chegou a um lugar mais estável. Olhem para o futuro juntas.",
        "active_pregnancy":  "Ela está grávida. Não mencione perda. Foque na gravidez dela.",
        "postnatal":         "Ela deu à luz recentemente. Suporte pós-natal apenas.",
        "general_support":   "Situação ainda não conhecida. Siga o ritmo dela.",
    },
    "sw": {
        "early_acute":       "Yuko katika wiki mbili za kwanza baada ya kupoteza. Uwepo wake ni muhimu.",
        "processing":        "Anaishi na kilichotokea. Kuwa wa kweli na thabiti.",
        "rebuilding":        "Anaanza kupata nguvu tena. Tambua mafanikio yake madogo.",
        "stabilised":        "Amefika mahali salama zaidi. Angalieni mbele pamoja, polepole.",
        "active_pregnancy":  "Yuko mjamzito. Usizungumzie upotezaji. Zingatia mimba yake.",
        "postnatal":         "Amejifungua hivi karibuni. Msaada wa baada ya kujifungua pekee.",
        "general_support":   "Hali yake haijulikani bado. Fuata mwelekeo wake.",
    },
}

_EMOTION_GUIDANCE = {
    "en": {
        "grief":        "She is in grief. Sit with her before anything else. Don't try to fix it.",
        "fear":         "She is scared. Name that before you say anything else.",
        "pain":         "She has reported a physical symptom. Take it seriously. Ask the right follow-up.",
        "hopeless":     "She sounds like she is losing hope. Full presence needed.",
        "lonely":       "She is feeling alone. Make her feel seen first.",
        "positive":     "She is in a good place today. Meet her there.",
        "casual":       "She is just greeting or checking in. Keep it warm and natural. Do NOT launch into grief or medical topics unprompted.",
        "question":     "She has a question. Answer it directly and fully. Do not deflect.",
        "anger":        "She is frustrated. Acknowledge it without becoming defensive.",
        "neutral":      "No strong emotional signal. Read her, respond naturally.",
        "seeking_info": "She wants a real answer. Give it fully and directly. Do not ask how she is feeling — answer her question first.",
        "reporting":    "She is describing her situation or symptoms. Respond to the CONTENT of what she said first.",
    },
    "fr": {
        "grief":        "Elle est en deuil. Reste avec elle avant toute chose.",
        "fear":         "Elle a peur. Nomme cela d'abord.",
        "pain":         "Symptôme physique signalé. Prends-le au sérieux.",
        "hopeless":     "Elle semble perdre espoir. Présence totale.",
        "lonely":       "Elle se sent seule. Fais-la se sentir vue d'abord.",
        "positive":     "Elle va bien aujourd'hui. Rejoins-la là.",
        "casual":       "Elle dit juste bonjour. Reste naturelle. Ne lance pas de sujet médical sans raison.",
        "question":     "Elle a une question. Réponds directement.",
        "anger":        "Elle est frustrée. Reconnais-le.",
        "neutral":      "Pas de signal fort. Réponds naturellement.",
        "seeking_info": "Elle veut une vraie réponse. Donne-la maintenant, directement.",
        "reporting":    "Elle décrit sa situation. Réponds au contenu de ce qu'elle a dit.",
    },
    "pt": {
        "grief":        "Ela está de luto. Fique com ela primeiro.",
        "fear":         "Ela está com medo. Reconheça isso primeiro.",
        "pain":         "Sintoma físico relatado. Leve a sério.",
        "hopeless":     "Ela parece estar perdendo esperança. Presença total.",
        "lonely":       "Ela está se sentindo sozinha. Faça-a sentir-se vista.",
        "positive":     "Ela está bem hoje. Encontre-a lá.",
        "casual":       "Ela está apenas cumprimentando. Seja natural. Não abra tópicos médicos sem razão.",
        "question":     "Ela tem uma pergunta. Responda diretamente.",
        "anger":        "Ela está frustrada. Reconheça.",
        "neutral":      "Nenhum sinal forte. Responda naturalmente.",
        "seeking_info": "Ela quer uma resposta real. Dê-a agora, diretamente.",
        "reporting":    "Ela descreve sua situação. Responda ao conteúdo do que ela disse.",
    },
    "sw": {
        "grief":        "Yuko katika huzuni. Kaa naye kwanza.",
        "fear":         "Anaogopa. Tambua hilo kwanza.",
        "pain":         "Dalili ya kimwili. Ichukue kwa uzito.",
        "hopeless":     "Anapoteza matumaini. Uwepo kamili.",
        "lonely":       "Anahisi upweke. Mfanye ahisi kuonekana.",
        "positive":     "Yuko vizuri leo. Mkutane naye huko.",
        "casual":       "Anasalimia tu. Kuwa wa kawaida. Usifungue mada za kimatibabu bila sababu.",
        "question":     "Ana swali. Jibu moja kwa moja.",
        "anger":        "Amekasirika. Tambua.",
        "neutral":      "Hakuna ishara kali. Jibu kiasili.",
        "seeking_info": "Anataka jibu halisi. Lipe sasa, moja kwa moja.",
        "reporting":    "Anaelezea hali yake. Jibu maudhui ya alichosema.",
    },
}

_WEIGHT_GUIDANCE = {
    "en": {
        "very_short": "Her message was very short. Match it — 1 or 2 sentences max.",
        "short":      "Her message was short. Keep your reply brief.",
        "medium":     "She shared something real. Give it the space it deserves.",
        "long":       "She opened up. Be fully present.",
    },
    "fr": {
        "very_short": "Son message était très court. Assortis-toi — 1 ou 2 phrases max.",
        "short":      "Son message était court. Reste brève.",
        "medium":     "Elle a partagé quelque chose de vrai. Donne-lui l'espace.",
        "long":       "Elle s'est ouverte. Sois pleinement présente.",
    },
    "pt": {
        "very_short": "Mensagem muito curta. Combine — 1 ou 2 frases.",
        "short":      "Mensagem curta. Seja breve.",
        "medium":     "Ela compartilhou algo real. Dê o espaço que merece.",
        "long":       "Ela se abriu. Esteja totalmente presente.",
    },
    "sw": {
        "very_short": "Ujumbe mfupi sana. Linganisha — sentensi 1-2.",
        "short":      "Ujumbe mfupi. Jibu kwa ufupi.",
        "medium":     "Alishiriki kitu halisi. Mpe nafasi.",
        "long":       "Alifunguka. Kuwa hapa kikamilifu.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(
    memory, user_id, db_session,
    lang="en", emotion="neutral",
    is_danger=False, is_crisis=False,
    weight="short", sess=None,
    clinical_records=None,
    channel="chat",
) -> str:
    from ..models import User, CHWCase

    user = db_session.query(User).get(user_id)
    name = user.name if user else "her"

    mc             = memory.to_context_dict() if memory else {}
    recovery_phase = mc.get("recovery_phase") or "processing"
    memory_summary = mc.get("memory_summary") or "Still getting to know her."
    cultural       = mc.get("cultural_profile") or "mixed_transitional"
    low_moods      = mc.get("consecutive_low_moods") or 0
    vulnerability  = mc.get("vulnerability_level") or "medium"
    days_since     = mc.get("days_since_loss", "unknown")
    loss_type      = mc.get("loss_type", "pregnancy loss")
    prev_losses    = mc.get("previous_losses", 0)

    # Guard: user_type overrides stale recovery_phase
    user_type = getattr(user, "user_type", None)
    if user_type == "pregnant" and recovery_phase not in ("active_pregnancy","postnatal"):
        recovery_phase = "active_pregnancy"
    elif user_type == "loss" and recovery_phase == "active_pregnancy":
        recovery_phase = "processing"

    has_chw = db_session.query(CHWCase).filter_by(
        patient_id=user_id, status="assigned"
    ).first() is not None

    # Filter fragments
    all_fragments  = mc.get("things_she_shared") or []
    PHYSICAL_FRAGS = {"Reported bleeding","Reported pain","Reported fever",
                      "Reported headaches","Mentioned a hospital visit","Mentioned a doctor"}
    fragments = all_fragments if emotion == "pain" else [
        f for f in all_fragments if f not in PHYSICAL_FRAGS
    ]

    tone_map = {
        "rural_conservative": {
            "en": "Traditional background. Family matters. Keep language simple, grounded, warm.",
            "fr": "Milieu traditionnel. La famille compte. Reste simple, ancrée, chaleureuse.",
            "pt": "Meio tradicional. Família importa. Seja simples, directa, calorosa.",
            "sw": "Mazingira ya kimapokeo. Familia ni muhimu. Kuwa rahisi, wa joto.",
        },
        "mixed_transitional": {
            "en": "Balances tradition and modern life. Be warm and clear.",
            "fr": "Équilibre tradition et moderne. Sois chaleureuse et claire.",
            "pt": "Equilibra tradição e vida moderna. Seja calorosa e clara.",
            "sw": "Anasawazisha mila na kisasa. Kuwa wa joto na wazi.",
        },
        "urban_educated": {
            "en": "Comfortable with health info. Be direct, informed, and human.",
            "fr": "À l'aise avec l'info santé. Sois directe, informée, humaine.",
            "pt": "Confortável com informações de saúde. Seja directa, informada, humana.",
            "sw": "Yupo tayari na taarifa za afya. Kuwa wa moja kwa moja, wa kibinadamu.",
        },
    }
    tone_note    = tone_map.get(cultural, tone_map["mixed_transitional"]).get(lang, tone_map["mixed_transitional"]["en"])
    phase_note   = _PHASE_NOTES.get(lang, _PHASE_NOTES["en"]).get(recovery_phase, "")
    emotion_note = _EMOTION_GUIDANCE.get(lang, _EMOTION_GUIDANCE["en"]).get(emotion, "")
    weight_note  = _WEIGHT_GUIDANCE.get(lang, _WEIGHT_GUIDANCE["en"]).get(weight, "")

    depth = sess.depth if sess else 0
    if depth <= 1:
        depth_note = "First exchange. Introduce yourself briefly and invite her to share."
    elif depth < 5:
        depth_note = "Still getting to know each other. Be warm and genuinely curious."
    else:
        depth_note = "You have real history with her now. Reference what she has shared the way a real person would."

    if sess and len(sess.emotion_history) >= 3:
        recent_emos = sess.recent_emotions(3)
        if all(e in ("grief","hopeless","lonely","fear") for e in recent_emos):
            depth_note += " She has been in a heavy emotional space — be especially gentle."

    grief_note   = sess.grief_note() if sess else ""
    user_status  = sess.get_user_status() if sess else "unknown"

    status_guidance = ""
    if user_status == "delivered" or recovery_phase == "postnatal":
        status_guidance = "She has recently given birth. Focus on postnatal care."
    elif user_status == "pregnant" or recovery_phase == "active_pregnancy":
        status_guidance = (
            "CRITICAL: This woman is CURRENTLY PREGNANT. "
            "NEVER mention loss, grief, miscarriage, or anything negative about pregnancy outcomes. "
            "You are her pregnancy companion. Focus ONLY on her current pregnancy health."
        )
    elif recovery_phase in ("early_acute","processing","rebuilding","stabilised"):
        status_guidance = "She has experienced a pregnancy loss. Acknowledge grief only when she brings it up."

    clinical_note = ""
    if clinical_records:
        summaries = "\n".join([f"- {r['diagnosis']}: {r['summary']}" for r in clinical_records])
        clinical_note = f"\n━━━ RELEVANT CLINICAL CONTEXT ━━━\n{summaries}\nUse this to inform — do not quote directly."

    alerts = ""
    if is_danger:
        alerts += "\n🚨 DANGER SIGNS: Tell her exactly what to do in the next 30 minutes. Direct. Do not soften."
    if is_crisis:
        alerts += "\n🚨 CRISIS SIGNAL: She may be in emotional crisis. Stay with her. Get her to real help."
    if low_moods >= 3:
        alerts += f"\n⚠️ {low_moods} consecutive low mood days. Be especially present."
    if vulnerability == "high" and not has_chw:
        alerts += "\n⚠️ High vulnerability, no CHW. Recommend CHW contact if in crisis."

    channel_instructions = {
        "ussd":  "USSD channel — 155 characters MAX. Plain text only. No markdown, bullets, emojis. Warm but incredibly concise.",
        "voice": "Voice call — 2-3 short spoken sentences. Natural. Numbers as words. Calm.",
        "chat":  "Chat interface — 2-4 sentences. 1 emoji only if natural. Warm and present.",
    }
    channel_note = channel_instructions.get(channel, channel_instructions["chat"])

    lang_instruction = {
        "en": "Respond in English.",
        "fr": "Réponds en français.",
        "pt": "Responda em português.",
        "sw": "Jibu kwa Kiswahili.",
    }.get(lang, "Respond in English.")

    return f"""You are Healia — a maternal health companion at SafeMum.
You are NOT a chatbot. You are a specialist — combining the knowledge of a maternal health doctor, a grief therapist, and a health educator. You have deep clinical knowledge AND human warmth.

{lang_instruction}
{channel_note}
{status_guidance}

━━━ NATURAL CONVERSATION RULES ━━━
- Sound like a real person. Use contractions naturally.
- Never ask more than ONE question per response.
- If she just told you something important, respond to it BEFORE asking anything.
- Give solutions and advice, not just questions.
- Call her {name} occasionally — not every sentence.

━━━ WHO SHE IS ━━━
Name: {name} | Days since loss: {days_since} | Loss type: {loss_type} | Previous losses: {prev_losses}
Recovery phase: {recovery_phase} | Vulnerability: {vulnerability} | CHW assigned: {has_chw}

━━━ WHAT YOU KNOW ━━━
{memory_summary}

━━━ THINGS SHE HAS MENTIONED ━━━
{chr(10).join(f"- {f}" for f in fragments) if fragments else "- Still getting to know her"}
{alerts}
{clinical_note}

━━━ CONTEXT ━━━
{tone_note} | {phase_note} | {emotion_note} | {weight_note} | {depth_note}
{grief_note}

━━━ THREE MODES ━━━
EDUCATOR — when she asks anything: answer immediately, fully, specifically. Teach her.
SUPPORT — when she is emotional: one sentence acknowledgement → then something useful.
CLINICAL — when she reports danger: skip softening, tell her what to do in 30 minutes.

━━━ HARD RULES ━━━
- NEVER ask "how are you feeling?" if she just told you. She told you — use it.
- NEVER ask the same question twice.
- NEVER remind her of her loss when she is asking about something else.
- When she asks a question, answer it. Fully. Do not redirect to emotions instead.
- Vary how you open replies. Not every message starts with her name or with sympathy.
- Sound like a real expert who genuinely cares — specific, warm, knowledgeable, human."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT (chat + voice)
# ─────────────────────────────────────────────────────────────────────────────

def chat(user_message: str, user_id: int, db_session, channel: str = "chat") -> dict:
    from ..models import Conversation, AIMemory, User, MedicalProfile, Pregnancy
    from .classifier import classify_risk
    from .context_builder import get_user_context

    sentiment = enhanced_sentiment_analysis(user_message)

    sess = session_store[user_id]
    sess.update_disclosed_info(user_message)

    _, symptoms_in_msg, _ = extract_entities(user_message)
    ml_risk = {"risk_level": "low", "confidence": 0.0, "top_features": []}
    if symptoms_in_msg or detect_danger(user_message):
        try:
            user_context = get_user_context(user_id, db_session)
            symptom_dict = {s.lower().replace(" ","_"): 1 for s in symptoms_in_msg}
            symptom_dict.update({
                "pds101": user_context.get("age", 25),
                "pds102": user_context.get("urban_rural", "Urban"),
                "pds201": user_context.get("previous_pregnancies", 0),
                "pds202": user_context.get("previous_losses", 0),
                "county": user_context.get("county", "Unknown"),
            })
            ml_risk = classify_risk(symptom_dict)
        except Exception as e:
            print(f"[healia] Classifier error: {e}")

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

    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id)
        db_session.add(memory)
        db_session.flush()
        _seed_memory_from_profile(memory, user_id, db_session)

    lang      = detect_language(user_message)
    emotion   = sentiment["emotion"] if sentiment else detect_emotion(user_message)
    is_danger = detect_danger(user_message)
    is_crisis = detect_crisis(user_message)
    weight    = message_weight(user_message)
    topics, symptoms, conditions = extract_entities(user_message)
    sess.update(emotion, topics, symptoms, conditions, lang)

    clinical_records = []
    if _should_use_clinical_data(sess.symptoms_seen, sess.conditions_seen, user_message, sess.depth):
        clinical_records = _query_dataset(sess.symptoms_seen, sess.conditions_seen)

    system = _build_system_prompt(
        memory=memory, user_id=user_id, db_session=db_session,
        lang=lang, emotion=emotion, is_danger=is_danger, is_crisis=is_crisis,
        weight=weight, sess=sess, clinical_records=clinical_records, channel=channel,
    )

    if is_danger or (ml_risk.get("risk_level") == "high" and ml_risk.get("confidence", 0) > 0.6):
        system += (
            f"\n\n🚨 CLINICAL OVERRIDE: Danger sign reported. "
            f"ML: risk={ml_risk.get('risk_level')}, confidence={ml_risk.get('confidence',0):.0%}. "
            f"Tell her exactly what to do now. One clear instruction. Then one question only if it helps her act."
        )

    model      = _USSD_MODEL if channel == "ussd" else (_VOICE_MODEL if channel == "voice" else _CHAT_MODEL)
    max_tokens = 55 if channel == "ussd" else (130 if channel == "voice" else 650)
    temp       = 0.4 if channel == "ussd" else (0.5 if channel == "voice" else 0.78)

    groq_msgs = [{"role": "system", "content": system}]
    recent    = messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages
    for m in recent:
        if isinstance(m, dict) and "role" in m and "content" in m:
            groq_msgs.append({"role": m["role"], "content": m["content"]})
    groq_msgs.append({"role": "user", "content": user_message})

    try:
        resp  = client.chat.completions.create(model=model, messages=groq_msgs, temperature=temp, max_tokens=max_tokens)
        reply = resp.choices[0].message.content.strip()
        if channel == "ussd":
            reply = _truncate_ussd(reply)
    except Exception as e:
        print(f"[healia] Groq error: {e}")
        reply = _fallback_reply(lang)

    actions = _detect_actions(reply, user_message, is_danger, is_crisis)

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
# USSD  — strict prompt + word-aware truncation
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_ussd(text: str) -> str:
    """
    Truncate to USSD_CHAR_LIMIT on a word boundary so the reply never
    ends mid-word or mid-sentence like "I'm here fo".
    Also strips markdown characters that have no meaning on a basic phone.
    """
    text = re.sub(r"[*#\-_`]", "", text).strip()
    if len(text) <= USSD_CHAR_LIMIT:
        return text
    # Try to cut at last sentence boundary within limit
    cut = text[:USSD_CHAR_LIMIT]
    for sep in (". ", "! ", "? ", "\n"):
        last = cut.rfind(sep)
        if last > USSD_CHAR_LIMIT // 2:
            return cut[:last + 1].strip()
    # Fall back to last word boundary
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space].strip()
    return cut


_USSD_SYSTEM_PROMPT = """You are SafeMum, a maternal health assistant for women via USSD on a basic phone.

LANGUAGE: Detect from user input. Respond in the same language.

ABSOLUTE RULES — these override everything else:
1. Hard limit: 140 characters. Count before you respond.
2. Plain text only. No asterisks, no bullets, no markdown, no emojis, no dashes.
3. Match the tone of what was said. Casual message = casual reply. Grief message = warm short reply.
4. If she just said "thank you" or "hello" — respond warmly and briefly. Do NOT open with "I'm sorry for your loss."
5. Assess risk proportionally. Most messages are NOT emergencies.
6. LOW risk: warm advice, suggest clinic when convenient.
7. MEDIUM risk: advise clinic within 24 hours.
8. HIGH risk (heavy bleeding, severe pain, unconscious): urgent referral only.
9. Never diagnose. Be warm, human, concise.
10. One thought per reply. No compound sentences with multiple clauses."""


def ask_ussd(user_message: str, history: list, topic: str = "general") -> str:
    system = _USSD_SYSTEM_PROMPT

    # Topic-specific instruction appended concisely
    topic_hints = {
        "grief":  "\nThis is a grief conversation. Lead with warmth. One short sentence of presence, then one gentle question or offer.",
        "health": "\nHealth concern. If not dangerous, give one clear piece of advice. If dangerous, say go to clinic now.",
        "clinic": "\nWoman looking for a clinic. Give the most practical next step. Be specific if you can.",
    }
    if topic in topic_hints:
        system += topic_hints[topic]

    risk = risk_level(user_message)
    risk_hint = {
        "high":   "[HIGH RISK — urgent but calm]",
        "medium": "[MEDIUM RISK — suggest clinic soon]",
        "low":    "[LOW RISK — reassure warmly]",
    }[risk]

    messages = history + [{"role": "user", "content": f"{risk_hint}\n{user_message}"}]

    try:
        response = client.chat.completions.create(
            model      = _USSD_MODEL,
            max_tokens = 50,           # ~140 chars — tighter than before
            temperature= 0.35,
            messages   = [{"role": "system", "content": system}, *messages],
        )
        reply = response.choices[0].message.content.strip()
        return _truncate_ussd(reply)
    except Exception as e:
        print(f"[healia] USSD error: {e}")
        return "Service unavailable. / Service indisponible. / Serviço indisponível."


# ─────────────────────────────────────────────────────────────────────────────
# VOICE
# ─────────────────────────────────────────────────────────────────────────────

_VOICE_SYSTEM_PROMPT = """You are SafeMum, a compassionate maternal health assistant on a voice helpline.

LANGUAGE: Detect from speech. Respond in the same language.

RULES:
1. 2-3 short spoken sentences per response. Natural pauses.
2. Assess risk proportionally. Do not escalate unless truly severe.
3. Light symptoms: warm advice + clinic suggestion. Moderate: clinic soon. Severe: urgent referral.
4. For grief: be patient, warm, empathetic. Let her feel heard first.
5. Never diagnose. Encourage professional support.
6. Speak numbers as words: "zero eight hundred" not "0800".
7. Be calm, never rushed."""


def ask_voice(user_speech: str, history: list, topic: str = "general") -> str:
    system = _VOICE_SYSTEM_PROMPT
    if topic == "grief":
        system += "\n\nThis woman is calling for grief support. Be present first. Let her feel heard."

    risk      = risk_level(user_speech)
    risk_hint = {"high":"[Risk: HIGH]","medium":"[Risk: MEDIUM — suggest clinic soon]","low":"[Risk: LOW — reassure warmly]"}[risk]
    messages  = history + [{"role": "user", "content": f"{risk_hint}\n{user_speech}"}]

    try:
        response = client.chat.completions.create(
            model=_VOICE_MODEL, max_tokens=130, temperature=0.5,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Voice error: {e}")
        return "I am sorry, I am having trouble right now. Please call zero eight hundred, seven two three, two five three for free support."


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_symptoms(selected_symptoms: list, user_id: int, ml_risk: dict, db_session) -> dict:
    from ..models import AIMemory
    memory     = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    memory_ctx = memory.to_context_dict() if memory else {}

    COMPLICATION_MAP = {
        "Heavy bleeding":       "potential haemorrhage",
        "Fever":                "potential infection/sepsis",
        "Severe pain":          "potential incomplete abortion",
        "Chest pain":           "potential pulmonary embolism",
        "Cold hands/feet":      "potential shock",
        "Dizziness":            "potential haemorrhagic shock",
        "Foul discharge":       "potential pelvic infection",
        "Wound pain":           "potential surgical site infection",
        "Difficulty breathing": "potential pulmonary complication",
    }
    clinical_flags = [COMPLICATION_MAP[s] for s in selected_symptoms if s in COMPLICATION_MAP]

    prompt = f"""You are Healia from SafeMum. A woman checked her symptoms.
Profile: {json.dumps(memory_ctx, indent=2)}
Symptoms: {', '.join(selected_symptoms) if selected_symptoms else 'None'}
ML risk: level={ml_risk.get('risk_level','unknown')}, confidence={ml_risk.get('confidence',0):.0%}
Clinical significance: {', '.join(clinical_flags) if clinical_flags else 'No high-risk flags'}

JSON only:
{{"risk_level":"emergency|urgent|monitor|stable","title":"max 8 words","message":"2 sentences specific to her","reply":"Healia speaks — warm, direct, 2-3 sentences","action":"emergency_alert|open_map|talk_to_healia|rest_and_monitor","trigger_emergency_alert":bool,"assign_chw":bool,"map_action":{{"filter":"post_loss_care|emergency|nearest","reason":"one sentence"}} or null}}

Rules: emergency=heavy bleeding OR chest pain OR cold+dizziness. urgent=fever+foul discharge OR severe pain. monitor=single mild symptom. stable=no concerning symptoms. Never say "see a doctor" — say exactly what to do next hour."""

    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":"Clinical maternal health AI. JSON only."},{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=600,
        )
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Symptom error: {e}")
        high = any(s in selected_symptoms for s in ["Heavy bleeding","Chest pain","Cold hands/feet","Dizziness"])
        return {"risk_level":"urgent" if high else "monitor","title":"Please get checked today",
                "message":"Some of what you are experiencing needs attention. Do not wait.",
                "reply":"I noticed some symptoms that concern me. Let me help you find the nearest facility right now.",
                "action":"open_map" if high else "talk_to_healia","trigger_emergency_alert":high,"assign_chw":True,
                "map_action":{"filter":"emergency","reason":"Urgent symptoms detected"} if high else None}


# ─────────────────────────────────────────────────────────────────────────────
# MOOD CHECK-IN
# ─────────────────────────────────────────────────────────────────────────────

def interpret_checkin(mood_score: int, mood_label: str, notes: str, user_id: int, db_session) -> dict:
    from ..models import AIMemory
    memory = db_session.query(AIMemory).filter_by(user_id=user_id).first()
    if not memory:
        memory = AIMemory(user_id=user_id); db_session.add(memory); db_session.flush()

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

    prompt = f"""You are Healia from SafeMum. A woman completed her check-in.
Profile: {json.dumps(mc, indent=2)}
Mood: {mood_score}/5 ({mood_label}) | Note: "{notes or 'nothing written'}" | Consecutive low: {memory.consecutive_low_moods}

JSON only: {{"reply":"2-3 sentences — reference her note or history — 1 emoji only if natural","flag_for_counsellor":bool,"assign_chw":bool,"urgency":"none|low|high","follow_up_message":"what to check tomorrow"}}

flag_for_counsellor=true if consecutive_low>=3 or notes mention hopelessness/self-harm. assign_chw=true if mood_score==1 AND high vulnerability AND no CHW. urgency=high if self-harm mentioned."""

    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":"Compassionate maternal health companion. JSON only."},{"role":"user","content":prompt}],
            temperature=0.5, max_tokens=400,
        )
        result = json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Check-in error: {e}")
        result = {"reply":"I see you showing up today, and that matters. Take it one moment at a time.",
                  "flag_for_counsellor": memory.consecutive_low_moods >= 3,
                  "assign_chw": False,
                  "urgency": "low" if mood_score <= 2 else "none",
                  "follow_up_message": "How are you feeling today?"}

    if result.get("flag_for_counsellor"):
        memory.flagged_for_counsellor = True
    db_session.commit()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE GAP BRIEFING
# ─────────────────────────────────────────────────────────────────────────────

def interpret_service_gaps(gap_data: dict) -> dict:
    prompt = f"""Health ministry briefing — SafeMum service gap analysis. Data: {json.dumps(gap_data, indent=2)}.
JSON only: {{"headline":"one sentence","top_priority_counties":["top 3"],"key_finding":"2 sentences","recommended_action":"one specific step","data_note":"one sentence on limitations"}}"""
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":"Public health analyst. JSON only."},{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=400,
        )
        return json.loads(_clean_json(r.choices[0].message.content))
    except Exception as e:
        print(f"[healia] Service gap error: {e}")
        return {"headline":"Analysis unavailable","top_priority_counties":[],"key_finding":"Unable to generate insight.",
                "recommended_action":"Review raw data.","data_note":"System error."}


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _seed_memory_from_profile(memory, user_id, db_session):
    from ..models import MedicalProfile, Pregnancy, User

    user      = db_session.query(User).get(user_id)
    user_type = getattr(user, "user_type", None)

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
        elif pregnancy.status == "active":
            memory.loss_type = None; memory.days_since_loss = None
            memory.recovery_phase = "active_pregnancy"
        elif pregnancy.status == "delivered":
            memory.loss_type = None; memory.recovery_phase = "postnatal"
        else:
            memory.loss_type = None; memory.days_since_loss = None
    else:
        if user_type == "pregnant":
            memory.loss_type = None; memory.days_since_loss = None
            memory.recovery_phase = "active_pregnancy"
        elif user_type == "loss":
            memory.loss_type = "pregnancy loss"; memory.recovery_phase = "processing"
        else:
            memory.recovery_phase = "general_support"

    days = memory.days_since_loss
    if days is not None:
        if   days <= 14: memory.recovery_phase = "early_acute"
        elif days <= 42: memory.recovery_phase = "processing"
        elif days <= 84: memory.recovery_phase = "rebuilding"
        else:            memory.recovery_phase = "stabilised"
    elif memory.recovery_phase not in ("active_pregnancy","postnatal","general_support"):
        memory.recovery_phase = "general_support"


def _rebuild_memory_summary(memory, recent_messages: list):
    if not recent_messages: return
    convo = "\n".join([f"{m['role'].upper()}: {m['content']}"
                       for m in recent_messages
                       if isinstance(m, dict) and "role" in m and "content" in m])
    prompt = f"""Read this conversation between Healia and a woman. Write a short paragraph (max 5 sentences) capturing who she is: emotional state, what she shared, recurring concerns, how she communicates. One caring person briefing another. No clinical labels. No bullets.\n\n{convo}"""
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=300,
        )
        memory.memory_summary = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[healia] Memory rebuild error: {e}")


def _extract_memory_fragments(user_message: str, memory):
    msg = user_message.lower(); fragments = list(memory.things_she_shared or [])
    triggers = [
        ("husband","Mentioned her husband"),("partner","Mentioned her partner"),
        ("child","Mentioned having a child"),("son","Mentioned her son"),
        ("daughter","Mentioned her daughter"),("mother","Mentioned her mother"),
        ("work","Mentioned work or job"),("sleep","Mentioned sleep issues"),
        ("guilt","Expressed feelings of guilt"),("afraid","Expressed fear"),
        ("scared","Expressed being scared"),("alone","Expressed feeling alone"),
        ("hope","Expressed hope"),("better","Said she is feeling better"),
        ("hospital","Mentioned a hospital visit"),("doctor","Mentioned a doctor"),
        ("bleeding","Reported bleeding"),("pain","Reported pain"),
        ("fever","Reported fever"),("headache","Reported headaches"),
        ("overthink","Mentioned overthinking"),("stress","Mentioned stress"),
        ("faith","Mentioned faith or religion"),("prayer","Mentioned prayer"),
        ("pregnant","Mentioned being pregnant"),("pregnancy","Discussed her pregnancy"),
        ("nutrition","Asked about nutrition"),("exercise","Asked about exercise"),
        ("prenatal","Asked about prenatal care"),
    ]
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
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_context_prefix(topic: str) -> str:
    contexts = {
        "health": "Woman is describing a health concern after pregnancy loss.",
        "clinic": "Woman is trying to find a nearby health facility.",
        "grief":  "Woman is seeking emotional and grief support after pregnancy loss.",
        "chw":    "Woman wants to connect with a Community Health Worker.",
    }
    return contexts.get(topic, "")

def is_emergency(text: str) -> bool:
    return detect_danger(text)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_reply(lang: str) -> str:
    msgs = {
        "en": "I'm here with you. I had a small technical issue — could you send that again?",
        "fr": "Je suis là avec toi. J'ai eu un petit problème technique — peux-tu renvoyer ça ?",
        "pt": "Estou aqui com você. Tive um pequeno problema técnico — pode enviar novamente?",
        "sw": "Niko hapa nawe. Nilikuwa na tatizo dogo la kiufundi — unaweza kutuma tena?",
    }
    return msgs.get(lang, msgs["en"])

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```"); raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"): raw = raw[4:]
    return raw.strip()

def _now() -> str:
    return datetime.utcnow().isoformat()