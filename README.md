# SafeMum AI — Backend API

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3--70B-green)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)

A Flask REST API backend for **SafeMum AI** — a post-pregnancy loss care platform for Sub-Saharan Africa. Built for the AI for Reproductive Health in Africa Hackathon 2026, Track I.

 [github.com/MushiehEdison/SAFEMUM-AI](https://github.com/MushiehEdison/SAFEMUM-AI)

---

## What this backend does

- Handles authentication for five actor types — patient, CHW, facility, NGO, admin — using JWT stored in httpOnly cookies
- Powers the AI health assistant using a 3-layer pipeline: scikit-learn ML models → Groq LLM interpretation → personalised app response
- Manages emergency alerts, smart referrals, CHW case assignments, and facility routing
- Serves real facility data seeded from 328 Kenyan health facilities
- Exposes USSD and voice call webhooks for Africa's Talking integration (offline access layer)
- Generates data-driven insights for the admin dashboard using trained ML models

---

## Tech stack

| Layer | Technology |
|---|---|
| Framework | Flask 3.x, Python 3.8+ |
| Database | PostgreSQL 16 via SQLAlchemy ORM |
| Migrations | Flask-Migrate (Alembic) |
| Auth | Flask-JWT-Extended — httpOnly cookies |
| Password hashing | Flask-Bcrypt |
| CORS | Flask-CORS |
| AI / LLM | Groq API — LLaMA3-70B |
| ML models | scikit-learn — Random Forest, Logistic Regression, Gradient Boosting, KMeans |
| Offline access | Africa's Talking — USSD and Voice webhooks |
| Production server | Gunicorn |
| Deployment | Railway |

---

## Folder structure

```
safemumAPI/
├── app.py                          ← entry point
├── requirements.txt
├── .env                            ← not committed
├── .env.example
└── SafeMumApp/
    ├── __init__.py                 ← app factory, create_app(), blueprint registration
    ├── config.py                   ← all config loaded from .env
    ├── models.py                   ← all 26 SQLAlchemy models in one file
    ├── services.py
    │
    ├── Routes/
    │   ├── patient/
    │   │   ├── auth.py             ← register, OTP login, logout, /me
    │   │   ├── home.py             ← home page data, recovery overview
    │   │   ├── chat.py             ← AI assistant conversations
    │   │   ├── recovery.py         ← SafeRecovery Hub, check-ins, community
    │   │   ├── reminders.py        ← reminders CRUD
    │   │   ├── emergency.py        ← emergency alert trigger
    │   │   ├── map.py              ← facility map, search, filter
    │   │   ├── profile.py          ← medical profile, pregnancy history
    │   │   └── voice_ai.py         ← voice call AI handler
    │   │
    │   ├── chw/
    │   │   ├── auth.py             ← register, login, logout
    │   │   ├── dashboard.py        ← stats, urgent cases, activity
    │   │   ├── cases.py            ← case list, case detail, update status
    │   │   ├── patients.py         ← patient lookup
    │   │   ├── profile.py          ← CHW profile, coverage area
    │   │   └── chw_community.py
    │   │
    │   ├── facility/
    │   │   ├── auth.py             ← register, login, logout
    │   │   ├── dashboard.py        ← incoming alerts, referrals, stats
    │   │   ├── alerts.py           ← alert management, acknowledge, resolve
    │   │   ├── referrals.py        ← referral accept/decline
    │   │   └── profile.py          ← facility profile, capabilities update
    │   │
    │   └── admin/
    │       ├── auth.py             ← admin login
    │       └── insight.py          ← insights, reports, heatmap data
    │
    ├── Ai_Analysis/
    │   ├── ai_assistant.py         ← main AI assistant logic
    │   ├── classifier.py           ← loads all ML models, exposes predict functions
    │   ├── context_builder.py      ← builds full user context from DB for LLM
    │   ├── interpreter.py          ← Groq LLM calls, prompt construction
    │   ├── dataset_interpreter.py  ← dataset knowledge for AI context
    │   ├── pipeline.py             ← orchestrates ML → LLM → response
    │   │
    │   ├── Models/                 ← trained .joblib files (not committed)
    │   │   ├── risk_classifier.joblib
    │   │   ├── repeat_loss_predictor.joblib
    │   │   ├── care_seeking_predictor.joblib
    │   │   ├── cultural_profile_segmenter.joblib
    │   │   ├── isolation_detector.joblib
    │   │   ├── facility_delivery_predictor.joblib
    │   │   ├── service_gap_cluster.joblib
    │   │   ├── vulnerability_index.csv
    │   │   └── service_gap_analysis.csv
    │   │
    │   ├── Datasets/               ← research CSV files (not committed)
    │   │   ├── ddi_pds_data.csv
    │   │   ├── ddi_hfs_data.csv
    │   │   ├── woman_final.csv
    │   │   ├── AKU_baseline.csv
    │   │   ├── AKU_endline.csv
    │   │   ├── pamanech_woman_data.csv
    │   │   └── W1 Mother Focal Child File-ANON.csv
    │   │
    │   └── Training/               ← run once to train models
    │       ├── train_risk_classifier.py
    │       ├── train_repeat_loss_predictor.py
    │       ├── train_care_seeking_predictor.py
    │       ├── train_cultural_profile_segmenter.py
    │       ├── train_isolation_detector.py
    │       ├── train_facility_delivery_predictor.py
    │       ├── train_service_gap_cluster.py
    │       └── build_social_vulnerability_index.py
    │
    ├── OfflineCom/                 ← offline access layer
    │   ├── ussd.py                 ← USSD webhook handler (Africa's Talking)
    │   ├── voice.py                ← voice call AI handler
    │   ├── ai.py                   ← offline AI logic
    │   ├── location_utils.py       ← GPS and location helpers
    │   └── session_store.py        ← USSD session management
    │
    └── utils/
        ├── decorators.py           ← role-based auth decorators
        ├── chw_assignment.py       ← automatic CHW matching logic
        └── __init__.py
```

---

## ML models

Eight models trained on five published research datasets covering 6,560 patient records, 328 health facilities, and 127 community health volunteers across Kenya.

| Model | Algorithm | Dataset | What it drives |
|---|---|---|---|
| Risk Classifier | Random Forest | ddi_pds_data.csv | Emergency alerts, referral urgency, mascot mood |
| Repeat Loss Predictor | Logistic Regression | ddi_pds_data.csv | CHW priority, reminder intensity |
| Care Seeking Predictor | Gradient Boosting | woman_final.csv | Post-referral CHW assignment |
| Service Gap Cluster | KMeans | ddi_pds + ddi_hfs | Admin heatmap, county prioritisation |
| Social Vulnerability Index | Composite score | W1 Mother Focal Child | SafeRecovery Hub routing, CHW assignment |
| Facility Delivery Predictor | Random Forest | pamanech_woman_data.csv | Prevention flag, home delivery risk |
| Cultural Profile Segmenter | KMeans | AKU_baseline.csv | AI assistant tone personalisation |
| Isolation Detector | Gradient Boosting | W1 Mother Focal Child | Proactive CHW outreach trigger |

---

## AI pipeline architecture

```
Woman reports symptoms
        ↓
Layer 1 — ML Models (scikit-learn)
classifier.py runs risk, repeat loss, care seeking, isolation, cultural models
        ↓
Layer 2 — Context Builder
context_builder.py queries DB — mood trend, behaviour, pregnancy history, care network
        ↓
Layer 3 — LLM Interpretation (Groq — LLaMA3-70B)
interpreter.py builds prompt, sends to Groq, parses structured JSON response
        ↓
App acts — warm chat message, CHW assigned, alert fired, mascot mood set
```

---

## Authentication

Five actor types, each with their own login and role-based access:

| Actor | Auth method | Token location |
|---|---|---|
| Patient | Passwordless OTP via phone | httpOnly cookie |
| CHW | Email + password | httpOnly cookie |
| Facility | Email + password | httpOnly cookie |
| NGO | Email + password | httpOnly cookie |
| Admin | Email + password | httpOnly cookie |

Role is baked into the JWT as a claim. Custom decorators protect every route:

```python
@bp.route('/profile', methods=['GET'])
@patient_required
def get_profile():
    user_id = get_current_user_id()
    ...
```

---

## API endpoints overview

```
Patient
  POST   /api/patient/auth/register
  POST   /api/patient/auth/request-otp
  POST   /api/patient/auth/login
  POST   /api/patient/auth/logout
  GET    /api/patient/auth/me
  GET    /api/patient/home
  GET    /api/patient/chat/conversations
  POST   /api/patient/chat/send
  POST   /api/patient/recovery/checkin
  GET    /api/patient/recovery/checkins
  GET    /api/patient/community/posts
  POST   /api/patient/community/posts
  GET    /api/patient/reminders
  POST   /api/patient/reminders
  POST   /api/patient/emergency/alert
  GET    /api/patient/map/facilities

CHW
  POST   /api/chw/auth/register
  POST   /api/chw/auth/login
  GET    /api/chw/dashboard
  GET    /api/chw/cases
  GET    /api/chw/cases/<id>
  PATCH  /api/chw/cases/<id>/update
  POST   /api/chw/cases/<id>/escalate

Facility
  POST   /api/facility/auth/register
  POST   /api/facility/auth/login
  GET    /api/facility/dashboard
  GET    /api/facility/alerts
  PATCH  /api/facility/alerts/<id>/acknowledge
  GET    /api/facility/referrals
  PATCH  /api/facility/referrals/<id>
  PUT    /api/facility/capabilities

Admin
  POST   /api/admin/auth/login
  GET    /api/admin/insights
  GET    /api/admin/insights/service-gaps
  GET    /api/admin/insights/reports

Offline
  POST   /api/ussd                  
  POST   /api/voice                 

Public
  GET    /api/ping                  ← health check
  GET    /api/facilities            ← public facility list for map
```

---



**Prerequisites:** Python 3.8+, PostgreSQL running locally or on Railway

```bash
# 1. Clone the repo
git clone https://github.com/MushiehEdison/safemumAPI.git
cd safemumAPI

# 2. Create and activate virtual environment
python -m venv safemum
safemum\Scripts\activate        # Windows
source safemum/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your values

# 5. Run database migrations
flask db upgrade

# 6. Start the server
python app.py
```

Server runs on `http://localhost:5000`

---

## Environment variables

Create a `.env` file at the root. See `.env.example` for the template.

```
DATABASE_URL=postgresql://user:password@localhost:5432/safemum
JWT_SECRET_KEY=your-long-random-secret-key
GROQ_API_KEY=your-groq-api-key
AFRICASTALKING_API_KEY=your-at-api-key
AFRICASTALKING_USERNAME=your-at-username
FRONTEND_URL=http://localhost:5173
FLASK_ENV=development
FLASK_APP=app.py
PORT=5000
```

---

## Training the ML models

Models are pre-trained and saved as `.joblib` files in `SafeMumApp/Ai_Analysis/Models/`. If you need to retrain:

```bash
cd SafeMumApp/Ai_Analysis/Training

python train_risk_classifier.py
python train_repeat_loss_predictor.py
python train_care_seeking_predictor.py
python train_cultural_profile_segmenter.py
python train_isolation_detector.py
python train_facility_delivery_predictor.py
python train_service_gap_cluster.py
python build_social_vulnerability_index.py
```

Datasets must be present in `SafeMumApp/Ai_Analysis/Datasets/` before running. Datasets and trained model files are excluded from version control via `.gitignore`.

---

## Deployment on Railway

1. Push repo to GitHub
2. Create new project on Railway and connect the repo
3. Add all environment variables from `.env` in the Railway dashboard
4. Set the start command: `gunicorn app:app`
5. Railway auto-deploys on every push to main

---

## Related

- [github.com/MushiehEdison/SAFEMUM-AI](https://github.com/yourname/SAFEMUM-AI)
- **Hackathon:** AI for Reproductive Health in Africa 2026 — Track I
- **Team:** SafeMum AI 
