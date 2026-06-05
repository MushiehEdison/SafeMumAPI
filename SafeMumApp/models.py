from . import db
from sqlalchemy.types import JSON
from datetime import datetime, date


# ─────────────────────────────────────────────
# EXISTING MODELS (unchanged)
# ─────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(15), nullable=False, unique=True)
    language = db.Column(db.String(32), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    user_type = db.Column(db.String(20), nullable=True)                # NEW: 'pregnant' | 'loss'
    latitude  = db.Column(db.Float, nullable=True)                     # NEW: patient location
    longitude = db.Column(db.Float, nullable=True)                     # NEW: patient location

    # Relationships
    conversations = db.relationship('Conversation', back_populates='user', lazy=True)
    medical_profile = db.relationship('MedicalProfile', back_populates='user', uselist=False, lazy=True)
    sessions = db.relationship('UserSession', back_populates='user', lazy=True)
    treatment_preferences = db.relationship('TreatmentPreference', back_populates='user', lazy=True)
    health_literacy = db.relationship('HealthLiteracy', back_populates='user', lazy=True)

    # SafeMum relationships
    pregnancies = db.relationship('Pregnancy', back_populates='user', lazy=True)
    notifications = db.relationship('Notification', back_populates='user', lazy=True)
    referrals = db.relationship('Referral', back_populates='patient', lazy=True)
    emergency_alerts = db.relationship('EmergencyAlert', back_populates='patient', lazy=True)
    tip_deliveries = db.relationship('TipDelivery', back_populates='patient', lazy=True)
    support_requests = db.relationship('SupportRequest', back_populates='patient', lazy=True)
    chw_cases = db.relationship('CHWCase', back_populates='patient', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'


class Conversation(db.Model):
    __tablename__ = 'conversation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.String(20), nullable=False, default='health_assistant')
    # type options: 'health_assistant', 'recovery_hub'
    messages = db.Column(JSON, nullable=False, default=[])
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', back_populates='conversations')
    message_indices = db.relationship('MessageIndex', back_populates='conversation', lazy=True)
    symptom_entries = db.relationship('SymptomEntry', back_populates='conversation', lazy=True)
    diagnoses = db.relationship('Diagnosis', back_populates='conversation', lazy=True)
    sentiment_records = db.relationship('SentimentRecord', back_populates='conversation', lazy=True)

    def __repr__(self):
        return f'<Conversation {self.id} for user {self.user_id}>'


class MedicalProfile(db.Model):
    __tablename__ = 'medical_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    marital_status = db.Column(db.String(20), nullable=True)
    nationality = db.Column(db.String(50), nullable=True)
    region = db.Column(db.String(50), nullable=True)
    city = db.Column(db.String(50), nullable=True)
    quarter = db.Column(db.String(50), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    profession = db.Column(db.String(100), nullable=True)
    emergency_contact = db.Column(db.String(100), nullable=True)
    emergency_relation = db.Column(db.String(50), nullable=True)
    emergency_phone = db.Column(db.String(20), nullable=True)
    blood_type = db.Column(db.String(5), nullable=True)
    genotype = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    chronic_conditions = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    primary_hospital = db.Column(db.String(100), nullable=True)
    primary_physician = db.Column(db.String(100), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    vaccination_history = db.Column(db.Text, nullable=True)
    last_dental_visit = db.Column(db.Date, nullable=True)
    last_eye_exam = db.Column(db.Date, nullable=True)
    lifestyle = db.Column(
        db.JSON,
        nullable=True,
        default=lambda: {
            'smokes': False,
            'alcohol': 'Never',
            'exercise': 'Never',
            'diet': 'Balanced'
        }
    )
    family_history = db.Column(db.Text, nullable=True)

    user = db.relationship('User', back_populates='medical_profile')

    @property
    def age(self):
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'dateOfBirth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'marital_status': self.marital_status,
            'nationality': self.nationality,
            'region': self.region,
            'city': self.city,
            'quarter': self.quarter,
            'address': self.address,
            'profession': self.profession,
            'emergency_contact': self.emergency_contact,
            'emergency_relation': self.emergency_relation,
            'emergency_phone': self.emergency_phone,
            'blood_type': self.blood_type,
            'genotype': self.genotype,
            'allergies': self.allergies,
            'chronic_conditions': self.chronic_conditions,
            'medications': self.medications,
            'primary_hospital': self.primary_hospital,
            'primary_physician': self.primary_physician,
            'medical_history': self.medical_history,
            'vaccination_history': self.vaccination_history,
            'lastDentalVisit': self.last_dental_visit.isoformat() if self.last_dental_visit else None,
            'lastEyeExam': self.last_eye_exam.isoformat() if self.last_eye_exam else None,
            'lifestyle': self.lifestyle or {},
            'family_history': self.family_history
        }


class MessageIndex(db.Model):
    __tablename__ = 'message_index'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    conversation = db.relationship('Conversation', back_populates='message_indices')

    def __repr__(self):
        return f'<MessageIndex {self.id} for conversation {self.convo_id}>'


class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    last_active = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='sessions')

    def __repr__(self):
        return f'<UserSession {self.id} for user {self.user_id}>'


class SymptomEntry(db.Model):
    __tablename__ = 'symptom_entries'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    symptom_name = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(20), nullable=True)
    reported_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    location = db.Column(db.String(100), nullable=True)

    conversation = db.relationship('Conversation', back_populates='symptom_entries')

    def __repr__(self):
        return f'<SymptomEntry {self.id} for conversation {self.convo_id}>'


class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    condition_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    requires_attention = db.Column(db.Boolean, default=False, nullable=False)

    conversation = db.relationship('Conversation', back_populates='diagnoses')

    def __repr__(self):
        return f'<Diagnosis {self.id} for conversation {self.convo_id}>'


class HealthAlert(db.Model):
    __tablename__ = 'health_alerts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    region = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f'<HealthAlert {self.id} - {self.title}>'


class SentimentRecord(db.Model):
    __tablename__ = 'sentiment_records'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sentiment_category = db.Column(db.String(20), nullable=False)
    percentage = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # SafeMum addition — flag if AI detected emotional distress
    ai_flag = db.Column(db.Boolean, default=False, nullable=False)
    referred_to_counsellor = db.Column(db.Boolean, default=False, nullable=False)

    conversation = db.relationship('Conversation', back_populates='sentiment_records')

    def __repr__(self):
        return f'<SentimentRecord {self.id} for conversation {self.convo_id}>'


class TreatmentPreference(db.Model):
    __tablename__ = 'treatment_preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    treatment_type = db.Column(db.String(100), nullable=False)
    preference_score = db.Column(db.Float, nullable=False)
    trend = db.Column(db.String(20), nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='treatment_preferences')

    def __repr__(self):
        return f'<TreatmentPreference {self.id} for user {self.user_id}>'


class HealthLiteracy(db.Model):
    __tablename__ = 'health_literacy'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age_group = db.Column(db.String(20), nullable=False)
    understanding_rate = db.Column(db.Float, nullable=False)
    engagement_rate = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='health_literacy')

    def __repr__(self):
        return f'<HealthLiteracy {self.id} for user {self.user_id}>'


class WorkflowMetric(db.Model):
    __tablename__ = 'workflow_metrics'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    change_percentage = db.Column(db.Float, nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<WorkflowMetric {self.id} - {self.metric_name}>'


class AIPerformance(db.Model):
    __tablename__ = 'ai_performance'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<AIPerformance {self.id} - {self.metric_name}>'


class CommunicationMetric(db.Model):
    __tablename__ = 'communication_metrics'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    current_value = db.Column(db.Float, nullable=False)
    previous_value = db.Column(db.Float, nullable=True)
    trend = db.Column(db.String(20), nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    time_range = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f'<CommunicationMetric {self.id} - {self.metric_name}>'


# ─────────────────────────────────────────────
# SAFEMUM MODELS (new additions)
# ─────────────────────────────────────────────

class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='admin')
    # role options: 'admin', 'super_admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Admin {self.email}>'


class Hospital(db.Model):
    __tablename__ = 'hospitals'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    county = db.Column(db.String(100), nullable=True)
    district = db.Column(db.String(100), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    facility_level = db.Column(db.String(50), nullable=True)
    is_verified = db.Column(db.Boolean, default=False, nullable=False) # NEW
    # inside Hospital:
    icu                    = db.Column(db.Boolean, default=False)
    available_beds         = db.Column(db.Integer, default=0)
    staff_on_duty          = db.Column(db.Integer, default=0)
    estimated_wait_minutes = db.Column(db.Integer, default=0)
    cap_reasons = db.Column(db.Text, nullable=True)
    
    # options: dispensary, health_centre, hospital, referral_hospital
    ownership = db.Column(db.String(30), nullable=True)
    # options: public, private, faith_based
    has_blood_bank = db.Column(db.Boolean, default=False)
    has_surgical = db.Column(db.Boolean, default=False)
    has_maternity = db.Column(db.Boolean, default=False)
    has_post_loss_care = db.Column(db.Boolean, default=False)
    is_available = db.Column(db.Boolean, default=True)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    referrals_received = db.relationship('Referral', back_populates='hospital', lazy=True)
    alerts_received = db.relationship('EmergencyAlert', back_populates='hospital', lazy=True)
    notifications_sent = db.relationship('Notification', back_populates='hospital', lazy=True)

    def __repr__(self):
        return f'<Hospital {self.name}>'


class CommunityHealthWorker(db.Model):
    __tablename__ = 'community_health_workers'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    institution = db.Column(db.String(200), nullable=True)
    coverage_area = db.Column(db.String(200), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    coverage_radius_km = db.Column(db.Float, default=5.0)
    speciality = db.Column(db.String(100), nullable=True)
    # options: nurse, midwife, volunteer, counsellor
    is_available = db.Column(db.Boolean, default=True)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False, nullable=False) 
    qualification = db.Column(db.String(50), nullable=True) 
    years_experience = db.Column(db.String(30), nullable=True)

    # Relationships
    cases = db.relationship('CHWCase', back_populates='chw', lazy=True)
    alerts = db.relationship('EmergencyAlert', back_populates='chw', lazy=True)
    referrals = db.relationship('Referral', back_populates='chw', lazy=True)

    def __repr__(self):
        return f'<CHW {self.full_name}>'




class Pregnancy(db.Model):
    __tablename__ = 'pregnancies'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    gestational_age_weeks = db.Column(db.Integer, nullable=True)
    expected_delivery = db.Column(db.Date, nullable=True)
    antenatal_visits_done = db.Column(db.Integer, default=0)
    last_visit_date = db.Column(db.Date, nullable=True)
    risk_level = db.Column(db.String(20), default='low')
    # options: low, moderate, high
    status = db.Column(db.String(20), default='active')
    # options: active, lost, delivered
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', back_populates='pregnancies')

    def __repr__(self):
        return f'<Pregnancy {self.id} for user {self.user_id} - {self.status}>'


class Referral(db.Model):
    __tablename__ = 'referrals'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospitals.id'), nullable=False)
    chw_id = db.Column(db.Integer, db.ForeignKey('community_health_workers.id'), nullable=True)
    reason = db.Column(db.Text, nullable=True)
    symptoms_reported = db.Column(db.Text, nullable=True)
    risk_level = db.Column(db.String(20), nullable=False)
    # options: moderate, high, emergency
    distance_km = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='pending')
    # options: pending, acknowledged, completed, declined
    alert_sent_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    decline_reason  = db.Column(db.Text, nullable=True)
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    completed_at    = db.Column(db.DateTime, nullable=True)

    patient = db.relationship('User', back_populates='referrals')
    hospital = db.relationship('Hospital', back_populates='referrals_received')
    chw = db.relationship('CommunityHealthWorker', back_populates='referrals')

    def __repr__(self):
        return f'<Referral {self.id} → Hospital {self.hospital_id} [{self.status}]>'


class EmergencyAlert(db.Model):
    __tablename__ = 'emergency_alerts'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospitals.id'), nullable=False)
    chw_id = db.Column(db.Integer, db.ForeignKey('community_health_workers.id'), nullable=True)
    symptoms_reported = db.Column(db.Text, nullable=True)
    risk_classification = db.Column(db.String(100), nullable=True)
    patient_latitude = db.Column(db.Float, nullable=True)
    patient_longitude = db.Column(db.Float, nullable=True)
    channel = db.Column(db.String(20), nullable=False, default='app')
    outcome = db.Column(db.Text, nullable=True)
    # options: app, ussd, voice, whatsapp
    status = db.Column(db.String(20), default='sent')
    # options: sent, acknowledged, resolved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('User', back_populates='emergency_alerts')
    hospital = db.relationship('Hospital', back_populates='alerts_received')
    chw = db.relationship('CommunityHealthWorker', back_populates='alerts')

    def __repr__(self):
        return f'<EmergencyAlert {self.id} for patient {self.patient_id} [{self.status}]>'


class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    hospital_id = db.Column(db.Integer, db.ForeignKey('hospitals.id'), nullable=True)
    chw_id = db.Column(db.Integer, db.ForeignKey('community_health_workers.id'), nullable=True)
    type = db.Column(db.String(50), nullable=False)
    # options: antenatal_reminder, danger_sign, hospital_alert, chw_alert, weekly_tip
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', back_populates='notifications')
    hospital = db.relationship('Hospital', back_populates='notifications_sent')

    def __repr__(self):
        return f'<Notification {self.id} for user {self.user_id} [{self.type}]>'


class PregnancyTip(db.Model):
    __tablename__ = 'pregnancy_tips'
    id = db.Column(db.Integer, primary_key=True)
    week_number = db.Column(db.Integer, nullable=False)  # 1 to 40
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=True)
    # options: nutrition, danger_signs, antenatal_visits, lifestyle, post_loss
    language = db.Column(db.String(32), nullable=False, default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    deliveries = db.relationship('TipDelivery', back_populates='tip', lazy=True)

    def __repr__(self):
        return f'<PregnancyTip week {self.week_number} - {self.title}>'


class TipDelivery(db.Model):
    __tablename__ = 'tip_deliveries'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tip_id = db.Column(db.Integer, db.ForeignKey('pregnancy_tips.id'), nullable=False)
    delivered_at = db.Column(db.DateTime, default=datetime.utcnow)
    channel = db.Column(db.String(20), nullable=False, default='app')
    # options: sms, whatsapp, app, ussd
    is_read = db.Column(db.Boolean, default=False)

    patient = db.relationship('User', back_populates='tip_deliveries')
    tip = db.relationship('PregnancyTip', back_populates='deliveries')

    def __repr__(self):
        return f'<TipDelivery tip {self.tip_id} → patient {self.patient_id}>'


class SupportRequest(db.Model):
    __tablename__ = 'support_requests'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ngo_id = db.Column(db.Integer, db.ForeignKey('ngos.id'), nullable=True)
    type = db.Column(db.String(50), nullable=False)
    # options: transport, financial_aid, counselling
    description = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='pending')
    # options: pending, accepted, resolved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('User', back_populates='support_requests')
    ngo = db.relationship('NGO', back_populates='support_requests')

    def __repr__(self):
        return f'<SupportRequest {self.id} [{self.type}] - {self.status}>'


class CHWCase(db.Model):
    __tablename__ = 'chw_cases'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    chw_id = db.Column(db.Integer, db.ForeignKey('community_health_workers.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='assigned')
    # options: assigned, contacted, visited, escalated, resolved
    notes = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = db.relationship('User', back_populates='chw_cases')
    chw = db.relationship('CommunityHealthWorker', back_populates='cases')

    def __repr__(self):
        return f'<CHWCase {self.id} patient {self.patient_id} → CHW {self.chw_id} [{self.status}]>'

class NGO(db.Model):
    __tablename__ = 'ngos'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    address = db.Column(db.String(255), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    services = db.Column(db.JSON, nullable=True)
    # example: ["transport", "financial_aid", "counselling"]
    coverage_area = db.Column(db.String(200), nullable=True)
    is_available = db.Column(db.Boolean, default=True)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    support_requests = db.relationship('SupportRequest', back_populates='ngo', lazy=True)

    def __repr__(self):
        return f'<NGO {self.name}>'



class CheckIn(db.Model):
    __tablename__ = 'check_ins'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mood       = db.Column(db.String(120), nullable=False)
    note       = db.Column(db.Text, nullable=True)
    conclusion = db.Column(db.Text, nullable=True) 
    chw_note         = db.Column(db.Text, nullable=True)                
    chw_responded_at = db.Column(db.DateTime, nullable=True)   
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('check_ins', lazy=True))


class CommunityPost(db.Model):
    __tablename__ = 'community_posts'
    id         = db.Column(db.Integer, primary_key=True)
    content    = db.Column(db.Text, nullable=False)       # always anonymous
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    replies = db.relationship('CommunityReply', back_populates='post',
                               order_by='CommunityReply.created_at', lazy=True)


class CommunityReply(db.Model):
    __tablename__ = 'community_replies'
    id         = db.Column(db.Integer, primary_key=True)
    post_id    = db.Column(db.Integer, db.ForeignKey('community_posts.id'), nullable=False)
    content    = db.Column(db.Text, nullable=False)   
    is_chw     = db.Column(db.Boolean, default=False, nullable=False)  
    chw_name   = db.Column(db.String(120), nullable=True)    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    post = db.relationship('CommunityPost', back_populates='replies')


class Reminder(db.Model):
    __tablename__ = 'reminders'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type         = db.Column(db.String(60), nullable=False)
    datetime_str = db.Column(db.String(60), nullable=False)   # human string, e.g. "Jun 3, 2025 at 08:00"
    note         = db.Column(db.Text, nullable=True)
    ai_message   = db.Column(db.Text, nullable=True)
    missed_count = db.Column(db.Integer, default=0, nullable=False)
    completed    = db.Column(db.Boolean, default=False, nullable=False)
    overdue      = db.Column(db.Boolean, default=False, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('reminders', lazy=True))



class AIMemory(db.Model):
    __tablename__ = 'ai_memory'
 
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
 
    # Core facts — populated from medical profile + pregnancy + user input
    loss_type           = db.Column(db.String(100), nullable=True)   # miscarriage, stillbirth, etc.
    days_since_loss     = db.Column(db.Integer, nullable=True)
    previous_losses     = db.Column(db.Integer, default=0)
    cultural_profile    = db.Column(db.String(60), nullable=True)    # from ML model
    vulnerability_level = db.Column(db.String(20), nullable=True)    # low / medium / high
    
    # Emotional patterns learned from conversations
    recurring_themes    = db.Column(db.JSON, nullable=True, default=list)
    # e.g. ["guilt", "partner support issues", "sleep problems"]
 
    things_she_shared   = db.Column(db.JSON, nullable=True, default=list)
    # Important things she said the AI should remember
    # e.g. ["lost her mother last year", "husband travels for work", "has two other children"]
 
    # Mood tracking for counsellor flag
    consecutive_low_moods    = db.Column(db.Integer, default=0)
    last_mood_score          = db.Column(db.Integer, nullable=True)   # 1-5
    total_checkins           = db.Column(db.Integer, default=0)
    flagged_for_counsellor   = db.Column(db.Boolean, default=False)
 
    # Recovery progress
    recovery_phase   = db.Column(db.String(50), nullable=True)
    # options: early_acute (0-2 weeks), processing (2-6 weeks),
    #          rebuilding (6-12 weeks), stabilised (12+ weeks)
 
    # App engagement
    last_active_days_ago  = db.Column(db.Integer, nullable=True)
    reminders_missed      = db.Column(db.Integer, default=0)
    checkin_streak        = db.Column(db.Integer, default=0)
 
    # AI-generated plain text summary — rebuilt every 10 messages
    memory_summary = db.Column(db.Text, nullable=True)
    # e.g. "Sarah lost her baby at 28 weeks in March 2025. She has two children.
    #        Her husband is supportive but travels. She struggles most with guilt
    #        and sleep. Mood has been low for 3 consecutive days. Prefers direct
    #        but warm communication."
 
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
 
    user = db.relationship('User', backref=db.backref('ai_memory', uselist=False, lazy=True))
 
    def to_context_dict(self):
        """Returns a clean dict the AI assistant uses as context."""
        return {
            'loss_type':             self.loss_type,
            'days_since_loss':       self.days_since_loss,
            'previous_losses':       self.previous_losses,
            'cultural_profile':      self.cultural_profile,
            'vulnerability_level':   self.vulnerability_level,
            'recurring_themes':      self.recurring_themes or [],
            'things_she_shared':     self.things_she_shared or [],
            'consecutive_low_moods': self.consecutive_low_moods,
            'last_mood_score':       self.last_mood_score,
            'recovery_phase':        self.recovery_phase,
            'last_active_days_ago':  self.last_active_days_ago,
            'reminders_missed':      self.reminders_missed,
            'checkin_streak':        self.checkin_streak,
            'memory_summary':        self.memory_summary,
        }
 
    def __repr__(self):
        return f'<AIMemory user={self.user_id}>'


class JournalEntry(db.Model):
    __tablename__ = "journal_entries"
 
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    content    = db.Column(db.Text, nullable=False)
    mood_tag   = db.Column(db.String(32), nullable=True)   # e.g. "Hopeful", "Heavy"
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
 
    user = db.relationship("User", backref=db.backref("journal_entries", lazy="dynamic"))
 
    def __repr__(self):
        return f"<JournalEntry id={self.id} user={self.user_id} mood={self.mood_tag}>"

class InsightReport(db.Model):
    __tablename__ = 'insight_reports'
    id         = db.Column(db.Integer, primary_key=True)
    tag        = db.Column(db.String(50))      # "Geography", "Care Gaps", etc.
    title      = db.Column(db.String(200))
    body       = db.Column(db.Text)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    period     = db.Column(db.String(20))      # "2025-05" monthly key


class DashboardSnapshot(db.Model):
    __tablename__ = 'dashboard_snapshots'
    id           = db.Column(db.Integer, primary_key=True)
    snapshot_key = db.Column(db.String(100))   # e.g. "loss_geography_2025-05"
    data         = db.Column(db.JSON)           # the actual chart data
    computed_at  = db.Column(db.DateTime, default=datetime.utcnow)