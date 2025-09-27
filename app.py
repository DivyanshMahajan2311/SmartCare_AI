from flask import Flask, render_template, Response, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sqlite3
from datetime import datetime
import io

import cv2
import time
from deepface import DeepFace


app = Flask(__name__)

app.secret_key = "your_super_secret_key_123"

@app.route('/')
def home():
    return render_template('home.html', active='home')

@app.route('/about')
def about():
    return render_template('about.html', active='about')

@app.route('/personal-care')
def personal_care():
    return render_template('personal_care.html', active='personal-care')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active='chatbot')

@app.route('/emotion-detection')
def emotion_detection():
    return render_template('emotion_detection.html', active='emotion-detection')

# PHQ-9 content (same text as your console app)
PHQ_QUESTIONS = [
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?",
    "3. Trouble falling/staying asleep, or sleeping too much?",
    "4. Feeling tired or having little energy?",
    "5. Poor appetite or overeating?",
    "6. Feeling bad about yourself, or that you are a failure?",
    "7. Trouble concentrating (school, reading, watching TV, etc.)?",
    "8. Moving or speaking slowly OR being restless/fidgety?",
    "9. Thoughts of being better off dead or of hurting yourself?"
]

PHQ_OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

def interpret_score(score):
    if score <= 4:
        return "🌱 Minimal or no depression"
    elif score <= 9:
        return "🙂 Mild depression"
    elif score <= 14:
        return "😐 Moderate depression"
    elif score <= 19:
        return "😟 Moderately severe depression"
    else:
        return "😢 Severe depression"

def precautions(score):
    if score <= 4:
        return [
            "✅ Keep a healthy daily routine.",
            "✅ Maintain regular sleep and exercise.",
            "✅ Talk to friends/family if stressed."
        ]
    elif score <= 9:
        return [
            "📝 Monitor mood regularly.",
            "✅ Keep good sleep, nutrition, and exercise habits.",
            "💬 Talk to a trusted adult or counselor if feeling stressed."
        ]
    elif score <= 14:
        return [
            "⚠ Consider consulting a school counselor or mental health professional.",
            "📝 Maintain a daily routine and journaling.",
            "💬 Talk openly with family/friends about feelings."
        ]
    elif score <= 19:
        return [
            "⚠ Schedule an appointment with a mental health professional.",
            "📝 Avoid isolation; stay in touch with supportive friends/family.",
            "🚨 Seek help immediately if having thoughts of self-harm."
        ]
    else:
        return [
            "🚨 High risk! Contact a mental health professional immediately.",
            "📞 Reach out to suicide helpline if feeling unsafe.",
            "💬 Talk to a trusted adult, family member, or counselor immediately."
        ]

@app.route('/depression-detection', methods=['GET', 'POST'])
def depression_detection():
    # On GET: render the form. On POST: collect answers, compute score, render results.
    results = None
    score = None
    advice = None
    responses = None

    if request.method == 'POST':
        # Collect answers q1..q9 from form. If missing, treat as 0.
        responses = []
        for i in range(1, len(PHQ_QUESTIONS) + 1):
            val = request.form.get(f"q{i}", "")
            try:
                iv = int(val)
            except (ValueError, TypeError):
                iv = 0
            # clamp to valid 0-3
            if iv < 0 or iv > 3:
                iv = 0
            responses.append(iv)

        score = sum(responses)
        results = interpret_score(score)
        advice = precautions(score)

    return render_template(
        'depression_detection.html',
        active='personal-care',   # keep "Personal Care" highlighted in nav (optional)
        questions=PHQ_QUESTIONS,
        options=PHQ_OPTIONS,
        responses=responses,
        score=score,
        result_text=results,
        advice=advice,
        enumerate=enumerate
    )
    
# Peer Pressure Part 

# ---------- Peer Pressure / School Pressure page (Flask integration of Streamlit logic) ----------

# Questions (same as your Streamlit script)
ACADEMIC_QUESTIONS = [
    "I feel stressed by my homework and assignments.",
    "I worry about my grades most of the time.",
    "I feel I must outperform my peers.",
    "I sacrifice sleep to complete schoolwork."
]

PEER_QUESTIONS = [
    "I do things because my friends expect me to.",
    "I change my behavior to fit in with my peers.",
    "I feel anxious if I’m not accepted by my friends."
]

OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

# Ensure VADER lexicon exists (download once if needed)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def detect_sentiment_label_confidence(text: str):
    if not text or not text.strip():
        return "Neutral", 0.0
    scores = sid.polarity_scores(text)
    compound = scores["compound"]
    label = "Negative" if compound < -0.2 else "Neutral" if compound < 0.2 else "Positive"
    confidence = abs(compound)
    return label, confidence

@app.route('/peer-pressure', methods=['GET', 'POST'])
def peer_pressure():
    # On GET: render form; on POST: compute and show results
    results = None
    combined_score = None
    stress_level = None
    text_sentiment = None
    text_confidence = None
    academic_total = 0
    peer_total = 0
    academic_responses = None
    peer_responses = None
    user_text = ""

    if request.method == 'POST':
        # collect academic responses a1..aN
        academic_responses = []
        for i in range(1, len(ACADEMIC_QUESTIONS) + 1):
            v = request.form.get(f"a{i}", "")
            try:
                iv = int(v)
            except (ValueError, TypeError):
                iv = 0
            if iv < 0 or iv > 3:
                iv = 0
            academic_responses.append(iv)
        academic_total = sum(academic_responses)

        # collect peer responses p1..pN
        peer_responses = []
        for i in range(1, len(PEER_QUESTIONS) + 1):
            v = request.form.get(f"p{i}", "")
            try:
                iv = int(v)
            except (ValueError, TypeError):
                iv = 0
            if iv < 0 or iv > 3:
                iv = 0
            peer_responses.append(iv)
        peer_total = sum(peer_responses)

        # text input
        user_text = request.form.get('user_text', '').strip()
        text_label, text_confidence = detect_sentiment_label_confidence(user_text)
        text_sentiment = text_label
        text_confidence = float(text_confidence)

        # Combined scoring (same heuristic as your Streamlit file)
        combined_score = academic_total + peer_total + (text_confidence * 10)

        if combined_score >= 20:
            stress_level = "High"
        elif combined_score >= 12:
            stress_level = "Moderate"
        else:
            stress_level = "Low"

        # small results dict to show if necessary
        results = {
            "academic_total": academic_total,
            "peer_total": peer_total,
            "text_sentiment": text_sentiment,
            "text_confidence": text_confidence,
            "combined_score": combined_score,
            "stress_level": stress_level
        }

    return render_template(
        'peer_pressure.html',
        active='personal-care',      # highlight Personal Care nav (keeps layout unchanged)
        academic_questions=ACADEMIC_QUESTIONS,
        peer_questions=PEER_QUESTIONS,
        options=OPTIONS,
        results=results,
        academic_total=academic_total,
        peer_total=peer_total,
        user_text=user_text
    )
    
# Reproductive Part 

# add these imports at top with others
import os
import joblib
import pandas as pd

# -----------------------------
# PCOS / Reproductive Problems (Flask integration)
# -----------------------------
FEATURES = [
    "Age", "Weight", "Height", "BMI", "Cycle(R/I)", "Weight gain(Y/N)",
    "Hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
    "Fast food (Y/N)", "Reg.Exercise(Y/N)"
]

# try to load model & scaler once at startup; if absent, keep as None and surface an error on the page
PCOS_MODEL = None
PCOS_SCALER = None
PCOS_LOAD_ERROR = None
_model_path = os.path.join(os.path.dirname(__file__), "pcos_svm_model.pkl")
_scaler_path = os.path.join(os.path.dirname(__file__), "pcos_scaler.pkl")

try:
    PCOS_MODEL = joblib.load(_model_path)
    PCOS_SCALER = joblib.load(_scaler_path)
except Exception as e:
    PCOS_MODEL = None
    PCOS_SCALER = None
    PCOS_LOAD_ERROR = str(e)

def predict_pcos_pcod(input_data):
    """
    input_data: list of values in the exact order of FEATURES
    returns: (label_str, probability_or_None)
    """
    if PCOS_MODEL is None or PCOS_SCALER is None:
        raise RuntimeError("Model or scaler not loaded: " + (PCOS_LOAD_ERROR or "unknown"))

    df_input = pd.DataFrame([input_data], columns=FEATURES)
    try:
        X_scaled = PCOS_SCALER.transform(df_input)
    except Exception as e:
        raise RuntimeError("Scaler transform failed: " + str(e))

    pred = PCOS_MODEL.predict(X_scaled)[0]
    prob = None
    if hasattr(PCOS_MODEL, "predict_proba"):
        try:
            prob = PCOS_MODEL.predict_proba(X_scaled)[0][1]
        except Exception:
            prob = None

    label = "Likely PCOS/PCOD" if pred == 1 else "Unlikely PCOS/PCOD"
    return label, prob

@app.route('/pcos_pcod', methods=['GET', 'POST'])
def pcos_pcod():
    """
    Renders the Reproductive Problems page (PCOS/PCOD prediction).
    On POST: reads form, computes BMI, predicts and shows result.
    """
    error = None
    result_label = None
    result_prob = None

    # default form values
    form_vals = {
        'Age': 18,
        'Weight': 68,
        'Height': 160,
        'BMI': None,
        'Cycle': 'Regular',
        'Weight_gain': 'No',
        'Hair_growth': 'No',
        'Skin_darkening': 'No',
        'Hair_loss': 'No',
        'Pimples': 'No',
        'Fast_food': 'No',
        'Exercise': 'Yes'
    }

    if request.method == 'POST':
        # read and sanitize inputs
        try:
            Age = int(request.form.get('Age', form_vals['Age']))
        except (ValueError, TypeError):
            Age = form_vals['Age']

        try:
            Weight = float(request.form.get('Weight', form_vals['Weight']))
        except (ValueError, TypeError):
            Weight = form_vals['Weight']

        try:
            Height = float(request.form.get('Height', form_vals['Height']))
        except (ValueError, TypeError):
            Height = form_vals['Height']

        # calculate BMI safely
        BMI = round(Weight / ((Height / 100) ** 2), 2) if Height > 0 else 0.0

        # binary fields
        Cycle = request.form.get('Cycle', form_vals['Cycle'])
        Cycle_RI = 1 if Cycle == 'Regular' else 0

        def yn_to_int(field_name, default='No'):
            val = request.form.get(field_name, default)
            return 1 if val == 'Yes' else 0

        Weight_gain_YN = yn_to_int('Weight_gain', 'No')
        Hair_growth_YN = yn_to_int('Hair_growth', 'No')
        Skin_darkening_YN = yn_to_int('Skin_darkening', 'No')
        Hair_loss_YN = yn_to_int('Hair_loss', 'No')
        Pimples_YN = yn_to_int('Pimples', 'No')
        Fast_food_YN = yn_to_int('Fast_food', 'No')
        Exercise_YN = yn_to_int('Exercise', 'Yes')

        # prepare input list matching FEATURES order
        input_data = [
            Age, Weight, Height, BMI, Cycle_RI, Weight_gain_YN,
            Hair_growth_YN, Skin_darkening_YN, Hair_loss_YN, Pimples_YN,
            Fast_food_YN, Exercise_YN
        ]

        # attempt prediction
        try:
            result_label, result_prob = predict_pcos_pcod(input_data)
        except Exception as e:
            error = str(e)

        # keep values to re-populate form
        form_vals.update({
            'Age': Age, 'Weight': Weight, 'Height': Height, 'BMI': BMI,
            'Cycle': Cycle,
            'Weight_gain': 'Yes' if Weight_gain_YN else 'No',
            'Hair_growth': 'Yes' if Hair_growth_YN else 'No',
            'Skin_darkening': 'Yes' if Skin_darkening_YN else 'No',
            'Hair_loss': 'Yes' if Hair_loss_YN else 'No',
            'Pimples': 'Yes' if Pimples_YN else 'No',
            'Fast_food': 'Yes' if Fast_food_YN else 'No',
            'Exercise': 'Yes' if Exercise_YN else 'No'
        })

    # render template (keeps active='personal-care' so nav remains same)
    return render_template(
        'pcos_pcod.html',
        active='personal-care',
        form_vals=form_vals,
        result_label=result_label,
        result_prob=result_prob,
        error=error,
        model_loaded=(PCOS_MODEL is not None and PCOS_SCALER is not None)
    )
    

# ---------- SmartCare routes ----------
from flask import render_template, request, redirect, url_for

# ---------------------------
# Clinical Knowledge (expandable)
# ---------------------------
CONDITIONS = {
    "PCOS": {
        "symptoms": ["irregular periods", "heavy periods", "acne", "weight gain", "excess hair", "infertility"],
        "advice": "Lifestyle modification, consult a gynecologist/endocrinologist for tests (hormones, ultrasound)."
    },
    "Anemia": {
        "symptoms": ["fatigue", "weakness", "pale skin", "dizziness", "shortness of breath"],
        "advice": "Get CBC (hemoglobin). Consider iron-rich diet and consult a physician."
    },
    "Endometriosis": {
        "symptoms": ["severe cramps", "pelvic pain", "painful intercourse", "infertility"],
        "advice": "Refer to gynecologist; they may recommend imaging and specialist care."
    },
    "Irregular Cycle (functional)": {
        "symptoms": ["irregular periods", "missed periods", "very long or short cycles"],
        "advice": "Track cycles for 3 months, see a doctor if persistent (hormone tests may be needed)."
    },
    "Normal Puberty": {
        "symptoms": ["mood swings", "growth spurts", "acne", "irregular periods (initial years)"],
        "advice": "Often normal; track and seek advice if severe or very delayed."
    }
}


# ---------------------------
# Doctor list (shared by SmartCare & Symptom Checker)
# ---------------------------
DOCTORS = [
    {"name": "Dr. Priya Sharma", "specialty": "Gynecologist", "clinic": "City Women's Clinic", "phone": "123-456-7890"},
    {"name": "Dr. Amit Verma", "specialty": "Pediatric Endocrinologist", "clinic": "Youth Health Center", "phone": "234-567-8901"},
    {"name": "Dr. S. Rao", "specialty": "General Practitioner", "clinic": "Family Care Clinic", "phone": "345-678-9012"},
]


# Flatten symptom set for UI (used by smartcare form)
try:
    ALL_KNOWN_SYMPTOMS = sorted({s for cond in CONDITIONS.values() for s in cond["symptoms"]})
except Exception:
    ALL_KNOWN_SYMPTOMS = []
# ------------------------------# --- smartcare flask integration (add to your existing app.py) ---
from flask import session, flash, send_file
import csv
import os

# ensure Flask app has a secret key
try:
    app.secret_key  # if already set, keep it
except Exception:
    app.secret_key = "replace-this-with-a-secure-secret"

DB_PATH = "teen_health.db"   # reuse your existing DB file




# Database helpers (reuse same connection pattern)
import sqlite3
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_smartcare_db():
    conn = get_db_conn()
    cur = conn.cursor()
    # smartcare_records stores entries per user (user_id nullable for Guest)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS smartcare_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            date TEXT,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            cycle_length_days INTEGER,
            mood TEXT,
            sleep_hours REAL,
            symptoms TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_users_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

# call both at startup
init_smartcare_db()
init_users_db()

# Save and fetch smartcare records
def save_smartcare_record_db(user_id, username, record: dict):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""INSERT INTO smartcare_records
                   (user_id, username, date, height_cm, weight_kg, bmi, cycle_length_days, mood, sleep_hours, symptoms, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id,
                 username,
                 record.get("date"),
                 record.get("height_cm"),
                 record.get("weight_kg"),
                 record.get("bmi"),
                 record.get("cycle_length_days"),
                 record.get("mood"),
                 record.get("sleep_hours"),
                 ",".join(record.get("symptoms", [])) if isinstance(record.get("symptoms", []), list) else record.get("symptoms",""),
                 record.get("notes","")
                ))
    conn.commit()
    conn.close()

def get_smartcare_history_for_user(user_id, username):
    conn = get_db_conn()
    cur = conn.cursor()
    if user_id is not None:
        cur.execute("SELECT * FROM smartcare_records WHERE user_id = ? ORDER BY date DESC", (user_id,))
    else:
        # show guest records saved under the guest username (usually 'Guest')
        cur.execute("SELECT * FROM smartcare_records WHERE username = ? ORDER BY date DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# user helpers (if not already defined elsewhere)
import hashlib, datetime

def hash_password(password: str):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def create_user(username, email, password):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), datetime.datetime.now().isoformat()))
        conn.commit()
        return True, "User created"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row['password_hash'] == hash_password(password):
        return True, row['id']
    return False, None


# ---------- SmartCare routes ----------
from flask import render_template, request, redirect, url_for

# ---------------------------
# Symptom checker helpers (Flask version of Streamlit logic)
# ---------------------------
def analyze_symptoms(selected_symptoms):
    """
    selected_symptoms: list[str]
    returns: list of tuples [(cond_name, {"score":..., "matched":[...], "advice":...}), ...]
    """
    scores = {}
    for cond_name, info in CONDITIONS.items():
        cond_symptoms = set(info["symptoms"])
        matched = cond_symptoms.intersection(set(selected_symptoms))
        score = len(matched) / max(len(cond_symptoms), 1)
        if score > 0:
            scores[cond_name] = {"score": score, "matched": list(matched), "advice": info["advice"]}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return sorted_scores

# def recommend_doctors_from_analysis(analysis_list):
#     """
#     analysis_list: output of analyze_symptoms (list of tuples)
#     returns: list of doctor dicts (deduplicated)
#     """
#     recs = []
#     for cond, _info in analysis_list:
#         if cond in ["PCOS", "Endometriosis", "Irregular Cycle (functional)"]:
#             recs.extend([d for d in DOCTORS if d["specialty"] == "Gynecologist"])
#         elif cond in ["Normal Puberty"]:
#             recs.extend([d for d in DOCTORS if d["specialty"] == "General Practitioner"])
#         else:
#             recs.extend(DOCTORS[:1])
#     # deduplicate
#     seen = set()
#     out = []
#     for d in recs:
#         key = (d["name"], d["clinic"])
#         if key not in seen:
#             out.append(d)
#             seen.add(key)
#     return out


def recommend_doctors_from_analysis(analysis_list):
    """
    analysis_list: output of analyze_symptoms (list of tuples)
    returns: list of doctor dicts (deduplicated). Uses global DOCTORS if available,
    otherwise uses a small built-in default to avoid NameError.
    """
    # Default fallback list (used only if global DOCTORS not defined)
    default_doctors = [
        {"name": "Dr. Priya Sharma", "specialty": "Gynecologist", "clinic": "City Women's Clinic", "phone": "123-456-7890"},
        {"name": "Dr. Amit Verma", "specialty": "Pediatric Endocrinologist", "clinic": "Youth Health Center", "phone": "234-567-8901"},
        {"name": "Dr. S. Rao", "specialty": "General Practitioner", "clinic": "Family Care Clinic", "phone": "345-678-9012"},
    ]

    # Ensure we have a doctors list available
    doctors_list = globals().get("DOCTORS", default_doctors)
    if not isinstance(doctors_list, list):
        doctors_list = default_doctors

    recs = []
    for cond, _info in analysis_list or []:
        # pick specialists according to condition
        if cond in ["PCOS", "Endometriosis", "Irregular Cycle (functional)"]:
            recs.extend([d for d in doctors_list if d.get("specialty", "").lower().startswith("gyn") or "gynec" in d.get("specialty","").lower()])
        elif cond in ["Normal Puberty"]:
            recs.extend([d for d in doctors_list if "general" in d.get("specialty","").lower() or "gp" in d.get("specialty","").lower()])
        else:
            # fallback: add the first doctor(s)
            recs.extend(doctors_list[:1])

    # deduplicate by (name, clinic)
    seen = set()
    out = []
    for d in recs:
        key = (d.get("name"), d.get("clinic"))
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out


# @app.route('/smartcare', methods=['GET', 'POST'])
# def smartcare():
#     """
#     Main SmartCare page:
#     - shows auth area (Login / Sign up / Continue as Guest)
#     - shows entry form (height, weight, mood, sleep, cycle, symptoms, notes)
#     - on POST: saves record for the logged-in user (or Guest) and redirects to history
#     """
#     # session user controls (works like Streamlit flow)
#     uid = session.get('user_id')      # may be None for guest
#     uname = session.get('username', 'Guest')

#     message = None
#     # handle form submission to save data
#     if request.method == 'POST' and request.form.get('action') == 'save_record':
#         # parse inputs
#         try:
#             height = float(request.form.get('height', 0))
#         except:
#             height = None
#         try:
#             weight = float(request.form.get('weight', 0))
#         except:
#             weight = None
#         bmi = None
#         if height and weight:
#             try:
#                 bmi = round(weight / ((height/100.0)**2), 2)
#             except:
#                 bmi = None
#         try:
#             cycle_length = int(request.form.get('cycle_length')) if request.form.get('cycle_length') else None
#         except:
#             cycle_length = None
#         mood = request.form.get('mood')
#         try:
#             sleep_hours = float(request.form.get('sleep_hours')) if request.form.get('sleep_hours') else None
#         except:
#             sleep_hours = None
#         # symptoms come as comma-separated (we allow both multi-select or plain text)
#         symptoms_raw = request.form.get('symptoms', '')
#         if request.form.getlist('symptoms_multi'):
#             symptoms = request.form.getlist('symptoms_multi')
#         else:
#             symptoms = [s.strip() for s in symptoms_raw.split(",")] if symptoms_raw else []
#         notes = request.form.get('notes','')

#         record = {
#             "date": datetime.date.today().isoformat(),
#             "height_cm": height,
#             "weight_kg": weight,
#             "bmi": bmi,
#             "cycle_length_days": cycle_length,
#             "mood": mood,
#             "sleep_hours": sleep_hours,
#             "symptoms": symptoms,
#             "notes": notes
#         }

#         # save with session user context (uid may be None for Guest)
#         username_to_save = uname or "Guest"
#         save_smartcare_record_db(uid, username_to_save, record)
#         flash("Record saved.")
#         return redirect(url_for('smartcare_history'))

#     # GET: render the page
#     return render_template('smartcare.html',
#                            username=uname,
#                            user_id=uid,
#                            all_symptoms=ALL_KNOWN_SYMPTOMS)

@app.route('/smartcare', methods=['GET', 'POST'])
def smartcare():
    """
    Main SmartCare page:
    - shows auth area (Login / Sign up / Continue as Guest)
    - shows entry form (height, weight, mood, sleep, cycle, symptoms, notes)
    - on POST: saves record for the logged-in user (or Guest) and redirects to history
    - supports Symptom Checker (action=analyze_symptoms)
    """
    uid = session.get('user_id')      # may be None for guest
    uname = session.get('username', 'Guest')

    message = None

    # Variables for symptom-checker results to pass to template
    analysis_results = None
    doctor_recs = None
    selected_symptoms = []

    # handle form submission to save data
    if request.method == 'POST':
        action = request.form.get('action', '')

        if action == 'save_record':
            # parse inputs (same as before)
            try:
                height = float(request.form.get('height', 0))
            except:
                height = None
            try:
                weight = float(request.form.get('weight', 0))
            except:
                weight = None
            bmi = None
            if height and weight:
                try:
                    bmi = round(weight / ((height/100.0)**2), 2)
                except:
                    bmi = None
            try:
                cycle_length = int(request.form.get('cycle_length')) if request.form.get('cycle_length') else None
            except:
                cycle_length = None
            mood = request.form.get('mood')
            try:
                sleep_hours = float(request.form.get('sleep_hours')) if request.form.get('sleep_hours') else None
            except:
                sleep_hours = None
            # symptoms come as comma-separated (we allow both multi-select or plain text)
            symptoms_raw = request.form.get('symptoms', '')
            if request.form.getlist('symptoms_multi'):
                symptoms = request.form.getlist('symptoms_multi')
            else:
                symptoms = [s.strip() for s in symptoms_raw.split(",")] if symptoms_raw else []
            notes = request.form.get('notes','')

            record = {
                "date": datetime.date.today().isoformat(),
                "height_cm": height,
                "weight_kg": weight,
                "bmi": bmi,
                "cycle_length_days": cycle_length,
                "mood": mood,
                "sleep_hours": sleep_hours,
                "symptoms": symptoms,
                "notes": notes
            }

            # save with session user context (uid may be None for Guest)
            username_to_save = uname or "Guest"
            save_smartcare_record_db(uid, username_to_save, record)
            flash("Record saved.")
            return redirect(url_for('smartcare_history'))

        elif action == 'analyze_symptoms':
            # Collect symptoms for analysis (multi-check or comma text)
            if request.form.getlist('symptoms_multi'):
                selected_symptoms = request.form.getlist('symptoms_multi')
            else:
                raw = request.form.get('symptoms', '')
                selected_symptoms = [s.strip() for s in raw.split(",") if s.strip()]
            # run analysis
            analysis_results = analyze_symptoms(selected_symptoms)
            doctor_recs = recommend_doctors_from_analysis(analysis_results)

    # GET or POST without redirect: render the page with current context
    return render_template('smartcare.html',
                           username=uname,
                           user_id=uid,
                           all_symptoms=ALL_KNOWN_SYMPTOMS,
                           analysis_results=analysis_results,
                           doctor_recs=doctor_recs,
                           selected_symptoms=selected_symptoms)

@app.route('/smartcare/history')
def smartcare_history():
    uid = session.get('user_id')
    uname = session.get('username', 'Guest')
    records = get_smartcare_history_for_user(uid, uname)
    # allow CSV download via query param ?download=1
    if request.args.get('download') == '1':
        # create CSV in-memory
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(["date","height_cm","weight_kg","bmi","cycle_length_days","mood","sleep_hours","symptoms","notes"])
        for r in records:
            cw.writerow([r.get('date'), r.get('height_cm'), r.get('weight_kg'), r.get('bmi'),
                         r.get('cycle_length_days'), r.get('mood'), r.get('sleep_hours'),
                         r.get('symptoms'), r.get('notes')])
        mem = io.BytesIO()
        mem.write(si.getvalue().encode('utf-8'))
        mem.seek(0)
        return send_file(mem, mimetype='text/csv', as_attachment=True,
                         download_name=f"smartcare_history_{uname}.csv")
    return render_template('smartcare_history.html', records=records, username=uname)

@app.route('/smartcare/signup', methods=['POST'])
def smartcare_signup():
    username = request.form.get('su_user')
    email = request.form.get('su_email','')
    password = request.form.get('su_pass')
    if not username or not password:
        flash("Please provide username and password for sign up.")
        return redirect(url_for('smartcare'))
    ok, msg = create_user(username, email, password)
    if ok:
        # auto-login after signup
        ok2, uid = authenticate_user(username, password)
        if ok2:
            session['user_id'] = uid
            session['username'] = username
            flash("Account created and logged in.")
    else:
        flash(msg)
    return redirect(url_for('smartcare'))

@app.route('/smartcare/login', methods=['POST'])
def smartcare_login():
    username = request.form.get('li_user')
    password = request.form.get('li_pass')
    ok, uid = authenticate_user(username, password)
    if ok:
        session['user_id'] = uid
        session['username'] = username
        flash(f"Logged in as {username}.")
    else:
        flash("Invalid credentials.")
    return redirect(url_for('smartcare'))

@app.route('/smartcare/guest', methods=['POST'])
def smartcare_guest():
    session['user_id'] = None
    session['username'] = 'Guest'
    flash("Continuing as Guest.")
    return redirect(url_for('smartcare'))

@app.route('/smartcare/logout')
def smartcare_logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash("Logged out.")
    return redirect(url_for('smartcare'))


from flask import Flask, render_template, Response
import cv2
import time
from deepface import DeepFace


@app.route('/face-emotion')
def face_emotion():
    return render_template('face_emotion.html', active='emotion-detection')


# Initialize camera
cap = cv2.VideoCapture(0)

# Variables for emotion detection
last_time = time.time()
emotion = ""

def generate_frames():
    global last_time, emotion
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = time.time()
            if current_time - last_time > 5:  # update every 5 seconds
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    last_time = current_time
                except:
                    pass

            # Overlay emotion text on frame
            cv2.putText(frame, emotion, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream to browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
