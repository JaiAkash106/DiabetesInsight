"""Streamlit app for diabetes risk prediction."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from src.predict import load_model, predict_single
from src.user_store import create_user, ensure_users_csv, upsert_user_prediction, user_exists

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/diabetes.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
TARGET_COL = os.getenv("TARGET_COL", "Outcome")
USERS_CSV_PATH = os.getenv("USERS_CSV_PATH", "data/users.csv")

# Expected input fields for UI
FEATURES = {
    "Pregnancies": {"label": "Pregnancies", "type": "number", "min": 0, "max": 20, "value": 1},
    "Glucose": {"label": "Glucose", "type": "number", "min": 0, "max": 250, "value": 100},
    "BloodPressure": {"label": "Blood Pressure", "type": "number", "min": 0, "max": 200, "value": 70},
    "SkinThickness": {"label": "Skin Thickness", "type": "number", "min": 0, "max": 100, "value": 20},
    "Insulin": {"label": "Insulin", "type": "number", "min": 0, "max": 900, "value": 80},
    "BMI": {"label": "BMI", "type": "number", "min": 10.0, "max": 70.0, "value": 25.0},
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "type": "number",
        "min": 0.0,
        "max": 3.0,
        "value": 0.5,
    },
    "GeneticRisk": {
        "label": "Genetic Risk (Yes/No)",
        "type": "radio",
        "options": ["No", "Yes"],
        "value": "No",
    },
    "Age": {"label": "Age", "type": "number", "min": 1, "max": 120, "value": 30},
}


st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
ensure_users_csv(USERS_CSV_PATH)


@st.cache_resource
def _load_model_cached(model_path: str):
    """Load and cache trained model for local predictions."""
    return load_model(model_path)

def _inject_modal_css() -> None:
    st.markdown(
        """
        <style>
        @keyframes riskModalIn {
            from { opacity: 0; transform: translate(-50%, -48%) scale(0.96); }
            to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }

        div[data-testid="stDialog"],
        .stDialog {
            position: fixed !important;
            inset: 0 !important;
            background: rgba(2, 6, 23, 0.55) !important;
            backdrop-filter: blur(4px) !important;
            -webkit-backdrop-filter: blur(4px) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 10000 !important;
        }

        div[data-testid="stDialogContent"],
        .stDialog div[role="dialog"] {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: min(460px, 92vw) !important;
            background: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            border-radius: 18px !important;
            padding: 24px !important;
            color: #e2e8f0 !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            animation: riskModalIn 260ms ease-out !important;
        }

        .risk-title {
            margin: 0 0 8px 0;
            text-align: center;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #93c5fd, #e0e7ff);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .risk-score {
            text-align: center;
            font-size: 42px;
            font-weight: 900;
            line-height: 1.05;
            margin: 0 0 14px 0;
            color: #f8fafc;
        }

        .risk-badge {
            display: inline-block;
            margin: 0 auto 14px auto;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
        }

        .risk-low { color: #22c55e; }
        .risk-mid { color: #f59e0b; }
        .risk-high { color: #ef4444; }

        .risk-badge.risk-low { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); }
        .risk-badge.risk-mid { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.45); }
        .risk-badge.risk-high { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.5); }

        .risk-glow-low { box-shadow: 0 0 0 1px rgba(34,197,94,0.25), 0 18px 40px rgba(34,197,94,0.22) !important; }
        .risk-glow-mid { box-shadow: 0 0 0 1px rgba(245,158,11,0.30), 0 18px 40px rgba(245,158,11,0.24) !important; }
        .risk-glow-high { box-shadow: 0 0 0 1px rgba(239,68,68,0.32), 0 18px 42px rgba(239,68,68,0.26) !important; }

        [data-testid="stProgressBar"] > div {
            background: rgba(148,163,184,0.25) !important;
            border-radius: 999px !important;
        }

        [data-testid="stProgressBar"] [role="progressbar"] {
            border-radius: 999px !important;
            background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444) !important;
        }

        div[data-testid="stDialogContent"] .stButton > button {
            width: 100% !important;
            margin-top: 12px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            transition: all .2s ease !important;
        }

        div[data-testid="stDialogContent"] .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 26px rgba(37,99,235,0.45) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "show_result" not in st.session_state:
    st.session_state.show_result = False


def _show_login() -> None:
    st.markdown("""
    <style>
    /* Full screen gradient */
    .stApp {
        background: linear-gradient(135deg, #0b1f3a 0%, #0a1630 48%, #081225 100%) !important;
    }

    /* Animation keyframes */
                
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
                
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* Animate Title */
    .login-title {
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
        animation-delay: 0.3s;
    }


    /* Animate Subtitle */
    .login-subtitle {
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
        animation-delay: 0.8s;
    }


    /* Animate Glass Card */
    [data-testid="stForm"] {
        opacity: 0;
        transform: translateY(40px);
        animation: fadeInUp 1.2s ease forwards;
        animation-delay: 2s;   
    }

         
    /* Login layout container */
    section.main > div.block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: none !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 100vh !important;
    }

    /* Kill extra layout wrapper */
    section.main,
    [data-testid="stMain"] {
        padding: 0 !important;
    }



    /* Wrapper */
    .login-panel {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Title */
    .login-title {
        font-size: 48px;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 8px;
        text-align: center;
    }

    /* Subtitle */
    .login-subtitle {
        font-size: 16px;
        color: #cbd5e1 !important;
        margin-bottom: 28px;
        text-align: center;
    }

    /* Glass card */
    [data-testid="stForm"] {
        width: min(480px, 92vw) !important;
        padding: 40px !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,0.12) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.25) !important;
        box-shadow: 0 25px 60px rgba(0,0,0,0.45) !important;

        margin-left: auto !important;
        margin-right: auto !important;
    }


    /* Input */
    [data-testid="stTextInput"] input {
        border-radius: 12px !important;
        padding: 12px !important;
        color: #000000 !important;
        background: rgba(248, 250, 252, 0.96) !important;
        caret-color: #2563eb !important;   /* 👈 THIS FIXES BLINKING CURSOR */
    }

    /* Remove weird focus outline override */
    [data-testid="stTextInput"] input:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.5) !important;
    }
    
    /* Placeholder text color */
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(0, 0, 0, 0.55) !important;  /* soft dark */
        font-weight: 500 !important;
    }



    /* Button */
    [data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 46px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 18px 40px rgba(37,99,235,0.5);
    }


</style>
""", unsafe_allow_html=True)



    st.markdown("<div class='login-panel'>", unsafe_allow_html=True)
    st.markdown("<h1 class='login-title'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>Create your profile to continue</p>", unsafe_allow_html=True)
    with st.form("login_form"):
        name = st.text_input("Name", placeholder="Enter your full name", label_visibility="collapsed").strip()
        submitted = st.form_submit_button("Continue", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not name:
            st.error("Name is required.")
            return
        if user_exists(name, USERS_CSV_PATH):
            st.error("User already exists")
            return
        if not create_user(name, USERS_CSV_PATH):
            st.error("User already exists")
            return

        st.session_state.logged_in = True
        st.session_state.user_name = name
        st.rerun()


def _main_app() -> None:
    _inject_modal_css()

    st.markdown("""
    <style>

    /* ===== Animated Background ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #0b1f3a 50%, #081225 100%);
        overflow-x: hidden;
    }

    /* Floating Glow */
    .stApp::before {
        content: "";
        position: fixed;
        width: 700px;
        height: 700px;
        background: radial-gradient(circle, rgba(59,130,246,0.25), transparent 60%);
        top: -200px;
        left: -200px;
        animation: floatGlow 10s ease-in-out infinite alternate;
        z-index: 0;
    }

    @keyframes floatGlow {
        from { transform: translate(0px, 0px); }
        to { transform: translate(160px, 120px); }
    }

    /* ===== Glass Dashboard Card ===== */
    section.main > div.block-container,
    [data-testid="stMainBlockContainer"] {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 40px 100px rgba(0,0,0,0.6);
        max-width: 1100px;
        position: relative;
        z-index: 1;
    }

    /* ===== Typography ===== */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    p, label {
        color: #cbd5e1 !important;
    }

    /* ===== Inputs ===== */
    .stNumberInput > div,
    .stTextInput > div {
        border-radius: 12px !important;
    }

    .stNumberInput input,
    .stTextInput input {
        background: rgba(255,255,255,0.96) !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #2563eb !important;
        opacity: 1 !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.45) !important;
    }

    .stNumberInput input::placeholder,
    .stTextInput input::placeholder {
        color: rgba(15,23,42,0.55) !important;
    }

    .stNumberInput button {
        border-radius: 10px !important;
    }


    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.25s ease;
        box-shadow: 0 15px 40px rgba(59,130,246,0.5);
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 25px 60px rgba(59,130,246,0.7);
    }

    /* ===== Progress Bar ===== */
    .stProgress > div > div {
        border-radius: 12px !important;
    }

    /* Section spacing */
    h3 {
        margin-top: 40px;
    }

    </style>
    """, unsafe_allow_html=True)


    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            "<h1 style='color: #1f4e79; font-size: 40px; margin-bottom: 0;'>Diabetes Risk Predictor</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(f"### Welcome, {st.session_state.user_name}")
    with right:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.last_result = None
            st.session_state.show_result = False
            st.rerun()

    st.write("Enter lifestyle and genetic information to estimate diabetes risk.")
    st.info("This tool is for educational purposes and not a medical diagnosis.")

    model_exists = Path(MODEL_PATH).exists()

    st.subheader("Model Status")
    if not model_exists:
        st.error(
            "Model file not found. Train once with: `python -m src.train_model` "
            "and ensure it creates `models/best_model.joblib`."
        )
    else:
        st.success("Model loaded and ready for prediction.")

    st.subheader("Health Metrics")

    col1, col2 = st.columns(2)
    input_data = {}

    for i, (key, cfg) in enumerate(FEATURES.items()):
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if cfg["type"] == "number":
                input_data[key] = st.number_input(
                    cfg["label"],
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["value"],
                )
            elif cfg["type"] == "radio":
                choice = st.radio(cfg["label"], cfg["options"], index=cfg["options"].index(cfg["value"]))
                input_data[key] = 1 if choice == "Yes" else 0
            else:
                input_data[key] = st.selectbox(cfg["label"], cfg["options"], index=cfg["options"].index(cfg["value"]))

    if st.button("Predict", type="primary"):
        if not model_exists:
            st.error("Please train the model first using `python -m src.train_model`.")
        else:
            with st.spinner("Processing..."):
                try:
                    model = _load_model_cached(MODEL_PATH)
                    label, proba = predict_single(model, input_data)

                    storage_payload = {
                        "pregnancies": input_data["Pregnancies"],
                        "glucose": input_data["Glucose"],
                        "blood_pressure": input_data["BloodPressure"],
                        "skin_thickness": input_data["SkinThickness"],
                        "insulin": input_data["Insulin"],
                        "bmi": input_data["BMI"],
                        "diabetes_pedigree_function": input_data["DiabetesPedigreeFunction"],
                        "genetic_risk": input_data["GeneticRisk"],
                        "age": input_data["Age"],
                    }
                    upsert_user_prediction(
                        name=st.session_state.user_name,
                        input_values=storage_payload,
                        prediction=label,
                        probability=proba,
                        csv_path=USERS_CSV_PATH,
                    )

                    st.session_state.last_result = {"label": label, "proba": proba}
                    st.session_state.show_result = True
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    if st.session_state.show_result and st.session_state.last_result:

        label = st.session_state.last_result["label"]
        proba = st.session_state.last_result["proba"]
        risk_percentage = int(proba * 100)

        # Color logic
        if risk_percentage < 40:
            status_color = "#16a34a"   # green
            status_text = "Low Risk"
        elif risk_percentage < 70:
            status_color = "#eab308"   # yellow
            status_text = "Moderate Risk"
        else:
            status_color = "#dc2626"   # red
            status_text = "High Risk"

        @st.dialog("Risk Analysis")
        def show_popup():
            if risk_percentage < 40:
                risk_class = "risk-low"
                glow_class = "risk-glow-low"
            elif risk_percentage < 70:
                risk_class = "risk-mid"
                glow_class = "risk-glow-mid"
            else:
                risk_class = "risk-high"
                glow_class = "risk-glow-high"

            st.markdown(
                (
                    f"<div class='{glow_class}' style='border-radius:14px; padding:14px;'>"
                    "<h2 class='risk-title'>Risk Analysis</h2>"
                    f"<div class='risk-score {risk_class}'>{risk_percentage}%</div>"
                    f"<div style='text-align:center;'><span class='risk-badge {risk_class}'>{status_text}</span></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            st.progress(risk_percentage / 100)

            if label == "Diabetic":
                st.error("High risk detected. Please consult a healthcare professional.")
            else:
                st.success("Maintain a healthy lifestyle and regular checkups.")

            if st.button("Close", key="risk_close_btn"):
                st.session_state.show_result = False
                st.rerun()

        show_popup()



if st.session_state.logged_in:
    _main_app()
else:
    _show_login()
