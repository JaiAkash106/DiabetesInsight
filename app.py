"""Production-ready Streamlit app for diabetes risk prediction."""

from __future__ import annotations

from pathlib import Path
from time import sleep
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import DATA_PATH, MODEL_PATH, TARGET_COL, USERS_CSV_PATH
from src.predict import (
    InputValidationError,
    ModelLoadError,
    PredictionError,
    load_model,
    predict_single,
)
from src.user_store import UserStoreError, create_user, ensure_users_csv, upsert_user_prediction, user_exists

DI_FAVICON_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#0f766e'/>
      <stop offset='100%' stop-color='#155e75'/>
    </linearGradient>
  </defs>
  <rect width='64' height='64' rx='14' fill='url(#g)'/>
  <path d='M32 10c8 12 12 18 12 25a12 12 0 1 1-24 0c0-7 4-13 12-25Z' fill='white' opacity='0.92'/>
</svg>
"""
DI_FAVICON_URI = f"data:image/svg+xml,{quote(DI_FAVICON_SVG)}"

FIELD_GROUPS = {
    "Personal Profile": [
        {
            "key": "Pregnancies",
            "label": "Pregnancies",
            "help": "Applicable when relevant for the patient profile.",
            "min": 0,
            "max": 20,
            "value": 1,
            "step": 1,
        },
        {"key": "Age", "label": "Age", "help": "Age in years.", "min": 1, "max": 120, "value": 30, "step": 1},
        {
            "key": "GeneticRisk",
            "label": "Family History of Diabetes",
            "help": "Choose Yes if there is known family history.",
            "type": "radio",
            "options": ["No", "Yes"],
            "value": "No",
        },
    ],
    "Vital Signs": [
        {"key": "Glucose", "label": "Glucose", "help": "Plasma glucose concentration.", "min": 0, "max": 250, "value": 100, "step": 1},
        {
            "key": "BloodPressure",
            "label": "Blood Pressure",
            "help": "Diastolic blood pressure (mm Hg).",
            "min": 0,
            "max": 200,
            "value": 70,
            "step": 1,
        },
        {
            "key": "SkinThickness",
            "label": "Skin Thickness",
            "help": "Triceps skin fold thickness (mm).",
            "min": 0,
            "max": 100,
            "value": 20,
            "step": 1,
        },
    ],
    "Metabolic Markers": [
        {"key": "Insulin", "label": "Insulin", "help": "2-Hour serum insulin.", "min": 0, "max": 900, "value": 80, "step": 1},
        {"key": "BMI", "label": "BMI", "help": "Body Mass Index.", "min": 10.0, "max": 70.0, "value": 25.0, "step": 0.1},
        {
            "key": "DiabetesPedigreeFunction",
            "label": "Diabetes Pedigree Function",
            "help": "Family risk score from the dataset.",
            "min": 0.0,
            "max": 3.0,
            "value": 0.5,
            "step": 0.01,
        },
    ],
}

THEMES = {
    "Light": {
        "bg_1": "#f4fbfa",
        "bg_2": "#e6f4f1",
        "card": "rgba(255, 255, 255, 0.92)",
        "text": "#0f172a",
        "muted": "#4b5563",
        "heading": "#0b1f3a",
        "card_text": "#0f172a",
        "dialog_surface": "#f8fcfb",
        "dialog_text": "#0b1f3a",
        "dialog_muted": "#475569",
        "soft_text": "#4b5563",
        "accent": "#0f766e",
        "accent_dark": "#155e75",
        "accent_soft": "#ccfbf1",
        "border": "rgba(15, 118, 110, 0.14)",
        "warning": "#b45309",
        "danger": "#b91c1c",
        "success": "#166534",
        "dialog_bg": "linear-gradient(180deg, #ffffff, #f0fdfa)",
    },
    "Dark": {
        "bg_1": "#06131f",
        "bg_2": "#0b2233",
        "card": "rgba(8, 24, 39, 0.92)",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "heading": "#ffffff",
        "card_text": "#f8fafc",
        "dialog_surface": "#f4f8f7",
        "dialog_text": "#081225",
        "dialog_muted": "#64748b",
        "soft_text": "#dbe7f3",
        "accent": "#22c55e",
        "accent_dark": "#67e8f9",
        "accent_soft": "rgba(34, 197, 94, 0.14)",
        "border": "rgba(148, 163, 184, 0.22)",
        "warning": "#f59e0b",
        "danger": "#f87171",
        "success": "#4ade80",
        "dialog_bg": "linear-gradient(180deg, #0f172a, #082f49)",
    },
}

st.set_page_config(page_title="Diabetes Insight", page_icon=DI_FAVICON_URI, layout="wide")


@st.cache_resource
def _load_model_cached(model_path: str):
    """Cache the trained model for repeated predictions."""
    return load_model(model_path)


def _default_form_values() -> dict:
    values = {}
    for fields in FIELD_GROUPS.values():
        for field in fields:
            if field.get("type") == "radio":
                values[field["key"]] = 1 if field["value"] == "Yes" else 0
            else:
                values[field["key"]] = field["value"]
    return values


def _init_state() -> None:
    defaults = {
        "logged_in": False,
        "user_name": "",
        "last_result": None,
        "show_result": False,
        "is_processing": False,
        "process_requested": False,
        "pending_input": None,
        "status_message": None,
        "status_level": "info",
        "form_values": _default_form_values(),
        "theme_name": "Light",
        "login_name_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_global_styles() -> None:
    theme = THEMES[st.session_state.theme_name]
    css = """
        <style>
        :root {
            --bg-1: __BG_1__;
            --bg-2: __BG_2__;
            --card: __CARD__;
            --text: __TEXT__;
            --muted: __MUTED__;
            --heading: __HEADING__;
            --card-text: __CARD_TEXT__;
            --dialog-surface: __DIALOG_SURFACE__;
            --dialog-text: __DIALOG_TEXT__;
            --dialog-muted: __DIALOG_MUTED__;
            --soft-text: __SOFT_TEXT__;
            --accent: __ACCENT__;
            --accent-dark: __ACCENT_DARK__;
            --accent-soft: __ACCENT_SOFT__;
            --border: __BORDER__;
            --warning: __WARNING__;
            --danger: __DANGER__;
            --success: __SUCCESS__;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 34%),
                radial-gradient(circle at bottom right, rgba(21,94,117,0.12), transparent 28%),
                linear-gradient(180deg, var(--bg-1), var(--bg-2));
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(255,255,255,0);
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            color: var(--heading) !important;
            letter-spacing: -0.02em;
        }

        p, label, div {
            color: var(--text);
        }

        .hero-card,
        .section-card,
        .metric-card,
        .login-card {
            background: var(--card);
            color: var(--card-text);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
        }

        .hero-card {
            padding: 2rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1rem 1.25rem;
            min-height: 120px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--heading);
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-subtext {
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.45rem;
        }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.35rem;
            padding: 0.9rem 1rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: color-mix(in srgb, var(--card) 88%, transparent);
        }

        .section-title h3 {
            margin: 0;
            font-size: 1.12rem;
        }

        .section-title span {
            font-size: 0.83rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-weight: 700;
        }

        .support-text {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 500;
            margin: 0.85rem 0 1.2rem 0;
        }

        .login-wrap {
            min-height: 92vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .login-card {
            width: min(520px, 100%);
            padding: 2.25rem;
            margin: 4rem auto 0 auto;
        }

        .login-kicker {
            display: inline-block;
            margin-bottom: 0.9rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .login-title {
            color: var(--heading);
            font-size: 2.4rem;
            font-weight: 900;
            margin-bottom: 0.45rem;
            text-shadow: 0 2px 18px rgba(15, 23, 42, 0.18);
        }

        .login-copy {
            color: var(--soft-text);
            font-weight: 500;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }

        .hero-copy {
            color: var(--soft-text);
            font-weight: 500;
            line-height: 1.6;
        }

        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 800;
            background: #ecfeff;
            color: var(--accent-dark);
            margin-bottom: 0.85rem;
        }

        [data-testid="column"]:has(.theme-switcher) [role="radiogroup"] {
            display: grid !important;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }

        [data-testid="column"]:has(.theme-switcher) label[data-baseweb="radio"] {
            margin: 0 !important;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.55rem 0.8rem;
            justify-content: center;
        }

        [data-testid="column"]:has(.theme-switcher) label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        div[data-testid="stForm"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stForm"] > div {
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 14px !important;
            background: #ffffff !important;
            border: 1px solid rgba(15, 118, 110, 0.16) !important;
        }

        div[data-testid="stRadio"] label {
            font-weight: 600;
        }

        div[data-testid="stRadio"] > div {
            gap: 0.8rem;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 14px !important;
            border: 1px solid transparent !important;
            font-weight: 800 !important;
            min-height: 2.9rem !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
            color: #ffffff !important;
            box-shadow: 0 14px 34px rgba(15,118,110,0.22);
        }

        .stButton > button:not([kind="primary"]) {
            background: #ffffff !important;
            color: var(--accent-dark) !important;
            border-color: rgba(15, 118, 110, 0.25) !important;
        }

        div[data-testid="stDialog"] {
            background: rgba(15, 23, 42, 0.32) !important;
            backdrop-filter: blur(4px);
        }

        div[data-testid="stDialogContent"] {
            border-radius: 24px !important;
            border: 1px solid rgba(15, 118, 110, 0.2) !important;
            padding: 1.5rem !important;
            color: var(--dialog-text) !important;
            background: __DIALOG_BG__ !important;
            box-shadow: 0 30px 90px rgba(2, 6, 23, 0.4) !important;
        }

        div[data-testid="stDialogContent"] * {
            color: inherit;
        }

        div[data-testid="stDialogContent"] [data-testid="stDialogHeader"] *,
        div[data-testid="stDialogContent"] h1,
        div[data-testid="stDialogContent"] h2,
        div[data-testid="stDialogContent"] h3 {
            color: var(--dialog-text) !important;
        }

        .result-shell {
            padding: 0.25rem;
        }

        .result-headline {
            color: var(--dialog-text) !important;
            font-size: 1.8rem;
            font-weight: 900;
            margin-bottom: 0.2rem;
        }

        .result-copy {
            color: var(--dialog-muted) !important;
            margin-bottom: 1rem;
            font-weight: 500;
            line-height: 1.55;
        }

        .result-score {
            color: var(--dialog-text) !important;
            font-size: 3rem;
            font-weight: 900;
            line-height: 1;
            margin: 0.35rem 0 0.5rem 0;
        }

        .result-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .safe {
            color: var(--success);
            background: rgba(22, 101, 52, 0.1);
        }

        .watch {
            color: var(--warning);
            background: rgba(180, 83, 9, 0.12);
        }

        .high {
            color: var(--danger);
            background: rgba(185, 28, 28, 0.12);
        }

        .theme-switch-shell {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1rem;
            }

            .hero-card,
            .metric-card,
            .login-card {
                border-radius: 20px;
            }
        }
        </style>
    """
    css = (
        css.replace("__BG_1__", theme["bg_1"])
        .replace("__BG_2__", theme["bg_2"])
        .replace("__CARD__", theme["card"])
        .replace("__TEXT__", theme["text"])
        .replace("__MUTED__", theme["muted"])
        .replace("__HEADING__", theme["heading"])
        .replace("__CARD_TEXT__", theme["card_text"])
        .replace("__DIALOG_SURFACE__", theme["dialog_surface"])
        .replace("__DIALOG_TEXT__", theme["dialog_text"])
        .replace("__DIALOG_MUTED__", theme["dialog_muted"])
        .replace("__SOFT_TEXT__", theme["soft_text"])
        .replace("__ACCENT__", theme["accent"])
        .replace("__ACCENT_DARK__", theme["accent_dark"])
        .replace("__ACCENT_SOFT__", theme["accent_soft"])
        .replace("__BORDER__", theme["border"])
        .replace("__WARNING__", theme["warning"])
        .replace("__DANGER__", theme["danger"])
        .replace("__SUCCESS__", theme["success"])
        .replace("__DIALOG_BG__", theme["dialog_bg"])
    )
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def _set_status(message: str, level: str) -> None:
    st.session_state.status_message = message
    st.session_state.status_level = level


def _clear_status() -> None:
    st.session_state.status_message = None
    st.session_state.status_level = "info"


def _show_status_banner() -> None:
    message = st.session_state.status_message
    if not message:
        return

    level = st.session_state.status_level
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def _model_status() -> tuple[bool, str]:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        return False, "Model file is missing. Train the model before running predictions."
    return True, f"Using model: {model_path.name}"


def _safe_prepare_storage() -> bool:
    try:
        ensure_users_csv(USERS_CSV_PATH)
        return True
    except UserStoreError as exc:
        _set_status(str(exc), "error")
        return False


def _render_theme_switcher() -> None:
    st.markdown("<div class='theme-switcher'></div>", unsafe_allow_html=True)
    selected = st.radio(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        key="theme_selector",
        horizontal=True,
        label_visibility="visible",
    )
    if selected != st.session_state.theme_name:
        st.session_state.theme_name = selected
        st.rerun()


def _show_login() -> None:
    theme_left, theme_mid, theme_right = st.columns([1.2, 1.2, 0.8])
    with theme_right:
        _render_theme_switcher()

    left, center, right = st.columns([1, 1.3, 1])
    with center:
        st.markdown(
            """
            <div class="login-card">
                <div class="status-pill">Secure screening access</div>
                <div class="login-title">Diabetes Insight</div>
                <div class="login-copy">
                    Secure access to the screening dashboard. Enter a user name to start a new prediction session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        name = st.text_input("Full Name", placeholder="Enter your full name", key="login_name_input").strip()
        submitted = st.button("Continue to Dashboard", type="primary", use_container_width=True)

    if submitted:
        if not name:
            _set_status("Name is required to continue.", "error")
            st.rerun()

        try:
            if user_exists(name, USERS_CSV_PATH):
                _set_status("User already exists. Please use a different name.", "warning")
                st.rerun()

            if not create_user(name, USERS_CSV_PATH):
                _set_status("User already exists. Please use a different name.", "warning")
                st.rerun()
        except UserStoreError as exc:
            _set_status(str(exc), "error")
            st.rerun()

        st.session_state.logged_in = True
        st.session_state.user_name = name
        _set_status("Login successful. You can now run a prediction.", "success")
        st.rerun()

    _show_status_banner()


def _render_metric(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_number_field(field: dict, values: dict) -> None:
    current_value = values.get(field["key"], field["value"])
    number_type = "float" if isinstance(field["value"], float) else "int"
    values[field["key"]] = st.number_input(
        field["label"],
        min_value=field["min"],
        max_value=field["max"],
        value=current_value,
        step=field["step"],
        help=field["help"],
        format="%.2f" if number_type == "float" else "%d",
    )


def _render_prediction_form(model_ready: bool) -> None:
    values = dict(st.session_state.form_values)
    with st.form("prediction_form", clear_on_submit=False):
        for section_name, fields in FIELD_GROUPS.items():
            st.markdown(
                f"""
                <div class="section-title">
                    <h3>{section_name}</h3>
                    <span>Required</span>
                </div>
                <div class="support-text">Provide realistic clinical values to evaluate diabetes risk.</div>
                """,
                unsafe_allow_html=True,
            )

            columns = st.columns(len(fields))
            for column, field in zip(columns, fields):
                with column:
                    if field.get("type") == "radio":
                        selected = "Yes" if values.get(field["key"], 0) == 1 else "No"
                        response = st.radio(
                            field["label"],
                            field["options"],
                            index=field["options"].index(selected),
                            horizontal=True,
                            help=field["help"],
                        )
                        values[field["key"]] = 1 if response == "Yes" else 0
                    else:
                        _render_number_field(field, values)
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.session_state.form_values = values
        submitted = st.form_submit_button(
            "Predict Diabetes Risk",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing or not model_ready,
        )

    if submitted:
        st.session_state.pending_input = dict(values)
        st.session_state.process_requested = True
        st.session_state.is_processing = True
        _clear_status()
        st.rerun()


def _store_prediction(input_data: dict, label: str, probability: float) -> None:
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
        probability=probability,
        csv_path=USERS_CSV_PATH,
    )


def _process_prediction_if_requested() -> None:
    if not st.session_state.process_requested:
        return

    progress = st.progress(0, text="Preparing secure prediction request...")
    pending_input = st.session_state.pending_input or {}
    try:
        progress.progress(20, text="Loading the trained model...")
        sleep(0.15)
        model = _load_model_cached(MODEL_PATH)

        with st.spinner("Running AI prediction..."):
            progress.progress(55, text="Validating health inputs...")
            sleep(0.15)
            label, proba = predict_single(model, pending_input)
            progress.progress(80, text="Saving session result...")
            sleep(0.15)
            _store_prediction(pending_input, label, proba)

        st.session_state.last_result = {"label": label, "proba": proba}
        st.session_state.show_result = True
        _set_status("Prediction completed successfully.", "success")
        progress.progress(100, text="Prediction completed.")
        sleep(0.2)
    except InputValidationError as exc:
        _set_status(str(exc), "error")
    except ModelLoadError as exc:
        _set_status(str(exc), "error")
    except UserStoreError as exc:
        _set_status(str(exc), "error")
    except PredictionError as exc:
        _set_status(str(exc), "error")
    except Exception:
        _set_status("Prediction failed. Please try again.", "error")
    finally:
        st.session_state.is_processing = False
        st.session_state.process_requested = False
        st.session_state.pending_input = None
        st.rerun()


def _show_result_dialog() -> None:
    if not st.session_state.show_result or not st.session_state.last_result:
        return

    label = st.session_state.last_result["label"]
    probability = st.session_state.last_result["proba"]
    risk_percentage = round(probability * 100)

    if risk_percentage < 40:
        badge_class = "safe"
        badge_text = "Low Risk"
        guidance = "Risk looks low. Continue healthy routines and periodic screening."
    elif risk_percentage < 70:
        badge_class = "watch"
        badge_text = "Moderate Risk"
        guidance = "Some indicators need attention. Review lifestyle habits and consult a clinician if needed."
    else:
        badge_class = "high"
        badge_text = "High Risk"
        guidance = "Risk looks elevated. Seek professional medical evaluation promptly."

    @st.dialog("Prediction Result")
    def _dialog() -> None:
        st.markdown(
            f"""
            <div class="result-shell">
                <div class="status-pill">AI Assessment Complete</div>
                <div class="result-headline">{label}</div>
                <div class="result-copy">Estimated probability of diabetes based on the supplied health indicators.</div>
                <div class="result-score">{risk_percentage}%</div>
                <div class="result-badge {badge_class}">{badge_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))
        if badge_class == "high":
            st.error(guidance)
        elif badge_class == "watch":
            st.warning(guidance)
        else:
            st.success(guidance)

        if st.button("Close Result", use_container_width=True, type="primary"):
            st.session_state.show_result = False
            st.rerun()

    _dialog()


def _main_app() -> None:
    model_ready, model_message = _model_status()

    left, mid, right = st.columns([4.6, 1.4, 1])
    with left:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="status-pill">Production-ready Streamlit workflow</div>
                <h1 style="margin-bottom:0.4rem;">Diabetes Risk Screening Dashboard</h1>
                <p class="hero-copy" style="margin-bottom:0.25rem;">
                    Welcome, <strong>{st.session_state.user_name}</strong>. Complete the structured form to generate a diabetes-risk prediction.
                </p>
                <p class="hero-copy" style="margin:0;">This tool supports academic evaluation and should not be used as a medical diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid:
        _render_theme_switcher()
    with right:
        if st.button("Logout", use_container_width=True):
            for key in ["logged_in", "user_name", "last_result", "show_result", "is_processing", "process_requested", "pending_input"]:
                st.session_state[key] = False if key in {"logged_in", "show_result", "is_processing", "process_requested"} else None
            st.session_state.user_name = ""
            st.session_state.last_result = None
            st.session_state.pending_input = None
            st.session_state.form_values = _default_form_values()
            _set_status("Session ended.", "info")
            st.rerun()

    metric_cols = st.columns(3)
    with metric_cols[0]:
        _render_metric("Model Status", "Ready" if model_ready else "Unavailable", model_message)
    with metric_cols[1]:
        dataset_name = Path(DATA_PATH).name
        _render_metric("Dataset", dataset_name, f"Target column: {TARGET_COL}")
    with metric_cols[2]:
        last_result = st.session_state.last_result
        summary = last_result["label"] if last_result else "No prediction yet"
        subtext = (
            f"Last probability: {round(last_result['proba'] * 100)}%" if last_result else "Run the form below to generate a result."
        )
        _render_metric("Latest Result", summary, subtext)

    if not model_ready:
        st.error("Model file not found. Run `python -m src.train_model` before deploying the app.")

    _show_status_banner()
    _render_prediction_form(model_ready=model_ready)
    _process_prediction_if_requested()
    _show_result_dialog()


_init_state()
_inject_global_styles()
storage_ready = _safe_prepare_storage()

if not storage_ready:
    _show_status_banner()
elif st.session_state.logged_in:
    _main_app()
else:
    _show_login()
