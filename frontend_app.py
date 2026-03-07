"""Draft Streamlit frontend that consumes the FastAPI backend."""

from __future__ import annotations

import time

import requests
import streamlit as st

from src.config import API_BASE_URL

st.set_page_config(page_title="Diabetes AI Client", layout="centered")
st.title("Diabetes AI Frontend (Draft)")
st.caption("Minimal UI for API integration demo")

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = ""


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "GeneticRisk",
    "Age",
]


def _headers() -> dict[str, str]:
    token = st.session_state.token
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, headers=_headers(), timeout=30)


def _api_get(path: str):
    return requests.get(f"{API_BASE_URL}{path}", headers=_headers(), timeout=30)


with st.expander("Backend Status", expanded=True):
    if st.button("Check /health"):
        try:
            response = _api_get("/health")
            st.write(response.status_code, response.json())
        except Exception as exc:  # pragma: no cover
            st.error(f"Backend not reachable: {exc}")


if not st.session_state.token:
    tab1, tab2 = st.tabs(["Register", "Login"])
    with tab1:
        with st.form("register"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account")
        if submitted:
            response = _api_post("/v1/auth/register", {"username": username, "password": password})
            st.write(response.status_code, response.json())

    with tab2:
        with st.form("login"):
            username = st.text_input("Login Username")
            password = st.text_input("Login Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            response = _api_post("/v1/auth/login", {"username": username, "password": password})
            body = response.json()
            if response.ok:
                st.session_state.token = body["access_token"]
                st.session_state.username = username
                st.success("Logged in")
                st.rerun()
            else:
                st.error(str(body))
else:
    st.success(f"Logged in as {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.username = ""
        st.rerun()

    with st.form("predict"):
        cols = st.columns(2)
        values: dict[str, float | int] = {}
        defaults = {
            "Pregnancies": 1,
            "Glucose": 110,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "GeneticRisk": 0,
            "Age": 30,
        }

        for index, name in enumerate(FEATURES):
            with cols[index % 2]:
                if name == "GeneticRisk":
                    values[name] = st.selectbox(name, options=[0, 1], index=0)
                else:
                    values[name] = st.number_input(name, value=float(defaults[name]))

        background = st.checkbox("Run as background job")
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"input": values}
        if not background:
            response = _api_post("/v1/predict", payload)
            st.write(response.status_code, response.json())
        else:
            queued = _api_post("/v1/predict/background", payload)
            if not queued.ok:
                st.error(str(queued.json()))
            else:
                job_id = queued.json()["job_id"]
                st.info(f"Queued job: {job_id}")
                placeholder = st.empty()
                for _ in range(30):
                    status_response = _api_get(f"/v1/predict/jobs/{job_id}")
                    data = status_response.json()
                    placeholder.write(data)
                    if data["status"] in {"completed", "failed"}:
                        break
                    time.sleep(1)

    st.subheader("Prediction History")
    history_response = _api_get("/v1/me/history?limit=10")
    if history_response.ok:
        body = history_response.json()
        st.json(body["items"])
    else:
        st.error(str(history_response.json()))
