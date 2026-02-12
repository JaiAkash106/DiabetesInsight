# Diabetes AI - Streamlit App

Diabetes risk prediction app with:
- Streamlit UI
- CSV-based user login/persistence
- Local model inference (no FastAPI required)

## Minimal Project Structure

- `app.py` Streamlit application
- `src/preprocessing.py` preprocessing helpers
- `src/train_model.py` model training script
- `src/predict.py` local prediction utilities
- `src/user_store.py` CSV user storage logic
- `data/diabetes.csv` training dataset
- `data/users.csv` user records and predictions
- `models/best_model.joblib` trained model
- `requirements.txt` dependencies

## Setup

```bash
python -m pip install -r requirements.txt
```

## Environment (Optional)

Copy `.env.example` to `.env` and adjust values if needed:

```bash
DATA_PATH=data/diabetes.csv
TARGET_COL=Outcome
MODEL_PATH=models/best_model.joblib
USERS_CSV_PATH=data/users.csv
```

## Train Model

```bash
python -m src.train_model
```

## Run App (Single Command)

```bash
python -m streamlit run app.py
```

## Notes

- `data/users.csv` is created automatically if missing.
- This tool is for educational use and is not a medical diagnosis.
