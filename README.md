# Diabetes Insight

Diabetes AI is a Streamlit-based diabetes risk prediction application built for academic evaluation. It keeps the project simple: a Streamlit frontend, a scikit-learn training pipeline, a prediction module, and CSV-based persistence for user sessions and latest prediction records.

## Project Overview

### Goal
Estimate diabetes risk from patient health indicators using a trained machine learning pipeline and present the result in a clean, user-friendly browser interface.

### Key Features
- Polished Streamlit login and prediction dashboard
- Grouped health-input sections with responsive layout
- Spinner, progress bar, and completion feedback during inference
- Styled result dialog with risk probability and guidance
- Graceful handling for invalid inputs, model failures, prediction failures, and CSV errors
- Deployment-ready configuration for Streamlit Cloud and Render

## Architecture

```text
User -> Streamlit UI (app.py)
     -> Input validation + loading states
     -> Prediction module (src/predict.py)
     -> Trained model (models/best_model.joblib)
     -> CSV storage (src/user_store.py / data/users.csv)
```

### Component Breakdown
- `app.py`
  - Streamlit frontend
  - login flow
  - grouped prediction form
  - loading/progress states
  - result dialog and user feedback
- `src/train_model.py`
  - loads data
  - trains Logistic Regression and Random Forest
  - saves the best pipeline
- `src/preprocessing.py`
  - missing-value imputation
  - scaling of numeric features
  - categorical encoding when needed
- `src/predict.py`
  - model loading
  - input validation
  - single-record prediction
  - prediction-safe exception handling
- `src/user_store.py`
  - creates the CSV store
  - creates users
  - updates prediction history
  - protects against read/write failures
- `models/best_model.joblib`
  - serialized trained ML pipeline
- `data/diabetes.csv`
  - training dataset

## AI Pipeline Explanation

### Training Flow
1. Load `data/diabetes.csv`
2. Split features and target (`Outcome`)
3. Build preprocessing with:
   - median imputation for numeric features
   - standard scaling for numeric features
   - most-frequent imputation and one-hot encoding for categorical features
4. Train:
   - Logistic Regression
   - Random Forest
5. Compare accuracy
6. Save the best-performing pipeline to `models/best_model.joblib`

### Prediction Flow
1. User logs in through the Streamlit interface
2. User enters health values in the prediction form
3. The app validates ranges and missing values
4. The trained model is loaded from disk
5. A prediction and probability are generated
6. The latest result is stored in `data/users.csv`
7. The UI shows a styled popup with the risk outcome

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd diabetes-ai
```

### 2. Create and activate a virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and adjust values if needed.

Example:
```env
DATA_PATH=data/diabetes.csv
MODEL_PATH=models/best_model.joblib
TARGET_COL=Outcome
USERS_CSV_PATH=data/users.csv
```

## Training Instructions

Train the model locally:

```bash
python -m src.train_model
```

Expected result:
- `models/best_model.joblib` is created or updated

## Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in the browser. The evaluator should be able to:
- enter a user name and gender
- fill in the health form
- click `Predict Diabetes Risk`
- see a spinner, progress indicator, and final risk popup

## Robustness and Error Handling

The application has been hardened to avoid crashes during evaluation.

### Handled cases
- invalid user inputs
- empty required values
- numeric out-of-range values
- missing model file
- model loading failure
- prediction failure
- CSV initialization failure
- CSV read/write failure

### User-visible messages
- `Invalid input value for <field>.`
- `<field> must be between min and max.`
- `Model file not found.`
- `Prediction failed. Please try again.`
- `Unable to save prediction history.`

## Deployment Instructions

The project is prepared for:
- Streamlit Cloud
- Render

### Option A: Streamlit Cloud

#### Files already prepared
- `.streamlit/config.toml`
- `.env.example`
- `requirements.txt`

#### Steps
1. Push the repository to GitHub.
2. Open Streamlit Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. In app settings, add environment variables if you want custom paths:
   - `DATA_PATH`
   - `MODEL_PATH`
   - `TARGET_COL`
   - `USERS_CSV_PATH`
6. Deploy.

#### Important note
For Streamlit Cloud, ensure these files are committed:
- `models/best_model.joblib`
- `data/diabetes.csv`

### Option B: Render

#### Files already prepared
- `render.yaml`
- `requirements.txt`

#### Steps
1. Push the repository to GitHub.
2. Create a new Render Web Service.
3. Connect the repository.
4. Render will detect `render.yaml`.
5. Confirm the start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

6. Add or confirm environment variables:
   - `DATA_PATH=data/diabetes.csv`
   - `MODEL_PATH=models/best_model.joblib`
   - `TARGET_COL=Outcome`
   - `USERS_CSV_PATH=data/users.csv`
7. Deploy and open the generated URL.

## Production Notes

### File Path Safety
`src/config.py` resolves relative paths against the project root so model and data files work reliably in hosted environments.

### Environment Variables
The app reads runtime values from `.env` locally and from platform environment variables in production.

### Model Loading
The model is cached with `st.cache_resource` to avoid repeated disk loads and improve hosted performance.

## Application Screenshots

### Login Page
![Login Page](docs/screenshots/Login page.png)

### Prediction Page
![Prediction Page](docs/screenshots/prediction page.png)

### Result Page
![Result Page](docs/screenshots/result page.png)

## Demo Explanation

A short presentation script is available in [docs/demo_script.md](/c:/Users/JAYA%20PRAKASH%20A/OneDrive/Documents/Mini%20project/diabetes-ai/docs/demo_script.md).

### Short Demo Summary
1. This project predicts diabetes risk from patient health metrics.
2. The frontend is built in Streamlit and the model pipeline is trained using scikit-learn.
3. Preprocessing handles missing values, scaling, and encoding before model training.
4. During prediction, the app validates inputs, loads the trained model, runs inference, and shows a risk score popup.
5. The latest user and prediction data are stored in a CSV file for simplicity.

## Submission Checklist
- `app.py` runs locally
- `models/best_model.joblib` is present
- `data/diabetes.csv` is present
- `requirements.txt` is updated
- deployment config is committed
- screenshots are added
- README is updated
