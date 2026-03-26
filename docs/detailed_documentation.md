# DiabetesInsight
## Detailed Project Documentation

Prepared by: Jai Akash
Program: B.Tech AIDS
Academic Level: 3rd Year
Document Date: March 26, 2026

## 1. Executive Summary
DiabetesInsight is a machine learning based Streamlit application that estimates diabetes risk from user-provided health indicators. The project combines a clean browser interface, a trained scikit-learn pipeline, and CSV-backed persistence to create a compact academic demonstration of predictive analytics in healthcare.

The system is designed for project evaluation, local demonstration, and basic cloud deployment. It guides a user from login to prediction, shows a probability-based result, and stores the latest prediction details in a simple tabular file.

## 2. Project Objectives
- Predict diabetes risk from commonly used medical and demographic features.
- Provide a simple and attractive interface that can be used by non-technical evaluators.
- Compare multiple machine learning models and keep the best one.
- Store user details and the latest prediction in a lightweight CSV file.
- Keep the project easy to deploy on platforms such as Streamlit Cloud or Render.

## 3. Problem Statement
Early diabetes screening is important because many patients do not recognize warning signs until the disease has already progressed. A lightweight prediction system can help demonstrate how machine learning can support screening by identifying risk patterns from historical health data.

This project does not replace clinical diagnosis. It is an academic decision-support prototype that shows how data preprocessing, model training, inference, and user-facing presentation can be combined into one application.

## 4. Technology Stack
- Frontend and UI: Streamlit
- Programming Language: Python
- Data Handling: pandas
- Machine Learning: scikit-learn
- Model Serialization: joblib
- Environment Configuration: python-dotenv
- Optional API Layer: FastAPI and Uvicorn
- Deployment Support: Streamlit Cloud and Render

## 5. Project Structure
```text
diabetes-ai/
|-- app.py
|-- frontend_app.py
|-- requirements.txt
|-- models/
|   `-- best_model.joblib
|-- data/
|   |-- diabetes.csv
|   `-- users.csv
|-- docs/
|   |-- detailed_documentation.md
|   |-- DiabetesInsight_Detailed_Documentation.pdf
|   |-- SRS.md
|   |-- schema.md
|   `-- Screenshots/
`-- src/
    |-- config.py
    |-- preprocessing.py
    |-- predict.py
    |-- train_model.py
    |-- user_store.py
    |-- auth.py
    |-- db.py
    |-- api.py
    `-- services/
```

## 6. System Architecture
The system follows a simple flow:

1. The user opens the Streamlit application.
2. The login screen captures a unique name and gender.
3. The main dashboard collects structured health inputs.
4. The prediction module validates the input and loads the trained model.
5. The model generates a label and probability score.
6. The result dialog shows the risk category and a short recommendation.
7. The latest user prediction is written to `data/users.csv`.

Architecture view:

```text
User
  -> Streamlit UI (app.py)
  -> Input Validation (src/predict.py)
  -> Trained Model (models/best_model.joblib)
  -> User Storage (src/user_store.py -> data/users.csv)
```

## 7. User Interface Flow
### 7.1 Login Page
The login page asks for:
- Full Name
- Gender

The application prevents duplicate user creation by checking `data/users.csv`. If the name already exists, the user is shown a warning message.

### 7.2 Prediction Dashboard
The dashboard groups input fields into logical sections:
- Personal Profile
- Vital Signs
- Metabolic Markers

Inputs collected by the application:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- GeneticRisk
- Age

The form uses widgets with predefined limits so the input stays within acceptable ranges. If the selected gender is male, the pregnancies field is automatically set to zero and disabled.

### 7.3 Result Dialog
After prediction, the application displays:
- Predicted class
- Estimated diabetes probability
- Risk category
- Recommendation message

Risk bands:
- Below 40 percent: Low Risk
- 40 to 69 percent: Moderate Risk
- 70 percent and above: High Risk

## 8. Machine Learning Workflow
### 8.1 Dataset
The model is trained using `data/diabetes.csv`. The expected target column is `Outcome`.

### 8.2 Preprocessing
The preprocessing pipeline is created in `src/preprocessing.py` and includes:
- Median imputation for numeric features
- Standard scaling for numeric features
- Most frequent imputation for categorical features
- One-hot encoding for categorical features when needed

### 8.3 Candidate Models
The training script evaluates two classification algorithms:
- Logistic Regression
- Random Forest Classifier

### 8.4 Model Selection
The script splits the data into train and test sets using stratification, evaluates both candidate models with accuracy, and saves the best-performing fitted pipeline to `models/best_model.joblib`.

### 8.5 Inference
During prediction:
1. Inputs are validated against field constraints.
2. A one-row pandas DataFrame is created.
3. The saved pipeline predicts the class label.
4. If available, `predict_proba` is used to compute the probability score.
5. The UI translates the numeric result into a readable diabetes risk assessment.

## 9. Module-Level Explanation
### `app.py`
- Builds the Streamlit interface
- Manages login state with `st.session_state`
- Renders the health form
- Displays progress and result dialogs
- Provides a documentation page and PDF download

### `src/config.py`
- Resolves file paths from environment variables
- Defines dataset, model, and storage paths
- Keeps runtime configuration centralized

### `src/preprocessing.py`
- Loads the dataset
- Builds the `ColumnTransformer`
- Applies imputation, scaling, and encoding

### `src/train_model.py`
- Trains Logistic Regression and Random Forest pipelines
- Compares model accuracy
- Saves the best model artifact

### `src/predict.py`
- Loads the trained model from disk
- Validates form inputs
- Runs single-record prediction safely
- Raises user-friendly exceptions for invalid states

### `src/user_store.py`
- Ensures `data/users.csv` exists
- Creates new users
- Prevents duplicate names
- Updates the latest prediction for the logged-in user

## 10. Data Storage Design
The project uses CSV storage instead of a full relational database. This keeps the application simple and easy to demonstrate.

Current `users.csv` schema:

| Column | Description |
| --- | --- |
| `name` | Unique user identifier |
| `pregnancies` | Number of pregnancies |
| `glucose` | Plasma glucose value |
| `blood_pressure` | Diastolic blood pressure |
| `skin_thickness` | Skin fold thickness |
| `insulin` | Serum insulin value |
| `bmi` | Body mass index |
| `diabetes_pedigree_function` | Family risk score |
| `genetic_risk` | Encoded family history flag |
| `age` | Age in years |
| `prediction` | Latest predicted class |
| `probability` | Latest predicted probability |

## 11. Validation and Error Handling
The project includes checks for:
- Missing model file
- Invalid input values
- Out-of-range numeric entries
- Dataset and storage file issues
- Prediction-time failures
- CSV read and write errors

Typical user-facing messages include:
- `Invalid input value for <field>.`
- `<field> must be between min and max.`
- `Model file not found.`
- `Prediction failed. Please try again.`
- `Unable to save prediction history.`

## 12. Execution Steps
### 12.1 Local Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 12.2 Train the Model
```bash
python -m src.train_model
```

### 12.3 Run the Application
```bash
streamlit run app.py
```

## 13. Deployment Notes
The project is prepared for deployment on:
- Streamlit Cloud
- Render

Important deployment files:
- `.streamlit/config.toml`
- `requirements.txt`
- `render.yaml`
- `.env.example`

Important runtime files that should be available:
- `models/best_model.joblib`
- `data/diabetes.csv`
- `data/users.csv` or write access to create it

## 14. Strengths of the Project
- Clean and responsive user interface
- Simple end-to-end workflow from input to result
- Uses an actual machine learning pipeline rather than rule-based scoring
- Modular source code for training, prediction, and storage
- Lightweight persistence suitable for academic demos
- Documentation and PDF support included inside the project

## 15. Current Limitations
- No password-based authentication
- CSV storage is not ideal for concurrent multi-user production use
- The model quality depends entirely on the dataset used for training
- The result is informational and not a medical diagnosis
- No advanced model explainability chart is currently shown

## 16. Future Enhancements
- Add secure authentication with password or email login
- Replace CSV storage with SQLite or PostgreSQL
- Show feature importance or explainability insights
- Add analytics dashboards for aggregate prediction trends
- Support multi-page admin reporting
- Add retraining controls from the UI for project demos

## 17. Conclusion
DiabetesInsight is a compact and well-structured academic machine learning project that demonstrates how health data can be transformed into a practical diabetes risk prediction workflow. The project covers the full pipeline from data preprocessing and model selection to user interaction, result presentation, and deployment readiness.

With the addition of a dedicated in-app documentation page and downloadable PDF report, the project is now easier to present, review, and submit for academic evaluation.
