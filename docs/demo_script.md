# Diabetes AI Demo Script

## 1. Project Purpose
This project predicts diabetes risk from patient health indicators using a trained machine learning model. It is designed as a simple, explainable screening tool for academic evaluation with a Streamlit user interface.

## 2. System Architecture
- `app.py` is the Streamlit frontend and session controller.
- `src/train_model.py` trains the ML pipeline and saves the best model.
- `src/preprocessing.py` handles missing values, scaling, and encoding.
- `src/predict.py` loads the model, validates user input, and runs inference.
- `src/user_store.py` stores login and latest prediction data in CSV format.
- `models/best_model.joblib` is the deployed trained pipeline.
- `data/diabetes.csv` is the source training dataset.

## 3. AI Pipeline
The training script loads the diabetes dataset, separates features and target, applies preprocessing through a scikit-learn `ColumnTransformer`, trains Logistic Regression and Random Forest models, compares accuracy, and saves the best-performing pipeline.

## 4. Prediction Process
The user logs in, fills the grouped health form, and clicks `Predict Diabetes Risk`. The app shows a spinner and progress bar, validates the inputs, loads the saved model, runs prediction, and presents the result in a dialog with the probability and risk level.

## 5. Storage Mechanism
User information and prediction outputs are stored in `data/users.csv`. The application creates the file if it does not exist, updates the current user's latest prediction, and handles CSV read/write failures gracefully.

## 6. Evaluation Talking Points
- The UI is organized into grouped health sections for clarity.
- The app includes loading states, success/error feedback, and a styled result popup.
- Input validation and exception handling prevent crashes.
- Path resolution and deployment config support hosted environments like Streamlit Cloud and Render.
