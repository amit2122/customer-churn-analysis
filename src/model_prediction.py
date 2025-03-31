import joblib
import pandas as pd

# Model paths
model_paths = {
    "Reinforcement Learning": "models/churn_rl_model.pkl",
    "Logistic Regression": "models/churn_logistic_regression.pkl",
    "XGBoost": "models/churn_xgboost.pkl",
}

def predict_churn(model_name, data):
    if model_name not in model_paths:
        raise ValueError("Invalid model selection. Choose from: Reinforcement Learning, Logistic Regression, XGBoost")

    # Load the correct model
    model = joblib.load(model_paths[model_name])
    
    # Ensure correct feature columns
    missing_cols = set(model.feature_names_in_) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")

    return model.predict(data)
