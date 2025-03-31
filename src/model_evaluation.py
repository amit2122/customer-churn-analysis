# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, classification_report

# def evaluate_model():
#     df = pd.read_csv("data/processed/feature_engineered_data.csv")

#     # Define features and target
#     X = df.drop(columns=['churned'])
#     y = df['churned']

#     # Load trained model
#     model = joblib.load("models/churn_rl_model.pkl")

#     # Predict and evaluate
#     y_pred = model.predict(X)

#     # print("Model Accuracy:", accuracy_score(y, y_pred))
#     print(f"✅ Model Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")  # Convert to percentage
#     print("Classification Report:\n", classification_report(y, y_pred))

# if __name__ == "__main__":
#     evaluate_model()



import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model paths
model_paths = {
    "Reinforcement Learning": "models/churn_rl_model.pkl",
    "Logistic Regression": "models/churn_logistic_regression.pkl",
    "XGBoost": "models/churn_xgboost.pkl",
}

def evaluate_model(model_name):
    if model_name not in model_paths:
        raise ValueError("Invalid model selection. Choose from: Reinforcement Learning, Logistic Regression, XGBoost")

    # Load processed data
    df = pd.read_csv("data/processed/feature_engineered_data.csv")

    # Define features and target
    X = df.drop(columns=['churned'])
    y = df['churned']

    # Load trained model
    model = joblib.load(model_paths[model_name])

    # Make predictions
    y_pred = model.predict(X)

    # Calculate performance metrics
    accuracy = accuracy_score(y, y_pred)
    class_report = classification_report(y, y_pred, target_names=["Not Churn", "Churn"])
    confusion = confusion_matrix(y, y_pred)

    return accuracy, class_report, confusion
