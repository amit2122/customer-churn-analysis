import joblib
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

def train_xgboost_model():
    df = pd.read_csv("data/processed/feature_engineered_data.csv")

    # Define features and target
    X = df.drop(columns=['churned'])
    y = df['churned']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/churn_xgboost.pkl")
    print("✅ XGBoost model trained & saved!")

if __name__ == "__main__":
    train_xgboost_model()
