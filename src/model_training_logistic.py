import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

def train_logistic_model():
    df = pd.read_csv("data/processed/feature_engineered_data.csv")

    # Define features and target
    X = df.drop(columns=['churned'])
    y = df['churned']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/churn_logistic_regression.pkl")
    print("✅ Logistic Regression model trained & saved!")

if __name__ == "__main__":
    train_logistic_model()
