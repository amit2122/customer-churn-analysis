import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data():
    df = pd.read_csv("data/raw/E-commerce Customer Behavior - Sheet1.csv")

    # Fill missing values (only for numeric columns)
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

    # Convert categorical variables into numerical format using One-Hot Encoding
    categorical_columns = ['Gender', 'City', 'Membership Type', 'Satisfaction Level']
    
    # One-Hot Encode without dropping any category
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

    # Save processed data
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("✅ Data preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
