import pandas as pd

def feature_engineering():
    df = pd.read_csv("data/processed/cleaned_data.csv")

    # Debugging: Print columns
    print("Available columns:", df.columns)

    if 'Days Since Last Purchase' in df.columns:
        # Modifying some customers greater than 65 days for testing
        df.loc[df.sample(frac=0.1).index, 'Days Since Last Purchase'] = 65  # Set 10% of customers as churned
        df['recency'] = df['Days Since Last Purchase']
        df['churned'] = df['recency'].apply(lambda x: 1 if x > 30 else 0)  # Mark churned if >30 days
    else:
        raise ValueError("Expected column 'Days Since Last Purchase' not found.")

    # Save updated dataset
    df.to_csv("data/processed/feature_engineered_data.csv", index=False)
    print("Feature Engineering completed!")

if __name__ == "__main__":
    feature_engineering()
