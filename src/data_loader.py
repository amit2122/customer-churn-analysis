import pandas as pd

def load_data():
    # Load raw datasets
    customers = pd.read_csv("data/raw/E-commerce Customer Behavior - Sheet1.csv")
    
    # Print basic info
    print(customers.info())
    print(customers.head())
    
    return customers

if __name__ == "__main__":
    df = load_data()
