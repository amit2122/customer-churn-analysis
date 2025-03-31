import sys
import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Add 'src' folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import modules
from model_prediction import predict_churn
from model_evaluation import evaluate_model
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering

# Load Available Models
model_options = ["Reinforcement Learning", "Logistic Regression", "XGBoost"]

# 🎯 **Streamlit UI**
st.title("📊 Customer Churn Prediction & Model Evaluation")

# 🌟 **Model Selection**
if "previous_model" not in st.session_state:
    st.session_state["previous_model"] = None
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None
if "evaluation_results" not in st.session_state:
    st.session_state["evaluation_results"] = None

model_choice = st.selectbox("🔍 Select Model for Prediction & Evaluation", model_options)

# ✅ **Reset Results When Model Changes**
if st.session_state["previous_model"] is not None and st.session_state["previous_model"] != model_choice:
    st.session_state["prediction_results"] = None
    st.session_state["evaluation_results"] = None
    st.session_state["previous_model"] = model_choice  # Store new model choice

st.session_state["previous_model"] = model_choice  # Store model selection

# 📌 **Upload Customer Data**
uploaded_file = st.file_uploader("📂 Upload Customer Data CSV", type=['csv'])

# ✅ **Handle New File Upload**
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["uploaded_data"] = df  # Store uploaded data
    st.session_state["prediction_results"] = None  # Clear previous predictions
    st.session_state["evaluation_results"] = None  # Clear previous evaluation
    st.write("✅ **Raw Data Preview:**", df.head())

# 📌 **Process Data & Feature Engineering (Only if Data Exists)**
if st.session_state["uploaded_data"] is not None:
    preprocess_data()
    feature_engineering()

    # Load processed data
    processed_df = pd.read_csv("data/processed/feature_engineered_data.csv")
    st.write("✅ **Processed Data:**", processed_df.head())

    # 🚀 **Predict Churn**
    if st.button("🚀 Predict Churn for All Customers"):
        processed_df['Churn Prediction'] = predict_churn(model_choice, processed_df.drop(columns=['churned'], errors='ignore'))
        st.session_state["prediction_results"] = processed_df  # Store predictions
        processed_df.to_csv("data/results/predictions.csv", index=False)  # Overwrite predictions file
        st.success("✅ Results saved as 'data/results/predictions.csv'.")

    # 📌 **Display Predictions (If Available)**
    if st.session_state["prediction_results"] is not None:
        st.write(f"✅ **Prediction Results using {model_choice}:**", 
                 st.session_state["prediction_results"][['Customer ID', 'Churn Prediction']])

    # 📊 **Evaluate Model**
    if st.button("📊 Evaluate Model Performance"):
        accuracy, class_report, confusion = evaluate_model(model_choice)
        st.session_state["evaluation_results"] = (accuracy, class_report, confusion)  # Store evaluation results

    # 📌 **Display Evaluation Results (If Available)**
    if st.session_state["evaluation_results"] is not None:
        accuracy, class_report, confusion = st.session_state["evaluation_results"]
        st.subheader(f"📊 {model_choice} Model Evaluation")
        st.write(f"✅ **{model_choice} Model Accuracy: {accuracy:.2%}**")
        st.text(f"🔍 Classification Report for {model_choice}:")
        st.text(class_report)
        st.text(f"📌 Confusion Matrix of {model_choice}:")
        st.write(confusion)
