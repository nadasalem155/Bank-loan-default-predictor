import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained XGBoost model and saved scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")  

# Selected feature columns
selected_features = [
    'HOUR_APPR_PROCESS_START', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR',
    'CNT_FAM_MEMBERS', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
    'CNT_CHILDREN', 'EXT_SOURCE_2', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH'
]

# Feature explanations with emojis (for sidebar only)
feature_explanations = {
    'HOUR_APPR_PROCESS_START': "🕒 Hour of application submission. Morning hours might suggest more discipline.",
    'NAME_CONTRACT_TYPE': "💳 Type of loan (Cash or Revolving).",
    'FLAG_OWN_CAR': "🚗 Owns a car? May reflect stability.",
    'CNT_FAM_MEMBERS': "👨‍👩‍👧‍👦 Number of family members. More members may increase expenses.",
    'AMT_GOODS_PRICE': "🛒 Price of goods being purchased. Larger amounts may increase risk.",
    'REGION_POPULATION_RELATIVE': "🏘️ Population density in the applicant's region.",
    'CNT_CHILDREN': "🧒 Number of children. More children may suggest financial responsibility.",
    'EXT_SOURCE_2': "📊 External credit score. Higher scores usually mean lower risk.",
    'AMT_INCOME_TOTAL': "💰 Total annual income. Higher income can lower default chances.",
    'DAYS_BIRTH': "🎂 Age in negative days. Older applicants are often more stable."
}

# App title
st.title("💸Loan Default Prediction App")

# Sidebar explanations
st.sidebar.header("Feature Descriptions")
for feature in selected_features:
    st.sidebar.markdown(f"*{feature}*: {feature_explanations.get(feature, '')}")

# Input form
st.header("Enter Applicant Information")

user_input = {}

# Create two columns
col1, col2 = st.columns(2)

with col1:
    user_input['HOUR_APPR_PROCESS_START'] = st.slider("🕒 HOUR_APPR_PROCESS_START", 0, 23, 12)
    user_input['NAME_CONTRACT_TYPE'] = st.selectbox("💳 NAME_CONTRACT_TYPE", ['Cash loans', 'Revolving loans'])
    user_input['FLAG_OWN_CAR'] = st.selectbox("🚗 FLAG_OWN_CAR", ['Y', 'N'])
    user_input['CNT_FAM_MEMBERS'] = st.number_input("👨‍👩‍👧‍👦 CNT_FAM_MEMBERS", min_value=1.0, step=1.0)
    user_input['AMT_GOODS_PRICE'] = st.number_input("🛒 AMT_GOODS_PRICE", min_value=0.0, step=1000.0)

with col2:
    user_input['REGION_POPULATION_RELATIVE'] = st.number_input("🏘️ REGION_POPULATION_RELATIVE", min_value=0.0, max_value=1.0, step=0.01)
    user_input['CNT_CHILDREN'] = st.number_input("🧒 CNT_CHILDREN", min_value=0, step=1)
    user_input['EXT_SOURCE_2'] = st.slider("📊 EXT_SOURCE_2", 0.0, 1.0, 0.5)
    user_input['AMT_INCOME_TOTAL'] = st.number_input("💰 AMT_INCOME_TOTAL", min_value=0.0, step=1000.0)
    user_input['DAYS_BIRTH'] = st.number_input("🎂 DAYS_BIRTH", min_value=-30000, max_value=-5000, step=100)

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical features
input_df['NAME_CONTRACT_TYPE'] = input_df['NAME_CONTRACT_TYPE'].map({'Cash loans': 0, 'Revolving loans': 1})
input_df['FLAG_OWN_CAR'] = input_df['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1})

# Scale using pre-fitted scaler
scaled_input = scaler.transform(input_df)

# Predict using the model
if st.button("Predict"):
    prediction = xgb_model.predict(scaled_input)
    proba = xgb_model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.markdown(f"<div style='padding:20px; background-color:#FFCCCC; color:#990000; border-radius:10px;'>⚠️ High Risk of Default! (Probability: {proba:.2f})</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:20px; background-color:#CCFFCC; color:#006600; border-radius:10px;'>✅ Low Risk of Default (Probability: {proba:.2f})</div>", unsafe_allow_html=True)
