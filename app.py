import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load trained XGBoost model and saved scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")  

# Selected feature columns
selected_features = [
    'HOUR_APPR_PROCESS_START', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR',
    'CNT_FAM_MEMBERS', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
    'CNT_CHILDREN', 'EXT_SOURCE_2', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH'
]

# Feature explanations for sidebar
feature_explanations = {
    'HOUR_APPR_PROCESS_START': "🕒 Hour of application submission.",
    'NAME_CONTRACT_TYPE': "💳 Type of loan (Cash or Revolving).",
    'FLAG_OWN_CAR': "🚗 Owns a car? May reflect stability.",
    'CNT_FAM_MEMBERS': "👨‍👩‍👧‍👦 Number of family members.",
    'AMT_GOODS_PRICE': "🛒 Price of goods being purchased.",
    'REGION_POPULATION_RELATIVE': "🏘️ Population density in applicant's region.",
    'CNT_CHILDREN': "🧒 Number of children.",
    'EXT_SOURCE_2': "📊 External credit score.",
    'AMT_INCOME_TOTAL': "💰 Total annual income.",
    'DAYS_BIRTH': "🎂 Age in negative days."
}

# App title
st.title("💸 Loan Default Prediction App")

# Sidebar explanations
st.sidebar.header("Feature Descriptions")
for feature in selected_features:
    st.sidebar.markdown(f"*{feature}*: {feature_explanations.get(feature, '')}")

# Input form
st.header("Enter Applicant Information")
user_input = {}

col1, col2 = st.columns(2)

with col1:
    user_input['HOUR_APPR_PROCESS_START'] = st.slider("🕒 HOUR_APPR_PROCESS_START", 0, 23, 12)
    user_input['NAME_CONTRACT_TYPE'] = st.selectbox("💳 NAME_CONTRACT_TYPE", ['Cash loans', 'Revolving loans'])
    user_input['FLAG_OWN_CAR'] = st.selectbox("🚗 FLAG_OWN_CAR", ['Y', 'N'])
    user_input['CNT_FAM_MEMBERS'] = st.number_input("👨‍👩‍👧‍👦 CNT_FAM_MEMBERS", min_value=1, step=1)
    user_input['AMT_GOODS_PRICE'] = st.number_input("🛒 AMT_GOODS_PRICE", min_value=0, step=1000)

with col2:
    user_input['REGION_POPULATION_RELATIVE'] = st.number_input("🏘️ REGION_POPULATION_RELATIVE", 0.0, 1.0, 0.5, 0.01)
    user_input['CNT_CHILDREN'] = st.number_input("🧒 CNT_CHILDREN", 0, step=1)
    user_input['EXT_SOURCE_2'] = st.slider("📊 EXT_SOURCE_2", 0.0, 1.0, 0.5)
    user_input['AMT_INCOME_TOTAL'] = st.number_input("💰 AMT_INCOME_TOTAL", min_value=0, step=1000)
    user_input['DAYS_BIRTH'] = st.number_input("🎂 DAYS_BIRTH", min_value=-30000, max_value=-5000, step=100)

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical columns using LabelEncoder
categorical_cols = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(['Cash loans','Revolving loans'] if col=='NAME_CONTRACT_TYPE' else ['N','Y'])
    input_df[col] = le.transform(input_df[col])

# Scale numerical features using the pre-fitted scaler
scaled_input = scaler.transform(input_df[selected_features])

# Prediction
if st.button("Predict"):
    prediction = xgb_model.predict(scaled_input)
    proba = xgb_model.predict_proba(scaled_input)[0]  # [Low Risk prob, High Risk prob]

    # Show probabilities
    st.markdown(f"✅ **Probability of Low Risk:** {proba[0]:.2f}")
    st.markdown(f"⚠️ **Probability of High Risk:** {proba[1]:.2f}")

    # Show risk message
    if prediction[0] == 1:
        st.markdown(f"<div style='padding:20px; background-color:#FFCCCC; color:#990000; border-radius:10px;'>High Risk of Default!</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:20px; background-color:#CCFFCC; color:#006600; border-radius:10px;'>Low Risk of Default</div>", unsafe_allow_html=True)