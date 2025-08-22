# 💸 Bank-Loan-default Prediction 

This project focuses on predicting the likelihood of loan default for applicants using machine learning. The goal is to build a reliable model and an interactive web app to help banks or financial institutions evaluate risk.  

---

## 🔗 Interactive Streamlit App
Check the live app here: [💻 Bank Loan Default Prediction App](https://bank-loan-default-predictor.streamlit.app/)

---

## 📊 Dataset Overview
- **Source:** Internal bank dataset  
- **Number of samples:** 307,511  
- **Target:** `TARGET` (0 = No default, 1 = Default)  
- **Features:** Various demographic, financial, and loan-related features  

---

## 🧹 Data Preprocessing
1. **Handling Missing Values (NAs)**  
   - Numeric columns (`AMT_ANNUITY`, `AMT_GOODS_PRICE`, `EXT_SOURCE_2`, `CNT_FAM_MEMBERS`) → Filled with **Median**  
   - Categorical columns (`OCCUPATION_TYPE`, `NAME_TYPE_SUITE`) → Filled with **Mode**  

2. **Encoding Categorical Variables**  
   - Used **LabelEncoder** to convert categorical features to numeric  

3. **Normalization**  
   - Applied **MinMaxScaler** to scale numeric features between 0 and 1  

4. **Handling Imbalanced Data**  
   - Used **SMOTE** twice:  
     1. Before training Decision Tree & Random Forest on all features  
     2. Before training XGBoost on Top 10 important features  

---

## 🌟 Feature Selection
- Selected **Top 10 Features** based on feature importance from XGBoost:
  - `HOUR_APPR_PROCESS_START` 🕒
  - `NAME_CONTRACT_TYPE` 💳
  - `FLAG_OWN_CAR` 🚗
  - `CNT_FAM_MEMBERS` 👨‍👩‍👧‍👦
  - `AMT_GOODS_PRICE` 🛒
  - `REGION_POPULATION_RELATIVE` 🏘️
  - `CNT_CHILDREN` 🧒
  - `EXT_SOURCE_2` 📊
  - `AMT_INCOME_TOTAL` 💰
  - `DAYS_BIRTH` 🎂

---

## 🤖 Machine Learning Models

| Model           | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | AUC     |
|----------------|--------------|--------------|-----------|--------|----------|---------|
| Decision Tree   | 0.8574       | 0.8255       | 0.8212    | 0.8322 | 0.8267   | 0.8874  |
| Random Forest   | 0.8789       | 0.8460       | 0.8311    | 0.8686 | 0.8494   | 0.9254  |
| **XGBoost**     | **0.9305**   | **0.9282**   | **0.9726**| **0.8813** | **0.9247** | **0.9629** |

- **Best model: XGBoost** (highest accuracy and AUC)

---

## 🛠️ Model Saving
- XGBoost model saved as: `xgb_model.pkl`  
- MinMaxScaler saved as: `scaler.pkl`  

---

## 💻 Streamlit Web App
- **Interactive interface** to input applicant information  
- Predicts **High/Low risk** of default  
- Shows **probability** for each class separately (High risk vs Low risk)  

**Features in the app:**  
- Hour of application 🕒  
- Loan type 💳  
- Car ownership 🚗  
- Family members 👨‍👩‍👧‍👦  
- Goods price 🛒  
- Region population 🏘️  
- Number of children 🧒  
- External credit score 📊  
- Annual income 💰  
- Age in days 🎂  

---

## 📈 How to Run the App
1. Install dependencies:  

pip install streamlit scikit-learn xgboost pandas numpy joblib

2. Run the Streamlit app:



streamlit run app.py

3. Enter applicant information and click Predict to see risk and probabilities.




---

##🔮 Results Interpretation

✅ Low Risk: Applicant is unlikely to default

⚠️ High Risk: Applicant has high probability of default


The app shows both probabilities (for High risk and Low risk) for deeper insight.


---

## 📌 Notes

-Dataset was imbalanced, so SMOTE was applied to balance classes

-Top 10 features were selected to simplify the app without losing predictive power

-Model evaluation metrics included Accuracy, Precision, Recall, F1 Score, and AUC



---
## 📽️ Presentation
Check the project presentation here: [ Project Presentation](presentation.pdf)
