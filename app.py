import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="Employee Burnout AI",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; color:#ff4b4b;'>Employee Burnout & Attrition Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------
# Load Models & Encoders
# ---------------------------------------------
burnout_model = joblib.load("burnout_model.pkl")
attrition_model = joblib.load("attrition_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ---------------------------------------------
# Input Section
# ---------------------------------------------
st.header("👤 Employee Information")

age = st.number_input("Age", 22, 60, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox(
    "Job Role",
    ["Developer", "Data Analyst", "Manager", "HR", "Sales"]
)
income = st.number_input("Monthly Income", 20000, 120000, 50000)
hours = st.slider("Work Hours Per Week", 35, 65, 45)
overtime = st.selectbox("Overtime", ["Yes", "No"])
job_sat = st.slider("Job Satisfaction (1–5)", 1, 5, 3)
wlb = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
years = st.number_input("Years at Company", 0, 15, 3)
promo_gap = st.slider("Years Since Last Promotion", 0, 7, 2)
manager_support = st.slider("Manager Support (1–5)", 1, 5, 3)

# ---------------------------------------------
# Encode Inputs
# ---------------------------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "JobRole": label_encoders["JobRole"].transform([job_role])[0],
    "MonthlyIncome": income,
    "WorkHoursPerWeek": hours,
    "Overtime": label_encoders["Overtime"].transform([overtime])[0],
    "JobSatisfaction": job_sat,
    "WorkLifeBalance": wlb,
    "YearsAtCompany": years,
    "PromotionGap": promo_gap,
    "ManagerSupport": manager_support
}])

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Risk", use_container_width=True)

if predict:
    st.markdown("---")
    st.header("📊 Prediction Results")

    # Burnout Prediction
    burnout_pred = burnout_model.predict(input_df)[0]
    burnout_label = label_encoders["BurnoutRisk"].inverse_transform([burnout_pred])[0]

    # Attrition Prediction
    attrition_prob = attrition_model.predict_proba(input_df)[0][1] * 100

    # ---------------------------------------------
    # Display Results
    # ---------------------------------------------
    if burnout_label == "High":
        st.error("🔥 Burnout Risk: HIGH")
    elif burnout_label == "Medium":
        st.warning("⚠️ Burnout Risk: MEDIUM")
    else:
        st.success("✅ Burnout Risk: LOW")

    st.markdown(
        f"<h3 style='color:red;'>Attrition Probability: {attrition_prob:.2f}%</h3>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------
    # Explainability using SHAP
    # ---------------------------------------------
    st.markdown("---")
    st.header("🧠 Why this Prediction? (Explainable AI)")

    explainer = shap.TreeExplainer(attrition_model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.bar_plot(
        shap_values[1][0],
        feature_names=input_df.columns,
        max_display=8
    )
    st.pyplot(fig)

    # ---------------------------------------------
    # HR Recommendations
    # ---------------------------------------------
    st.markdown("---")
    st.header("💡 HR Action Recommendations")

    if burnout_label == "High":
        st.write("🔴 Reduce workload and overtime immediately")
        st.write("🔴 Improve manager support")
        st.write("🔴 Offer wellness or leave options")
    elif burnout_label == "Medium":
        st.write("🟡 Monitor workload")
        st.write("🟡 Consider role growth or promotion")
    else:
        st.write("🟢 Employee is healthy and engaged")
