import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="Employee Burnout AI", layout="centered")

# ---------------------------------------------
# Theme
# ---------------------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #f8f9fa, #e9ecef); }
h1, h2, h3 { color: #b02a37; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Employee Burnout & Attrition Predictor</h1>", unsafe_allow_html=True)
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
job_role = st.selectbox("Job Role", ["Developer", "Data Analyst", "Manager", "HR", "Sales"])
income = st.number_input("Monthly Income", 20000, 120000, 50000, step=1000)
hours = st.number_input("Work Hours Per Week", 35, 65, 45)
overtime = st.selectbox("Overtime", ["Yes", "No"])
job_sat = st.selectbox("Job Satisfaction (1–5)", [1,2,3,4,5], index=2)
wlb = st.selectbox("Work-Life Balance (1–5)", [1,2,3,4,5], index=2)
years = st.number_input("Years at Company", 0, 15, 3)
promo_gap = st.number_input("Years Since Last Promotion", 0, 7, 2)
manager_support = st.selectbox("Manager Support (1–5)", [1,2,3,4,5], index=2)

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
predict = st.button("🔍 Predict Risk", use_container_width=True)

if predict:
    st.markdown("---")
    st.header("📊 Prediction Results")

    # -----------------------------------------
    # Burnout Prediction
    # -----------------------------------------
    burnout_pred = burnout_model.predict(input_df)[0]
    burnout_label = {0:"Low",1:"Medium",2:"High"}[burnout_pred]

    # -----------------------------------------
    # Attrition Prediction (ML)
    # -----------------------------------------
    input_df_attrition = input_df.copy()
    input_df_attrition["BurnoutRisk"] = burnout_pred
    ml_prob = attrition_model.predict_proba(input_df_attrition)[0][1] * 100

    # -----------------------------------------
    # FEATURE RISK SCORE (RULE-BASED)
    # -----------------------------------------
    feature_score = 0

    if hours > 55: feature_score += 15
    if overtime == "Yes": feature_score += 10
    if job_sat <= 2: feature_score += 20
    if wlb <= 2: feature_score += 15
    if manager_support <= 2: feature_score += 15
    if promo_gap >= 4: feature_score += 10
    if income < 30000: feature_score += 10

    feature_score = min(feature_score, 100)

    # -----------------------------------------
    # FINAL HYBRID SCORE
    # -----------------------------------------
    final_risk = (0.6 * ml_prob) + (0.4 * feature_score)

    # -----------------------------------------
    # Risk Display (COLOR CHANGES ALWAYS)
    # -----------------------------------------
    if final_risk >= 70:
        st.error(f"🔴 HIGH Attrition Risk: {final_risk:.1f}%")
        risk_level = "HIGH"
    elif final_risk >= 40:
        st.warning(f"🟡 MEDIUM Attrition Risk: {final_risk:.1f}%")
        risk_level = "MEDIUM"
    else:
        st.success(f"🟢 LOW Attrition Risk: {final_risk:.1f}%")
        risk_level = "LOW"

    # -----------------------------------------
    # Burnout Display
    # -----------------------------------------
    if burnout_label == "High":
        st.error("🔥 Burnout Risk: HIGH")
    elif burnout_label == "Medium":
        st.warning("⚠️ Burnout Risk: MEDIUM")
    else:
        st.success("✅ Burnout Risk: LOW")

    # -----------------------------------------
    # HR RECOMMENDATIONS (NOW DYNAMIC)
    # -----------------------------------------
    st.markdown("---")
    st.header("💡 HR Action Recommendations")

    if risk_level == "HIGH":
        st.write("🔴 Immediate intervention required")
        if hours > 55: st.write("• Reduce excessive work hours")
        if overtime == "Yes": st.write("• Limit overtime")
        if job_sat <= 2: st.write("• Address job dissatisfaction")
        if manager_support <= 2: st.write("• Improve manager support")
        if promo_gap >= 4: st.write("• Review promotion opportunities")

    elif risk_level == "MEDIUM":
        st.write("🟡 Preventive actions recommended")
        if wlb <= 3: st.write("• Improve work-life balance")
        if job_sat <= 3: st.write("• Increase recognition & feedback")
        st.write("• Regular check-ins")

    else:
        st.write("🟢 Employee is stable")
        st.write("• Maintain engagement")
        st.write("• Support career growth")
