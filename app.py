import streamlit as st
import pandas as pd
import joblib

# Load model and metadata
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Churn Intelligence Simulator", layout="centered")

st.title("📉 Customer Churn Intelligence & Retention Simulator")
st.markdown("Predict churn risk and simulate retention strategies.")

st.header("Customer Profile")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20, 120, 70)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(input_df)

# Align with training columns
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

churn_prob = model.predict_proba(input_encoded)[0][1]

st.subheader("📊 Prediction Results")
st.metric("Churn Probability", f"{churn_prob:.2%}")

if churn_prob > 0.6:
    st.error("High churn risk")
elif churn_prob > 0.3:
    st.warning("Medium churn risk")
else:
    st.success("Low churn risk")


st.subheader("💡 Retention Recommendation")

if churn_prob > 0.6:
    st.write("Offer 15% discount + upgrade to long-term contract")
elif churn_prob > 0.3:
    st.write("Offer add-on services (Security / Support)")
else:
    st.write("No action needed")

revenue_at_risk = churn_prob * monthly_charges
st.metric("Estimated Monthly Revenue at Risk", f"${revenue_at_risk:.2f}")
