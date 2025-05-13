import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Load the saved model
model = joblib.load("random_forest_model.pkl")

# Feature names and class labels
FEATURES = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
CLASS_LABELS = {0: 'Non-Diabetic (N)', 1: 'Diabetic (Y)', 2: 'Pre-Diabetic (P)'}

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Predicts likelihood of Diabetes, Pre-Diabetes, or Non-Diabetes based on health metrics.")

st.sidebar.header("Input Health Metrics")
def user_input():
    return pd.DataFrame([{
        'AGE': st.sidebar.number_input("Age (years)", 0, 120, 50, 1),
        'Gender': 1 if st.sidebar.radio("Gender", ["Male", "Female"], 0) == "Male" else 0,
        'Urea': st.sidebar.number_input("Urea (mg/dl)", value=4.7, step=0.1),
        'Cr': st.sidebar.number_input("Creatinine (mg/dl)", value=46.0, step=0.1),
        'HbA1c': st.sidebar.number_input("HbA1c (%)", value=4.9, step=0.1),
        'Chol': st.sidebar.number_input("Cholesterol (mg/dl)", value=4.2, step=0.1),
        'TG': st.sidebar.number_input("Triglycerides (mg/dl)", value=0.9, step=0.1),
        'HDL': st.sidebar.number_input("HDL (mg/dl)", value=2.4, step=0.1),
        'LDL': st.sidebar.number_input("LDL (mg/dl)", value=1.4, step=0.1),
        'VLDL': st.sidebar.number_input("VLDL (mg/dl)", value=0.5, step=0.1),
        'BMI': st.sidebar.number_input("BMI (kg/m²)", value=24.0, step=0.1)
    }])

data = user_input()

st.subheader("Your Input Data")
st.write(data)

# Prediction
prediction = model.predict(data[FEATURES])[0]
prediction_proba = model.predict_proba(data[FEATURES])[0] * 100

st.subheader("Prediction Result")
st.write(f"Predicted Status: **{CLASS_LABELS[prediction]}**")

st.subheader("Prediction Probabilities")
st.write(pd.DataFrame([prediction_proba], columns=CLASS_LABELS.values()).style.format("{:.2f}%"))

# Pie Chart for Probabilities
fig, ax = plt.subplots()
ax.pie(prediction_proba, labels=CLASS_LABELS.values(), autopct="%.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99"])
ax.set_title("Prediction Probabilities")

buf = BytesIO()
plt.savefig(buf, format="png")
st.image(buf)
buf.close()

# Additional Insights
st.subheader("Additional Insights")
st.write("### Risk Assessment")
if prediction == 1:
    st.warning("High risk of Diabetes. Please consult a healthcare professional.")
elif prediction == 2:
    st.info("Pre-Diabetic condition detected. Lifestyle changes are recommended.")
else:
    st.success("No Diabetes detected. Maintain a healthy lifestyle.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
