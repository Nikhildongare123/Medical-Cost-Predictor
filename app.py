import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title
st.set_page_config(page_title="Medical Cost Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Predict insurance charges based on patient information.")

# Input fields (match model's feature names: age, bmi, children, smoker_yes)
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, step=0.1)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.radio("Smoker", ["No", "Yes"])

# Convert smoker to binary (1 for Yes, 0 for No)
smoker_yes = 1 if smoker == "Yes" else 0

# Prepare input array
input_data = np.array([[age, bmi, children, smoker_yes]])

# Prediction
if st.sidebar.button("Predict Cost"):
    prediction = model.predict(input_data)
    st.subheader("📊 Estimated Insurance Cost")
    st.success(f"${prediction[0]:,.2f}")

# Optional: show feature importance (coeffs)
if st.checkbox("Show model coefficients"):
    feature_names = ["age", "bmi", "children", "smoker_yes"]
    coefs = model.coef_
    intercept = model.intercept_
    st.write("Intercept:", round(intercept, 2))
    df_coef = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    st.dataframe(df_coef)
