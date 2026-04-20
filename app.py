import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .developer-info {
        text-align: center;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
    }
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except:
        st.error("❌ Model file not found! Please make sure 'model.pkl' is in the same directory.")
        return None

model = load_model()

# Header
st.markdown('<div class="main-header">🏥 Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)

if model is not None:
    # Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Patient Information")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        age = st.slider("📅 Age (years)", 18, 100, 30)
        bmi = st.slider("⚖️ BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
        children = st.selectbox("👶 Number of Children", [0, 1, 2, 3, 4, 5])
        smoker = st.radio("🚬 Smoker?", ["No", "Yes"], horizontal=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Health Analysis")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        # BMI Analysis
        if bmi < 18.5:
            st.warning("⚠️ Underweight - Consider gaining weight")
        elif bmi < 25:
            st.success("✅ Normal weight - Keep it up!")
        elif bmi < 30:
            st.warning("⚠️ Overweight - Consider exercise")
        else:
            st.error("❌ Obese - High health risk")
        
        # Age Analysis
        if age < 30:
            st.info("🌟 Young - Lower risk category")
        elif age < 50:
            st.info("📈 Middle age - Moderate risk")
        else:
            st.warning("⚠️ Senior - Higher risk category")
        
        # Smoking Impact
        if smoker == "Yes":
            st.error("🔥 Smoker - Significantly higher premiums")
        else:
            st.success("🍃 Non-smoker - Better rates")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction
    st.markdown("---")
    
    # Prepare data
    smoker_yes = 1 if smoker == "Yes" else 0
    input_data = np.array([[age, bmi, children, smoker_yes]])
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 PREDICT INSURANCE COST", use_container_width=True):
            prediction = model.predict(input_data)[0]
            
            # Show result
            st.markdown(f"""
            <div class="prediction-box">
                <h3>💰 Estimated Annual Insurance Cost</h3>
                <h1 style="font-size: 3.5rem; margin: 0;">${prediction:,.2f}</h1>
                <p style="font-size: 1.2rem;">Monthly Premium: ${prediction/12:,.2f}</p>
                <p style="margin-top: 1rem;">✨ Prediction completed successfully ✨</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional tips
            if smoker == "Yes":
                st.warning("💡 **Tip:** Quitting smoking can reduce your insurance cost by up to 50%!")
            if bmi > 30:
                st.warning("💡 **Tip:** Reducing BMI to normal range can lower your premium significantly!")
    
    # Feature Importance
    st.markdown("---")
    st.subheader("📊 Feature Impact Analysis")
    
    features = ['Age', 'BMI', 'Children', 'Smoker']
    coefficients = model.coef_
    
    # Create DataFrame
    impact_df = pd.DataFrame({
        'Feature': features,
        'Impact': coefficients,
        'Effect': ['Increases' if x > 0 else 'Decreases' for x in coefficients]
    })
    
    # Display as columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Age",
            value=f"${coefficients[0]:,.2f}",
            delta="per year" if coefficients[0] > 0 else "decrease"
        )
    
    with col2:
        st.metric(
            label="⚖️ BMI",
            value=f"${coefficients[1]:,.2f}",
            delta="per unit" if coefficients[1] > 0 else "decrease"
        )
    
    with col3:
        st.metric(
            label="👶 Children",
            value=f"${coefficients[2]:,.2f}",
            delta="per child" if coefficients[2] > 0 else "decrease"
        )
    
    with col4:
        st.metric(
            label="🚬 Smoker",
            value=f"${coefficients[3]:,.2f}",
            delta="additional cost" if coefficients[3] > 0 else "savings"
        )
    
    # Sample predictions
    st.markdown("---")
    with st.expander("📚 View Sample Predictions"):
        st.markdown("""
        ### Example Insurance Costs:
        
        | Age | BMI | Children | Smoker | Estimated Cost |
        |-----|-----|----------|--------|---------------|
        | 30 | 25 | 0 | No | $5,000 - $8,000 |
        | 50 | 30 | 2 | No | $10,000 - $15,000 |
        | 30 | 25 | 0 | Yes | $20,000 - $25,000 |
        | 50 | 35 | 2 | Yes | $35,000 - $45,000 |
        
        > **Note:** Smokers pay significantly higher premiums due to health risks
        """)
    
    # Developer Section
    st.markdown("---")
    st.markdown(f"""
    <div class="developer-info">
        <h3>👨‍💻 Developed by <span style="color:#667eea">Nikhil Dongare</span></h3>
        <p style="font-size: 1.1rem;">Data Scientist | Machine Learning Engineer</p>
        <a href="https://www.linkedin.com/in/nikhil-dongare-5958092ba" target="_blank">
            <button style="
                background-color: #0077b5;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            ">
                🔗 Connect on LinkedIn
            </button>
        </a>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
            📊 Machine Learning Model | 🏥 Healthcare Analytics | 💡 Insurance Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("""
    ### ⚠️ Model Not Found
    
    Please make sure you have the `model.pkl` file in the same directory as this app.
    
    **Steps to fix:**
    1. Check if model.pkl exists
    2. Verify the file name is correct
    3. Ensure the file is not corrupted
    """)

# Footer
st.markdown("""
<p style="text-align: center; color: #666; padding: 1rem; border-top: 1px solid #e0e0e0; margin-top: 2rem;">
    © 2026 Medical Insurance Cost Predictor | Powered by Linear Regression | All Rights Reserved
</p>
""", unsafe_allow_html=True)
