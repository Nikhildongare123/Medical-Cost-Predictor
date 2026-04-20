import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        background-color: #27ae60;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .developer-info {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict your annual medical insurance charges accurately")

# Create two columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Patient Information")
    
    # Input fields with better styling
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Age of the patient (18-100 years)"
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=15.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="BMI = weight(kg) / height(m)² (Normal range: 18.5-24.9)"
    )
    
    children = st.selectbox(
        "Number of Children/Dependents",
        options=[0, 1, 2, 3, 4, 5],
        help="Number of children covered by the insurance"
    )
    
    smoker = st.radio(
        "Smoking Status",
        options=["No", "Yes"],
        horizontal=True,
        help="Does the patient smoke?"
    )

with col2:
    st.subheader("📊 Health Metrics Analysis")
    
    # BMI Category
    if bmi < 18.5:
        bmi_category = "Underweight ⚠️"
        bmi_color = "orange"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal ✅"
        bmi_color = "green"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight ⚠️"
        bmi_color = "orange"
    else:
        bmi_category = "Obese ❌"
        bmi_color = "red"
    
    st.metric("BMI Category", bmi_category)
    
    # Age group
    if age < 30:
        age_group = "Young Adult"
    elif age < 50:
        age_group = "Middle Age"
    else:
        age_group = "Senior"
    
    st.metric("Age Group", age_group)
    
    # Risk indicator
    risk_factors = 0
    if bmi >= 30:
        risk_factors += 1
    if smoker == "Yes":
        risk_factors += 1
    if age >= 50:
        risk_factors += 1
    
    if risk_factors >= 2:
        risk_level = "High Risk 🔴"
    elif risk_factors == 1:
        risk_level = "Moderate Risk 🟡"
    else:
        risk_level = "Low Risk 🟢"
    
    st.metric("Risk Level", risk_level)

# Prediction button and result
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

# Convert inputs
smoker_yes = 1 if smoker == "Yes" else 0

# Prepare input array
input_data = np.array([[age, bmi, children, smoker_yes]])

with col2:
    if st.button("🔮 Predict Insurance Cost", use_container_width=True, type="primary"):
        if model is not None:
            prediction = model.predict(input_data)[0]
            
            # Display prediction with animation
            st.balloons()
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Estimated Annual Insurance Cost</h3>
                <h1 style="font-size: 3rem; margin: 0;">${prediction:,.2f}</h1>
                <p>Monthly premium: ${prediction/12:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.info("💡 **Insight**: " + (
                "Your smoking status has the biggest impact on insurance costs." if smoker == "Yes"
                else "Maintaining a healthy BMI can help reduce your insurance costs."
            ))
        else:
            st.error("Model not loaded properly. Please check model.pkl file.")

# Feature importance visualization
st.markdown("---")
st.subheader("📈 How Each Factor Affects Your Insurance Cost")

if model is not None:
    feature_names = ['Age', 'BMI', 'Children', 'Smoker']
    coefficients = model.coef_
    
    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact on Cost': coefficients,
        'Absolute Impact': np.abs(coefficients)
    })
    
    # Sort by absolute impact
    coef_df = coef_df.sort_values('Absolute Impact', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        coef_df,
        x='Impact on Cost',
        y='Feature',
        orientation='h',
        title='Feature Impact on Insurance Cost',
        color='Impact on Cost',
        color_continuous_scale='RdYlGn',
        text='Impact on Cost'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Change in Insurance Cost ($)",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.caption("""
    *Positive impact = increases insurance cost | Negative impact = decreases insurance cost
    Being a smoker typically increases the cost by the largest margin.*
    """)

# BMI Visualization
st.subheader("📊 BMI & Cost Relationship")
bmi_range = np.arange(15, 51, 1)
predicted_costs = []

for bmi_val in bmi_range:
    temp_data = np.array([[age, bmi_val, children, smoker_yes]])
    if model is not None:
        cost = model.predict(temp_data)[0]
        predicted_costs.append(cost)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=bmi_range,
    y=predicted_costs,
    mode='lines+markers',
    name='Predicted Cost',
    line=dict(color='#27ae60', width=3),
    marker=dict(size=6)
))

fig2.update_layout(
    title=f'Insurance Cost vs BMI (Age: {age}, Smoker: {smoker})',
    xaxis_title='BMI',
    yaxis_title='Predicted Insurance Cost ($)',
    hovermode='closest',
    height=450
)

# Add vertical line for current BMI
fig2.add_vline(x=bmi, line_dash="dash", line_color="red", 
               annotation_text=f"Your BMI: {bmi}", annotation_position="top")

st.plotly_chart(fig2, use_container_width=True)

# Developer section with LinkedIn
st.markdown("---")
st.markdown(f"""
<div class="developer-info">
    <h4>👨‍💻 Developed by <strong>Nikhil Dongare</strong></h4>
    <p>Data Scientist | Machine Learning Engineer</p>
    <a href="https://www.linkedin.com/in/nikhil-dongare-5958092ba" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" 
             alt="LinkedIn">
    </a>
    <br><br>
    <small>📧 For queries and collaboration, connect on LinkedIn</small>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2026 Medical Insurance Predictor | Powered by Machine Learning</p>", unsafe_allow_html=True)
