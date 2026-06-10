import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Australian Salary Predictor", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    pipeline = joblib.load('model_pipeline.joblib')
    features = joblib.load('model_features.joblib')
    return pipeline, features

try:
    pipeline, model_features = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🇦🇺 Australian Job Salary Predictor")
st.markdown("Predict median salary based on job market data.")

# Input Form
with st.form("prediction_form"):
    st.header("Job Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        experience = st.number_input("Experience (Numeric Level)", min_value=1, max_value=20, value=3)
        job_title_enc = st.number_input("Job Title Code", min_value=0, value=5)
        industry_enc = st.number_input("Industry Code", min_value=0, value=5)
        city_enc = st.number_input("City Code", min_value=0, value=5)
        
    with col2:
        month = st.slider("Month", 1, 12, 6)
        is_high_demand = st.checkbox("Is High Demand Job?", value=True)
        remote_flag = st.checkbox("Remote Available?", value=False)
        metro_city = st.checkbox("Metro City?", value=True)
        
    with col3:
        # Categorical handling
        openings_level = st.selectbox("Openings Level", ["High", "Medium", "Low"])
        cost_region = st.selectbox("Cost Region", ["High", "Medium", "Low"])
        
    st.header("Market Metrics")
    col4, col5 = st.columns(2)
    with col4:
        job_market_pressure = st.slider("Job Market Pressure (0-1)", 0.0, 1.0, 0.5)
        competitiveness_score = st.slider("Competitiveness Score (0-100)", 0.0, 100.0, 50.0)
    with col5:
        seasonal_hiring_score = st.slider("Seasonal Hiring Score", 0, 10, 5)
        demand_index = st.number_input("Demand Index", min_value=0, value=100)
        number_of_openings = st.number_input("Number of Openings", min_value=0, value=10)

    # Hidden/Less important inputs (defaulted based on averages or sliders if user wants precision)
    with st.expander("Advanced Features (Encoded)"):
        c1, c2 = st.columns(2)
        with c1:
            region_enc = st.number_input("Region Code", value=5)
            edu_req_enc = st.number_input("Education Req Code", value=2)
            job_type_enc = st.number_input("Job Type Code", value=2)
        with c2:
            skills_req_enc = st.number_input("Skills Req Code", value=5)
            hiring_trend_enc = st.number_input("Hiring Trend Code", value=5)
            season_enc = st.number_input("Season Code", value=1)

    submit = st.form_submit_button("Predict Salary")

if submit:
    # Prepare input DataFrame
    input_data = {
        'month': month,
        'number_of_openings': number_of_openings,
        'demand_index': demand_index,
        'is_high_demand': 1 if is_high_demand else 0,
        'experience_numeric': experience,
        'remote_flag': 1 if remote_flag else 0,
        'metro_city': 1 if metro_city else 0,
        'job_market_pressure': job_market_pressure,
        'competitiveness_score': competitiveness_score,
        'seasonal_hiring_score': seasonal_hiring_score,
        'job_title_encoded': job_title_enc,
        'industry_encoded': industry_enc,
        'city_encoded': city_enc,
        'region_encoded': region_enc,
        'education_requirement_encoded': edu_req_enc,
        'job_type_encoded': job_type_enc,
        'skills_required_encoded': skills_req_enc,
        'hiring_trend_encoded': hiring_trend_enc,
        'season_encoded': season_enc,
        # Handling OneHot for 'openings_level'. Training had 'openings_level_Low', 'openings_level_Medium'
        # Base category is likely 'High' since it was dropped first?
        # Logic: if selected is Low -> Low=1, Medium=0. If Medium -> Low=0, Medium=1. High -> Both 0.
        'openings_level_Low': 1 if openings_level == 'Low' else 0,
        'openings_level_Medium': 1 if openings_level == 'Medium' else 0,
        # Same for cost_region
        'cost_region_Low': 1 if cost_region == 'Low' else 0,
        'cost_region_Medium': 1 if cost_region == 'Medium' else 0,
    }
    
    # Ensure columns match model feature order
    # Any missing columns (if model has more) fill with 0
    df_input = pd.DataFrame([input_data])
    
    # Reindex to match training columns
    df_input = df_input.reindex(columns=model_features, fill_value=0)
    
    prediction = pipeline.predict(df_input)[0]
    
    st.balloons()
    st.metric(label="Predicted Median Salary (AUD)", value=f"${prediction:,.2f}")
    
    st.info("Note: Prediction is based on a Random Forest model trained on Australian job market data.")
