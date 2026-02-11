"""
Flight Fare Prediction Web Application

A Streamlit web application for predicting flight fares using a trained Random Forest model.

Author: Data Engineering Team
Date: 2026-02-09
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.app.model_service import ModelService
from src.app.utils import (
    AIRLINES, AIRPORTS, AIRCRAFT_TYPES, CLASSES, BOOKING_SOURCES, STOPOVERS,
    validate_date, format_currency, get_airport_name
)

# Page configuration
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .info-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize model service
@st.cache_resource
def get_model_service():
    """Initialize and cache the model service."""
    return ModelService()

model_service = get_model_service()

# Header
st.markdown('<div class="main-header"> Flight Fare Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get instant fare predictions powered by Machine Learning</div>', unsafe_allow_html=True)

# Sidebar for model info
with st.sidebar:
    st.header("ℹ About")
    st.write("""
    This application uses a trained **Random Forest** model to predict flight fares 
    for flights in Bangladesh and international routes.
    """)
    
    # Load model and display info
    try:
        model = model_service.load_model()
        st.success(" Model loaded successfully!")
        
        model_info = model_service.get_model_info()
        st.markdown("### Model Performance")
        st.metric("R² Score", "0.67")
        st.metric("RMSE", "46,898 BDT")
        st.metric("MAE", "28,096 BDT")
        
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")

# Main content
st.markdown("---")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Flight Details")
    
    # Airline selection
    airline = st.selectbox(
        "Airline *",
        options=AIRLINES,
        help="Select the airline for your flight"
    )
    
    # Source and destination
    source_code = st.selectbox(
        "Departure Airport *",
        options=list(AIRPORTS.keys()),
        format_func=lambda x: f"{x} - {get_airport_name(x)}",
        help="Select departure airport"
    )
    
    destination_code = st.selectbox(
        "Arrival Airport *",
        options=list(AIRPORTS.keys()),
        format_func=lambda x: f"{x} - {get_airport_name(x)}",
        help="Select destination airport"
    )
    
    # Departure date and time
    st.markdown("**Departure Date & Time ***")
    departure_col1, departure_col2 = st.columns(2)
    with departure_col1:
        departure_date = st.date_input(
            "Date",
            value=datetime.now() + timedelta(days=30),
            min_value=datetime.now().date(),
            label_visibility="collapsed"
        )
    with departure_col2:
        departure_time = st.time_input(
            "Time",
            value=datetime.strptime("10:00", "%H:%M").time(),
            label_visibility="collapsed"
        )
    
    # Arrival date and time
    st.markdown("**Arrival Date & Time ***")
    arrival_col1, arrival_col2 = st.columns(2)
    with arrival_col1:
        arrival_date = st.date_input(
            "Date",
            value=departure_date,
            min_value=departure_date,
            label_visibility="collapsed"
        )
    with arrival_col2:
        arrival_time = st.time_input(
            "Time",
            value=datetime.strptime("12:00", "%H:%M").time(),
            label_visibility="collapsed"
        )

with col2:
    st.subheader(" Booking Details")
    
    # Travel class
    travel_class = st.radio(
        "Travel Class *",
        options=CLASSES,
        horizontal=True,
        help="Select your preferred travel class"
    )
    
    # Aircraft type
    aircraft_type = st.selectbox(
        "Aircraft Type *",
        options=AIRCRAFT_TYPES,
        help="Select the aircraft type"
    )
    
    # Stopovers
    stopovers = st.selectbox(
        "Stopovers *",
        options=STOPOVERS,
        help="Select flight type (direct or with stops)"
    )
    
    # Booking source
    booking_source = st.selectbox(
        "Booking Source *",
        options=BOOKING_SOURCES,
        help="Select where you're booking from"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

# Combine date and time
departure_datetime = datetime.combine(departure_date, departure_time)
arrival_datetime = datetime.combine(arrival_date, arrival_time)

# Validation
validation_errors = []

if source_code == destination_code:
    validation_errors.append(" Source and destination cannot be the same")

if arrival_datetime <= departure_datetime:
    validation_errors.append(" Arrival time must be after departure time")

if not validate_date(departure_datetime):
    validation_errors.append(" Departure date must be in the future")

# Display validation errors
if validation_errors:
    for error in validation_errors:
        st.error(error)

# Prediction button
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

with predict_col2:
    predict_button = st.button(
        "Predict Fare",
        type="primary",
        use_container_width=True,
        disabled=len(validation_errors) > 0
    )

# Make prediction
if predict_button:
    try:
        with st.spinner("Analyzing flight data and making prediction..."):
            # Prepare input data
            user_input = {
                'airline': airline,
                'source': source_code,
                'source_name': get_airport_name(source_code),
                'destination': destination_code,
                'destination_name': get_airport_name(destination_code),
                'departure_datetime': departure_datetime,
                'arrival_datetime': arrival_datetime,
                'stopovers': stopovers,
                'aircraft_type': aircraft_type,
                'travel_class': travel_class,
                'booking_source': booking_source,
            }
            
            # Get prediction
            predicted_fare, metadata = model_service.predict(user_input)
            
            # Display result
            st.success("Prediction Complete!")
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin-top: 0;">Predicted Flight Fare</h3>
                <div class="prediction-value">{format_currency(predicted_fare)}</div>
                <p style="color: #666; margin-bottom: 0;">
                    <strong>Route:</strong> {source_code} ({get_airport_name(source_code)}) → 
                    {destination_code} ({get_airport_name(destination_code)})
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                duration = (arrival_datetime - departure_datetime).total_seconds() / 3600
                st.metric("Flight Duration", f"{duration:.1f} hours")
            
            with col_info2:
                days_before = (departure_datetime - datetime.now()).days
                st.metric("Days Until Departure", f"{days_before} days")
            
            with col_info3:
                st.metric("Travel Class", travel_class)
            
            # Model info
            with st.expander("Prediction Details"):
                st.write(f"**Model Type:** {metadata['model_type']}")
                st.write(f"**Features Used:** {metadata['features_used']}")
                st.write(f"**Confidence:** {metadata['confidence']}")
                st.info("""
                This prediction is based on historical flight data and may vary from actual fares. 
                Factors like demand, availability, and promotions can affect real-time pricing.
                """)
            
    except Exception as e:
        st.error(f"Prediction Failed: {str(e)}")
        st.write("Please check your inputs and try again.")
        
        # Debug info (can be removed in production)
        with st.expander("Debug Information"):
            st.write(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with ❤️ using Streamlit | Model: Random Forest (R² = 0.67)</p>
    <p style="font-size: 0.8rem;">Data Engineering Project - Flight Fare Prediction</p>
</div>
""", unsafe_allow_html=True)
