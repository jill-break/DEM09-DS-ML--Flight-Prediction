"""
Flight Fare Prediction Web Application

A Streamlit web application for predicting flight fares using a trained ML model.

Author: Data Engineering Team
Date: 2026-02-09
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.app.model_service import ModelService
from src.app.utils import (
    AIRLINES, AIRPORTS, AIRCRAFT_TYPES, CLASSES, BOOKING_SOURCES, STOPOVERS,
    validate_date, format_currency, get_airport_name
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fare Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
    }
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 0.3rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* ── Card Styling ── */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .glass-card h3 {
        color: #38bdf8;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Prediction Result ── */
    .result-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(129, 140, 248, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .result-card .fare {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }
    .result-card .route {
        color: #cbd5e1;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .result-card .route .arrow {
        color: #38bdf8;
        font-weight: 600;
        margin: 0 0.3rem;
    }

    /* ── Stat Chips ── */
    .stat-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .stat-chip {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 140px;
    }
    .stat-chip .stat-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .stat-chip .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* ── Sidebar Polish ── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    .sidebar-metric {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-metric .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-metric .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #38bdf8;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #475569;
        font-size: 0.8rem;
    }
    .footer a {
        color: #38bdf8;
        text-decoration: none;
    }

    /* ── Button Override ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #818cf8) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.3) !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── INIT MODEL SERVICE ─────────────────────────────────────────────────────
@st.cache_resource
def get_model_service():
    """Initialize and cache the model service."""
    service = ModelService()
    service.load_model()
    return service

model_service = get_model_service()

# ─── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon"></div>
    <h1>Flight Fare Predictor</h1>
    <p>Instant fare estimates powered by Machine Learning</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Model Info")

    try:
        model_info = model_service.get_model_info()
        model_type = model_info.get('model_type', 'Gradient Boosting')
        st.markdown(f"Using **{model_type}** to predict fares for Bangladesh domestic & international routes.")

        # Load evaluation metrics
        eval_path = Path('models/evaluation_results.json')
        if eval_path.exists():
            eval_results = json.loads(eval_path.read_text())
            best_name = max(eval_results, key=lambda k: eval_results[k].get('r2_score', 0))
            m = eval_results[best_name]

            st.caption(f"Best: **{best_name}**")

            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{m['r2_score']:.4f}</div>
            </div>
            <div class="sidebar-metric">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{m['rmse']:,.0f} BDT</div>
            </div>
            <div class="sidebar-metric">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{m['mae']:,.0f} BDT</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("R² Score", "0.68")
            st.metric("RMSE", "46,194 BDT")
            st.metric("MAE", "28,007 BDT")

    except Exception as e:
        st.error(f" {str(e)}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("###  Tips")
    st.markdown("""
    - Book **early** — fares rise as departure nears
    - **Direct** flights cost more than stopovers
    - **Business/First** class can be 3-5× Economy
    """)

# ─── FORM ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card"><h3>Flight Details</h3>', unsafe_allow_html=True)

    airline = st.selectbox("Airline", options=AIRLINES, help="Select the airline")

    source_code = st.selectbox(
        "Departure Airport",
        options=list(AIRPORTS.keys()),
        format_func=lambda x: f"{x} — {get_airport_name(x)}",
    )

    destination_code = st.selectbox(
        "Arrival Airport",
        options=list(AIRPORTS.keys()),
        index=1,
        format_func=lambda x: f"{x} — {get_airport_name(x)}",
    )

    st.markdown("**Departure**")
    dep_c1, dep_c2 = st.columns(2)
    with dep_c1:
        departure_date = st.date_input(
            "Dep Date",
            value=datetime.now() + timedelta(days=30),
            min_value=datetime.now().date(),
            label_visibility="collapsed",
        )
    with dep_c2:
        departure_time = st.time_input(
            "Dep Time",
            value=datetime.strptime("10:00", "%H:%M").time(),
            label_visibility="collapsed",
        )

    st.markdown("**Arrival**")
    arr_c1, arr_c2 = st.columns(2)
    with arr_c1:
        arrival_date = st.date_input(
            "Arr Date",
            value=departure_date,
            min_value=departure_date,
            label_visibility="collapsed",
        )
    with arr_c2:
        arrival_time = st.time_input(
            "Arr Time",
            value=datetime.strptime("12:00", "%H:%M").time(),
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card"><h3> Booking Details</h3>', unsafe_allow_html=True)

    travel_class = st.radio("Travel Class", options=CLASSES, horizontal=True)

    aircraft_type = st.selectbox("Aircraft Type", options=AIRCRAFT_TYPES)

    stopovers = st.selectbox("Stopovers", options=STOPOVERS)

    booking_source = st.selectbox("Booking Source", options=BOOKING_SOURCES)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── VALIDATION ──────────────────────────────────────────────────────────────
departure_datetime = datetime.combine(departure_date, departure_time)
arrival_datetime = datetime.combine(arrival_date, arrival_time)

validation_errors = []
if source_code == destination_code:
    validation_errors.append("Source and destination cannot be the same")
if arrival_datetime <= departure_datetime:
    validation_errors.append("Arrival must be after departure")
if not validate_date(departure_datetime):
    validation_errors.append("Departure must be in the future")

for err in validation_errors:
    st.error(err)

# ─── PREDICT BUTTON ──────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    predict_button = st.button(
        " Predict Fare",
        type="primary",
        use_container_width=True,
        disabled=len(validation_errors) > 0,
    )

# ─── PREDICTION RESULT ───────────────────────────────────────────────────────
if predict_button:
    try:
        with st.spinner("Analyzing flight data…"):
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

            predicted_fare, metadata = model_service.predict(user_input)

        # ── Result Card ──
        src_name = get_airport_name(source_code)
        dst_name = get_airport_name(destination_code)
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Fare</div>
            <div class="fare">{format_currency(predicted_fare)}</div>
            <div class="route">
                {source_code} ({src_name})
                <span class="arrow">→</span>
                {destination_code} ({dst_name})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Stat Chips ──
        duration = (arrival_datetime - departure_datetime).total_seconds() / 3600
        days_before = (departure_datetime - datetime.now()).days

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip">
                <div class="stat-value">{duration:.1f} hrs</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{days_before} days</div>
                <div class="stat-label">Until Departure</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{travel_class}</div>
                <div class="stat-label">Class</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{stopovers}</div>
                <div class="stat-label">Stops</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Expandable Details ──
        with st.expander(" Prediction Details"):
            det1, det2, det3 = st.columns(3)
            det1.metric("Model", metadata['model_type'])
            det2.metric("Features", metadata['features_used'])
            det3.metric("Confidence", metadata['confidence'])
            st.info(
                "This prediction is based on historical flight data and may vary "
                "from actual fares. Demand, availability, and promotions affect "
                "real-time pricing."
            )

    except Exception as e:
        st.error(f" Prediction Failed: {str(e)}")
        with st.expander("Debug Info"):
            st.code(str(e))

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
footer_model = model_service.get_model_info().get('model_type', 'ML Model')
st.markdown(f"""
<div class="footer">
    Built using Streamlit &nbsp;·&nbsp; Model: {footer_model}
    <br>Data Engineering Project — Flight Fare Prediction
</div>
""", unsafe_allow_html=True)
