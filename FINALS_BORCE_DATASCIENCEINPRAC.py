import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configurations & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MIAA Flight Forecasting Dashboard",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for a clean modern interface
st.markdown("""
    <style>
    .main-title { font-size:32px; font-weight:bold; color:#1E3A8A; margin-bottom:10px; }
    .section-title { font-size:24px; font-weight:bold; color:#1E3A8A; margin-top:20px; }
    .highlight { background-color: #F0FDF4; padding: 15px; border-left: 5px solid #22C55E; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/airport.png", width=80)
st.sidebar.title("MIAA Analytics")
page = st.sidebar.radio("Navigate Workspace", ["Business Understanding", "Data Exploration (EDA)"])

# ---------------------------------------------------------
# Helper Function: Load & Process Data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Replace this path with your actual dataset path if deploying
    file_path = 'PH_Airports_Arrivals_and_Departures.csv'
    
    # Simulating data fallback for testing/demonstration without real file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Generate artificial data mimicking the real structure for runtime safety
        date_range = pd.date_range(start="2018-01-01", end="2023-11-05", freq="H")
        df = pd.DataFrame({
            'firstSeen': date_range,
            'icao24': np.random.choice(['RPLL', 'RPLC', 'RPVM'], len(date_range)),
            'estDepartureAirport': np.random.choice(['MNL', 'CRK', 'CEB', np.nan], len(date_range)),
            'estArrivalAirport': np.random.choice(['MNL', 'CRK', 'CEB', np.nan], len(date_range))
        })
    
    # Data Processing
    df['date'] = pd.to_datetime(df['firstSeen'])
    df['date_only'] = df['date'].dt.date
    
    # Aggregating Daily Flights
    daily = df.groupby('date_only').size().reset_index(name='total_flights')
    daily['date_only'] = pd.to_datetime(daily['date_only'])
    daily = daily.sort_values('date_only')
    
    # Calculate null summary
    null_summary = df.isna().sum().to_frame(name="Missing Values")
    
    return df, daily, null_summary

# Load data assets
df, daily, null_summary = load_data()


# ---------------------------------------------------------
# Page 1: Business Understanding
# ---------------------------------------------------------
if page == "Business Understanding":
    st.markdown('<div class="main-title">Business Understanding</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### **Organization**")
        st.write("**Manila International Airport Authority (MIAA)** — Managing Ninoy Aquino International Airport (MNL) and major Philippine gateways.")
        
        st.markdown("### **Business Problem**")
        st.info(
            "Airport managers currently rely on historical averages and seasonal intuition to allocate staff for "
            "security, customs, and ground handling. This leads to either costly overstaffing or long passenger queues "
            "during unexpected surges. A data-driven forecast of daily flight volumes is needed to optimize shift scheduling, "
            "reduce wait times, and control labor costs."
        )
        
        st.markdown("### **Project Objective**")
        st.markdown(
            '<div class="highlight"><strong>Goal:</strong> Forecast the total number of daily departures and '
            'arrivals at Philippine International Airports for the next 30 days.</div>', 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### **Key Operational Questions**")
        questions = [
            "How has total daily flight volume evolved over the past 2-3 years? Are there clear growth or decline trends?",
            "Are there specific months or weeks like Christmas or summer with higher volatility or unusual traffic?",
            "Can we build a model to accurately predict daily flight counts 7, 14, and 30 days ahead?",
            "Are the model's predictions good enough to support operational decisions (e.g., error less than 15 flights per day)?",
            "What risks like sudden lockdowns or typhoons should managers be aware of that the model might miss?"
        ]
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}.** {q}")


# ---------------------------------------------------------
# Page 2: Data Understanding (EDA)
# ---------------------------------------------------------
elif page == "Data Exploration (EDA)":
    st.markdown('<div class="main-title">Data Understanding & Exploratory Analysis</div>', unsafe_allow_html=True)
    
    # Metric Summary Row
    st.markdown("### **Key Dataset Metrics**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Row Records (Raw)", f"{df.shape[0]:,}")
    m2.metric("Total Days Tracked", f"{daily.shape[0]:,}")
    m3.metric("Average Daily Flights", f"{int(daily['total_flights'].mean())} flights")
    m4.metric("Peak Single-Day Traffic", f"{daily['total_flights'].max()} flights")
    
    st.markdown("---")
    
    # Layout with columns for properties and missing values
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### **Daily Time Series Statistical Summary**")
        st.dataframe(daily.describe().T, use_container_width=True)
        
        st.markdown("### **Missing Data Profile (Raw Data)**")
        st.dataframe(null_summary, use_container_width=True)

    with col_right:
        st.markdown("### **Distribution of Daily Flights**")
        fig_hist = px.histogram(
            daily, 
            x='total_flights', 
            nbins=30,
            title="Frequency Distribution of Daily Flight Records",
            labels={'total_flights': 'Number of Flights'},
            color_discrete_sequence=['#3B82F6']
        )
        fig_hist.update_layout(bargap=0.05, yaxis_title="Count of Days")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    
    # Dynamic Volumetric Timeline Plot
    st.markdown("### **Flight Volumetric Historical Timeline**")
    fig_line = px.line(
        daily, 
        x='date_only', 
        y='total_flights',
        title="Daily Flight Volume Trends over Time at Philippine International Airports",
        labels={'date_only': 'Timeline Date', 'total_flights': 'Total Volume Count'},
        color_discrete_sequence=['#1E3A8A']
    )
    fig_line.update_xaxes(rangeslider_visible=True) # Adds interactive time zoom slider bar below plot
    st.plotly_chart(fig_line, use_container_width=True)
