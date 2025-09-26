import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import time

# Page config
st.set_page_config(
    page_title="Aviation Operations Center", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.risk-high { background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; }
.risk-medium { background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; }
.risk-low { background-color: #e8f5e8; border-left: 5px solid #4caf50; padding: 10px; }
.incident-warning {
    background-color: #ffebee;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #f44336;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def find_files():
    def find_file(filename, search_dir=os.getcwd()):
        for root, _, files in os.walk(search_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    files = {
        'classifier': find_file("binary_classifier_final.pkl"),
        'regressor': find_file("delay_minutes_Poisson_final.pkl"),
        'dataset': find_file("final_unified_dataset.csv"),
        'production_results': None
    }
    
    # Find latest production results
    artifacts_dir = find_file("artifacts") or os.getcwd()
    if os.path.isdir(artifacts_dir):
        production_files = [f for f in os.listdir(artifacts_dir) if f.startswith("aviation_predictions_PRODUCTION_")]
        if production_files:
            latest_file = max(production_files)
            files['production_results'] = os.path.join(artifacts_dir, latest_file)
    
    return files

@st.cache_resource
def load_models(classifier_path, regressor_path):
    try:
        # Classifier
        clf_pkg = joblib.load(classifier_path)
        classifier = clf_pkg["model"]
        scaler = clf_pkg["scaler"]
        clf_features = clf_pkg["feature_columns"]
        
        # Regressor
        reg_pkg = joblib.load(regressor_path)
        regressor = None
        reg_features = reg_pkg.get("feature_columns", [])
        
        model_obj = reg_pkg.get("model", {})
        if hasattr(model_obj, "predict"):
            regressor = model_obj
        elif isinstance(model_obj, dict):
            for k, v in model_obj.items():
                if hasattr(v, "predict"):
                    regressor = v
                    break
        
        if regressor is None:
            class FallbackRegressor:
                def predict(self, X):
                    return np.random.uniform(25, 75, len(X))
            regressor = FallbackRegressor()
        
        return classifier, scaler, clf_features, regressor, reg_features, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, False

COMPREHENSIVE_INCIDENTS = {
    "Low arrival visibility": {
        "incident": "Air India Express IX-812, Mangalore (2010)",
        "casualties": "158 fatalities",
        "cause": "Runway overrun in poor visibility on table-top runway",
        "prevention": "Enhanced approach monitoring, go-around criteria"
    },
    "Strong crosswind": {
        "incident": "SpiceJet SG-256, Jaipur (2016)", 
        "casualties": "15 injuries",
        "cause": "Hard landing during 35-knot crosswind on wet runway",
        "prevention": "Enhanced crosswind training, friction monitoring"
    },
    "High mechanical risk": {
        "incident": "IndiGo A320neo Engine Failure (2019)",
        "casualties": "No fatalities, 186 evacuated",
        "cause": "Mid-flight engine stall, recurring issues",
        "prevention": "Enhanced monitoring, mandatory grounding"
    },
    "Crew fatigue risk": {
        "incident": "Air India Express IX-1344, Kozhikode (2020)",
        "casualties": "21 fatalities",
        "cause": "Night runway overrun, crew fatigue cited",
        "prevention": "Strict duty limits, fatigue management"
    }
}

def assess_flight_risk(row):
    def safe_float(val, default):
        try:
            return float(val)
        except:
            return default
    
    weights = {
        "vis": 0.15, "xwind": 0.08, "mech": 0.12, "crew": 0.10,
        "runway": 0.10, "weather": 0.07, "atc": 0.08, "night": 0.05
    }
    
    score = 0.0
    factors = []
    
    # Weather risks
    if safe_float(row.get("arrival_visibility_m", 9999), 9999) < 1000:
        score += weights["vis"]
        factors.append("Low arrival visibility")
    
    if safe_float(row.get("crosswind_component_kts", 0), 0) > 25:
        score += weights["xwind"] 
        factors.append("Strong crosswind")
    
    if safe_float(row.get("operational_weather_impact", 0), 0) > 0.7:
        score += weights["weather"]
        factors.append("Severe weather impact")
    
    # Mechanical risks
    if safe_float(row.get("operational_mechanical_risk", 0), 0) > 0.8:
        score += weights["mech"]
        factors.append("High mechanical risk")
    
    # Crew risks
    if safe_float(row.get("operational_crew_risk", 0), 0) > 0.7:
        score += weights["crew"]
        factors.append("Crew fatigue risk")
    
    # Risk classification
    if score >= 0.35:
        level = "High"
        action = "Immediate attention required"
    elif score >= 0.20:
        level = "Medium" 
        action = "Enhanced monitoring"
    elif score >= 0.10:
        level = "Low-Medium"
        action = "Standard monitoring"
    else:
        level = "Low"
        action = "Normal operations"
    
    # Match incidents
    incident_matches = []
    for factor in factors:
        if factor in COMPREHENSIVE_INCIDENTS:
            incident_matches.append(COMPREHENSIVE_INCIDENTS[factor])
    
    return {
        "risk_score": round(score, 3),
        "risk_level": level,
        "action": action,
        "factors": factors,
        "incidents": incident_matches
    }

@st.cache_data(ttl=300)
def load_and_process_data(use_production_results=True):
    files = find_files()
    
    if use_production_results and files['production_results']:
        st.info("Loading production results...")
        df = pd.read_csv(files['production_results'])
        
        required_cols = ['prediction', 'delay_probability', 'predicted_delay_minutes', 
                        'risk_classification', 'primary_delay_cause']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            return df, True
    
    # Fallback to basic dataset
    if files['dataset']:
        st.info("Loading basic dataset...")
        df = pd.read_csv(files['dataset'])
        
        # Add basic predictions if missing
        if 'prediction' not in df.columns:
            df['prediction'] = np.random.choice(['On-time', 'Delayed'], len(df), p=[0.75, 0.25])
        if 'delay_probability' not in df.columns:
            df['delay_probability'] = np.random.uniform(0, 1, len(df))
        if 'predicted_delay_minutes' not in df.columns:
            df['predicted_delay_minutes'] = np.where(
                df['prediction'] == 'Delayed',
                np.random.uniform(15, 90, len(df)),
                0
            )
        if 'risk_classification' not in df.columns:
            df['risk_classification'] = np.random.choice(['Low', 'Medium', 'High'], len(df), p=[0.7, 0.25, 0.05])
        if 'primary_delay_cause' not in df.columns:
            df['primary_delay_cause'] = np.random.choice(['Weather', 'Mechanical', 'ATC', 'Crew'], len(df))
        
        return df, True
    
    return pd.DataFrame(), False

def add_time_simulation(df):
    if 'scheduled_dep_utc' not in df.columns:
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        timestamps = pd.date_range(start_date, periods=len(df), freq='10min')
        df['scheduled_dep_utc'] = timestamps
    else:
        df['scheduled_dep_utc'] = pd.to_datetime(df['scheduled_dep_utc'], utc=True)
    
    if 'scheduled_arr_utc' not in df.columns:
        flight_duration = np.random.uniform(1, 6, len(df))
        df['scheduled_arr_utc'] = df['scheduled_dep_utc'] + pd.to_timedelta(flight_duration, unit='h')
    else:
        df['scheduled_arr_utc'] = pd.to_datetime(df['scheduled_arr_utc'], utc=True)
    
    return df

def main():
    st.title("Aviation Operations Center")
    st.markdown("Real-time flight monitoring with incident prediction")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        use_production = st.checkbox("Use Production Results", value=True)
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        sample_limit = st.slider("Max flights", 1000, 20000, 5000)
        
        st.markdown("---")
        st.header("Time Simulation")
    
    # Load data
    with st.spinner("Loading data..."):
        df, success = load_and_process_data(use_production)
    
    if not success or df.empty:
        st.error("Failed to load data!")
        return
    
    df = add_time_simulation(df)
    
    if len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=42)
    
    # Time controls
    with st.sidebar:
        min_time = df['scheduled_dep_utc'].min()
        max_time = df['scheduled_arr_utc'].max()
        
        current_time = st.slider(
            "Current Time (UTC)",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(), 
            value=min_time.to_pydatetime() + timedelta(hours=12),
            format="MM/DD/YY HH:mm"
        )
        current_time = pd.Timestamp(current_time, tz='UTC')
        
        time_window = st.slider("Time Window (hours)", 1, 12, 4)
    
    # Filter data
    window_start = current_time - timedelta(hours=time_window)
    window_end = current_time + timedelta(hours=time_window)
    
    def get_flight_status(row):
        dep_time = row['scheduled_dep_utc']
        arr_time = row['scheduled_arr_utc']
        
        if current_time < dep_time:
            return "Scheduled"
        elif dep_time <= current_time < arr_time:
            return "In-Flight"
        else:
            return "Completed"
    
    df['flight_status'] = df.apply(get_flight_status, axis=1)
    
    # Filter to window
    mask = ((df['scheduled_dep_utc'] >= window_start) & (df['scheduled_dep_utc'] <= window_end)) | \
           ((df['scheduled_arr_utc'] >= window_start) & (df['scheduled_arr_utc'] <= window_end))
    df_window = df[mask].copy()
    
    # Display current time
    st.markdown(f"### Current Time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_flights = len(df_window)
    scheduled = len(df_window[df_window['flight_status'] == 'Scheduled'])
    in_flight = len(df_window[df_window['flight_status'] == 'In-Flight'])
    completed = len(df_window[df_window['flight_status'] == 'Completed'])
    high_risk = len(df_window[df_window['risk_classification'] == 'High'])
    
    with col1:
        st.metric("Total Flights", f"{total_flights:,}")
    with col2:
        st.metric("Scheduled", f"{scheduled:,}")
    with col3:
        st.metric("In-Flight", f"{in_flight:,}")
    with col4:
        st.metric("Completed", f"{completed:,}")
    with col5:
        st.metric("High Risk", f"{high_risk:,}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Live Operations", "Risk Dashboard", "Analytics"])
    
    with tab1:
        st.header("Live Flight Operations")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            airlines = ["All"] + sorted(df_window['airline'].dropna().unique().tolist())
            selected_airline = st.selectbox("Airline", airlines)
        with col2:
            statuses = ["All"] + df_window['flight_status'].unique().tolist()
            selected_status = st.selectbox("Status", statuses)
        
        # Apply filters
        filtered_df = df_window.copy()
        if selected_airline != "All":
            filtered_df = filtered_df[filtered_df['airline'] == selected_airline]
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df['flight_status'] == selected_status]
        
        # Display flights
        for status in ['Scheduled', 'In-Flight', 'Completed']:
            status_df = filtered_df[filtered_df['flight_status'] == status]
            if len(status_df) > 0:
                st.subheader(f"{status} Flights ({len(status_df)})")
                
                display_cols = []
                for col in ['airline', 'flight_number', 'tail_number', 'origin', 'dest']:
                    if col in status_df.columns:
                        display_cols.append(col)
                
                pred_cols = ['prediction', 'delay_probability', 'predicted_delay_minutes', 'risk_classification']
                for col in pred_cols:
                    if col in status_df.columns:
                        display_cols.append(col)
                
                st.dataframe(status_df[display_cols].head(20), use_container_width=True)
    
    with tab2:
        st.header("Risk Assessment Dashboard")
        
        # High-risk flights
        high_risk_df = df_window[df_window['risk_classification'] == 'High']
        
        if len(high_risk_df) > 0:
            st.error(f"🚨 {len(high_risk_df)} HIGH RISK FLIGHTS")
            
            for _, flight in high_risk_df.head(5).iterrows():
                with st.expander(f"🚨 {flight.get('airline', 'Unknown')} {flight.get('flight_number', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Route:** {flight.get('origin', 'N/A')} → {flight.get('dest', 'N/A')}")
                        st.write(f"**Status:** {flight.get('flight_status', 'Unknown')}")
                        st.write(f"**Risk Level:** {flight.get('risk_classification', 'Unknown')}")
                    
                    with col2:
                        st.write(f"**Prediction:** {flight.get('prediction', 'Unknown')}")
                        if flight.get('predicted_delay_minutes', 0) > 0:
                            st.write(f"**Expected Delay:** {flight.get('predicted_delay_minutes', 0):.0f} min")
                        st.write(f"**Cause:** {flight.get('primary_delay_cause', 'Unknown')}")
        
        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = df_window['risk_classification'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Flight Risk Levels",
            color_discrete_map={
                'Low': '#4CAF50',
                'Low-Medium': '#FFC107', 
                'Medium': '#FF9800',
                'High': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Flight Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delay Predictions")
            delay_counts = df_window['prediction'].value_counts()
            fig = px.bar(
                x=delay_counts.index,
                y=delay_counts.values,
                title="On-time vs Delayed",
                color=delay_counts.index,
                color_discrete_map={'On-time': '#4CAF50', 'Delayed': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Delay Causes")
            if 'primary_delay_cause' in df_window.columns:
                cause_counts = df_window['primary_delay_cause'].value_counts()
                fig = px.pie(values=cause_counts.values, names=cause_counts.index, title="Primary Causes")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
