import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Aviation Operations Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def debug_file_system():
    """Debug function to show what files are available"""
    st.subheader("🔍 File System Debug")
    
    # Current working directory
    cwd = os.getcwd()
    st.write(f"**Current Directory:** `{cwd}`")
    
    # List all files in current directory
    files_here = os.listdir('.')
    st.write("**Files in current directory:**")
    for f in sorted(files_here):
        if os.path.isfile(f):
            size = os.path.getsize(f) / 1024  # KB
            st.write(f"📄 `{f}` ({size:.1f} KB)")
        elif os.path.isdir(f):
            st.write(f"📁 `{f}/`")
    
    # Look for specific aviation files
    st.write("**Looking for aviation files:**")
    patterns = [
        "*.csv",
        "*aviation*",
        "*flight*",
        "*prediction*",
        "artifacts/*.pkl",
        "*.pkl"
    ]
    
    found_files = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            st.write(f"Pattern `{pattern}`: {len(matches)} files found")
            for match in matches[:5]:  # Show first 5
                found_files.append(match)
                st.write(f"  - `{match}`")
        else:
            st.write(f"Pattern `{pattern}`: ❌ No files found")
    
    return found_files

def create_sample_data():
    """Create sample aviation data for testing"""
    np.random.seed(42)
    n_flights = 1000
    
    airlines = ['AI', 'IX', '6E', 'UK', 'SG', 'G8']
    airports = ['DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD', 'COK', 'AMD']
    
    data = {
        'flight_id': [f"FL{i:06d}" for i in range(n_flights)],
        'airline': np.random.choice(airlines, n_flights),
        'origin': np.random.choice(airports, n_flights),
        'destination': np.random.choice(airports, n_flights),
        'scheduled_departure': pd.date_range('2024-01-01', periods=n_flights, freq='H'),
        'actual_departure': None,
        'delay_minutes': np.random.poisson(15, n_flights),
        'status': np.random.choice(['Scheduled', 'In-Flight', 'Completed', 'Delayed'], n_flights, p=[0.3, 0.2, 0.4, 0.1]),
        'risk_level': np.random.choice(['High', 'Medium', 'Low'], n_flights, p=[0.1, 0.3, 0.6]),
        'incident_probability': np.random.beta(2, 50, n_flights),  # Most flights low risk
        'weather_score': np.random.uniform(0, 1, n_flights),
        'mechanical_score': np.random.uniform(0, 1, n_flights),
        'crew_score': np.random.uniform(0, 1, n_flights),
        'atc_score': np.random.uniform(0, 1, n_flights)
    }
    
    df = pd.DataFrame(data)
    
    # Add actual departure times
    df['actual_departure'] = df['scheduled_departure'] + pd.to_timedelta(df['delay_minutes'], unit='min')
    
    # Add some incident warnings for high-risk flights
    high_risk = df['risk_level'] == 'High'
    df.loc[high_risk, 'incident_warning'] = np.random.choice([
        'Similar to Mangalore IX-812 - 158 fatalities',
        'Weather pattern matches Air India Express crash',
        'Mechanical issues similar to fatal incidents',
        'Crew fatigue levels concerning'
    ], high_risk.sum())
    
    return df

def load_aviation_data():
    """Load aviation data with fallbacks"""
    
    # Try to load production results first
    production_files = glob.glob("*aviation_predictions_PRODUCTION*.csv")
    if production_files:
        st.success(f"✅ Found production file: {production_files[0]}")
        try:
            return pd.read_csv(production_files[0])
        except Exception as e:
            st.error(f"Error loading production file: {e}")
    
    # Try to load main dataset
    dataset_files = ['final_unified_dataset.csv', 'aviation_dataset.csv', 'flight_data.csv']
    for filename in dataset_files:
        if os.path.exists(filename):
            st.success(f"✅ Found dataset: {filename}")
            try:
                df = pd.read_csv(filename)
                # Add prediction columns if missing
                if 'incident_probability' not in df.columns:
                    df['incident_probability'] = np.random.beta(2, 50, len(df))
                if 'risk_level' not in df.columns:
                    df['risk_level'] = pd.cut(df['incident_probability'], 
                                             bins=[0, 0.1, 0.3, 1.0], 
                                             labels=['Low', 'Medium', 'High'])
                return df
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
    
    # Fallback to sample data
    st.warning("⚠️ No aviation data files found. Using sample data for demonstration.")
    return create_sample_data()

def main():
    st.title("🛫 Aviation Operations Center")
    st.markdown("*Real-time flight monitoring with incident prediction*")
    
    # Sidebar for debug mode
    st.sidebar.title("🔧 Debug & Controls")
    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)
    
    if debug_mode:
        with st.expander("🔍 File System Debug", expanded=True):
            found_files = debug_file_system()
    
    # Load data
    try:
        with st.spinner("Loading aviation data..."):
            df = load_aviation_data()
        
        st.success(f"✅ Loaded {len(df):,} flight records")
        
        # Show data info
        if debug_mode:
            with st.expander("📊 Data Information"):
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Columns:**", list(df.columns))
                st.write("**Sample Data:**")
                st.dataframe(df.head())
                
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    st.write("**Risk Distribution:**")
                    st.write(risk_counts)
        
        # Main dashboard
        create_dashboard(df)
        
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.write("**Error Details:**")
        st.code(str(e))
        
        if st.button("🔄 Try Again"):
            st.experimental_rerun()

def create_dashboard(df):
    """Create the main dashboard"""
    
    # Time simulation controls
    st.header("⏰ Time Simulation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        simulation_speed = st.selectbox("Simulation Speed", ["1x", "2x", "5x", "10x"])
    
    with col2:
        if 'scheduled_departure' in df.columns:
            min_date = pd.to_datetime(df['scheduled_departure']).min().date()
            max_date = pd.to_datetime(df['scheduled_departure']).max().date()
            current_date = st.date_input("Current Date", min_date, min_value=min_date, max_value=max_date)
        else:
            current_date = st.date_input("Current Date", datetime.now().date())
    
    with col3:
        current_time = st.time_input("Current Time", datetime.now().time())
    
    # Filters
    st.header("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'airline' in df.columns:
            airlines = st.multiselect("Airlines", df['airline'].unique(), default=df['airline'].unique()[:3])
        else:
            airlines = []
    
    with col2:
        if 'status' in df.columns:
            statuses = st.multiselect("Status", df['status'].unique(), default=df['status'].unique())
        else:
            statuses = []
    
    with col3:
        if 'risk_level' in df.columns:
            risk_levels = st.multiselect("Risk Level", df['risk_level'].unique(), default=df['risk_level'].unique())
        else:
            risk_levels = []
    
    with col4:
        if 'origin' in df.columns and 'destination' in df.columns:
            routes = [f"{row['origin']}-{row['destination']}" for _, row in df[['origin', 'destination']].drop_duplicates().iterrows()]
            selected_routes = st.multiselect("Routes", routes[:10], default=routes[:3])
        else:
            selected_routes = []
    
    # Apply filters
    filtered_df = df.copy()
    if airlines and 'airline' in df.columns:
        filtered_df = filtered_df[filtered_df['airline'].isin(airlines)]
    if statuses and 'status' in df.columns:
        filtered_df = filtered_df[filtered_df['status'].isin(statuses)]
    if risk_levels and 'risk_level' in df.columns:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
    
    # Key Metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flights", f"{len(filtered_df):,}")
    
    with col2:
        if 'risk_level' in filtered_df.columns:
            high_risk = len(filtered_df[filtered_df['risk_level'] == 'High'])
            st.metric("High Risk Flights", high_risk, delta=f"{high_risk/len(filtered_df)*100:.1f}%")
        else:
            st.metric("High Risk Flights", "N/A")
    
    with col3:
        if 'delay_minutes' in filtered_df.columns:
            avg_delay = filtered_df['delay_minutes'].mean()
            st.metric("Avg Delay (min)", f"{avg_delay:.1f}")
        else:
            st.metric("Avg Delay (min)", "N/A")
    
    with col4:
        if 'incident_probability' in filtered_df.columns:
            avg_risk = filtered_df['incident_probability'].mean() * 100
            st.metric("Avg Risk Score", f"{avg_risk:.2f}%")
        else:
            st.metric("Avg Risk Score", "N/A")
    
    # Charts
    st.header("📈 Analytics")
    
    # Risk distribution
    if 'risk_level' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            risk_dist = filtered_df['risk_level'].value_counts()
            fig = px.pie(values=risk_dist.values, names=risk_dist.index, title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'airline' in filtered_df.columns:
                airline_risk = filtered_df.groupby(['airline', 'risk_level']).size().unstack(fill_value=0)
                fig = px.bar(airline_risk, title="Risk by Airline", barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
    
    # Flight table
    st.header("✈️ Flight Monitor")
    
    # Add alerts for high-risk flights
    if 'risk_level' in filtered_df.columns:
        high_risk_flights = filtered_df[filtered_df['risk_level'] == 'High']
        if len(high_risk_flights) > 0:
            st.warning(f"⚠️ {len(high_risk_flights)} HIGH RISK flights require immediate attention!")
            
            # Show sample high-risk flights
            if 'incident_warning' in high_risk_flights.columns:
                for _, flight in high_risk_flights.head(3).iterrows():
                    if pd.notna(flight.get('incident_warning')):
                        st.error(f"🚨 Flight {flight.get('flight_id', 'Unknown')}: {flight['incident_warning']}")
    
    # Display flight table
    display_columns = [col for col in ['flight_id', 'airline', 'origin', 'destination', 'status', 'risk_level', 'delay_minutes', 'incident_probability'] if col in filtered_df.columns]
    
    if display_columns:
        st.dataframe(
            filtered_df[display_columns].head(20),
            use_container_width=True
        )
    else:
        st.warning("No suitable columns found for flight display")

if __name__ == "__main__":
    main()
