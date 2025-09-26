import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

def generate_realistic_aviation_data(n_flights=10000):
    """Generate comprehensive realistic aviation dataset"""
    
    print("🛫 Generating realistic aviation dataset...")
    
    # Indian airline data with realistic fleet info
    airlines_data = {
        'Air India': {
            'code': 'AI', 
            'aircraft_types': ['Boeing 787-8', 'Boeing 777-300ER', 'Airbus A320neo', 'Boeing 737-800'],
            'fleet_size': 120,
            'safety_score': 85
        },
        'IndiGo': {
            'code': '6E', 
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo', 'ATR 72-600'],
            'fleet_size': 280,
            'safety_score': 92
        },
        'SpiceJet': {
            'code': 'SG', 
            'aircraft_types': ['Boeing 737-800', 'Boeing 737 MAX 8', 'Bombardier Q400'],
            'fleet_size': 90,
            'safety_score': 78
        },
        'Vistara': {
            'code': 'UK', 
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo', 'Boeing 787-9'],
            'fleet_size': 60,
            'safety_score': 95
        },
        'GoFirst': {
            'code': 'G8', 
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo'],
            'fleet_size': 55,
            'safety_score': 82
        },
        'AirAsia India': {
            'code': 'I5', 
            'aircraft_types': ['Airbus A320neo'],
            'fleet_size': 30,
            'safety_score': 88
        }
    }
    
    # Indian airports with coordinates
    airports_data = {
        'DEL': {'name': 'Delhi', 'lat': 28.5562, 'lng': 77.1000, 'elevation': 777, 'traffic_level': 'Very High'},
        'BOM': {'name': 'Mumbai', 'lat': 19.0896, 'lng': 72.8656, 'elevation': 11, 'traffic_level': 'Very High'},
        'BLR': {'name': 'Bangalore', 'lat': 13.1986, 'lng': 77.7066, 'elevation': 3000, 'traffic_level': 'High'},
        'MAA': {'name': 'Chennai', 'lat': 12.9941, 'lng': 80.1709, 'elevation': 52, 'traffic_level': 'High'},
        'CCU': {'name': 'Kolkata', 'lat': 22.6547, 'lng': 88.4467, 'elevation': 16, 'traffic_level': 'High'},
        'HYD': {'name': 'Hyderabad', 'lat': 17.2403, 'lng': 78.4294, 'elevation': 1742, 'traffic_level': 'High'},
        'COK': {'name': 'Kochi', 'lat': 10.1520, 'lng': 76.4019, 'elevation': 106, 'traffic_level': 'Medium'},
        'AMD': {'name': 'Ahmedabad', 'lat': 23.0726, 'lng': 72.6263, 'elevation': 189, 'traffic_level': 'Medium'},
        'PNQ': {'name': 'Pune', 'lat': 18.5822, 'lng': 73.9197, 'elevation': 1942, 'traffic_level': 'Medium'},
        'JAI': {'name': 'Jaipur', 'lat': 26.8247, 'lng': 75.8127, 'elevation': 1263, 'traffic_level': 'Medium'}
    }
    
    # Generate aircraft registry
    def generate_tail_numbers(airline_code, count):
        """Generate realistic Indian aircraft tail numbers"""
        tail_numbers = []
        for i in range(count):
            # Indian aircraft registration format: VT-XXX
            suffix = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))
            tail_numbers.append(f'VT-{suffix}')
        return tail_numbers
    
    # Create base dataset
    flights_data = []
    
    for i in range(n_flights):
        # Select airline
        airline_name = np.random.choice(list(airlines_data.keys()))
        airline_info = airlines_data[airline_name]
        
        # Select aircraft type
        aircraft_type = np.random.choice(airline_info['aircraft_types'])
        
        # Generate tail number
        tail_number = f"VT-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3, replace=True).tobytes().decode('utf-8')}"
        
        # Select origin and destination
        origin = np.random.choice(list(airports_data.keys()))
        destination = np.random.choice([apt for apt in airports_data.keys() if apt != origin])
        
        # Generate flight timing
        base_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        scheduled_departure = base_date + timedelta(
            hours=np.random.randint(6, 23),
            minutes=np.random.choice([0, 15, 30, 45])
        )
        
        # Aircraft age and maintenance factors
        aircraft_age_years = np.random.uniform(1, 20)
        flight_hours = np.random.uniform(5000, 80000)
        cycles = flight_hours / 1.5  # Approximate flight cycles
        
        # Time since last maintenance (days)
        last_maintenance_days = np.random.uniform(1, 180)
        
        # Weather conditions
        origin_weather = airports_data[origin]
        dest_weather = airports_data[destination]
        
        # Seasonal weather factors
        month = scheduled_departure.month
        monsoon_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
        winter_fog_factor = 1.3 if month in [12, 1, 2] and origin in ['DEL', 'JAI'] else 1.0
        
        weather_score = np.random.uniform(0.2, 1.0) * monsoon_factor * winter_fog_factor
        weather_score = min(weather_score, 1.0)
        
        # Technical factors
        engine_health = max(0, 100 - aircraft_age_years * 2 - np.random.uniform(0, 20))
        structural_integrity = max(0, 100 - aircraft_age_years * 1.5 - cycles/1000 - np.random.uniform(0, 15))
        avionics_status = max(0, 100 - aircraft_age_years * 1 - np.random.uniform(0, 10))
        
        # Maintenance score based on time since last maintenance
        maintenance_score = max(0, 100 - last_maintenance_days/2 - np.random.uniform(0, 20))
        
        # Crew factors
        pilot_experience_hours = np.random.uniform(500, 15000)
        pilot_experience = min(100, pilot_experience_hours / 150)
        crew_rest_hours = np.random.uniform(8, 24)
        crew_fatigue_factor = max(0, min(100, crew_rest_hours * 4))
        
        # Air Traffic Control factors
        traffic_multiplier = {
            'Very High': 1.5, 'High': 1.2, 'Medium': 1.0, 'Low': 0.8
        }
        origin_traffic = traffic_multiplier[airports_data[origin]['traffic_level']]
        dest_traffic = traffic_multiplier[airports_data[destination]['traffic_level']]
        atc_delay_probability = (origin_traffic + dest_traffic) / 2
        atc_score = max(0, 100 - atc_delay_probability * 30 - np.random.uniform(0, 20))
        
        # Calculate overall risk factors
        technical_risk = (
            (100 - engine_health) * 0.4 +
            (100 - structural_integrity) * 0.3 +
            (100 - avionics_status) * 0.2 +
            (100 - maintenance_score) * 0.1
        )
        
        human_risk = (
            (100 - pilot_experience) * 0.7 +
            (100 - crew_fatigue_factor) * 0.3
        )
        
        environmental_risk = (
            (weather_score * 100) * 0.7 +
            (100 - atc_score) * 0.3
        )
        
        # Overall incident probability
        incident_probability = (
            technical_risk * 0.5 +
            human_risk * 0.3 +
            environmental_risk * 0.2
        ) / 100
        
        # Add airline safety score influence
        incident_probability *= (100 - airline_info['safety_score']) / 100
        
        # Ensure probability is between 0 and 1
        incident_probability = max(0, min(1, incident_probability))
        
        # Determine risk level
        if incident_probability > 0.7:
            risk_level = 'CRITICAL'
        elif incident_probability > 0.5:
            risk_level = 'HIGH'
        elif incident_probability > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Calculate delay
        base_delay = np.random.poisson(8)  # Base delay in minutes
        
        # Weather delays
        if weather_score > 0.7:
            weather_delay = np.random.poisson(30)
        elif weather_score > 0.5:
            weather_delay = np.random.poisson(15)
        else:
            weather_delay = 0
        
        # Technical delays
        if engine_health < 70 or maintenance_score < 60:
            technical_delay = np.random.poisson(45)
        elif engine_health < 85:
            technical_delay = np.random.poisson(20)
        else:
            technical_delay = 0
        
        # ATC delays
        if atc_score < 70:
            atc_delay = np.random.poisson(25)
        elif atc_score < 85:
            atc_delay = np.random.poisson(10)
        else:
            atc_delay = 0
        
        total_delay = base_delay + weather_delay + technical_delay + atc_delay
        
        # Flight status
        if total_delay > 60:
            status = 'DELAYED'
        elif np.random.random() > 0.7:
            status = 'IN-FLIGHT'
        elif np.random.random() > 0.5:
            status = 'COMPLETED'
        else:
            status = 'SCHEDULED'
        
        # Actual departure time
        actual_departure = scheduled_departure + timedelta(minutes=total_delay)
        
        flight_data = {
            # Basic flight info
            'flight_id': f'{airline_info["code"]}{1000 + i}',
            'airline': airline_name,
            'airline_code': airline_info['code'],
            'aircraft_type': aircraft_type,
            'tail_number': tail_number,
            'origin': origin,
            'destination': destination,
            'origin_name': airports_data[origin]['name'],
            'destination_name': airports_data[destination]['name'],
            
            # Timing
            'scheduled_departure': scheduled_departure,
            'actual_departure': actual_departure,
            'delay_minutes': total_delay,
            'status': status,
            
            # Aircraft characteristics
            'aircraft_age_years': aircraft_age_years,
            'flight_hours': flight_hours,
            'cycles': cycles,
            'last_maintenance_days': last_maintenance_days,
            
            # Technical factors
            'engine_health': engine_health,
            'structural_integrity': structural_integrity,
            'avionics_status': avionics_status,
            'maintenance_score': maintenance_score,
            
            # Human factors
            'pilot_experience': pilot_experience,
            'crew_fatigue_factor': crew_fatigue_factor,
            
            # Environmental factors
            'weather_score': weather_score,
            'atc_score': atc_score,
            
            # Risk assessment
            'technical_risk': technical_risk,
            'human_risk': human_risk,
            'environmental_risk': environmental_risk,
            'incident_probability': incident_probability,
            'risk_level': risk_level,
            
            # Location data
            'origin_lat': airports_data[origin]['lat'],
            'origin_lng': airports_data[origin]['lng'],
            'dest_lat': airports_data[destination]['lat'],
            'dest_lng': airports_data[destination]['lng'],
            
            # Current position (for in-flight aircraft)
            'current_lat': airports_data[origin]['lat'] + 
                          (airports_data[destination]['lat'] - airports_data[origin]['lat']) * np.random.random(),
            'current_lng': airports_data[origin]['lng'] + 
                          (airports_data[destination]['lng'] - airports_data[origin]['lng']) * np.random.random(),
            'altitude': np.random.randint(25000, 42000) if status == 'IN-FLIGHT' else 0,
            'speed': np.random.randint(400, 550) if status == 'IN-FLIGHT' else 0,
            'heading': np.random.randint(0, 360),
            
            # Delay breakdown
            'weather_delay': weather_delay,
            'technical_delay': technical_delay,
            'atc_delay': atc_delay,
            'base_delay': base_delay
        }
        
        flights_data.append(flight_data)
    
    df = pd.DataFrame(flights_data)
    print(f"✅ Generated {len(df)} flight records")
    return df

def train_ml_models(df):
    """Train machine learning models for predictions"""
    
    print("🤖 Training ML models...")
    
    # Prepare features for modeling
    feature_columns = [
        'aircraft_age_years', 'flight_hours', 'cycles', 'last_maintenance_days',
        'engine_health', 'structural_integrity', 'avionics_status', 'maintenance_score',
        'pilot_experience', 'crew_fatigue_factor', 'weather_score', 'atc_score'
    ]
    
    X = df[feature_columns]
    
    # Binary incident classification (high risk vs low risk)
    y_incident = (df['incident_probability'] > 0.5).astype(int)
    
    # Delay prediction
    y_delay = df['delay_minutes']
    
    # Train incident classifier
    X_train, X_test, y_inc_train, y_inc_test = train_test_split(X, y_incident, test_size=0.2, random_state=42)
    
    incident_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    incident_classifier.fit(X_train, y_inc_train)
    
    # Train delay predictor
    _, _, y_delay_train, y_delay_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    
    delay_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    delay_predictor.fit(X_train, y_delay_train)
    
    # Evaluate models
    inc_pred = incident_classifier.predict(X_test)
    delay_pred = delay_predictor.predict(X_test)
    
    print("📊 Model Performance:")
    print("Incident Classifier:")
    print(classification_report(y_inc_test, inc_pred))
    print(f"Delay Predictor RMSE: {np.sqrt(mean_squared_error(y_delay_test, delay_pred)):.2f}")
    
    return incident_classifier, delay_predictor, feature_columns

def create_guardian_eye_streamlit():
    """Create Guardian Eye Streamlit dashboard"""
    
    st.set_page_config(
        page_title="Guardian Eye - Aviation Operations Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for Guardian Eye theme
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
    }
    .metric-card {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .risk-critical {
        border-left: 5px solid #DC2626;
        background: rgba(220, 38, 38, 0.1);
    }
    .risk-high {
        border-left: 5px solid #F59E0B;
        background: rgba(245, 158, 11, 0.1);
    }
    .risk-medium {
        border-left: 5px solid #3B82F6;
        background: rgba(59, 130, 246, 0.1);
    }
    .risk-low {
        border-left: 5px solid #10B981;
        background: rgba(16, 185, 129, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #3B82F6; font-size: 3rem; margin: 0;">🛡️ GUARDIAN EYE</h1>
            <p style="color: #9CA3AF; font-size: 1.2rem;">Aviation Operations Center</p>
            <p style="color: #6B7280;">Real-time Flight Safety Monitoring & Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    if not os.path.exists('aviation_dataset.csv'):
        with st.spinner("🛫 Generating aviation dataset..."):
            df = generate_realistic_aviation_data(5000)
            df.to_csv('aviation_dataset.csv', index=False)
            st.success("✅ Aviation dataset created!")
    else:
        df = pd.read_csv('aviation_dataset.csv')
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Flight Selection")
    
    # Airline filter
    airlines = ['All Airlines'] + sorted(df['airline'].unique())
    selected_airline = st.sidebar.selectbox("Select Airline", airlines)
    
    # Filter data based on airline
    if selected_airline != 'All Airlines':
        filtered_df = df[df['airline'] == selected_airline]
    else:
        filtered_df = df
    
    # Aircraft type filter
    aircraft_types = ['All Types'] + sorted(filtered_df['aircraft_type'].unique())
    selected_aircraft_type = st.sidebar.selectbox("Select Aircraft Type", aircraft_types)
    
    # Filter by aircraft type
    if selected_aircraft_type != 'All Types':
        filtered_df = filtered_df[filtered_df['aircraft_type'] == selected_aircraft_type]
    
    # Tail number filter
    tail_numbers = ['All Aircraft'] + sorted(filtered_df['tail_number'].unique())
    selected_tail = st.sidebar.selectbox("Select Tail Number", tail_numbers)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Aircraft", len(filtered_df), delta=None)
    
    with col2:
        in_flight = len(filtered_df[filtered_df['status'] == 'IN-FLIGHT'])
        st.metric("In Flight", in_flight, delta=f"{in_flight/len(filtered_df)*100:.1f}%")
    
    with col3:
        critical_risk = len(filtered_df[filtered_df['risk_level'] == 'CRITICAL'])
        st.metric("Critical Risk", critical_risk, delta="⚠️" if critical_risk > 0 else "✅")
    
    with col4:
        avg_risk = filtered_df['incident_probability'].mean() * 100
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%", delta=None)
    
    # Aircraft-specific analysis
    if selected_tail != 'All Aircraft':
        aircraft_data = filtered_df[filtered_df['tail_number'] == selected_tail].iloc[0]
        
        st.markdown("---")
        st.markdown(f"## 🛩️ Aircraft Analysis: {selected_tail}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Aircraft Information")
            st.write(f"**Airline:** {aircraft_data['airline']}")
            st.write(f"**Type:** {aircraft_data['aircraft_type']}")
            st.write(f"**Status:** {aircraft_data['status']}")
            st.write(f"**Age:** {aircraft_data['aircraft_age_years']:.1f} years")
            st.write(f"**Flight Hours:** {aircraft_data['flight_hours']:,.0f}")
            
        with col2:
            st.markdown("### Risk Assessment")
            risk_color = {
                'CRITICAL': '#DC2626',
                'HIGH': '#F59E0B', 
                'MEDIUM': '#3B82F6',
                'LOW': '#10B981'
            }[aircraft_data['risk_level']]
            
            st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                RISK LEVEL: {aircraft_data['risk_level']}
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Incident Probability:** {aircraft_data['incident_probability']*100:.2f}%")
        
        # Health metrics
        st.markdown("### 📊 Health Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Engine Health", f"{aircraft_data['engine_health']:.1f}%")
        with col2:
            st.metric("Structural Integrity", f"{aircraft_data['structural_integrity']:.1f}%")
        with col3:
            st.metric("Avionics Status", f"{aircraft_data['avionics_status']:.1f}%")
        with col4:
            st.metric("Maintenance Score", f"{aircraft_data['maintenance_score']:.1f}%")
        
        # Risk breakdown chart
        risk_data = {
            'Risk Factor': ['Technical', 'Human', 'Environmental'],
            'Risk Score': [
                aircraft_data['technical_risk'],
                aircraft_data['human_risk'], 
                aircraft_data['environmental_risk']
            ]
        }
        
        fig = px.bar(
            x=risk_data['Risk Factor'],
            y=risk_data['Risk Score'],
            title=f"Risk Breakdown for {selected_tail}",
            color=risk_data['Risk Score'],
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Fleet overview charts
        st.markdown("---")
        st.markdown("## 📈 Fleet Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = filtered_df['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={
                    'CRITICAL': '#DC2626',
                    'HIGH': '#F59E0B',
                    'MEDIUM': '#3B82F6', 
                    'LOW': '#10B981'
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Aircraft type risk analysis
            type_risk = filtered_df.groupby('aircraft_type')['incident_probability'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=type_risk.index,
                y=type_risk.values * 100,
                title="Average Risk by Aircraft Type",
                color=type_risk.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Aircraft list
        st.markdown("### ✈️ Aircraft Monitor")
        
        # Sort by risk level
        display_df = filtered_df.sort_values('incident_probability', ascending=False)[
            ['tail_number', 'airline', 'aircraft_type', 'status', 'risk_level', 'incident_probability', 'delay_minutes']
        ].head(20)
        
        display_df['incident_probability'] = (display_df['incident_probability'] * 100).round(2)
        display_df.columns = ['Tail Number', 'Airline', 'Aircraft Type', 'Status', 'Risk Level', 'Risk %', 'Delay (min)']
        
        st.dataframe(display_df, use_container_width=True)

def main():
    """Main execution function"""
    
    print("🛡️ Starting Guardian Eye Aviation Operations Center...")
    
    # Check if we need to generate data
    if not os.path.exists('aviation_dataset.csv'):
        print("📊 No dataset found. Generating realistic aviation data...")
        df = generate_realistic_aviation_data(5000)
        df.to_csv('aviation_dataset.csv', index=False)
        print("✅ Dataset saved as 'aviation_dataset.csv'")
        
        # Train ML models
        incident_classifier, delay_predictor, feature_columns = train_ml_models(df)
        
        # Save models
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/incident_classifier.pkl', 'wb') as f:
            pickle.dump(incident_classifier, f)
        with open('artifacts/delay_predictor.pkl', 'wb') as f:
            pickle.dump(delay_predictor, f)
        with open('artifacts/feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f)
        
        print("✅ ML models saved to artifacts/")
    
    # Run Streamlit dashboard
    create_guardian_eye_streamlit()

if __name__ == "__main__":
    # For command line execution
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Generate data only
        df = generate_realistic_aviation_data(10000)
        df.to_csv('aviation_dataset.csv', index=False)
        print("✅ Aviation dataset generated successfully!")
        
        # Train and save models
        incident_classifier, delay_predictor, feature_columns = train_ml_models(df)
        os.makedirs('artifacts', exist_ok=True)
        
        with open('artifacts/incident_classifier.pkl', 'wb') as f:
            pickle.dump(incident_classifier, f)
        with open('artifacts/delay_predictor.pkl', 'wb') as f:
            pickle.dump(delay_predictor, f)
        with open('artifacts/feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f)
        
        print("✅ ML models trained and saved!")
    else:
        # Run Streamlit dashboard
        main()

