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

# Set random seed for reproducibility
np.random.seed(42)

# -------- helpers -------------------------------------------------
def py_int(x) -> int:
    return int(np.asarray(x).item() if np.ndim(x) == 0 else x)

def py_float(x) -> float:
    return float(np.asarray(x).item() if np.ndim(x) == 0 else x)

def rand_tail():
    # VT-ABC format without numpy bytes/decoding shenanigans
    return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

# -------- data gen -----------------------------------------------
def generate_realistic_aviation_data(n_flights=10000):
    """Generate comprehensive realistic aviation dataset (NumPy-safe)."""
    print("🛫 Generating realistic aviation dataset...")

    airlines_data = {
        'Air India': {'code': 'AI','aircraft_types': ['Boeing 787-8','Boeing 777-300ER','Airbus A320neo','Boeing 737-800'],'fleet_size': 120,'safety_score': 85},
        'IndiGo': {'code': '6E','aircraft_types': ['Airbus A320neo','Airbus A321neo','ATR 72-600'],'fleet_size': 280,'safety_score': 92},
        'SpiceJet': {'code': 'SG','aircraft_types': ['Boeing 737-800','Boeing 737 MAX 8','Bombardier Q400'],'fleet_size': 90,'safety_score': 78},
        'Vistara': {'code': 'UK','aircraft_types': ['Airbus A320neo','Airbus A321neo','Boeing 787-9'],'fleet_size': 60,'safety_score': 95},
        'GoFirst': {'code': 'G8','aircraft_types': ['Airbus A320neo','Airbus A321neo'],'fleet_size': 55,'safety_score': 82},
        'AirAsia India': {'code': 'I5','aircraft_types': ['Airbus A320neo'],'fleet_size': 30,'safety_score': 88}
    }

    airports_data = {
        'DEL': {'name': 'Delhi','lat': 28.5562,'lng': 77.1000,'elevation': 777,'traffic_level': 'Very High'},
        'BOM': {'name': 'Mumbai','lat': 19.0896,'lng': 72.8656,'elevation': 11,'traffic_level': 'Very High'},
        'BLR': {'name': 'Bangalore','lat': 13.1986,'lng': 77.7066,'elevation': 3000,'traffic_level': 'High'},
        'MAA': {'name': 'Chennai','lat': 12.9941,'lng': 80.1709,'elevation': 52,'traffic_level': 'High'},
        'CCU': {'name': 'Kolkata','lat': 22.6547,'lng': 88.4467,'elevation': 16,'traffic_level': 'High'},
        'HYD': {'name': 'Hyderabad','lat': 17.2403,'lng': 78.4294,'elevation': 1742,'traffic_level': 'High'},
        'COK': {'name': 'Kochi','lat': 10.1520,'lng': 76.4019,'elevation': 106,'traffic_level': 'Medium'},
        'AMD': {'name': 'Ahmedabad','lat': 23.0726,'lng': 72.6263,'elevation': 189,'traffic_level': 'Medium'},
        'PNQ': {'name': 'Pune','lat': 18.5822,'lng': 73.9197,'elevation': 1942,'traffic_level': 'Medium'},
        'JAI': {'name': 'Jaipur','lat': 26.8247,'lng': 75.8127,'elevation': 1263,'traffic_level': 'Medium'}
    }

    flights_data = []

    for i in range(n_flights):
        airline_name = np.random.choice(list(airlines_data.keys()))
        airline_info = airlines_data[airline_name]
        aircraft_type = np.random.choice(airline_info['aircraft_types'])
        tail_number = rand_tail()

        origin = np.random.choice(list(airports_data.keys()))
        destination = np.random.choice([apt for apt in airports_data.keys() if apt != origin])

        base_date = datetime(2024, 1, 1) + timedelta(days=py_int(np.random.randint(0, 365)))
        scheduled_departure = base_date + timedelta(
            hours=py_int(np.random.randint(6, 23)),
            minutes=py_int(np.random.choice([0, 15, 30, 45]))
        )

        aircraft_age_years = py_float(np.random.uniform(1, 20))
        flight_hours = py_float(np.random.uniform(5000, 80000))
        cycles = py_float(flight_hours / 1.5)
        last_maintenance_days = py_float(np.random.uniform(1, 180))

        month = scheduled_departure.month
        monsoon_factor = 1.5 if month in [6, 7, 8, 9] else 1.0
        winter_fog_factor = 1.3 if month in [12, 1, 2] and origin in ['DEL', 'JAI'] else 1.0
        weather_score = py_float(np.random.uniform(0.2, 1.0) * monsoon_factor * winter_fog_factor)
        weather_score = min(weather_score, 1.0)

        engine_health = py_float(max(0, 100 - aircraft_age_years * 2 - np.random.uniform(0, 20)))
        structural_integrity = py_float(max(0, 100 - aircraft_age_years * 1.5 - cycles/1000 - np.random.uniform(0, 15)))
        avionics_status = py_float(max(0, 100 - aircraft_age_years * 1 - np.random.uniform(0, 10)))
        maintenance_score = py_float(max(0, 100 - last_maintenance_days/2 - np.random.uniform(0, 20)))

        pilot_experience_hours = py_float(np.random.uniform(500, 15000))
        pilot_experience = py_float(min(100, pilot_experience_hours / 150))
        crew_rest_hours = py_float(np.random.uniform(8, 24))
        crew_fatigue_factor = py_float(max(0, min(100, crew_rest_hours * 4)))

        traffic_multiplier = {'Very High': 1.5, 'High': 1.2, 'Medium': 1.0, 'Low': 0.8}
        origin_traffic = traffic_multiplier[airports_data[origin]['traffic_level']]
        dest_traffic = traffic_multiplier[airports_data[destination]['traffic_level']]
        atc_delay_probability = (origin_traffic + dest_traffic) / 2
        atc_score = py_float(max(0, 100 - atc_delay_probability * 30 - np.random.uniform(0, 20)))

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
        incident_probability = (
            technical_risk * 0.5 + human_risk * 0.3 + environmental_risk * 0.2
        ) / 100
        incident_probability *= (100 - airline_info['safety_score']) / 100
        incident_probability = max(0.0, min(1.0, py_float(incident_probability)))

        if incident_probability > 0.7:
            risk_level = 'CRITICAL'
        elif incident_probability > 0.5:
            risk_level = 'HIGH'
        elif incident_probability > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        base_delay = py_int(np.random.poisson(8))
        weather_delay = py_int(np.random.poisson(30)) if weather_score > 0.7 else (
                        py_int(np.random.poisson(15)) if weather_score > 0.5 else 0)
        technical_delay = py_int(np.random.poisson(45)) if (engine_health < 70 or maintenance_score < 60) else (
                          py_int(np.random.poisson(20)) if engine_health < 85 else 0)
        atc_delay = py_int(np.random.poisson(25)) if atc_score < 70 else (
                    py_int(np.random.poisson(10)) if atc_score < 85 else 0)
        total_delay = py_int(base_delay + weather_delay + technical_delay + atc_delay)

        status = 'DELAYED' if total_delay > 60 else (
            'IN-FLIGHT' if np.random.random() > 0.7 else ('COMPLETED' if np.random.random() > 0.5 else 'SCHEDULED')
        )
        actual_departure = scheduled_departure + timedelta(minutes=py_int(total_delay))

        # current position (keep simple linear interp; cast to float)
        frac = py_float(np.random.random())
        o = airports_data[origin]; d = airports_data[destination]
        current_lat = py_float(o['lat'] + (d['lat'] - o['lat']) * frac)
        current_lng = py_float(o['lng'] + (d['lng'] - o['lng']) * frac)

        flight_data = {
            'flight_id': f'{airline_info["code"]}{1000 + i}',
            'airline': airline_name,
            'airline_code': airline_info['code'],
            'aircraft_type': aircraft_type,
            'tail_number': tail_number,
            'origin': origin,
            'destination': destination,
            'origin_name': o['name'],
            'destination_name': d['name'],

            'scheduled_departure': scheduled_departure,
            'actual_departure': actual_departure,
            'delay_minutes': py_int(total_delay),
            'status': status,

            'aircraft_age_years': py_float(aircraft_age_years),
            'flight_hours': py_float(flight_hours),
            'cycles': py_float(cycles),
            'last_maintenance_days': py_float(last_maintenance_days),

            'engine_health': engine_health,
            'structural_integrity': structural_integrity,
            'avionics_status': avionics_status,
            'maintenance_score': maintenance_score,

            'pilot_experience': pilot_experience,
            'crew_fatigue_factor': crew_fatigue_factor,

            'weather_score': weather_score,
            'atc_score': atc_score,

            'technical_risk': py_float(technical_risk),
            'human_risk': py_float(human_risk),
            'environmental_risk': py_float(environmental_risk),
            'incident_probability': py_float(incident_probability),
            'risk_level': risk_level,

            'origin_lat': py_float(o['lat']),
            'origin_lng': py_float(o['lng']),
            'dest_lat': py_float(d['lat']),
            'dest_lng': py_float(d['lng']),
            'current_lat': current_lat,
            'current_lng': current_lng,
            'altitude': py_int(np.random.randint(25000, 42000)) if status == 'IN-FLIGHT' else 0,
            'speed': py_int(np.random.randint(400, 550)) if status == 'IN-FLIGHT' else 0,
            'heading': py_int(np.random.randint(0, 360)),

            'weather_delay': py_int(weather_delay),
            'technical_delay': py_int(technical_delay),
            'atc_delay': py_int(atc_delay),
            'base_delay': py_int(base_delay),
        }

        flights_data.append(flight_data)

    df = pd.DataFrame(flights_data)
    print(f"✅ Generated {len(df)} flight records")
    return df

# -------- ML ------------------------------------------------------
def train_ml_models(df):
    """Train machine learning models for predictions (robust types)."""
    print("🤖 Training ML models...")

    feature_columns = [
        'aircraft_age_years','flight_hours','cycles','last_maintenance_days',
        'engine_health','structural_integrity','avionics_status','maintenance_score',
        'pilot_experience','crew_fatigue_factor','weather_score','atc_score'
    ]

    # ensure numeric dtype
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    y_incident = (df['incident_probability'].astype(float) > 0.5).astype(int)
    y_delay = df['delay_minutes'].astype(int)

    X_train, X_test, y_inc_train, y_inc_test = train_test_split(X, y_incident, test_size=0.2, random_state=42)
    incident_classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    incident_classifier.fit(X_train, y_inc_train)

    _, _, y_delay_train, y_delay_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    delay_predictor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    delay_predictor.fit(X_train, y_delay_train)

    inc_pred = incident_classifier.predict(X_test)
    delay_pred = delay_predictor.predict(X_test)

    print("📊 Model Performance:")
    print("Incident Classifier:")
    print(classification_report(y_inc_test, inc_pred))
    print(f"Delay Predictor RMSE: {np.sqrt(mean_squared_error(y_delay_test, delay_pred)):.2f}")

    return incident_classifier, delay_predictor, feature_columns

# -------- UI ------------------------------------------------------
def create_guardian_eye_streamlit():
    """Create Guardian Eye Streamlit dashboard"""
    st.set_page_config(page_title="Guardian Eye - Aviation Operations Center", page_icon="🛡️", layout="wide")

    st.markdown("""
    <style>
    .main{background:linear-gradient(135deg,#0f1419 0%,#1a2332 50%,#0f1419 100%)}
    .stApp{background:linear-gradient(135deg,#0f1419 0%,#1a2332 50%,#0f1419 100%)}
    .metric-card{background:rgba(0,0,0,.4);border:1px solid rgba(59,130,246,.3);border-radius:10px;padding:20px;backdrop-filter:blur(10px)}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:20px;">
            <h1 style="color:#3B82F6;font-size:3rem;margin:0;">🛡️ GUARDIAN EYE</h1>
            <p style="color:#9CA3AF;font-size:1.2rem;">Aviation Operations Center</p>
            <p style="color:#6B7280;">Real-time Flight Safety Monitoring & Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)

    # Load or create dataset; ensure datetime parsed
    if not os.path.exists('aviation_dataset.csv'):
        with st.spinner("🛫 Generating aviation dataset..."):
            df = generate_realistic_aviation_data(5000)
            # save with ISO format to keep times clean
            df.to_csv('aviation_dataset.csv', index=False)
            st.success("✅ Aviation dataset created!")
    else:
        df = pd.read_csv(
            'aviation_dataset.csv',
            parse_dates=['scheduled_departure','actual_departure'],
            infer_datetime_format=True
        )

    st.sidebar.markdown("## 🎛️ Flight Selection")

    airlines = ['All Airlines'] + sorted(df['airline'].dropna().unique().tolist())
    selected_airline = st.sidebar.selectbox("Select Airline", airlines)

    filtered_df = df if selected_airline == 'All Airlines' else df[df['airline'] == selected_airline]

    aircraft_types = ['All Types'] + sorted(filtered_df['aircraft_type'].dropna().unique().tolist())
    selected_aircraft_type = st.sidebar.selectbox("Select Aircraft Type", aircraft_types)
    if selected_aircraft_type != 'All Types':
        filtered_df = filtered_df[filtered_df['aircraft_type'] == selected_aircraft_type]

    tail_numbers = ['All Aircraft'] + sorted(filtered_df['tail_number'].dropna().unique().tolist())
    selected_tail = st.sidebar.selectbox("Select Tail Number", tail_numbers)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Aircraft", int(len(filtered_df)))
    with col2:
        in_flight = int((filtered_df['status'] == 'IN-FLIGHT').sum())
        denom = max(1, len(filtered_df))
        st.metric("In Flight", in_flight, delta=f"{in_flight/denom*100:.1f}%")
    with col3:
        critical_risk = int((filtered_df['risk_level'] == 'CRITICAL').sum())
        st.metric("Critical Risk", critical_risk, delta="⚠️" if critical_risk > 0 else "✅")
    with col4:
        avg_risk = float(filtered_df['incident_probability'].astype(float).mean() * 100) if len(filtered_df) else 0.0
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%", delta=None)

    if selected_tail != 'All Aircraft' and (filtered_df['tail_number'] == selected_tail).any():
        aircraft_data = filtered_df[filtered_df['tail_number'] == selected_tail].iloc[0]

        st.markdown("---")
        st.markdown(f"## 🛩️ Aircraft Analysis: {selected_tail}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Aircraft Information")
            st.write(f"**Airline:** {aircraft_data['airline']}")
            st.write(f"**Type:** {aircraft_data['aircraft_type']}")
            st.write(f"**Status:** {aircraft_data['status']}")
            st.write(f"**Age:** {py_float(aircraft_data['aircraft_age_years']):.1f} years")
            st.write(f"**Flight Hours:** {py_float(aircraft_data['flight_hours']):,.0f}")

        with c2:
            st.markdown("### Risk Assessment")
            risk_color_map = {'CRITICAL':'#DC2626','HIGH':'#F59E0B','MEDIUM':'#3B82F6','LOW':'#10B981'}
            risk_color = risk_color_map.get(str(aircraft_data['risk_level']), '#10B981')
            st.markdown(
                f"<div style='background:{risk_color};color:white;padding:10px;border-radius:5px;text-align:center;font-weight:bold;'>"
                f"RISK LEVEL: {aircraft_data['risk_level']}</div>",
                unsafe_allow_html=True
            )
            st.write(f"**Incident Probability:** {py_float(aircraft_data['incident_probability'])*100:.2f}%")

        st.markdown("### 📊 Health Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Engine Health", f"{py_float(aircraft_data['engine_health']):.1f}%")
        with m2: st.metric("Structural Integrity", f"{py_float(aircraft_data['structural_integrity']):.1f}%")
        with m3: st.metric("Avionics Status", f"{py_float(aircraft_data['avionics_status']):.1f}%")
        with m4: st.metric("Maintenance Score", f"{py_float(aircraft_data['maintenance_score']):.1f}%")

        risk_data = {
            'Risk Factor': ['Technical','Human','Environmental'],
            'Risk Score': [
                py_float(aircraft_data['technical_risk']),
                py_float(aircraft_data['human_risk']),
                py_float(aircraft_data['environmental_risk'])
            ]
        }
        fig = px.bar(x=risk_data['Risk Factor'], y=risk_data['Risk Score'],
                     title=f"Risk Breakdown for {selected_tail}", color=risk_data['Risk Score'],
                     color_continuous_scale='Reds')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("---")
        st.markdown("## 📈 Fleet Analytics")

        c1, c2 = st.columns(2)
        with c1:
            risk_counts = filtered_df['risk_level'].value_counts()
            if not risk_counts.empty:
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Level Distribution",
                             color_discrete_map={'CRITICAL':'#DC2626','HIGH':'#F59E0B','MEDIUM':'#3B82F6','LOW':'#10B981'})
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data to chart yet.")

        with c2:
            if len(filtered_df):
                type_risk = filtered_df.groupby('aircraft_type')['incident_probability'].mean().sort_values(ascending=False)
                fig = px.bar(x=type_risk.index, y=(type_risk.values * 100), title="Average Risk by Aircraft Type",
                             color=type_risk.values, color_continuous_scale='Reds')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ✈️ Aircraft Monitor")
        if len(filtered_df):
            display_df = filtered_df.sort_values('incident_probability', ascending=False)[
                ['tail_number','airline','aircraft_type','status','risk_level','incident_probability','delay_minutes']
            ].head(20).copy()
            display_df['incident_probability'] = (display_df['incident_probability'].astype(float) * 100).round(2)
            display_df.columns = ['Tail Number','Airline','Aircraft Type','Status','Risk Level','Risk %','Delay (min)']
            st.dataframe(display_df, use_container_width=True)

# -------- entry ---------------------------------------------------
def main():
    print("🛡️ Starting Guardian Eye Aviation Operations Center...")

    if not os.path.exists('aviation_dataset.csv'):
        print("📊 No dataset found. Generating realistic aviation data...")
        df = generate_realistic_aviation_data(5000)
        df.to_csv('aviation_dataset.csv', index=False)
        print("✅ Dataset saved as 'aviation_dataset.csv'")

        incident_classifier, delay_predictor, feature_columns = train_ml_models(df)
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/incident_classifier.pkl','wb') as f: pickle.dump(incident_classifier,f)
