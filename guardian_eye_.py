# guardian_eye_fixed.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import pickle

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# Helpers
# -----------------------------
def py_int(x) -> int:
    return int(np.asarray(x).item() if np.ndim(x) == 0 else x)

def py_float(x) -> float:
    return float(np.asarray(x).item() if np.ndim(x) == 0 else x)

def rand_tail() -> str:
    # VT-ABC format
    return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

def safe_column_access(df, columns):
    """Safely access columns that exist in the dataframe"""
    available_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}")
    
    return available_cols

# -----------------------------
# Static reference data
# -----------------------------
AIRLINES = {
    'Air India': {
        'code': 'AI',
        'types': ['Boeing 787-8','Boeing 777-300ER','Airbus A320neo','Boeing 737-800'],
        'safety_score': 85
    },
    'IndiGo': {
        'code': '6E',
        'types': ['Airbus A320neo','Airbus A321neo','ATR 72-600'],
        'safety_score': 92
    },
    'SpiceJet': {
        'code': 'SG',
        'types': ['Boeing 737-800','Boeing 737 MAX 8','Bombardier Q400'],
        'safety_score': 78
    },
    'Vistara': {
        'code': 'UK',
        'types': ['Airbus A320neo','Airbus A321neo','Boeing 787-9'],
        'safety_score': 95
    },
    'GoFirst': {
        'code': 'G8',
        'types': ['Airbus A320neo','Airbus A321neo'],
        'safety_score': 82
    },
    'AirAsia India': {
        'code': 'I5',
        'types': ['Airbus A320neo'],
        'safety_score': 88
    }
}

AIRPORTS = {
    'DEL': {'name': 'Delhi',      'lat': 28.5562, 'lng': 77.1000, 'traffic': 'Very High'},
    'BOM': {'name': 'Mumbai',     'lat': 19.0896, 'lng': 72.8656, 'traffic': 'Very High'},
    'BLR': {'name': 'Bangalore',  'lat': 13.1986, 'lng': 77.7066, 'traffic': 'High'},
    'MAA': {'name': 'Chennai',    'lat': 12.9941, 'lng': 80.1709, 'traffic': 'High'},
    'CCU': {'name': 'Kolkata',    'lat': 22.6547, 'lng': 88.4467, 'traffic': 'High'},
    'HYD': {'name': 'Hyderabad',  'lat': 17.2403, 'lng': 78.4294, 'traffic': 'High'},
    'COK': {'name': 'Kochi',      'lat': 10.1520, 'lng': 76.4019, 'traffic': 'Medium'},
    'AMD': {'name': 'Ahmedabad',  'lat': 23.0726, 'lng': 72.6263, 'traffic': 'Medium'},
    'PNQ': {'name': 'Pune',       'lat': 18.5822, 'lng': 73.9197, 'traffic': 'Medium'},
    'JAI': {'name': 'Jaipur',     'lat': 26.8247, 'lng': 75.8127, 'traffic': 'Medium'}
}

TRAFFIC_MULT = {'Very High': 1.5, 'High': 1.2, 'Medium': 1.0, 'Low': 0.8}

# -----------------------------
# Data generation (balanced, flight-wise)
# -----------------------------
def generate_realistic_aviation_data(n_days: int = 30) -> pd.DataFrame:
    """
    Structured generator (flight-wise), with realistic delays and all aircraft/risk fields.
    """
    print("🛫 Generating structured aviation dataset ...")

    # Fixed domestic routes per airline (simple, realistic)
    airlines_routes = {
        "Air India":     [("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL")],
        "IndiGo":        [("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR")],
        "SpiceJet":      [("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM")],
        "Vistara":       [("DEL","BOM"),("BOM","DEL"),("MAA","BLR"),("BLR","MAA")],
        "GoFirst":       [("DEL","PNQ"),("PNQ","DEL"),("BOM","AMD"),("AMD","BOM")],
        "AirAsia India": [("BLR","DEL"),("DEL","BLR"),("MAA","CCU"),("CCU","MAA")]
    }

    flights = []
    flight_counter = 1000

    for airline, routes in airlines_routes.items():
        info = AIRLINES[airline]
        code = info['code']
        for (origin, dest) in routes:
            route_block_minutes = np.random.randint(90, 180)  # 1.5h to 3h
            for d in range(n_days):
                flight_counter += 1
                # schedule windows between 06:00 and 23:00
                sched_date = datetime(2024, 1, 1) + timedelta(days=d)
                dep_hour = np.random.randint(6, 23)
                dep_min = np.random.choice([0, 10, 20, 30, 40, 50])
                scheduled_departure = sched_date.replace(hour=dep_hour, minute=dep_min, second=0, microsecond=0)
                scheduled_arrival = scheduled_departure + timedelta(minutes=route_block_minutes)

                # Balanced delay distribution:
                # ~70% on-time (within +/-10), 20% small delay (15-45), 8% larger delay (45-120), 2% early
                r = np.random.rand()
                if r < 0.70:
                    delay = np.random.randint(-10, 11)  # on time window
                elif r < 0.90:
                    delay = np.random.randint(15, 46)   # small delay
                elif r < 0.98:
                    delay = np.random.randint(45, 121)  # larger delay
                else:
                    delay = np.random.randint(-30, -11) # early

                # Pick aircraft type & tail
                aircraft_type = np.random.choice(info['types'])
                tail_number = rand_tail()

                # Seasonal/weather factors
                monsoon_factor = 1.3 if scheduled_departure.month in [6,7,8,9] else 1.0
                fog_factor = 1.2 if (scheduled_departure.month in [12,1,2] and origin in ['DEL','JAI']) else 1.0
                weather_score = min(1.0, py_float(np.random.uniform(0.2, 1.0) * monsoon_factor * fog_factor))

                # Technical/crew factors
                aircraft_age_years = py_float(np.random.uniform(1, 20))
                flight_hours = py_float(np.random.uniform(5000, 80000))
                cycles = py_float(flight_hours/1.6)
                last_maintenance_days = py_float(np.random.uniform(1, 180))

                engine_health = py_float(max(0, 100 - aircraft_age_years*2 - np.random.uniform(0, 20)))
                structural_integrity = py_float(max(0, 100 - aircraft_age_years*1.3 - cycles/1000 - np.random.uniform(0, 12)))
                avionics_status = py_float(max(0, 100 - aircraft_age_years*0.8 - np.random.uniform(0, 8)))
                maintenance_score = py_float(max(0, 100 - last_maintenance_days/2 - np.random.uniform(0, 15)))

                pilot_exp_hours = py_float(np.random.uniform(800, 12000))
                pilot_experience = py_float(min(100, pilot_exp_hours/120))
                crew_rest_hours = py_float(np.random.uniform(8, 24))
                crew_fatigue_factor = py_float(max(0, min(100, crew_rest_hours * 4)))

                # ATC
                origin_traffic = TRAFFIC_MULT[AIRPORTS[origin]['traffic']]
                dest_traffic = TRAFFIC_MULT[AIRPORTS[dest]['traffic']]
                atc_pressure = (origin_traffic + dest_traffic)/2
                atc_score = py_float(max(0, 100 - atc_pressure*25 - np.random.uniform(0, 15)))

                # Risk aggregation
                technical_risk = (
                    (100 - engine_health)*0.4 +
                    (100 - structural_integrity)*0.3 +
                    (100 - avionics_status)*0.2 +
                    (100 - maintenance_score)*0.1
                )
                human_risk = (
                    (100 - pilot_experience)*0.7 +
                    (100 - crew_fatigue_factor)*0.3
                )
                environmental_risk = (
                    (weather_score*100)*0.7 +
                    (100 - atc_score)*0.3
                )
                incident_probability = (
                    technical_risk*0.5 + human_risk*0.3 + environmental_risk*0.2
                )/100.0
                incident_probability *= (100 - info['safety_score'])/100
                incident_probability = max(0.0, min(1.0, py_float(incident_probability)))
                
                if incident_probability > 0.7:
                    risk_level = 'CRITICAL'
                elif incident_probability > 0.5:
                    risk_level = 'HIGH'
                elif incident_probability > 0.3:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'

                # Delay breakdown model to explain "why"
                base_delay = 0
                # weather
                weather_delay = np.random.poisson(10) if weather_score > 0.7 else (np.random.poisson(5) if weather_score > 0.5 else 0)
                # technical
                technical_delay = np.random.poisson(25) if (engine_health < 70 or maintenance_score < 60) else (np.random.poisson(10) if engine_health < 85 else 0)
                # atc
                atc_delay = np.random.poisson(15) if atc_score < 70 else (np.random.poisson(7) if atc_score < 85 else 0)

                # If the stochastic 'delay' we chose earlier is positive, distribute it across components
                if delay > 0:
                    # total modeled
                    modeled = base_delay + weather_delay + technical_delay + atc_delay
                    if modeled == 0:
                        base_delay = delay
                    else:
                        scale = delay / modeled
                        weather_delay = py_int(weather_delay*scale)
                        technical_delay = py_int(technical_delay*scale)
                        atc_delay = py_int(atc_delay*scale)
                        base_delay = py_int(max(0, delay - (weather_delay+technical_delay+atc_delay)))

                # Status & expected arrival
                expected_arrival = scheduled_arrival + timedelta(minutes=delay)
                if delay > 15:
                    status = "DELAYED"
                elif delay < -10:
                    status = "EARLY"
                else:
                    status = "ON-TIME"

                # Position along great-circle (simple linear)
                frac = py_float(np.random.uniform(0, 1))
                o, d_ = AIRPORTS[origin], AIRPORTS[dest]
                current_lat = py_float(o['lat'] + (d_['lat'] - o['lat'])*frac)
                current_lng = py_float(o['lng'] + (d_['lng'] - o['lng'])*frac)

                # Dominant delay reason for UI
                reasons = {
                    "Weather": weather_delay,
                    "Technical": technical_delay,
                    "ATC": atc_delay,
                    "Other": base_delay
                }
                dominant_reason = max(reasons, key=reasons.get)
                dominant_minutes = reasons[dominant_reason]

                flights.append({
                    "flight_id": f"{code}{flight_counter}",
                    "airline": airline,
                    "airline_code": code,
                    "aircraft_type": aircraft_type,
                    "tail_number": tail_number,

                    "origin": origin,
                    "destination": dest,
                    "origin_name": o['name'],
                    "destination_name": d_['name'],

                    "scheduled_departure": scheduled_departure,
                    "scheduled_arrival": scheduled_arrival,
                    "expected_arrival": expected_arrival,
                    "delay_minutes": int(delay),
                    "status": status,

                    # health & risk
                    "aircraft_age_years": aircraft_age_years,
                    "flight_hours": flight_hours,
                    "cycles": cycles,
                    "last_maintenance_days": last_maintenance_days,

                    "engine_health": engine_health,
                    "structural_integrity": structural_integrity,
                    "avionics_status": avionics_status,
                    "maintenance_score": maintenance_score,

                    "pilot_experience": pilot_experience,
                    "crew_fatigue_factor": crew_fatigue_factor,

                    "weather_score": weather_score,
                    "atc_score": atc_score,

                    "technical_risk": py_float(technical_risk),
                    "human_risk": py_float(human_risk),
                    "environmental_risk": py_float(environmental_risk),
                    "incident_probability": incident_probability,
                    "risk_level": risk_level,

                    "origin_lat": float(o['lat']),
                    "origin_lng": float(o['lng']),
                    "dest_lat": float(d_['lat']),
                    "dest_lng": float(d_['lng']),
                    "current_lat": current_lat,
                    "current_lng": current_lng,
                    "altitude": int(np.random.randint(25000, 42000)) if status != "ON-TIME" and np.random.rand()>0.5 else 0,
                    "speed": int(np.random.randint(400, 550)) if status != "ON-TIME" and np.random.rand()>0.5 else 0,
                    "heading": int(np.random.randint(0, 360)),

                    # delay breakdown
                    "weather_delay": int(weather_delay),
                    "technical_delay": int(technical_delay),
                    "atc_delay": int(atc_delay),
                    "base_delay": int(base_delay),
                    "dominant_delay_reason": dominant_reason,
                    "dominant_delay_minutes": int(dominant_minutes)
                })

    df = pd.DataFrame(flights)
    print(f"✅ Generated {len(df)} records across {len(airlines_routes)} airlines/routes")
    return df

# -----------------------------
# (Optional) Model training
# -----------------------------
def train_ml_models(df: pd.DataFrame):
    """Keep models for future predictions (safe types)."""
    print("🤖 Training ML models...")
    feature_columns = [
        'aircraft_age_years','flight_hours','cycles','last_maintenance_days',
        'engine_health','structural_integrity','avionics_status','maintenance_score',
        'pilot_experience','crew_fatigue_factor','weather_score','atc_score'
    ]
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    y_incident = (df['incident_probability'].astype(float) > 0.5).astype(int)
    y_delay = df['delay_minutes'].astype(int)

    X_train, X_test, y_inc_train, y_inc_test = train_test_split(X, y_incident, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_inc_train)

    _, _, y_delay_train, y_delay_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_delay_train)

    inc_pred = clf.predict(X_test)
    delay_pred = reg.predict(X_test)
    print("📊 Model Performance:")
    print(classification_report(y_inc_test, inc_pred))
    print(f"Delay RMSE: {np.sqrt(mean_squared_error(y_delay_test, delay_pred)):.2f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/incident_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("artifacts/delay_predictor.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open("artifacts/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

# -----------------------------
# UI
# -----------------------------
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye - Aviation Operations Center", page_icon="🛡️", layout="wide")

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
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #3B82F6; font-size: 3rem; margin: 0;">🛡️ GUARDIAN EYE</h1>
        <p style="color: #9CA3AF; font-size: 1.2rem;">Aviation Operations Center</p>
        <p style="color: #6B7280;">Real-time Flight Safety Monitoring & Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Load or create dataset
    csv_path = "aviation_dataset.csv"
    if not os.path.exists(csv_path):
        with st.spinner("🛫 Generating realistic aviation dataset..."):
            df = generate_realistic_aviation_data(n_days=30)
            df.to_csv(csv_path, index=False)
            st.success("✅ Aviation dataset created!")
    else:
        df = pd.read_csv(csv_path)

    # Debug: Show available columns
    with st.expander("🔍 Debug Info"):
        st.write("**Available columns:**", list(df.columns))
        st.write("**Dataset shape:**", df.shape)
        st.write("**Sample data:**")
        st.dataframe(df.head(3))

    # Parse dates safely if present
    date_columns = ["scheduled_departure", "scheduled_arrival", "expected_arrival"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Flight Selection")
    
    # Airline filter
    if 'airline' in df.columns:
        airlines = ["All Airlines"] + sorted(df["airline"].dropna().unique().tolist())
        sel_airline = st.sidebar.selectbox("Select Airline", airlines)
        
        if sel_airline == "All Airlines":
            tmp = df.copy()
        else:
            tmp = df[df["airline"] == sel_airline].copy()
    else:
        st.sidebar.error("No 'airline' column found")
        tmp = df.copy()
        sel_airline = "All Airlines"

    # Aircraft type filter
    if 'aircraft_type' in tmp.columns:
        types = ["All Types"] + sorted(tmp["aircraft_type"].dropna().unique().tolist())
        sel_type = st.sidebar.selectbox("Select Aircraft Type", types)
        if sel_type != "All Types":
            tmp = tmp[tmp["aircraft_type"] == sel_type].copy()
    else:
        sel_type = "All Types"

    # Tail number filter
    if 'tail_number' in tmp.columns:
        tails = ["All Aircraft"] + sorted(tmp["tail_number"].dropna().unique().tolist())
        sel_tail = st.sidebar.selectbox("Select Tail Number", tails)
        if sel_tail != "All Aircraft":
            tmp = tmp[tmp["tail_number"] == sel_tail].copy()
    else:
        sel_tail = "All Aircraft"

    # Route filters
    if 'origin' in tmp.columns:
        origins = ["All Origins"] + sorted(tmp["origin"].dropna().unique().tolist())
        sel_origin = st.sidebar.selectbox("Select Origin", origins)
        if sel_origin != "All Origins":
            tmp = tmp[tmp["origin"] == sel_origin].copy()

    if 'destination' in tmp.columns:
        dests = ["All Destinations"] + sorted(tmp["destination"].dropna().unique().tolist())
        sel_dest = st.sidebar.selectbox("Select Destination", dests)
        if sel_dest != "All Destinations":
            tmp = tmp[tmp["destination"] == sel_dest].copy()

    # Current time display
    current_time = datetime.now()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Alert level based on risk
    if 'risk_level' in tmp.columns:
        critical_count = len(tmp[tmp['risk_level'] == 'CRITICAL'])
        high_count = len(tmp[tmp['risk_level'] == 'HIGH'])
        
        if critical_count > 0:
            alert_level = "CRITICAL"
            alert_color = "#DC2626"
        elif high_count > 2:
            alert_level = "HIGH"
            alert_color = "#F59E0B"
        elif high_count > 0:
            alert_level = "ELEVATED"
            alert_color = "#3B82F6"
        else:
            alert_level = "NORMAL"
            alert_color = "#10B981"
        
        st.sidebar.markdown(f"""
        <div style="background: {alert_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin: 10px 0;">
            ALERT LEVEL: {alert_level}
        </div>
        """, unsafe_allow_html=True)

    # Main KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flights", len(tmp), delta=None)
    
    with col2:
        if 'status' in tmp.columns:
            on_time = len(tmp[tmp["status"] == "ON-TIME"])
            delayed = len(tmp[tmp["status"] == "DELAYED"])
            st.metric("On-Time", on_time, delta=f"{on_time/len(tmp)*100:.1f}%" if len(tmp) > 0 else "0%")
        else:
            st.metric("On-Time", "N/A")
    
    with col3:
        if 'risk_level' in tmp.columns:
            high_crit = len(tmp[tmp["risk_level"].isin(["HIGH","CRITICAL"])])
            st.metric("High/Critical Risk", high_crit, delta="⚠️" if high_crit > 0 else "✅")
        else:
            st.metric("High/Critical Risk", "N/A")
    
    with col4:
        if 'delay_minutes' in tmp.columns:
            avg_delay = float(tmp["delay_minutes"].mean()) if len(tmp) > 0 else 0.0
            st.metric("Avg Delay (min)", f"{avg_delay:.1f}")
        else:
            st.metric("Avg Delay (min)", "N/A")

    st.markdown("---")

    # 3D Globe Map
    if len(tmp) > 0 and all(col in tmp.columns for col in ['current_lat', 'current_lng', 'origin_lat', 'origin_lng', 'dest_lat', 'dest_lng']):
        st.markdown("### 🌍 Global Flight Tracking")
        
        fig = go.Figure()

        # Flight routes as lines
        for _, r in tmp.head(100).iterrows():  # Limit for performance
            try:
                fig.add_trace(go.Scattergeo(
                    lon=[r["origin_lng"], r["dest_lng"]],
                    lat=[r["origin_lat"], r["dest_lat"]],
                    mode="lines",
                    line=dict(width=1, color="rgba(100,150,255,0.4)"),
                    hoverinfo="skip",
                    showlegend=False
                ))
            except:
                continue

        # Aircraft current positions
        if 'risk_level' in tmp.columns:
            color_map = {"LOW":"#10B981","MEDIUM":"#3B82F6","HIGH":"#F59E0B","CRITICAL":"#DC2626"}
            
            hover_text = []
            for _, r in tmp.iterrows():
                text = f"{r.get('flight_id', 'Unknown')} | {r.get('airline', 'Unknown')}<br>"
                text += f"{r.get('origin', 'Unknown')} → {r.get('destination', 'Unknown')}<br>"
                text += f"Risk: {r.get('risk_level', 'Unknown')} | Delay: {r.get('delay_minutes', 0)} min"
                hover_text.append(text)
            
            fig.add_trace(go.Scattergeo(
                lon=tmp["current_lng"],
                lat=tmp["current_lat"],
                text=hover_text,
                mode="markers",
                marker=dict(
                    size=8,
                    color=[color_map.get(risk, "#6B7280") for risk in tmp["risk_level"]],
                    line=dict(width=1, color="white")
                ),
                hoverinfo="text",
                name="Aircraft",
                showlegend=False
            ))

        fig.update_geos(
            projection_type="orthographic",
            showcountries=True,
            showcoastlines=True,
            showland=True,
            landcolor="rgb(20,30,40)",
            oceancolor="rgb(5,12,24)",
            showocean=True,
            lataxis_showgrid=True,
            lonaxis_showgrid=True,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Flight Schedule Table
    st.markdown("### ✈️ Flight Schedule Monitor")
    if len(tmp) > 0:
        # Define preferred columns and what's available
        preferred_cols = [
            "flight_id", "airline", "aircraft_type", "tail_number",
            "origin", "destination", "scheduled_departure", "scheduled_arrival", 
            "expected_arrival", "delay_minutes", "status", "risk_level"
        ]
        
        # Add delay reason columns if available
        if 'dominant_delay_reason' in tmp.columns:
            preferred_cols.extend(["dominant_delay_reason", "dominant_delay_minutes"])
        
        # Get available columns
        available_cols = safe_column_access(tmp, preferred_cols)
        
        if available_cols:
            show_df = tmp[available_cols].copy()
            
            # Format datetime columns
            for col in ["scheduled_departure", "scheduled_arrival", "expected_arrival"]:
                if col in show_df.columns:
                    show_df[col] = pd.to_datetime(show_df[col], errors='coerce')
                    show_df[col] = show_df[col].dt.strftime("%Y-%m-%d %H:%M")
            
            # Sort by departure time if available
            if 'scheduled_departure' in show_df.columns:
                show_df = show_df.sort_values("scheduled_departure")
            
            # Display top 50 flights
            st.dataframe(show_df.head(50), use_container_width=True, hide_index=True)
        else:
            st.error("No suitable columns available for flight display")
    else:
        st.info("No flights match the current filters.")

    # Aircraft-specific detailed panel
    if sel_tail != "All Aircraft" and len(tmp) > 0:
        st.markdown("---")
        st.markdown(f"## 🛩️ Aircraft Detail: {sel_tail}")
        
        row = tmp.iloc[0]
        
        # Aircraft info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Aircraft Information")
            if 'airline' in row:
                st.write(f"**Airline:** {row['airline']}")
            if 'aircraft_type' in row:
                st.write(f"**Type:** {row['aircraft_type']}")
            if 'status' in row:
                st.write(f"**Status:** {row['status']}")
            if 'aircraft_age_years' in row:
                st.write(f"**Age:** {row['aircraft_age_years']:.1f} years")
            if 'flight_hours' in row:
                st.write(f"**Flight Hours:** {row['flight_hours']:,.0f}")
        
        with col2:
            st.markdown("### Risk Assessment")
            if 'risk_level' in row:
                risk_colors = {
                    'CRITICAL': '#DC2626',
                    'HIGH': '#F59E0B', 
                    'MEDIUM': '#3B82F6',
                    'LOW': '#10B981'
                }
                risk_color = risk_colors.get(row['risk_level'], '#6B7280')
                
                st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; margin: 10px 0;">
                    RISK LEVEL: {row['risk_level']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'incident_probability' in row:
                st.write(f"**Incident Probability:** {row['incident_probability']*100:.2f}%")

        # Health metrics
        health_cols = ['engine_health', 'structural_integrity', 'avionics_status', 'maintenance_score']
        available_health = [col for col in health_cols if col in row]
        
        if available_health:
            st.markdown("### 📊 Health Metrics")
            health_row_cols = st.columns(len(available_health))
            
            for i, col in enumerate(available_health):
                with health_row_cols[i]:
                    col_name = col.replace('_', ' ').title()
                    st.metric(col_name, f"{row[col]:.1f}%")

        # Risk breakdown chart
        risk_cols = ['technical_risk', 'human_risk', 'environmental_risk']
        available_risk = [col for col in risk_cols if col in row]
        
        if len(available_risk) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Risk Breakdown")
                risk_data = {
                    'Factor': [col.replace('_risk', '').title() for col in available_risk],
                    'Score': [row[col] for col in available_risk]
                }
                
                fig = px.bar(
                    x=risk_data['Factor'],
                    y=risk_data['Score'],
                    title="Risk Factor Analysis",
                    color=risk_data['Score'],
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### ⏱️ Delay Analysis")
                delay_cols = ['weather_delay', 'technical_delay', 'atc_delay', 'base_delay']
                available_delay = [col for col in delay_cols if col in row]
                
                if available_delay:
                    for col in available_delay:
                        delay_name = col.replace('_delay', '').replace('_', ' ').title()
                        if col == 'base_delay':
                            delay_name = 'Other'
                        st.write(f"**{delay_name}:** {int(row[col])} min")
                    
                    if 'dominant_delay_reason' in row:
                        st.markdown(f"**Primary Cause:** {row['dominant_delay_reason']}")

    # Fleet analytics (when viewing all aircraft)
    if sel_tail == "All Aircraft" and len(tmp) > 10:
        st.markdown("---")
        st.markdown("## 📈 Fleet Analytics")
        
        col1, col2 = st.columns(2)
        
        # Risk distribution
        if 'risk_level' in tmp.columns:
            with col1:
                risk_counts = tmp['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Fleet Risk Distribution",
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
        
        # Airline performance
        if 'airline' in tmp.columns and 'delay_minutes' in tmp.columns:
            with col2:
                airline_perf = tmp.groupby('airline')['delay_minutes'].mean().sort_values(ascending=True)
                fig = px.bar(
                    x=airline_perf.values,
                    y=airline_perf.index,
                    title="Average Delay by Airline",
                    orientation='h',
                    color=airline_perf.values,
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Entry point
# -----------------------------
def main():
    """Main execution function"""
    
    # Ensure data exists
    csv_path = "aviation_dataset.csv"
    if not os.path.exists(csv_path):
        print("🛫 No dataset found. Generating realistic aviation data...")
        df = generate_realistic_aviation_data(n_days=30)
        df.to_csv(csv_path, index=False)
        print("✅ Dataset saved!")
        
        # Train models (optional)
        try:
            train_ml_models(df)
            print("✅ ML models trained and saved!")
        except Exception as e:
            print(f"Model training warning: {e}")

    # Run Streamlit dashboard
    create_guardian_eye_streamlit()

if __name__ == "__main__":
    main()
