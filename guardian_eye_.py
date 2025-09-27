# guardian_eye_final.py
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
# Data generation
# -----------------------------
def generate_realistic_aviation_data(n_days: int = 30) -> pd.DataFrame:
    """Generate structured aviation dataset"""
    print("🛫 Generating structured aviation dataset ...")

    airlines_routes = {
        "Air India":     [("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL"),("DEL","MAA"),("MAA","DEL")],
        "IndiGo":        [("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR"),("BOM","BLR"),("BLR","BOM")],
        "SpiceJet":      [("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM"),("DEL","PNQ"),("PNQ","DEL")],
        "AirAsia India": [("BLR","DEL"),("DEL","BLR"),("MAA","CCU"),("CCU","MAA"),("BOM","AMD"),("AMD","BOM")]
    }

    flights = []
    flight_counter = 1000

    for airline, routes in airlines_routes.items():
        info = AIRLINES[airline]
        code = info['code']
        for (origin, dest) in routes:
            route_block_minutes = np.random.randint(90, 180)
            for d in range(n_days):
                flight_counter += 1
                sched_date = datetime(2024, 1, 1) + timedelta(days=d)
                dep_hour = np.random.randint(6, 23)
                dep_min = np.random.choice([0, 10, 20, 30, 40, 50])
                scheduled_departure = sched_date.replace(hour=dep_hour, minute=dep_min, second=0, microsecond=0)
                scheduled_arrival = scheduled_departure + timedelta(minutes=route_block_minutes)

                # Balanced delay distribution
                r = np.random.rand()
                if r < 0.70:
                    delay = np.random.randint(-10, 11)
                elif r < 0.90:
                    delay = np.random.randint(15, 46)
                elif r < 0.98:
                    delay = np.random.randint(45, 121)
                else:
                    delay = np.random.randint(-30, -11)

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

                # Delay breakdown
                base_delay = 0
                weather_delay = np.random.poisson(10) if weather_score > 0.7 else (np.random.poisson(5) if weather_score > 0.5 else 0)
                technical_delay = np.random.poisson(25) if (engine_health < 70 or maintenance_score < 60) else (np.random.poisson(10) if engine_health < 85 else 0)
                atc_delay = np.random.poisson(15) if atc_score < 70 else (np.random.poisson(7) if atc_score < 85 else 0)

                if delay > 0:
                    modeled = base_delay + weather_delay + technical_delay + atc_delay
                    if modeled == 0:
                        base_delay = delay
                    else:
                        scale = delay / modeled
                        weather_delay = py_int(weather_delay*scale)
                        technical_delay = py_int(technical_delay*scale)
                        atc_delay = py_int(atc_delay*scale)
                        base_delay = py_int(max(0, delay - (weather_delay+technical_delay+atc_delay)))

                expected_arrival = scheduled_arrival + timedelta(minutes=delay)
                if delay > 15:
                    status = "DELAYED"
                elif delay < -10:
                    status = "EARLY"
                else:
                    status = "ON-TIME"

                frac = py_float(np.random.uniform(0, 1))
                o, d_ = AIRPORTS[origin], AIRPORTS[dest]
                current_lat = py_float(o['lat'] + (d_['lat'] - o['lat'])*frac)
                current_lng = py_float(o['lng'] + (d_['lng'] - o['lng'])*frac)

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
# UI
# -----------------------------
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye - Aviation Operations Center", page_icon="🛡️", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
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

    # Parse dates safely
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

    # Current time display
    current_time = datetime.now()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🕐 Current Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh
    st.sidebar.markdown("*Auto-refresh: 30s*")
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    # Alert level
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

    # Enhanced Main KPIs
    st.markdown("### 📊 Real-Time Operations Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_flights = len(tmp)
        st.metric("🛫 Total Flights", total_flights, delta=None)
    
    with col2:
        if 'status' in tmp.columns:
            on_time = len(tmp[tmp["status"] == "ON-TIME"])
            on_time_pct = (on_time/len(tmp)*100) if len(tmp) > 0 else 0
            delta_color = "normal" if on_time_pct >= 70 else "inverse"
            st.metric("✅ On-Time", on_time, delta=f"{on_time_pct:.1f}%", delta_color=delta_color)
        else:
            st.metric("✅ On-Time", "N/A")
    
    with col3:
        if 'status' in tmp.columns:
            delayed = len(tmp[tmp["status"] == "DELAYED"])
            delayed_pct = (delayed/len(tmp)*100) if len(tmp) > 0 else 0
            delta_color = "inverse" if delayed_pct > 20 else "normal"
            st.metric("⏱️ Delayed", delayed, delta=f"{delayed_pct:.1f}%", delta_color=delta_color)
        else:
            st.metric("⏱️ Delayed", "N/A")
    
    with col4:
        if 'risk_level' in tmp.columns:
            high_crit = len(tmp[tmp["risk_level"].isin(["HIGH","CRITICAL"])])
            risk_pct = (high_crit/len(tmp)*100) if len(tmp) > 0 else 0
            delta_color = "inverse" if high_crit > 0 else "normal"
            st.metric("⚠️ High Risk", high_crit, delta=f"{risk_pct:.1f}%", delta_color=delta_color)
        else:
            st.metric("⚠️ High Risk", "N/A")
    
    with col5:
        if 'delay_minutes' in tmp.columns:
            avg_delay = float(tmp["delay_minutes"].mean()) if len(tmp) > 0 else 0.0
            delta_color = "inverse" if avg_delay > 15 else "normal"
            st.metric("⏱️ Avg Delay", f"{avg_delay:.1f}m", delta_color=delta_color)
        else:
            st.metric("⏱️ Avg Delay", "N/A")

    # Critical alert banner
    if 'risk_level' in tmp.columns:
        critical_flights = tmp[tmp['risk_level'] == 'CRITICAL']
        if len(critical_flights) > 0:
            st.error(f"🚨 CRITICAL ALERT: {len(critical_flights)} flights require immediate attention!")

    st.markdown("---")

    # Enhanced Real-World Globe
    if len(tmp) > 0:
        st.markdown("### 🌍 Guardian Eye - Global Flight Tracking")
        
        # Create globe figure
        globe_fig = go.Figure()

        # Add flight routes
        if all(col in tmp.columns for col in ['origin_lat', 'origin_lng', 'dest_lat', 'dest_lng']):
            risk_color_map = {
                "CRITICAL": "#DC2626",
                "HIGH": "#F59E0B", 
                "MEDIUM": "#3B82F6",
                "LOW": "#10B981"
            }
            
            for _, r in tmp.head(100).iterrows():
                if all(pd.notna([r['origin_lat'], r['origin_lng'], r['dest_lat'], r['dest_lng']])):
                    route_color = risk_color_map.get(r.get('risk_level', 'LOW'), "#6B7280")
                    
                    globe_fig.add_trace(go.Scattergeo(
                        lon=[r["origin_lng"], r["dest_lng"]],
                        lat=[r["origin_lat"], r["dest_lat"]],
                        mode="lines",
                        line=dict(width=3, color=route_color),
                        opacity=0.7,
                        hoverinfo="skip",
                        showlegend=False
                    ))

        # Add aircraft positions
        if all(col in tmp.columns for col in ['current_lat', 'current_lng', 'risk_level']):
            hover_text = []
            colors = []
            sizes = []
            
            for _, r in tmp.iterrows():
                text = f"✈️ <b>{r.get('flight_id', 'Unknown')}</b><br>"
                text += f"🏢 {r.get('airline', 'Unknown')}<br>"
                text += f"📍 <b>{r.get('origin', '')} → {r.get('destination', '')}</b><br>"
                text += f"🛩️ {r.get('aircraft_type', 'Unknown')}<br>"
                text += f"🏷️ {r.get('tail_number', 'Unknown')}<br>"
                text += f"⚠️ Risk Level: <b>{r.get('risk_level', 'Unknown')}</b><br>"
                text += f"⏱️ Delay: <b>{r.get('delay_minutes', 0)} min</b>"
                hover_text.append(text)
                
                color = risk_color_map.get(r.get('risk_level', 'LOW'), "#6B7280")
                colors.append(color)
                
                # Size based on risk level
                if r.get('risk_level') == 'CRITICAL':
                    sizes.append(20)
                elif r.get('risk_level') == 'HIGH':
                    sizes.append(16)
                elif r.get('risk_level') == 'MEDIUM':
                    sizes.append(12)
                else:
                    sizes.append(10)
            
            globe_fig.add_trace(go.Scattergeo(
                lon=tmp["current_lng"],
                lat=tmp["current_lat"],
                text=hover_text,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol="triangle-up",
                    line=dict(width=2, color="white"),
                    opacity=0.9
                ),
                hoverinfo="text",
                hovertemplate="<extra></extra>%{text}",
                name="✈️ Aircraft",
                showlegend=True
            ))

        # Add airports
        airport_data = []
        for code, info in AIRPORTS.items():
            airport_data.append({
                'code': code,
                'name': info['name'],
                'lat': info['lat'],
                'lng': info['lng'],
                'traffic': info['traffic']
            })
        
        airport_df = pd.DataFrame(airport_data)
        traffic_sizes = {'Very High': 25, 'High': 20, 'Medium': 15, 'Low': 10}
        sizes = [traffic_sizes.get(traffic, 15) for traffic in airport_df['traffic']]
        
        hover_text = [
            f"🏢 <b>{row['code']} - {row['name']}</b><br>📊 Traffic Level: <b>{row['traffic']}</b>"
            for _, row in airport_df.iterrows()
        ]
        
        globe_fig.add_trace(go.Scattergeo(
            lon=airport_df['lng'],
            lat=airport_df['lat'],
            text=hover_text,
            mode="markers",
            marker=dict(
                size=sizes,
                color='#FFD700',
                symbol='square',
                line=dict(width=2, color='black'),
                opacity=0.9
            ),
            hoverinfo="text",
            hovertemplate="<extra></extra>%{text}",
            name="🏢 Airports",
            showlegend=True
        ))

        # Configure globe
        globe_fig.update_geos(
            projection_type="orthographic",
            showland=True,
            landcolor="rgb(40, 60, 40)",
            showocean=True,
            oceancolor="rgb(20, 50, 100)",
            showlakes=True,
            lakecolor="rgb(30, 70, 140)",
            showcoastlines=True,
            coastlinecolor="rgb(200, 200, 200)",
            showcountries=True,
            countrycolor="rgb(150, 150, 150)",
            showframe=False,
            bgcolor="rgba(5, 12, 24, 1)",
            center=dict(lat=20, lon=77),
            projection_rotation=dict(lat=20, lon=77, roll=0)
        )
        
        globe_fig.update_layout(
            title=dict(
                text="🛡️ Guardian Eye - Real-Time Global Aviation Operations Center",
                x=0.5,
                font=dict(size=20, color='white', family="Arial Black")
            ),
            paper_bgcolor='rgba(0, 0, 0, 0.9)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='white', size=12),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor="white",
                borderwidth=1
            )
        )

        # Display globe and stats
        col_globe, col_stats = st.columns([3, 1])
        
        with col_globe:
            st.plotly_chart(globe_fig, use_container_width=True)
        
        with col_stats:
            st.markdown("#### 🎯 Operations Control")
            
            # Risk level statistics
            if 'risk_level' in tmp.columns:
                st.markdown("**📊 Risk Distribution:**")
                risk_counts = tmp['risk_level'].value_counts()
                total_flights = len(tmp)
                
                for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    count = risk_counts.get(risk_level, 0)
                    percentage = (count / total_flights * 100) if total_flights > 0 else 0
                    
                    if risk_level == 'CRITICAL':
                        st.markdown(f"🔴 **Critical:** {count} ({percentage:.1f}%)")
                    elif risk_level == 'HIGH':
                        st.markdown(f"🟡 **High:** {count} ({percentage:.1f}%)")
                    elif risk_level == 'MEDIUM':
                        st.markdown(f"🔵 **Medium:** {count} ({percentage:.1f}%)")
                    else:
                        st.markdown(f"🟢 **Low:** {count} ({percentage:.1f}%)")
            
            # Status distribution
            if 'status' in tmp.columns:
                st.markdown("**🔄 Flight Status:**")
                status_counts = tmp['status'].value_counts()
                for status, count in status_counts.items():
                    if status == 'DELAYED':
                        st.markdown(f"⏱️ **Delayed:** {count}")
                    elif status == 'ON-TIME':
                        st.markdown(f"✅ **On-Time:** {count}")
                    else:
                        st.markdown(f"📋 **{status}:** {count}")
            
            # Airline distribution
            if 'airline' in tmp.columns:
                st.markdown("**🏢 By Airline:**")
                airline_counts = tmp['airline'].value_counts()
                for airline, count in airline_counts.items():
                    st.markdown(f"• **{airline}:** {count}")
            
            st.markdown("---")
            st.markdown("**🎮 Globe Controls:**")
            st.markdown("""
            - **Rotate:** Click & drag
            - **Zoom:** Mouse wheel
            - **Pan:** Shift + drag
            - **Reset:** Double-click
            """)
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
    else:
        st.info("🔍 No flights match current filters")

    st.markdown("---")

    # Flight Schedule Table
    st.markdown("### ✈️ Flight Schedule Monitor")
    if len(tmp) > 0:
        preferred_cols = [
            "flight_id", "airline", "aircraft_type", "tail_number",
            "origin", "destination", "scheduled_departure", "scheduled_arrival", 
            "expected_arrival", "delay_minutes", "status", "risk_level"
        ]
        
        available_cols = safe_column_access(tmp, preferred_cols)
        
        if available_cols:
            show_df = tmp[available_cols].copy()
            
            # Format datetime columns
            for col in ["scheduled_departure", "scheduled_arrival", "expected_arrival"]:
                if col in show_df.columns:
                    show_df[col] = pd.to_datetime(show_df[col], errors='coerce')
                    show_df[col] = show_df[col].dt.strftime("%Y-%m-%d %H:%M")
            
            if 'scheduled_departure' in show_df.columns:
                show_df = show_df.sort_values("scheduled_departure")
            
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
            st.markdown("### 📈 Risk Analysis")
            risk_data = {
                'Factor': [col.replace('_risk', '').title() for col in available_risk],
                'Score': [row[col] for col in available_risk]
            }
            
            risk_fig = px.bar(
                x=risk_data['Factor'],
                y=risk_data['Score'],
                title="Risk Factor Breakdown",
                color=risk_data['Score'],
                color_continuous_scale='Reds'
            )
            risk_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(risk_fig, use_container_width=True)

def main():
    """Main execution function"""
    csv_path = "aviation_dataset.csv"
    if not os.path.exists(csv_path):
        print("🛫 No dataset found. Generating realistic aviation data...")
        df = generate_realistic_aviation_data(n_days=30)
        df.to_csv(csv_path, index=False)
        print("✅ Dataset saved!")

    create_guardian_eye_streamlit()

if __name__ == "__main__":
    main()
