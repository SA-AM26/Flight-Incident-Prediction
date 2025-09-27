# guardian_eye.py
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
# Data generation (balanced, flight-wise)
# -----------------------------
def generate_realistic_aviation_data(n_days: int = 30) -> pd.DataFrame:
    """
    Structured generator (flight-wise), with realistic delays and all aircraft/risk fields.
    Produces:
      scheduled_departure, scheduled_arrival, expected_arrival (a.k.a Actual/ETA),
      delay_minutes, status, risk metrics, positions, etc.
    """
    print("🛫 Generating structured aviation dataset ...")

    # Fixed domestic routes per airline (simple, realistic)
    airlines_routes = {
        "Air India":     [("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL")],
        "IndiGo":        [("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR")],
        "SpiceJet":      [("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM")],
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

                # Risk aggregation (same scheme as before)
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
                    "tail_number": rand_tail(),

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

    # Header
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0;">
            <h1 style="margin:0;font-size:40px;color:#3B82F6;">🛡️ GUARDIAN EYE</h1>
            <div style="color:#9CA3AF;">Real-time Flight Safety Monitoring & Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Load or create dataset
    csv_path = "aviation_dataset.csv"
    if not os.path.exists(csv_path):
        with st.spinner("Generating dataset..."):
            df = generate_realistic_aviation_data(n_days=30)
            df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    # Parse dates safely if present
    for col in ["scheduled_departure", "scheduled_arrival", "expected_arrival"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Sidebar filters
    st.sidebar.subheader("🎛️ Filters")
    airlines = ["All"] + sorted(df["airline"].dropna().unique().tolist())
    sel_airline = st.sidebar.selectbox("Airline", airlines)

    tmp = df if sel_airline == "All" else df[df["airline"] == sel_airline]

    types = ["All"] + sorted(tmp["aircraft_type"].dropna().unique().tolist())
    sel_type = st.sidebar.selectbox("Aircraft Type", types)
    if sel_type != "All":
        tmp = tmp[tmp["aircraft_type"] == sel_type]

    origins = ["All"] + sorted(tmp["origin"].dropna().unique().tolist())
    sel_origin = st.sidebar.selectbox("Origin", origins)
    if sel_origin != "All":
        tmp = tmp[tmp["origin"] == sel_origin]

    dests = ["All"] + sorted(tmp["destination"].dropna().unique().tolist())
    sel_dest = st.sidebar.selectbox("Destination", dests)
    if sel_dest != "All":
        tmp = tmp[tmp["destination"] == sel_dest]

    tails = ["All"] + sorted(tmp["tail_number"].dropna().unique().tolist())
    sel_tail = st.sidebar.selectbox("Tail Number", tails)

    if sel_tail != "All":
        tmp = tmp[tmp["tail_number"] == sel_tail]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Flights (filtered)", len(tmp))
    with c2:
        on_time = int((tmp["status"] == "ON-TIME").sum())
        delayed = int((tmp["status"] == "DELAYED").sum())
        st.metric("On-Time / Delayed", f"{on_time} / {delayed}")
    with c3:
        high_crit = int((tmp["risk_level"].isin(["HIGH","CRITICAL"])).sum())
        st.metric("High/Critical Risk", high_crit)
    with c4:
        avg_delay = float(tmp["delay_minutes"].mean()) if len(tmp) else 0.0
        st.metric("Avg Delay (min)", f"{avg_delay:.1f}")

    st.markdown("---")

    # Map (globe-style) of current positions + routes
    if len(tmp):
        fig = go.Figure()

        # routes as lines
        for _, r in tmp.head(200).iterrows():  # limit for performance
            fig.add_trace(go.Scattergeo(
                lon=[r["origin_lng"], r["dest_lng"]],
                lat=[r["origin_lat"], r["dest_lat"]],
                mode="lines",
                line=dict(width=1, color="rgba(100,150,255,0.4)"),
                hoverinfo="skip",
                showlegend=False
            ))

        # aircraft markers (color by risk)
        color_map = {"LOW":"#10B981","MEDIUM":"#3B82F6","HIGH":"#F59E0B","CRITICAL":"#DC2626"}
        fig.add_trace(go.Scattergeo(
            lon=tmp["current_lng"],
            lat=tmp["current_lat"],
            text=tmp.apply(lambda r:
                           f"{r['flight_id']} | {r['airline']}<br>"
                           f"{r['origin']} → {r['destination']}<br>"
                           f"Risk: {r['risk_level']} | Delay: {r['delay_minutes']} min",
                           axis=1),
            mode="markers",
            marker=dict(
                size=6,
                color=tmp["risk_level"].map(color_map),
                line=dict(width=0.5, color="white")
            ),
            hoverinfo="text",
            name="Aircraft"
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
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flights match the current filters.")

    st.markdown("### ✈️ Flights (Schedule vs Expected)")
    if len(tmp):
        view_cols = [
            "flight_id","airline","aircraft_type","tail_number",
            "origin","destination",
            "scheduled_departure","scheduled_arrival","expected_arrival",
            "delay_minutes","status","risk_level","dominant_delay_reason","dominant_delay_minutes"
        ]
        show = tmp[view_cols].sort_values(["scheduled_departure","airline","flight_id"]).head(200).copy()
        show["scheduled_departure"] = show["scheduled_departure"].dt.strftime("%Y-%m-%d %H:%M")
        show["scheduled_arrival"]   = show["scheduled_arrival"].dt.strftime("%Y-%m-%d %H:%M")
        show["expected_arrival"]    = show["expected_arrival"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No rows to display.")

    # If a single tail is selected, show detailed panel
    if sel_tail != "All" and len(tmp):
        st.markdown("---")
        st.subheader(f"Aircraft Detail: {sel_tail}")
        row = tmp.iloc[0]

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Engine Health", f"{py_float(row['engine_health']):.1f}%")
        i2.metric("Structural Integrity", f"{py_float(row['structural_integrity']):.1f}%")
        i3.metric("Avionics Status", f"{py_float(row['avionics_status']):.1f}%")
        i4.metric("Maintenance Score", f"{py_float(row['maintenance_score']):.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            # Risk breakdown bar
            risk_df = pd.DataFrame({
                "Factor": ["Technical","Human","Environmental"],
                "Score": [row["technical_risk"], row["human_risk"], row["environmental_risk"]]
            })
            fig2 = px.bar(risk_df, x="Factor", y="Score", title="Risk Breakdown",
                          color="Score", color_continuous_scale="Reds")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("**Delay Explanation**")
            st.write(f"- Dominant Reason: **{row['dominant_delay_reason']}** "
                     f"({int(row['dominant_delay_minutes'])} min)")
            st.write(f"- Weather: {int(row['weather_delay'])} min, "
                     f"Technical: {int(row['technical_delay'])} min, "
                     f"ATC: {int(row['atc_delay'])} min, Other: {int(row['base_delay'])} min")

# -----------------------------
# Entry
# -----------------------------
def main():
    # Ensure data exists, optionally (re)train models once
    if not os.path.exists("aviation_dataset.csv"):
        df = generate_realistic_aviation_data(n_days=30)
        df.to_csv("aviation_dataset.csv", index=False)
        # Train models (optional)
        try:
            train_ml_models(df)
        except Exception as e:
            print("Model training warning:", e)

    create_guardian_eye_streamlit()

if __name__ == "__main__":
    main()
