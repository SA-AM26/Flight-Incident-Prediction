# guardian_eye.py
# One-file Streamlit app: data generator + ML models + 3D Guardian Eye dashboard

import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="🛡️ Guardian Eye", page_icon="🛡️", layout="wide")
np.random.seed(42)

DATA_CSV = "aviation_dataset.csv"
ARTIFACTS_DIR = "artifacts"
INCIDENT_PKL = os.path.join(ARTIFACTS_DIR, "incident_classifier.pkl")
DELAY_PKL = os.path.join(ARTIFACTS_DIR, "delay_predictor.pkl")
FEATURES_PKL = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

# ---------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------
def generate_realistic_aviation_data(n_flights: int = 10000) -> pd.DataFrame:
    airlines_data = {
        "Air India": {
            "code": "AI",
            "aircraft_types": ["Boeing 787-8", "Boeing 777-300ER", "Airbus A320neo", "Boeing 737-800"],
            "safety_score": 85,
        },
        "IndiGo": {
            "code": "6E",
            "aircraft_types": ["Airbus A320neo", "Airbus A321neo", "ATR 72-600"],
            "safety_score": 92,
        },
        "SpiceJet": {
            "code": "SG",
            "aircraft_types": ["Boeing 737-800", "Boeing 737 MAX 8", "Bombardier Q400"],
            "safety_score": 78,
        },
        "Vistara": {
            "code": "UK",
            "aircraft_types": ["Airbus A320neo", "Airbus A321neo", "Boeing 787-9"],
            "safety_score": 95,
        },
        "GoFirst": {"code": "G8", "aircraft_types": ["Airbus A320neo", "Airbus A321neo"], "safety_score": 82},
        "AirAsia India": {"code": "I5", "aircraft_types": ["Airbus A320neo"], "safety_score": 88},
    }

    airports = {
        "DEL": {"name": "Delhi", "lat": 28.5562, "lng": 77.1000, "traffic": "Very High"},
        "BOM": {"name": "Mumbai", "lat": 19.0896, "lng": 72.8656, "traffic": "Very High"},
        "BLR": {"name": "Bengaluru", "lat": 13.1986, "lng": 77.7066, "traffic": "High"},
        "MAA": {"name": "Chennai", "lat": 12.9941, "lng": 80.1709, "traffic": "High"},
        "CCU": {"name": "Kolkata", "lat": 22.6547, "lng": 88.4467, "traffic": "High"},
        "HYD": {"name": "Hyderabad", "lat": 17.2403, "lng": 78.4294, "traffic": "High"},
        "COK": {"name": "Kochi", "lat": 10.1520, "lng": 76.4019, "traffic": "Medium"},
        "AMD": {"name": "Ahmedabad", "lat": 23.0726, "lng": 72.6263, "traffic": "Medium"},
        "PNQ": {"name": "Pune", "lat": 18.5822, "lng": 73.9197, "traffic": "Medium"},
        "JAI": {"name": "Jaipur", "lat": 26.8247, "lng": 75.8127, "traffic": "Medium"},
        "GOI": {"name": "Goa", "lat": 15.3800, "lng": 73.8314, "traffic": "Medium"},
        "TRV": {"name": "Trivandrum", "lat": 8.4821, "lng": 76.9209, "traffic": "Medium"},
        "IXB": {"name": "Bagdogra", "lat": 26.6812, "lng": 88.3286, "traffic": "Low"},
        "IXC": {"name": "Chandigarh", "lat": 30.6735, "lng": 76.7885, "traffic": "Low"},
        "VNS": {"name": "Varanasi", "lat": 25.4524, "lng": 82.8593, "traffic": "Low"},
        "SXR": {"name": "Srinagar", "lat": 33.9871, "lng": 74.7743, "traffic": "Low"},
    }

    def reg() -> str:
        return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

    flights = []
    traffic_mult = {"Very High": 1.5, "High": 1.2, "Medium": 1.0, "Low": 0.8}

    for i in range(n_flights):
        airline = np.random.choice(list(airlines_data.keys()))
        info = airlines_data[airline]
        a_type = np.random.choice(info["aircraft_types"])
        tail = reg()

        origin = np.random.choice(list(airports.keys()))
        dest_choices = [a for a in airports.keys() if a != origin]
        dest = np.random.choice(dest_choices)

        base_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 366))
        sched_dep = base_date + timedelta(hours=np.random.randint(5, 23), minutes=int(np.random.choice([0, 15, 30, 45])))

        age = np.random.uniform(1, 20)
        hours = np.random.uniform(5_000, 80_000)
        cycles = hours / 1.5
        last_maint_days = np.random.uniform(1, 180)

        monsoon = 1.5 if sched_dep.month in [6, 7, 8, 9] else 1.0
        winter_fog = 1.3 if (sched_dep.month in [12, 1, 2] and origin in ["DEL", "JAI"]) else 1.0
        weather_score = min(1.0, np.random.uniform(0.2, 1.0) * monsoon * winter_fog)

        engine = max(0, 100 - age * 2 - np.random.uniform(0, 20))
        structure = max(0, 100 - age * 1.5 - cycles / 1000 - np.random.uniform(0, 15))
        avionics = max(0, 100 - age * 1 - np.random.uniform(0, 10))
        maint = max(0, 100 - last_maint_days / 2 - np.random.uniform(0, 20))

        pilot_exp_hrs = np.random.uniform(500, 15000)
        pilot_exp = min(100, pilot_exp_hrs / 150)
        crew_rest = np.random.uniform(8, 24)
        crew_fatigue = max(0, min(100, crew_rest * 4))

        atc_prob = (traffic_mult[airports[origin]["traffic"]] + traffic_mult[airports[dest]["traffic"]]) / 2
        atc_score = max(0, 100 - atc_prob * 30 - np.random.uniform(0, 20))

        tech_risk = (100 - engine) * 0.4 + (100 - structure) * 0.3 + (100 - avionics) * 0.2 + (100 - maint) * 0.1
        human_risk = (100 - pilot_exp) * 0.7 + (100 - crew_fatigue) * 0.3
        env_risk = weather_score * 100 * 0.7 + (100 - atc_score) * 0.3

        incident_prob = (tech_risk * 0.5 + human_risk * 0.3 + env_risk * 0.2) / 100
        incident_prob *= (100 - info["safety_score"]) / 100
        incident_prob = float(np.clip(incident_prob, 0, 1))

        if incident_prob > 0.7:
            risk_level = "CRITICAL"
        elif incident_prob > 0.5:
            risk_level = "HIGH"
        elif incident_prob > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        base_delay = np.random.poisson(8)
        weather_delay = np.random.poisson(30) if weather_score > 0.7 else (np.random.poisson(15) if weather_score > 0.5 else 0)
        tech_delay = np.random.poisson(45) if (engine < 70 or maint < 60) else (np.random.poisson(20) if engine < 85 else 0)
        atc_delay = np.random.poisson(25) if atc_score < 70 else (np.random.poisson(10) if atc_score < 85 else 0)
        total_delay = int(base_delay + weather_delay + tech_delay + atc_delay)

        status = "DELAYED" if total_delay > 60 else ("IN-FLIGHT" if np.random.random() > 0.7 else ("COMPLETED" if np.random.random() > 0.5 else "SCHEDULED"))
        actual_dep = sched_dep + timedelta(minutes=total_delay)

        flights.append(
            {
                "flight_id": f'{info["code"]}{1000 + i}',
                "airline": airline,
                "aircraft_code": info["code"],
                "aircraft_type": a_type,
                "tail_number": tail,
                "origin": origin,
                "destination": dest,
                "scheduled_departure": sched_dep,
                "actual_departure": actual_dep,
                "delay_minutes": total_delay,
                "status": status,
                "aircraft_age_years": age,
                "flight_hours": hours,
                "cycles": cycles,
                "last_maintenance_days": last_maint_days,
                "engine_health": engine,
                "structural_integrity": structure,
                "avionics_status": avionics,
                "maintenance_score": maint,
                "pilot_experience": pilot_exp,
                "crew_fatigue_factor": crew_fatigue,
                "weather_score": weather_score,
                "atc_score": atc_score,
                "technical_risk": tech_risk,
                "human_risk": human_risk,
                "environmental_risk": env_risk,
                "incident_probability": incident_prob,
                "risk_level": risk_level,
                "current_lat": airports[origin]["lat"] + (airports[dest]["lat"] - airports[origin]["lat"]) * np.random.random(),
                "current_lng": airports[origin]["lng"] + (airports[dest]["lng"] - airports[origin]["lng"]) * np.random.random(),
            }
        )

    return pd.DataFrame(flights)

# ---------------------------------------------------------------------
# ML training & loading
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "aircraft_age_years",
    "flight_hours",
    "cycles",
    "last_maintenance_days",
    "engine_health",
    "structural_integrity",
    "avionics_status",
    "maintenance_score",
    "pilot_experience",
    "crew_fatigue_factor",
    "weather_score",
    "atc_score",
]

@st.cache_data(show_spinner=False)
def ensure_data_and_models():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_CSV):
        df = generate_realistic_aviation_data(8000)
        df.to_csv(DATA_CSV, index=False)
    else:
        df = pd.read_csv(DATA_CSV, parse_dates=["scheduled_departure", "actual_departure"])

    # --- ensure numeric ---
    X_full = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    # --- adaptive incident threshold ---
    thresholds = [0.5, 0.45, 0.4, 0.35, 0.30]
    y_incident = None
    for t in thresholds:
        y_try = (df["incident_probability"].astype(float) > t).astype(int)
        vc = y_try.value_counts()
        if len(vc.index) == 2 and min(vc.values) >= 0.03 * len(y_try):
            y_incident = y_try
            break
    if y_incident is None:
        q = float(df["incident_probability"].quantile(0.7))
        y_incident = (df["incident_probability"].astype(float) > q).astype(int)

    y_delay = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0).astype(int)

    need_train = not (os.path.exists(INCIDENT_PKL) and os.path.exists(DELAY_PKL) and os.path.exists(FEATURES_PKL))
    if need_train:
        X_train, _, y_inc_train, _ = train_test_split(X_full, y_incident, test_size=0.2, random_state=42, stratify=y_incident)
        _, _, y_d_train, _ = train_test_split(X_full, y_delay, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_inc_train)

        reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        reg.fit(X_train, y_d_train)

        with open(INCIDENT_PKL, "wb") as f:
            pickle.dump(clf, f)
        with open(DELAY_PKL, "wb") as f:
            pickle.dump(reg, f)
        with open(FEATURES_PKL, "wb") as f:
            pickle.dump(FEATURE_COLUMNS, f)

    with open(INCIDENT_PKL, "rb") as f:
        clf = pickle.load(f)
    with open(DELAY_PKL, "rb") as f:
        reg = pickle.load(f)

    return df, clf, reg

# ---------------------------------------------------------------------
# 3D Globe (Plotly)
# ---------------------------------------------------------------------
def globe_with_aircraft(df: pd.DataFrame) -> go.Figure:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    def latlng_to_xyz(lat, lng, r=1.02):
        lat = np.deg2rad(lat)
        lng = np.deg2rad(lng)
        return r * np.cos(lat) * np.cos(lng), r * np.cos(lat) * np.sin(lng), r * np.sin(lat)

    color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "dodgerblue", "LOW": "limegreen"}
    lats, lngs = df["current_lat"].values, df["current_lng"].values
    colors = df["risk_level"].map(color_map).values
    Xp, Yp, Zp = latlng_to_xyz(lats, lngs)

    fig = go.Figure(
        data=[
            go.Surface(x=x, y=y, z=z, opacity=0.25, showscale=False, colorscale="Blues"),
            go.Scatter3d(
                x=Xp, y=Yp, z=Zp,
                mode="markers",
                marker=dict(size=4, color=colors),
                text=df.apply(lambda r: f"{r['tail_number']} • {r['airline']} • {r['aircraft_type']}", axis=1),
                hovertemplate="%{text}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------
def run_predictions(df_row: pd.Series, clf, reg) -> dict:
    X = df_row[FEATURE_COLUMNS].values.reshape(1, -1)
    try:
        proba_row = clf.predict_proba(X)[0]
        if hasattr(clf, "classes_"):
            if len(clf.classes_) == 2:
                idx1 = int(np.where(clf.classes_ == 1)[0][0])
                inc_proba = float(proba_row[idx1])
            elif len(clf.classes_) == 1:
                inc_proba = float(proba_row[0]) if clf.classes_[0] == 1 else 0.0
            else:
                inc_proba = 0.0
        else:
            inc_proba = 0.0
    except Exception:
        inc_proba = 0.0

    delay_pred = float(reg.predict(X)[0])
    return {"incident_probability": inc_proba, "predicted_delay_minutes": delay_pred}

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def app():
    st.markdown("## 🛡️ Guardian Eye — Aviation Operations Center")

    df, clf, reg = ensure_data_and_models()

    # Sidebar filters
    st.sidebar.header("🎛️ Aircraft Selection")
    airlines = ["All Airlines"] + sorted(df["airline"].unique().tolist())
    sel_airline = st.sidebar.selectbox("Airline", airlines)

    sub = df.copy()
    if sel_airline != "All Airlines":
