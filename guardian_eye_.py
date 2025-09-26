# guardian_eye.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========
# Utilities
# ==========
def _seed():
    np.random.seed(42)

def _risk_level(p):
    if p > 0.7:
        return "CRITICAL"
    if p > 0.5:
        return "HIGH"
    if p > 0.3:
        return "MEDIUM"
    return "LOW"

def _dominant_reason(w, t, a):
    parts = []
    if w >= 10: parts.append("Weather disruption")
    if t >= 15: parts.append("Technical inspection")
    if a >= 10: parts.append("ATC congestion")
    return ", ".join(parts) if parts else "On-time operations"

# ===========================
# 1) Generate synthetic data
# ===========================
def generate_realistic_aviation_data(n_flights=3000):
    _seed()
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]
    aircraft_types = [
        "A320", "A321", "A350", "ATR72", "B737", "B787"
    ]
    airports = {
        "DEL": (28.5562, 77.1000),
        "BOM": (19.0896, 72.8656),
        "BLR": (13.1986, 77.7066),
        "MAA": (12.9941, 80.1709),
        "CCU": (22.6547, 88.4467),
        "HYD": (17.2403, 78.4294),
        "COK": (10.1520, 76.4019),
        "AMD": (23.0726, 72.6263),
        "PNQ": (18.5822, 73.9197),
        "JAI": (26.8247, 75.8127),
        "GOI": (15.3800, 73.8310),
        "TRV": (8.4821, 76.9200),
        "IXB": (26.6812, 88.3286),
        "IXC": (30.6735, 76.7885),
        "VNS": (25.4524, 82.8593),
        "SXR": (33.9871, 74.7749),
    }

    rows = []
    start_date = datetime(2024, 1, 1)
    for i in range(n_flights):
        airline = np.random.choice(airlines)
        aircraft = np.random.choice(aircraft_types)
        origin, dest = np.random.choice(list(airports.keys()), 2, replace=False)

        # Schedule
        sched = start_date + timedelta(days=np.random.randint(0, 365), hours=np.random.randint(5, 23),
                                       minutes=np.random.choice([0, 15, 30, 45]))

        # Factors (drivers)
        # Weather a bit stronger in monsoon-ish months
        month = sched.month
        monsoon_boost = 1.4 if month in [6, 7, 8, 9] else 1.0
        winter_fog = 1.25 if month in [12, 1, 2] and origin in ["DEL", "JAI", "IXB", "IXC"] else 1.0

        weather_factor = np.random.poisson(8 * monsoon_boost * winter_fog)
        technical_factor = np.random.poisson(10 if np.random.rand() < 0.25 else 3)
        atc_factor = np.random.poisson(6 if origin in ["DEL", "BOM", "BLR", "MAA", "HYD"] else 3)

        # Delay components
        delay_weather = np.random.poisson(weather_factor // 2)
        delay_tech = np.random.poisson(technical_factor)
        delay_atc = np.random.poisson(atc_factor // 2 + 1)
        total_delay = delay_weather + delay_tech + delay_atc

        status = "DELAYED" if total_delay > 30 else ("IN-FLIGHT" if np.random.rand() > 0.5 else "ON-TIME")

        # Risk proxy from factors (bounded 0..1)
        risk_proxy = np.clip(0.55 * (technical_factor / 40) + 0.3 * (weather_factor / 40) + 0.25 * (atc_factor / 40), 0, 1)
        # airline safety modifier
        airline_safety = {
            "Vistara": 0.9, "IndiGo": 0.92, "Air India": 0.85,
            "AirAsia India": 0.88, "GoFirst": 0.82, "SpiceJet": 0.78
        }[airline]
        incident_probability = float(np.clip(risk_proxy * (1 - (airline_safety - 0.75)), 0, 1))
        risk_level = _risk_level(incident_probability)

        rows.append({
            "flight_id": f"{airline[:2].upper()}{1000 + i}",
            "airline": airline,
            "aircraft_type": aircraft,
            "tail_number": f"VT-{chr(65 + (i % 26))}{chr(65 + ((i+1) % 26))}{chr(65 + ((i+2) % 26))}",
            "origin": origin, "destination": dest,
            "scheduled_departure": sched,
            "actual_departure": sched + timedelta(minutes=int(total_delay)),
            "status": status,
            "delay_minutes": int(total_delay),
            "delay_reason": _dominant_reason(delay_weather, delay_tech, delay_atc),
            "incident_probability": incident_probability,
            "risk_level": risk_level,
            # raw drivers
            "weather_factor": int(weather_factor),
            "technical_factor": int(technical_factor),
            "atc_factor": int(atc_factor),
            # coordinates (spawn around origin)
            "lat": float(airports[origin][0] + np.random.uniform(-1.0, 1.0)),
            "lng": float(airports[origin][1] + np.random.uniform(-1.0, 1.0)),
            # some extra aircraft “health” flavor
            "engine_health": float(np.clip(100 - technical_factor * 2 + np.random.randn() * 6, 0, 100)),
            "structural_integrity": float(np.clip(100 - (technical_factor * 1.2) + np.random.randn() * 6, 0, 100)),
            "avionics_status": float(np.clip(100 - (technical_factor * 0.8) + np.random.randn() * 6, 0, 100)),
            "maintenance_score": float(np.clip(100 - (technical_factor * 1.5) + np.random.randn() * 6, 0, 100)),
            "flight_hours": int(np.random.uniform(2_000, 60_000)),
            "last_maintenance_days": int(np.random.uniform(1, 180)),
        })

    df = pd.DataFrame(rows)
    return df

# ==========================
# 2) Train simple ML models
# ==========================
def train_models(df: pd.DataFrame):
    X = df[["weather_factor", "technical_factor", "atc_factor"]]
    y_incident = (df["risk_level"].isin(["HIGH", "CRITICAL"])).astype(int)
    y_delay = df["delay_minutes"]

    X_train, X_test, y_train_inc, y_test_inc = train_test_split(X, y_incident, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train, y_train_inc)

    reg = RandomForestRegressor(n_estimators=120, random_state=42)
    reg.fit(X_train, y_delay.loc[X_train.index])

    return clf, reg

def explain_delay_reason(row):
    parts = []
    if row["weather_factor"] >= 10: parts.append("Heavy weather impact")
    if row["technical_factor"] >= 12: parts.append("Possible technical/MX hold")
    if row["atc_factor"] >= 8: parts.append("ATC congestion/slot issues")
    return parts if parts else ["No major delay drivers detected"]

def contributions(clf, reg, row):
    # simple normalized “contributions” using feature importances × scaled value
    feats = ["weather_factor", "technical_factor", "atc_factor"]
    vals = np.array([row[f] for f in feats], dtype=float)

    # scale to [0..1] with a soft cap for readability
    scaled = np.clip(vals / (vals.max() + 1e-9), 0, 1)

    clf_imp = getattr(clf, "feature_importances_", np.array([1/3, 1/3, 1/3]))
    reg_imp = getattr(reg, "feature_importances_", np.array([1/3, 1/3, 1/3]))

    inc_contrib = (scaled * clf_imp) / (np.sum(scaled * clf_imp) + 1e-9)
    delay_contrib = (scaled * reg_imp) / (np.sum(scaled * reg_imp) + 1e-9)

    return dict(zip(feats, inc_contrib)), dict(zip(feats, delay_contrib))

# =====================
# 3) Streamlit UI
# =====================
st.set_page_config(page_title="🛡️ Guardian Eye", layout="wide")
st.title("🛡️ Guardian Eye – Real-time Aviation Risk & Delay Intelligence")

st.sidebar.header("⚙️ Controls")
regen = st.sidebar.button("🔄 Regenerate Dataset")

DATA_FILE = "aviation_dataset.csv"
if regen or not os.path.exists(DATA_FILE):
    st.sidebar.info("Generating dataset…")
    df_all = generate_realistic_aviation_data(3000)
    df_all.to_csv(DATA_FILE, index=False)
else:
    df_all = pd.read_csv(DATA_FILE, parse_dates=["scheduled_departure", "actual_departure"])

# Train models (kept in memory so selects are instant)
clf, reg = train_models(df_all)

# Filters
airlines = ["All Airlines"] + sorted(df_all["airline"].unique())
airline_sel = st.sidebar.selectbox("Airline", airlines, index=0)

df = df_all.copy()
if airline_sel != "All Airlines":
    df = df[df["airline"] == airline_sel]

types = ["All Types"] + sorted(df["aircraft_type"].unique())
type_sel = st.sidebar.selectbox("Aircraft Type", types, index=0)
if type_sel != "All Types":
    df = df[df["aircraft_type"] == type_sel]

tails = ["All Aircraft"] + sorted(df["tail_number"].unique())
tail_sel = st.sidebar.selectbox("Tail Number", tails, index=0)
if tail_sel != "All Aircraft":
    df = df[df["tail_number"] == tail_sel]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Flights", len(df))
c2.metric("Delayed Flights", int((df["status"] == "DELAYED").sum()))
c3.metric("Critical Risk", int((df["risk_level"] == "CRITICAL").sum()))
c4.metric("Avg Delay (min)", round(float(df["delay_minutes"].mean() if len(df) else 0), 1))

st.divider()

# ========= 3D Globe =========
st.subheader("🌍 Global Aircraft Tracking (3D-style)")
if len(df):
    # map colored by risk probability
    view_state = pdk.ViewState(latitude=22, longitude=79, zoom=4.3, pitch=40, bearing=0)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lng", "lat"],
        get_radius=45000,
        get_fill_color=[
            "incident_probability * 255",
            "(1 - incident_probability) * 180",
            "80",
            "200"
        ],
        pickable=True,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
    )
    tooltip = {
        "html": "<b>{flight_id}</b><br/>Airline: {airline}<br/>Type: {aircraft_type}<br/>Risk: {risk_level}<br/>Delay: {delay_minutes} min<br/>Reason: {delay_reason}",
        "style": {"backgroundColor": "rgba(30,41,59,0.9)", "color": "white", "fontSize": "12px"}
    }
    deck = pdk.Deck(map_style="mapbox://styles/mapbox/dark-v11", initial_view_state=view_state,
                    layers=[scatter], tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)
else:
    st.info("No flights match your filters.")

st.divider()

# ========= Flight table with ML predictions =========
st.subheader("📋 Flight Monitor + ML Predictions")
view_df = df.copy()
if len(view_df) > 300:
    view_df = view_df.sample(300, random_state=42)

X = view_df[["weather_factor", "technical_factor", "atc_factor"]]
view_df["pred_incident_flag"] = clf.predict(X)
view_df["pred_incident_prob"] = clf.predict_proba(X)[:, 1]
view_df["pred_incident"] = np.where(view_df["pred_incident_flag"] == 1, "HIGH RISK", "LOW RISK")
view_df["pred_delay"] = reg.predict(X).round(1)

# simple reason highlight
reasons = []
for _, r in view_df.iterrows():
    reasons.append(_dominant_reason(r["weather_factor"], r["technical_factor"], r["atc_factor"]))
view_df["predicted_reason"] = reasons

st.dataframe(
    view_df[[
        "flight_id", "airline", "aircraft_type", "tail_number",
        "status", "risk_level", "incident_probability",
        "pred_incident", "pred_incident_prob", "delay_minutes", "pred_delay", "predicted_reason"
    ]].sort_values("pred_incident_prob", ascending=False),
    use_container_width=True
)

st.divider()

# ========= “Click flight” style: select one flight for deep dive =========
st.subheader("🛠️ Selected Flight – Detailed Risk & Delay Explanation")
# (Streamlit can't capture pydeck on-map clicks directly, so we provide a selector synced with the table)
flight_ids = ["-- Select a flight --"] + list(view_df["flight_id"])
pick = st.selectbox("Choose a flight (mirrors the globe markers & table above):", flight_ids, index=0)

if pick != "-- Select a flight --":
    sel = view_df[view_df["flight_id"] == pick].iloc[0]
    st.markdown(f"### ✈️ {sel['flight_id']} • {sel['airline']} • {sel['aircraft_type']} • Tail {sel['tail_number']}")

    # Compute contributions
    inc_contrib, delay_contrib = contributions(clf, reg, sel)

    left, right = st.columns(2)

    with left:
        st.markdown("#### 🔮 Incident Prediction")
        st.metric("Predicted Incident Risk", f"{'HIGH' if sel['pred_incident_flag']==1 else 'LOW'}",
                  delta=f"{sel['pred_incident_prob']*100:.1f}% probability")
        st.write("**Drivers (relative contribution):**")
        st.progress(float(inc_contrib["weather_factor"]))
        st.caption(f"Weather factor: {inc_contrib['weather_factor']*100:.1f}%")
        st.progress(float(inc_contrib["technical_factor"]))
        st.caption(f"Technical factor: {inc_contrib['technical_factor']*100:.1f}%")
        st.progress(float(inc_contrib["atc_factor"]))
        st.caption(f"ATC factor: {inc_contrib['atc_factor']*100:.1f}%")

        st.write("**Reasoning:**")
        for line in explain_delay_reason(sel):
            st.write(f"- {line}")

    with right:
        st.markdown("#### ⏱️ Delay Prediction")
        st.metric("Predicted Delay (min)", f"{sel['pred_delay']:.1f}",
                  delta=f"Actual: {sel['delay_minutes']} min")
        st.write("**Drivers (relative contribution):**")
        st.progress(float(delay_contrib["weather_factor"]))
        st.caption(f"Weather factor: {delay_contrib['weather_factor']*100:.1f}%")
        st.progress(float(delay_contrib["technical_factor"]))
        st.caption(f"Technical factor: {delay_contrib['technical_factor']*100:.1f}%")
        st.progress(float(delay_contrib["atc_factor"]))
        st.caption(f"ATC factor: {delay_contrib['atc_factor']*100:.1f}%")

        st.write("**Operational Context**")
        st.write(f"- Status: **{sel['status']}**")
        st.write(f"- Risk Level: **{sel['risk_level']}** (p={sel['incident_probability']:.2f})")
        st.write(f"- Original Delay Reason: **{sel['delay_reason']}**")
        st.write(f"- Engine Health: **{sel['engine_health']:.1f}%**, Avionics: **{sel['avionics_status']:.1f}%**")
        st.write(f"- Maintenance Score: **{sel['maintenance_score']:.1f}%**, Last MX: **{sel['last_maintenance_days']} days** ago")

st.caption("Tip: Use the Airline / Type / Tail filters to scope the globe & table. The selector above opens a deep-dive panel for that flight.")


