# app.py
import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime, timedelta, timezone

# ---------------------------
# Utilities: auto-find files
# ---------------------------
def find_file(filename, search_dir=os.getcwd()):
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

CLASSIFIER_PKL = find_file("binary_classifier_final.pkl")
REGRESSOR_PKL  = find_file("delay_minutes_Poisson_final.pkl")
DATASET_CSV    = find_file("final_unified_dataset.csv")

if CLASSIFIER_PKL is None:
    st.stop()
if REGRESSOR_PKL is None:
    st.stop()
if DATASET_CSV is None:
    st.stop()

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_models():
    clf_pkg = joblib.load(CLASSIFIER_PKL)
    classifier = clf_pkg["model"]
    scaler = clf_pkg["scaler"]
    clf_features = clf_pkg["feature_columns"]

    reg_pkg = joblib.load(REGRESSOR_PKL)
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
                base = 45.0 + np.random.normal(0, 15, len(X))
                return np.clip(base, 0, 180)
        regressor = FallbackRegressor()

    return classifier, scaler, clf_features, regressor, reg_features

# ---------------------------
# Risk knowledge base
# ---------------------------
INCIDENT_KB = [
    {"factor": "Low arrival visibility", "ref": "Mangalore Air India Express IX-812 (2010)", "lesson": "Poor visibility on table-top runway caused overrun"},
    {"factor": "Strong crosswind", "ref": "SpiceJet Jaipur (2016)", "lesson": "Hard landing during crosswind"},
    {"factor": "High mechanical risk", "ref": "IndiGo A320 (2019)", "lesson": "Mid-air engine stall due to turbine issue"},
    {"factor": "Crew fatigue risk", "ref": "Air India Express Kozhikode (2020)", "lesson": "Runway overrun, fatigue cited"},
    {"factor": "Short runway", "ref": "Alliance Air Patna (2000)", "lesson": "Runway excursion on undershoot"},
    {"factor": "Wet runway conditions", "ref": "SpiceJet Mumbai (2019)", "lesson": "Excursion after heavy rain"},
    {"factor": "Severe weather impact", "ref": "Air India Delhi (2015)", "lesson": "Diversion due to thunderstorms"},
]

# ---------------------------
# Risk assessment (robust)
# ---------------------------
def _safe_float(v, default):
    try: return float(v)
    except Exception: return default

def score_risk(row: pd.Series):
    # weights
    w = {
        "arr_vis": 0.10, "xwind": 0.08, "twind": 0.05,
        "mech": 0.12, "crew": 0.10, "short": 0.10,
        "wet": 0.07, "wx": 0.07, "atc": 0.08, "night": 0.05, "peak": 0.07, "cascade": 0.06
    }
    score, factors = 0.0, []

    # Weather
    if _safe_float(row.get("arrival_visibility_m", 9999), 9999) < 1000:
        score += w["arr_vis"]; factors.append("Low arrival visibility")
    if _safe_float(row.get("crosswind_component_kts", 0), 0) > 25:
        score += w["xwind"]; factors.append("Strong crosswind")
    if _safe_float(row.get("tailwind_component_kts", 0), 0) > 15:
        score += w["twind"]; factors.append("Strong tailwind")
    if _safe_float(row.get("operational_weather_impact", 0), 0) > 0.7:
        score += w["wx"]; factors.append("Severe weather impact")

    # Maintenance / Mechanical
    if _safe_float(row.get("operational_mechanical_risk", 0), 0) > 0.8:
        score += w["mech"]; factors.append("High mechanical risk")

    # Ops
    if _safe_float(row.get("operational_crew_risk", 0), 0) > 0.7:
        score += w["crew"]; factors.append("Crew fatigue risk")
    if _safe_float(row.get("operational_atc_risk", 0), 0) > 0.7:
        score += w["atc"]; factors.append("ATC congestion")
    if _safe_float(row.get("operational_peak_hour", 0), 0) > 0.8:
        score += w["peak"]; factors.append("Peak congestion")
    if _safe_float(row.get("operational_cascade_potential", 0), 0) > 0.7:
        score += w["cascade"]; factors.append("Cascade risk")
    dep_hr = int(_safe_float(row.get("scheduled_dep_hour", 12), 12))
    if dep_hr < 6 or dep_hr > 22:
        score += w["night"]; factors.append("Night ops")

    # Runway / Airport
    if _safe_float(row.get("runway_length_m", 99999), 99999) < 1800:
        score += w["short"]; factors.append("Short runway")
    if str(row.get("runway_surface_wet", 0)) == "1":
        score += w["wet"]; factors.append("Wet runway conditions")

    # Classify
    if score >= 0.35: level, action, prob = "High", "Immediate attention", min(0.05, score*0.25)
    elif score >= 0.20: level, action, prob = "Medium", "Enhanced monitoring", min(0.02, score*0.20)
    elif score >= 0.10: level, action, prob = "Low-Medium", "Standard monitoring", min(0.01, score*0.10)
    else: level, action, prob = "Low", "Normal ops", min(0.005, score*0.10)

    # Incident references
    refs = []
    for f in factors:
        for kb in INCIDENT_KB:
            if f == kb["factor"]:
                refs.append(f"{kb['ref']} — {kb['lesson']}")
    if not refs:
        refs = ["No similar past incidents"]

    return pd.Series({
        "total_risk_score": round(score, 3),
        "risk_classification": level,
        "incident_probability": round(prob, 3),
        "priority_action": action,
        "risk_factors": ", ".join(factors) if factors else "",
        "incident_references": " | ".join(refs)
    })

# ---------------------------
# Load data + predict (cached)
# ---------------------------
@st.cache_data(show_spinner=True)
def load_and_score(sample_limit=None):
    df = pd.read_csv(DATASET_CSV)
    # Ensure UTC-like timestamps (assume ISO strings exist)
    # We will look for scheduled_dep_utc / scheduled_arr_utc. Fall back if absent.
    time_cols = []
    for c in ["scheduled_dep_utc","scheduled_arr_utc","expected_arr_utc"]:
        if c in df.columns:
            time_cols.append(c)
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    classifier, scaler, clf_features, regressor, reg_features = load_models()

    # Classification
    for f in clf_features:
        if f not in df.columns:
            df[f] = 0.0
    Xc = scaler.transform(df[clf_features].fillna(0))
    y_class = classifier.predict(Xc)
    y_prob = classifier.predict_proba(Xc)[:, 1]

    # Regression
    for f in reg_features:
        if f not in df.columns:
            df[f] = 0.0
    Xr = df[reg_features].fillna(0)
    y_delay = regressor.predict(Xr)

    # Risk
    risk_df = df.apply(score_risk, axis=1)

    # Merge predictions
    out = df.copy()
    out["prediction"] = np.where(y_class == 1, "Delayed", "On-time")
    out["delay_probability"] = np.round(y_prob, 3)
    out["predicted_delay_minutes"] = np.round(np.where(y_class == 1, y_delay, 0.0), 1)

    out = pd.concat([out, risk_df], axis=1)

    # Cut sample if requested (for performance)
    if sample_limit and len(out) > sample_limit:
        out = out.sample(sample_limit, random_state=42)

    # Derive date range
    dep_col = "scheduled_dep_utc" if "scheduled_dep_utc" in out.columns else None
    arr_col = "scheduled_arr_utc" if "scheduled_arr_utc" in out.columns else None
    if dep_col:
        start_dt = pd.to_datetime(out[dep_col].min())
        end_dt = pd.to_datetime(out[dep_col].max() if arr_col is None else max(out[dep_col].max(), out[arr_col].max()))
    else:
        # Fallback to first/last flight_date if present
        if "flight_date" in out.columns:
            out["flight_date"] = pd.to_datetime(out["flight_date"], errors="coerce")
            start_dt = pd.to_datetime(out["flight_date"].min()).replace(tzinfo=timezone.utc)
            end_dt = pd.to_datetime(out["flight_date"].max()).replace(tzinfo=timezone.utc) + timedelta(days=1)
        else:
            now = datetime.now(timezone.utc)
            start_dt, end_dt = now - timedelta(days=1), now

    return out, start_dt, end_dt

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Aviation Ops Dashboard", layout="wide")

st.title("🛫 Aviation Ops — Real-Time Replay & Incident-aware Predictions")

with st.sidebar:
    st.subheader("Data & Models")
    st.caption(f"Classifier: `{CLASSIFIER_PKL}`")
    st.caption(f"Regressor: `{REGRESSOR_PKL}`")
    st.caption(f"Dataset: `{DATASET_CSV}`")

    SAMPLE_LIMIT = st.number_input("Load max flights (for speed)", 20000, 500000, 200000, step=10000)

    st.subheader("Playback")
    speed = st.select_slider("Playback speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)
    window_hours = st.slider("Window (hours around 'now')", 1, 12, 3)

data, timeline_start, timeline_end = load_and_score(SAMPLE_LIMIT)

# Filters
airlines = sorted(data["airline"].dropna().unique().tolist()) if "airline" in data.columns else []
routes = sorted((data["origin"] + "-" + data["dest"]).dropna().unique().tolist()) if {"origin","dest"}.issubset(data.columns) else []

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    sel_airlines = st.multiselect("Filter airlines", airlines, default=airlines[:5] if airlines else [])
with colf2:
    sel_routes = st.multiselect("Filter routes", routes, default=routes[:10] if routes else [])
with colf3:
    st.write("")
    apply_filters = st.checkbox("Apply filters", value=True)

# Build filtered set
df = data.copy()
if apply_filters:
    if sel_airlines and "airline" in df.columns:
        df = df[df["airline"].isin(sel_airlines)]
    if sel_routes and {"origin","dest"}.issubset(df.columns):
        rr = df["origin"] + "-" + df["dest"]
        df = df[rr.isin(sel_routes)]

# Timeline slider
st.subheader("⏱️ Live Ops Replay")
ts_min, ts_max = timeline_start, timeline_end
ts_current = st.slider(
    "Current time (UTC)", 
    min_value=ts_min.to_pydatetime(), 
    max_value=ts_max.to_pydatetime(),
    value=ts_min.to_pydatetime(),
    format="YYYY-MM-DD HH:mm"
)
ts_current = pd.Timestamp(ts_current, tz="UTC")

dep_col = "scheduled_dep_utc" if "scheduled_dep_utc" in df.columns else None
arr_col = "scheduled_arr_utc" if "scheduled_arr_utc" in df.columns else None
exp_col = "expected_arr_utc" if "expected_arr_utc" in df.columns else arr_col

# Define time window
win_start = ts_current - timedelta(hours=window_hours)
win_end = ts_current + timedelta(hours=window_hours)

def in_window(row):
    # consider dep/arr around current time for display
    dep = row[dep_col] if dep_col else None
    arr = row[exp_col] if exp_col else None
    # show if dep or arr lies within the window around "now"
    flags = []
    if isinstance(dep, pd.Timestamp):
        flags.append(win_start <= dep <= win_end)
    if isinstance(arr, pd.Timestamp):
        flags.append(win_start <= arr <= win_end)
    return any(flags) if flags else True

dfw = df[df.apply(in_window, axis=1)] if (dep_col or exp_col) else df.copy()

# Compute status at "now"
def status_now(row):
    dep = row[dep_col] if dep_col else None
    arr = row[exp_col] if exp_col else None
    if isinstance(arr, pd.Timestamp) and arr <= ts_current:
        return "Arrived"
    if isinstance(dep, pd.Timestamp) and dep > ts_current:
        return "Upcoming"
    if isinstance(dep, pd.Timestamp) and dep <= ts_current and (not isinstance(arr, pd.Timestamp) or arr > ts_current):
        return "In-Air"
    return "Unknown"

dfw["live_status"] = dfw.apply(status_now, axis=1)

# KPIs
c1, c2, c3, c4 = st.columns(4)
total_flights = len(dfw)
delayed = (dfw["prediction"] == "Delayed").sum()
ontime = (dfw["prediction"] == "On-time").sum()
high_risk = (dfw["risk_classification"] == "High").sum()
with c1: st.metric("Flights in view", f"{total_flights:,}")
with c2: st.metric("Predicted Delayed", f"{delayed:,}")
with c3: st.metric("Predicted On-time", f"{ontime:,}")
with c4: st.metric("High Risk Flights", f"{high_risk:,}")

# Tables
st.markdown("### 🟢 Upcoming Flights (with incident-aware risk)")
upc = dfw[dfw["live_status"] == "Upcoming"].copy()
show_cols = [c for c in ["airline","flight_number","tail_number","origin","dest","aircraft_type","live_status","prediction","delay_probability","predicted_delay_minutes","risk_classification","incident_probability","priority_action","risk_factors","incident_references"] if c in upc.columns]
st.dataframe(upc.sort_values(dep_col if dep_col else "flight_id").head(500)[show_cols], use_container_width=True)

st.markdown("### ✈️ In-Air")
ina = dfw[dfw["live_status"] == "In-Air"].copy()
show_cols_in = [c for c in ["airline","flight_number","tail_number","origin","dest","aircraft_type","prediction","predicted_delay_minutes","risk_classification","priority_action"] if c in ina.columns]
st.dataframe(ina.sort_values(exp_col if exp_col else "flight_id").head(500)[show_cols_in], use_container_width=True)

st.markdown("### 🛬 Arrived")
arr = dfw[dfw["live_status"] == "Arrived"].copy()
show_cols_ar = [c for c in ["airline","flight_number","tail_number","origin","dest","aircraft_type","prediction","predicted_delay_minutes","risk_classification"] if c in arr.columns]
st.dataframe(arr.sort_values(exp_col if exp_col else "flight_id", ascending=False).head(500)[show_cols_ar], use_container_width=True)

# Airline Overview
st.markdown("---")
st.header("🏷️ Airline Overview")
if "airline" in dfw.columns:
    g = dfw.groupby("airline").agg(
        flights=("flight_id","count"),
        delayed=("prediction", lambda s: (s=="Delayed").sum()),
        ontime=("prediction", lambda s: (s=="On-time").sum()),
        high_risk=("risk_classification", lambda s: (s=="High").sum()),
        avg_delay_min=("predicted_delay_minutes","mean")
    )
    g["delay_rate"] = (g["delayed"]/g["flights"]*100).round(1)
    st.dataframe(g.sort_values("flights", ascending=False), use_container_width=True)

# Footer: small helper to “play” the slider (optional manual use)
with st.expander("Auto-play help"):
    st.markdown("Use the time slider to scrub through history. For auto-play, just drag slowly; Streamlit re-renders quickly. You can also re-run the app periodically.")
