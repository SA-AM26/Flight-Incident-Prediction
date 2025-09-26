import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# Auto-refresh every 30s (without full rerun spam)
# =====================================================
st_autorefresh(interval=30_000, key="guardian_eye_autorefresh")
st.set_page_config(page_title="🛡️ Guardian Eye", layout="wide")

# =====================================================
# Airport reference (used if your CSV lacks lat/lon)
# =====================================================
AIRPORTS = {
    # Core Indian hubs
    "DEL": {"name": "Delhi", "lat": 28.5562, "lng": 77.1000},
    "BOM": {"name": "Mumbai", "lat": 19.0896, "lng": 72.8656},
    "BLR": {"name": "Bengaluru", "lat": 13.1986, "lng": 77.7066},
    "MAA": {"name": "Chennai", "lat": 12.9941, "lng": 80.1709},
    "CCU": {"name": "Kolkata", "lat": 22.6547, "lng": 88.4467},
    "HYD": {"name": "Hyderabad", "lat": 17.2403, "lng": 78.4294},
    "COK": {"name": "Kochi", "lat": 10.1520, "lng": 76.4019},
    "AMD": {"name": "Ahmedabad", "lat": 23.0726, "lng": 72.6263},
    "PNQ": {"name": "Pune", "lat": 18.5822, "lng": 73.9197},
    "JAI": {"name": "Jaipur", "lat": 26.8247, "lng": 75.8127},
    # Extended list we used earlier
    "GOI": {"name": "Goa", "lat": 15.3800, "lng": 73.8310},
    "TRV": {"name": "Trivandrum", "lat": 8.4821, "lng": 76.9201},
    "IXB": {"name": "Bagdogra", "lat": 26.6812, "lng": 88.3286},
    "IXC": {"name": "Chandigarh", "lat": 30.6735, "lng": 76.7885},
    "VNS": {"name": "Varanasi", "lat": 25.4510, "lng": 82.8593},
    "SXR": {"name": "Srinagar", "lat": 33.9871, "lng": 74.7740},
}

# =====================================================
# Helpers
# =====================================================
def coerce_dt(s: pd.Series):
    """Safely coerce datetime columns."""
    return pd.to_datetime(s, errors="coerce")

def add_missing_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time columns exist with standard names."""
    df = df.copy()
    # Standardize column names if needed
    if "Scheduled Departure" not in df.columns and "scheduled_departure" in df.columns:
        df.rename(columns={"scheduled_departure": "Scheduled Departure"}, inplace=True)
    if "Actual Arrival" not in df.columns:
        # Some datasets have 'actual_departure' as the only realized time; use that for display
        if "actual_departure" in df.columns:
            df.rename(columns={"actual_departure": "Actual Arrival"}, inplace=True)
        elif "expected_arr_utc" in df.columns:
            df.rename(columns={"expected_arr_utc": "Actual Arrival"}, inplace=True)

    # Coerce to datetime if present
    for c in ["Scheduled Departure", "Actual Arrival"]:
        if c in df.columns:
            df[c] = coerce_dt(df[c])

    return df

def fill_airport_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Fill origin/destination lat/lng from reference if missing."""
    df = df.copy()

    # If origin/dest coords are missing, fill from AIRPORTS
    if "origin_lat" not in df.columns:
        df["origin_lat"] = df["origin"].map(lambda x: AIRPORTS.get(str(x), {}).get("lat", np.nan))
    if "origin_lng" not in df.columns:
        df["origin_lng"] = df["origin"].map(lambda x: AIRPORTS.get(str(x), {}).get("lng", np.nan))
    if "dest_lat" not in df.columns:
        df["dest_lat"] = df["destination"].map(lambda x: AIRPORTS.get(str(x), {}).get("lat", np.nan))
    if "dest_lng" not in df.columns:
        df["dest_lng"] = df["destination"].map(lambda x: AIRPORTS.get(str(x), {}).get("lng", np.nan))

    return df

def estimate_current_position(row: pd.Series, now_utc: datetime) -> tuple[float, float]:
    """
    Estimate current position between origin and destination.
    If current_lat/current_lng exist, use them. Else, interpolate based on time progress.
    """
    # Use real-time columns if present
    if "current_lat" in row and pd.notna(row.get("current_lat")) \
       and "current_lng" in row and pd.notna(row.get("current_lng")):
        return float(row["current_lat"]), float(row["current_lng"])

    # Fallback interpolation using time fraction between scheduled and arrival
    o_lat, o_lng, d_lat, d_lng = row["origin_lat"], row["origin_lng"], row["dest_lat"], row["dest_lng"]

    sd = row.get("Scheduled Departure", pd.NaT)
    ar = row.get("Actual Arrival", pd.NaT)

    # If we don't have times, just show origin
    if pd.isna(sd) or pd.isna(ar) or sd == ar:
        return float(o_lat), float(o_lng)

    # Compute progress [0..1]
    total = (ar - sd).total_seconds()
    elapsed = (now_utc - sd).total_seconds()
    frac = max(0.0, min(1.0, elapsed / total)) if total > 0 else 0.0

    # Linear interpolation (great-circle would be ideal, but this is fast & fine for viz)
    lat = o_lat + (d_lat - o_lat) * frac
    lng = o_lng + (d_lng - o_lng) * frac
    return float(lat), float(lng)

def risk_to_color(risk: str) -> str:
    return {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "royalblue", "LOW": "limegreen"}.get(str(risk), "lightgray")

# Fake incident history narratives (used when high/critical)
INCIDENT_HISTORY = {
    "engine_health": "Past events show engine wear leading to in-flight shutdowns—needs boroscope inspection.",
    "maintenance_score": "Deferred maintenance previously resulted in ground delays and MEL usage—schedule rectification.",
    "weather_score": "Route sees seasonal low vis / convective weather—consider alt routing and added fuel.",
    "atc_score": "ATC congestion patterns on this corridor cause holding and late sequencing—adjust slot allocations.",
    "structural_integrity": "Fatigue findings on similar tails—recommend detailed visual and NDT checks before next sector.",
}

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    if not os.path.exists("aviation_dataset.csv"):
        st.error("Missing `aviation_dataset.csv`. Please place your dataset in the repo root.")
        st.stop()

    df = pd.read_csv("aviation_dataset.csv")
    df = add_missing_time_columns(df)

    # Ensure required categorical cols exist
    required = ["airline", "aircraft_type", "tail_number", "origin", "destination",
                "status", "delay_minutes", "risk_level", "incident_probability"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan  # create empty if absent to avoid KeyErrors

    # Ensure numeric health/risk fields exist
    numeric_defaults = {
        "engine_health": 80.0,
        "maintenance_score": 80.0,
        "weather_score": 0.3,   # normalized 0..1
        "atc_score": 85.0,      # 0..100
        "structural_integrity": 85.0,
    }
    for c, v in numeric_defaults.items():
        if c not in df.columns:
            df[c] = v

    # Coordinates
    df = fill_airport_coords(df)

    return df

df = load_data()

# =====================================================
# Sidebar filters
# =====================================================
st.sidebar.title("🎛️ Flight Filters")

airlines = ["All Airlines"] + sorted(df["airline"].dropna().unique().tolist())
selected_airline = st.sidebar.selectbox("Airline", airlines)

aircraft_types = ["All Types"] + sorted(df["aircraft_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Aircraft Type", aircraft_types)

origins = ["All Origins"] + sorted(df["origin"].dropna().unique().tolist())
selected_origin = st.sidebar.selectbox("Origin", origins)

destinations = ["All Destinations"] + sorted(df["destination"].dropna().unique().tolist())
selected_destination = st.sidebar.selectbox("Destination", destinations)

tails = ["All Aircraft"] + sorted(df["tail_number"].dropna().unique().tolist())
selected_tail = st.sidebar.selectbox("Tail Number", tails)

# Apply filters
flt = df.copy()
if selected_airline != "All Airlines":
    flt = flt[flt["airline"] == selected_airline]
if selected_type != "All Types":
    flt = flt[flt["aircraft_type"] == selected_type]
if selected_origin != "All Origins":
    flt = flt[flt["origin"] == selected_origin]
if selected_destination != "All Destinations":
    flt = flt[flt["destination"] == selected_destination]
if selected_tail != "All Aircraft":
    flt = flt[flt["tail_number"] == selected_tail]

# =====================================================
# Header
# =====================================================
st.markdown(
    """
    <div style="text-align:center; padding:14px; background:#0c1220; border-radius:12px; 
                border:1px solid rgba(59,130,246,.35);">
        <h1 style="color:#60A5FA; margin:0;">🛡️ GUARDIAN EYE</h1>
        <div style="color:#93A3B5;">Real-time Aviation Operations • Risk & Delay Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =====================================================
# KPI Row
# =====================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Flights (filtered)", len(flt))
c2.metric("In-Flight", int((flt["status"] == "IN-FLIGHT").sum()))
c3.metric("Delayed", int((flt["status"] == "DELAYED").sum()))
c4.metric("Critical Risk", int((flt["risk_level"] == "CRITICAL").sum()))

# =====================================================
# Globe (Plotly orthographic) + routes & markers
# =====================================================
st.subheader("🌍 Fleet Globe View (live)")

# Build small working subset for performance on globe
now_utc = datetime.utcnow().replace(tzinfo=timezone.utc).replace(tzinfo=None)
work = flt.copy()

# Estimate positions (or use current_lat/current_lng if present)
lat_list, lng_list, color_list, hover_list = [], [], [], []
for _, r in work.iterrows():
    try:
        lat, lng = estimate_current_position(r, now_utc)
        lat_list.append(lat)
        lng_list.append(lng)
        color_list.append(risk_to_color(r.get("risk_level", "")))
        hover_list.append(
            f"{r.get('tail_number','?')} | {r.get('airline','?')} | "
            f"{r.get('origin','?')}→{r.get('destination','?')} | "
            f"Risk: {r.get('risk_level','?')} | Delay: {r.get('delay_minutes',0)}m"
        )
    except Exception:
        # If anything goes off, skip that row gracefully
        continue

fig = go.Figure()

# Route lines (thin, translucent)
route_sample = work.head(150)  # limit for clarity
for _, r in route_sample.iterrows():
    try:
        fig.add_trace(go.Scattergeo(
            lon=[r["origin_lng"], r["dest_lng"]],
            lat=[r["origin_lat"], r["dest_lat"]],
            mode="lines",
            line=dict(width=1, color="rgba(180,200,255,0.25)"),
            hoverinfo="skip",
            showlegend=False
        ))
    except Exception:
        pass

# Aircraft markers
fig.add_trace(go.Scattergeo(
    lon=lng_list,
    lat=lat_list,
    text=hover_list,
    hoverinfo="text",
    mode="markers",
    marker=dict(size=5, color=color_list, opacity=0.95),
    name="Aircraft"
))

# Gentle rotation based on time (changes each refresh)
spin = (datetime.utcnow().timestamp() / 12) % 360  # slow spin
fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=spin, lat=15),
    showcountries=True,
    showcoastlines=True,
    landcolor="rgb(10,30,60)",
    oceancolor="rgb(5,10,25)",
    showocean=True,
    showland=True,
    bgcolor="rgba(0,0,0,0)"
)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    height=480
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Flight table
# =====================================================
st.subheader("✈️ Flight Monitor")

show_cols = [
    "flight_id", "airline", "aircraft_type", "tail_number",
    "origin", "destination", "Scheduled Departure", "Actual Arrival",
    "status", "delay_minutes", "risk_level", "incident_probability"
]
for c in show_cols:
    if c not in flt.columns:
        flt[c] = np.nan

tbl = flt[show_cols].copy()
tbl["incident_probability"] = (pd.to_numeric(tbl["incident_probability"], errors="coerce").fillna(0) * 100).round(2)
st.dataframe(tbl.head(100), use_container_width=True)

# =====================================================
# Detailed panel (if a single aircraft selected)
# =====================================================
if selected_tail != "All Aircraft" and not flt.empty:
    flight = flt.iloc[0]
    st.markdown("---")
    st.subheader(f"🛩️ Aircraft Detail • {selected_tail}")

    i1, i2 = st.columns(2)
    with i1:
        st.write(f"**Airline:** {flight.get('airline','')}")
        st.write(f"**Type:** {flight.get('aircraft_type','')}")
        st.write(f"**Route:** {flight.get('origin','?')} → {flight.get('destination','?')}")
        st.write(f"**Status:** {flight.get('status','')}")
        sd = flight.get("Scheduled Departure", pd.NaT)
        ar = flight.get("Actual Arrival", pd.NaT)
        st.write(f"**Scheduled Departure:** {sd}")
        st.write(f"**Planned/Actual Arrival:** {ar}")
        st.write(f"**Delay (min):** {int(pd.to_numeric(flight.get('delay_minutes', 0), errors='coerce') or 0)}")

    with i2:
        risk_level = str(flight.get("risk_level", "LOW"))
        risk_badge = {
            "CRITICAL": "🔴 CRITICAL",
            "HIGH": "🟠 HIGH",
            "MEDIUM": "🔵 MEDIUM",
            "LOW": "🟢 LOW"
        }.get(risk_level, "🟢 LOW")
        st.write(f"**Risk Level:** {risk_badge}")
        ip = float(pd.to_numeric(flight.get("incident_probability", 0), errors="coerce") or 0) * 100
        st.write(f"**Incident Probability:** {ip:.2f}%")

        # Reasons (from history mapping) when HIGH/CRITICAL
        if risk_level in ["HIGH", "CRITICAL"]:
            st.error("⚠️ Historical risk drivers detected:")
            if float(flight.get("engine_health", 100)) < 60:
                st.write("• " + INCIDENT_HISTORY["engine_health"])
            if float(flight.get("maintenance_score", 100)) < 60:
                st.write("• " + INCIDENT_HISTORY["maintenance_score"])
            if float(flight.get("weather_score", 0)) > 0.7:
                st.write("• " + INCIDENT_HISTORY["weather_score"])
            if float(flight.get("atc_score", 100)) < 70:
                st.write("• " + INCIDENT_HISTORY["atc_score"])
            if float(flight.get("structural_integrity", 100)) < 60:
                st.write("• " + INCIDENT_HISTORY["structural_integrity"])

    # Next flight impact check
    st.markdown("### ⏳ Next Flight Impact Check")
    if risk_level in ["HIGH", "CRITICAL"]:
        st.warning(
            f"Tail `{flight.get('tail_number','')}` shows **{risk_level} risk**. "
            f"This may impact the **next departure from {flight.get('destination','?')}**."
        )
        # If arrival time exists, assume next sector in 4 hours
        ar = flight.get("Actual Arrival", pd.NaT)
        if pd.notna(ar):
            next_dep = ar + timedelta(hours=4)
            mins_left = int((next_dep - datetime.now()).total_seconds() / 60)
            if mins_left > 0:
                color = "🟢" if mins_left >= 120 else ("🟠" if mins_left >= 60 else "🔴")
                st.write(f"Next departure in **{mins_left} min** {color}")
            else:
                st.error("Next sector window already exceeded—re-slot or swap tail advised.")
        else:
            st.info("Arrival time unavailable—cannot compute turnaround window.")
    else:
        st.success("No immediate action required for next sector.")

# =====================================================
# Risk analytics
# =====================================================
st.markdown("---")
st.subheader("📊 Risk Analytics")

risk_counts = df["risk_level"].value_counts()
fig_risk = px.pie(
    names=risk_counts.index,
    values=risk_counts.values,
    title="Fleet Risk Distribution",
    color=risk_counts.index,
    color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "royalblue", "LOW": "limegreen"},
)
fig_risk.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_risk, use_container_width=True)
