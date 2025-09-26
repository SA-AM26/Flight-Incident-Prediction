import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Guardian Eye - Aviation Dashboard",
                   page_icon="🛡️",
                   layout="wide")

# -----------------------------
# AUTO REFRESH CLOCK
# -----------------------------
st_autorefresh(interval=1000, key="clock_refresh")

# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1 style='color:#4FC3F7;'>🛡️ Guardian Eye - Aviation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:grey;'>Real-time flight safety & delay monitoring</p>", unsafe_allow_html=True)

with col2:
    now = datetime.now()
    st.markdown("### ⏱️ Live Clock")
    st.markdown(f"<div style='font-size:22px; color:#00FFAA;'>{now.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:grey;'>{now.strftime('%d %B %Y')}</div>", unsafe_allow_html=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
if not os.path.exists("aviation_dataset.csv"):
    st.error("❌ Dataset not found! Please generate 'aviation_dataset.csv' first.")
    st.stop()

df = pd.read_csv("aviation_dataset.csv")

# Ensure datetime parsing
if "scheduled_departure" in df.columns:
    df["scheduled_departure"] = pd.to_datetime(df["scheduled_departure"], errors="coerce")
if "actual_departure" in df.columns:
    df["actual_departure"] = pd.to_datetime(df["actual_departure"], errors="coerce")

# -----------------------------
# INCIDENT HISTORY (SIMULATED)
# -----------------------------
incident_reasons = {
    "CRITICAL": [
        "⚠️ Engine vibration exceeded safe threshold",
        "⚠️ Hydraulic failure during pre-flight check",
        "⚠️ Severe weather forecasted on route",
        "⚠️ Landing gear malfunction reported"
    ],
    "HIGH": [
        "⚠️ Delayed A-check maintenance overdue",
        "⚠️ Pilot fatigue flagged (>14hrs duty)",
        "⚠️ Avionics warning during taxiing",
        "⚠️ ATC congestion at destination airport"
    ],
    "MEDIUM": [
        "⚠️ Minor technical snag reported",
        "⚠️ Weather instability expected en route",
        "⚠️ Crew rest below recommended hours"
    ]
}

action_plan = {
    "CRITICAL": "🚨 Flight must be GROUNDED. Emergency maintenance required before next departure.",
    "HIGH": "🟠 Extra inspection recommended before departure. Delay likely until cleared.",
    "MEDIUM": "🔵 Monitor closely. Possible short delay if condition persists.",
    "LOW": "🟢 Normal operations."
}

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🎛️ Flight Filters")

airlines = ["All Airlines"] + sorted(df["airline"].dropna().unique())
aircraft_types = ["All Types"] + sorted(df["aircraft_type"].dropna().unique())
origins = ["All Origins"] + sorted(df["origin"].dropna().unique())
destinations = ["All Destinations"] + sorted(df["destination"].dropna().unique())

selected_airline = st.sidebar.selectbox("Airline", airlines)
selected_aircraft_type = st.sidebar.selectbox("Aircraft Type", aircraft_types)
selected_origin = st.sidebar.selectbox("Origin Airport", origins)
selected_destination = st.sidebar.selectbox("Destination Airport", destinations)

filtered_df = df.copy()
if selected_airline != "All Airlines":
    filtered_df = filtered_df[filtered_df["airline"] == selected_airline]
if selected_aircraft_type != "All Types":
    filtered_df = filtered_df[filtered_df["aircraft_type"] == selected_aircraft_type]
if selected_origin != "All Origins":
    filtered_df = filtered_df[filtered_df["origin"] == selected_origin]
if selected_destination != "All Destinations":
    filtered_df = filtered_df[filtered_df["destination"] == selected_destination]

# -----------------------------
# SUMMARY STATS
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Flights", len(filtered_df))
col2.metric("In-Flight", len(filtered_df[filtered_df["status"] == "IN-FLIGHT"]))
col3.metric("Delayed", len(filtered_df[filtered_df["status"] == "DELAYED"]))
col4.metric("Critical Risk", len(filtered_df[filtered_df["risk_level"] == "CRITICAL"]))
col5.metric("Avg Delay (min)", round(filtered_df["delay_minutes"].mean(), 1))

# -----------------------------
# 3D GLOBE (Pydeck)
# -----------------------------
if "current_lat" in filtered_df.columns and "current_lng" in filtered_df.columns:
    st.markdown("### 🌍 Global Aircraft Tracking")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position=["current_lng", "current_lat"],
        get_color=[
            "255 * (risk_level == 'CRITICAL')",
            "200 * (risk_level == 'HIGH')",
            "100 * (risk_level == 'MEDIUM')",
            "50 * (risk_level == 'LOW')",
        ],
        get_radius=50000,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=78, zoom=3, pitch=30)

    st.pydeck_chart(pdk.Deck(layers=[layer],
                             initial_view_state=view_state,
                             map_style="mapbox://styles/mapbox/dark-v10"))
else:
    st.warning("⚠️ No geolocation data available for flights.")

# -----------------------------
# FLIGHT TABLE WITH REASONS
# -----------------------------
st.markdown("### ✈️ Flight Monitor (with Delay & Risk Reasons)")

display_df = filtered_df.copy()

# Add Delay Reason Column
def get_delay_reason(row):
    reasons = []
    if row.get("weather_delay", 0) > 15:
        reasons.append("Weather")
    if row.get("technical_delay", 0) > 15:
        reasons.append("Technical")
    if row.get("atc_delay", 0) > 10:
        reasons.append("ATC")
    return ", ".join(reasons) if reasons else "On-Time"

display_df["Delay Reason"] = display_df.apply(get_delay_reason, axis=1)

# Add Incident History + Action
def get_incident_info(row):
    if row["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM"]:
        reason = np.random.choice(incident_reasons[row["risk_level"]])
        action = action_plan[row["risk_level"]]
        return f"{reason}\n{action}"
    else:
        return "🟢 No incidents. Safe to operate."

display_df["Incident History"] = display_df.apply(get_incident_info, axis=1)

# Final columns
cols = [
    "flight_id", "airline", "aircraft_type", "origin", "destination",
    "scheduled_departure", "actual_departure", "status",
    "risk_level", "delay_minutes", "Delay Reason", "Incident History"
]

st.dataframe(display_df[cols].head(20), use_container_width=True)

# -----------------------------
# INCIDENT LOG PANEL (WATCHLIST)
# -----------------------------
st.markdown("### 🚨 Incident Watchlist (High & Critical Risk Flights)")

incident_watchlist = display_df[display_df["risk_level"].isin(["CRITICAL", "HIGH"])]

if incident_watchlist.empty:
    st.success("🟢 No High or Critical risk flights at the moment. All safe!")
else:
    for _, row in incident_watchlist.iterrows():
        st.markdown(f"""
        <div style="background:#1E1E1E; border-left:5px solid {'#FF0000' if row['risk_level']=='CRITICAL' else '#FFA500'};
                    padding:10px; margin-bottom:10px; border-radius:5px;">
            <h4 style="color:#4FC3F7;">✈️ Flight {row['flight_id']} | {row['airline']} ({row['aircraft_type']})</h4>
            <p style="color:white;">🛫 {row['origin']} ➝ {row['destination']}</p>
            <p style="color:grey;">Scheduled: {row['scheduled_departure']} | Actual: {row['actual_departure']}</p>
            <p style="color:{'red' if row['risk_level']=='CRITICAL' else 'orange'};">
                <b>Risk Level: {row['risk_level']}</b>
            </p>
            <p style="color:white;">Reason: {row['Incident History'].splitlines()[0]}</p>
            <p style="color:#00FFAA;">Action: {action_plan[row['risk_level']]}</p>
        </div>
        """, unsafe_allow_html=True)
