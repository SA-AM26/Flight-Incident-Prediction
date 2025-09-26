import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st.set_page_config(page_title="Guardian Eye - Aviation Dashboard", layout="wide")

# 🔄 Auto-refresh every 30 seconds (refreshes KPIs & globe)
st_autorefresh(interval=30 * 1000, key="data_refresh")

# --- Utility: Generate sample aviation data ---
def generate_sample_data(n=200):
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]
    aircraft_types = ["Boeing 737-800", "Boeing 737 MAX 8", "Airbus A320neo",
                      "Airbus A321", "ATR 72-600", "Bombardier Q400", "Boeing 787-8", "Airbus A350-900"]
    airports = {
        "DEL": (28.5562, 77.1000), "BOM": (19.0896, 72.8656),
        "BLR": (13.1986, 77.7066), "MAA": (12.9941, 80.1709),
        "CCU": (22.6547, 88.4467), "HYD": (17.2403, 78.4294),
        "COK": (10.1520, 76.4019), "AMD": (23.0726, 72.6263),
        "PNQ": (18.5822, 73.9197), "JAI": (26.8247, 75.8127)
    }

    records = []
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    for i in range(n):
        airline = np.random.choice(airlines)
        aircraft_type = np.random.choice(aircraft_types)
        origin, dest = np.random.choice(list(airports.keys()), 2, replace=False)
        tail_number = f"VT-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))}"

        dep_time = base_time + timedelta(minutes=np.random.randint(0, 60*24))
        delay = max(0, int(np.random.normal(30, 20)))
        arr_time = dep_time + timedelta(hours=2, minutes=delay)

        incident_prob = np.random.rand()
        if incident_prob > 0.7:
            risk_level = "CRITICAL"
            reason = "Engine fault history"
        elif incident_prob > 0.5:
            risk_level = "HIGH"
            reason = "Poor weather + maintenance overdue"
        elif incident_prob > 0.3:
            risk_level = "MEDIUM"
            reason = "ATC congestion"
        else:
            risk_level = "LOW"
            reason = "Normal operations"

        records.append({
            "Flight ID": f"FL{i+1:04d}",
            "Airline": airline,
            "Aircraft Type": aircraft_type,
            "Tail Number": tail_number,
            "Origin": origin,
            "Destination": dest,
            "Scheduled Departure": dep_time,
            "Actual Arrival": arr_time,
            "Delay Minutes": delay,
            "Incident Probability": incident_prob,
            "Risk Level": risk_level,
            "Reason": reason,
            "Latitude": airports[origin][0] + np.random.uniform(-1, 1),
            "Longitude": airports[origin][1] + np.random.uniform(-1, 1),
            "Origin Lat": airports[origin][0],
            "Origin Lng": airports[origin][1],
            "Dest Lat": airports[dest][0],
            "Dest Lng": airports[dest][1]
        })

    return pd.DataFrame(records)

# --- Load or generate data ---
if not os.path.exists("aviation_dataset.csv"):
    df = generate_sample_data(8000)
    df.to_csv("aviation_dataset.csv", index=False)
else:
    df = pd.read_csv("aviation_dataset.csv", parse_dates=["Scheduled Departure", "Actual Arrival"])

# --- Sidebar Filters ---
st.sidebar.header("✈️ Flight Selection")
airline_choice = st.sidebar.selectbox("Airline", ["All"] + sorted(df["Airline"].unique()))
aircraft_choice = st.sidebar.selectbox("Aircraft Type", ["All"] + sorted(df["Aircraft Type"].unique()))
origin_choice = st.sidebar.selectbox("Origin", ["All"] + sorted(df["Origin"].unique()))
dest_choice = st.sidebar.selectbox("Destination", ["All"] + sorted(df["Destination"].unique()))

filtered = df.copy()
if airline_choice != "All":
    filtered = filtered[filtered["Airline"] == airline_choice]
if aircraft_choice != "All":
    filtered = filtered[filtered["Aircraft Type"] == aircraft_choice]
if origin_choice != "All":
    filtered = filtered[filtered["Origin"] == origin_choice]
if dest_choice != "All":
    filtered = filtered[filtered["Destination"] == dest_choice]

# --- Live Clock ---
st_autorefresh(interval=1000, key="clock_refresh")
current_time = datetime.now()
clock_col = st.sidebar.container()
clock_col.subheader("⏱️ Live Clock")
clock_col.write(f"**{current_time.strftime('%H:%M:%S')}**")
clock_col.write(current_time.strftime("%d %B %Y"))

# --- Dashboard Header ---
st.markdown("<h1 style='color:#3B82F6;'>🛡️ Guardian Eye - Aviation Dashboard</h1>", unsafe_allow_html=True)
st.write("Real-time flight safety & delay monitoring")

# --- KPIs ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Flights", len(filtered))
kpi2.metric("In-Flight", len(filtered[filtered["Delay Minutes"] < 15]))
kpi3.metric("Delayed", len(filtered[filtered["Delay Minutes"] > 15]))
kpi4.metric("Critical Risk", len(filtered[filtered["Risk Level"] == "CRITICAL"]))
kpi5.metric("Avg Delay (min)", round(filtered["Delay Minutes"].mean(), 1))

# --- Globe Visualization ---
st.subheader("🌍 Global Aircraft Tracking")

def show_globe(df):
    fig = go.Figure()

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    fig.add_surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.3, showscale=False)

    # Flight paths
    for _, row in df.iterrows():
        arc_lats = np.linspace(row["Origin Lat"], row["Dest Lat"], 30)
        arc_lons = np.linspace(row["Origin Lng"], row["Dest Lng"], 30)
        arc_z = np.linspace(0.02, 0.1, 30)

        fig.add_trace(go.Scatter3d(
            x=np.cos(np.radians(arc_lons)) * np.cos(np.radians(arc_lats)),
            y=np.sin(np.radians(arc_lons)) * np.cos(np.radians(arc_lats)),
            z=np.sin(np.radians(arc_lats)) + arc_z,
            mode="lines",
            line=dict(color="yellow", width=2),
            opacity=0.4,
            hoverinfo="skip"
        ))

    # Aircraft markers
    fig.add_trace(go.Scatter3d(
        x=np.cos(np.radians(df["Longitude"])) * np.cos(np.radians(df["Latitude"])),
        y=np.sin(np.radians(df["Longitude"])) * np.cos(np.radians(df["Latitude"])),
        z=np.sin(np.radians(df["Latitude"])),
        mode="markers",
        marker=dict(
            size=np.clip(df["Delay Minutes"]/15, 4, 15),
            color=df["Incident Probability"],
            colorscale="Reds",
            opacity=0.9
        ),
        text=[
            f"✈ {row['Flight ID']} | {row['Airline']} ({row['Aircraft Type']})<br>"
            f"{row['Origin']} → {row['Destination']}<br>"
            f"Departure: {row['Scheduled Departure']}<br>"
            f"Arrival: {row['Actual Arrival']}<br>"
            f"Delay: {row['Delay Minutes']} min<br>"
            f"Risk: {row['Risk Level']} ({row['Reason']})"
            for _, row in df.iterrows()
        ],
        hoverinfo="text"
    ))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

show_globe(filtered)

# --- High Risk Flights Requiring Action ---
st.subheader("⚠️ Flights Requiring Action (High/Critical Risk)")
action_df = filtered[filtered["Risk Level"].isin(["HIGH", "CRITICAL"])][
    ["Flight ID", "Airline", "Aircraft Type", "Origin", "Destination", 
     "Scheduled Departure", "Delay Minutes", "Risk Level", "Reason"]
].sort_values("Risk Level", ascending=False)

if action_df.empty:
    st.success("✅ No immediate actions required. All flights are safe.")
else:
    st.dataframe(action_df, use_container_width=True)
