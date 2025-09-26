import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(
    page_title="🛡️ Guardian Eye - Aviation Ops Center",
    page_icon="🛩️",
    layout="wide"
)

# Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="datarefresh")

# ================================================================
# LOAD DATA
# ================================================================
DATASET = "aviation_dataset.csv"

if not os.path.exists(DATASET):
    st.error(f"Dataset not found: {DATASET}. Please upload your dataset.")
    st.stop()

# Parse datetime columns if available
try:
    df = pd.read_csv(
        DATASET,
        parse_dates=["Scheduled Departure", "Expected Arrival"]
    )
except Exception:
    df = pd.read_csv(DATASET)
    # If no datetime parsing worked, coerce manually
    for col in ["Scheduled Departure", "Expected Arrival"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# ================================================================
# CLEANUP + ENRICH DATA
# ================================================================
# Standardize column names
df.columns = [c.strip().replace("_", " ").title() for c in df.columns]

# Add Flight Status if not present
if "Delay Minutes" in df.columns:
    df["Flight Status"] = np.where(df["Delay Minutes"] > 15, "DELAYED", "ON-TIME")
else:
    df["Flight Status"] = "UNKNOWN"

# Add Risk Level if not present
if "Incident Probability" in df.columns:
    df["Risk Level"] = pd.cut(
        df["Incident Probability"],
        bins=[-0.01, 0.3, 0.5, 0.7, 1],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
else:
    df["Risk Level"] = "UNKNOWN"

# ================================================================
# SIDEBAR FILTERS
# ================================================================
st.sidebar.header("🎛️ Filters")

airlines = ["All"] + sorted(df["Airline"].dropna().unique().tolist())
airline_filter = st.sidebar.selectbox("Airline", airlines)

aircraft_types = ["All"] + sorted(df["Aircraft Type"].dropna().unique().tolist())
aircraft_filter = st.sidebar.selectbox("Aircraft Type", aircraft_types)

tail_numbers = ["All"] + sorted(df["Tail Number"].dropna().unique().tolist())
tail_filter = st.sidebar.selectbox("Tail Number", tail_numbers)

origins = ["All"] + sorted(df["Origin"].dropna().unique().tolist())
origin_filter = st.sidebar.selectbox("Origin", origins)

destinations = ["All"] + sorted(df["Destination"].dropna().unique().tolist())
destination_filter = st.sidebar.selectbox("Destination", destinations)

# Apply filters
filtered_df = df.copy()
if airline_filter != "All":
    filtered_df = filtered_df[filtered_df["Airline"] == airline_filter]
if aircraft_filter != "All":
    filtered_df = filtered_df[filtered_df["Aircraft Type"] == aircraft_filter]
if tail_filter != "All":
    filtered_df = filtered_df[filtered_df["Tail Number"] == tail_filter]
if origin_filter != "All":
    filtered_df = filtered_df[filtered_df["Origin"] == origin_filter]
if destination_filter != "All":
    filtered_df = filtered_df[filtered_df["Destination"] == destination_filter]

# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div style="text-align:center; padding:10px;">
  <h1 style="color:#3B82F6;">🛡️ Guardian Eye</h1>
  <p style="color:#9CA3AF;">Real-Time Flight Safety & Delay Risk Dashboard</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col2:
    st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))
    st.metric("Date", datetime.now().strftime("%d %b %Y"))

# ================================================================
# METRICS SUMMARY
# ================================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Flights", len(filtered_df))
with col2:
    st.metric("On-Time Flights", len(filtered_df[filtered_df["Flight Status"]=="ON-TIME"]))
with col3:
    st.metric("Delayed Flights", len(filtered_df[filtered_df["Flight Status"]=="DELAYED"]))
with col4:
    st.metric("High Risk Flights", len(filtered_df[filtered_df["Risk Level"]=="HIGH"]))
with col5:
    st.metric("Critical Risk Flights", len(filtered_df[filtered_df["Risk Level"]=="CRITICAL"]))

# ================================================================
# TIMELINE VISUALIZATION
# ================================================================
st.markdown("## 📊 Flight Timeline (Scheduled vs Expected)")

if "Scheduled Departure" in filtered_df.columns and "Expected Arrival" in filtered_df.columns:
    timeline_df = filtered_df.dropna(subset=["Scheduled Departure", "Expected Arrival"])

    fig = go.Figure()

    for _, row in timeline_df.iterrows():
        color = "blue" if row["Flight Status"] == "ON-TIME" else "red"
        fig.add_trace(go.Scatter(
            x=[row["Scheduled Departure"], row["Expected Arrival"]],
            y=[row["Flight Id"], row["Flight Id"]],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=6),
            name=row["Flight Id"],
            hovertemplate=(
                f"Flight: {row['Flight Id']}<br>"
                f"Airline: {row['Airline']}<br>"
                f"Route: {row['Origin']} → {row['Destination']}<br>"
                f"Scheduled: {row['Scheduled Departure']}<br>"
                f"Expected: {row['Expected Arrival']}<br>"
                f"Delay: {row['Delay Minutes']} min<br>"
                f"Status: {row['Flight Status']}"
            )
        ))

    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title="Flight ID",
        showlegend=False,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Scheduled/Expected times not available in dataset.")

# ================================================================
# TABLE VIEW
# ================================================================
st.markdown("## ✈️ Flights Overview")

st.dataframe(
    filtered_df[
        ["Flight Id","Airline","Aircraft Type","Tail Number","Origin","Destination",
         "Scheduled Departure","Expected Arrival","Flight Status","Risk Level","Delay Minutes"]
    ].sort_values("Scheduled Departure").reset_index(drop=True),
    use_container_width=True
)

# ================================================================
# RISK ALERTS
# ================================================================
st.markdown("## 🚨 Risk Alerts")

high_risk_flights = filtered_df[filtered_df["Risk Level"].isin(["HIGH","CRITICAL"])]

if high_risk_flights.empty:
    st.success("✅ No high-risk flights detected at the moment.")
else:
    for _, row in high_risk_flights.iterrows():
        st.error(
            f"⚠️ Flight {row['Flight Id']} ({row['Airline']} - {row['Aircraft Type']}) "
            f"from {row['Origin']} to {row['Destination']} shows **{row['Risk Level']} RISK** "
            f"with delay {row['Delay Minutes']} min. This condition **might lead to incident** "
            f"if not addressed before next departure."
        )
