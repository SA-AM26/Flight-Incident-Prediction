import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st
import plotly.express as px

# ============================================================
# Helpers
# ============================================================
np.random.seed(42)

def py_int(x): return int(np.asarray(x).item() if np.ndim(x) == 0 else x)
def py_float(x): return float(np.asarray(x).item() if np.ndim(x) == 0 else x)
def rand_tail(): return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

# ============================================================
# Data Generator
# ============================================================
def generate_realistic_aviation_data(n_flights=2000):
    airlines_data = {
        'Air India': {'code': 'AI','aircraft_types': ['Airbus A320neo','Boeing 787-8'],'safety_score': 85},
        'IndiGo': {'code': '6E','aircraft_types': ['Airbus A320neo','Airbus A321neo','ATR 72-600'],'safety_score': 92},
        'SpiceJet': {'code': 'SG','aircraft_types': ['Boeing 737-800','Bombardier Q400'],'safety_score': 78},
        'AirAsia India': {'code': 'I5','aircraft_types': ['Airbus A320neo'],'safety_score': 88}
    }

    airports = ['DEL','BOM','BLR','MAA','CCU','HYD','COK','AMD','PNQ','JAI']

    flights = []
    for i in range(n_flights):
        airline_name = np.random.choice(list(airlines_data.keys()))
        airline_info = airlines_data[airline_name]
        aircraft_type = np.random.choice(airline_info['aircraft_types'])
        tail_number = rand_tail()

        origin, destination = np.random.choice(airports, 2, replace=False)

        base_date = datetime(2024,1,1) + timedelta(days=py_int(np.random.randint(0, 30)))
        scheduled_departure = base_date + timedelta(hours=py_int(np.random.randint(5,23)),
                                                    minutes=py_int(np.random.choice([0,15,30,45])))
        scheduled_arrival = scheduled_departure + timedelta(hours=np.random.randint(1,3))
        
        # Assign realistic status distribution
        p = np.random.random()
        if p < 0.1:
            status = "EARLY"
            delay_minutes = -np.random.randint(5,15)
        elif p < 0.25:
            status = "DELAYED"
            delay_minutes = np.random.randint(20,120)
        else:
            status = "ON-TIME"
            delay_minutes = 0

        actual_departure = scheduled_departure + timedelta(minutes=max(0, delay_minutes))
        expected_arrival = scheduled_arrival + timedelta(minutes=delay_minutes)

        # Risk simulation
        engine_health = py_float(np.random.uniform(60,100))
        weather_score = py_float(np.random.uniform(0.3,1.0))
        atc_score = py_float(np.random.uniform(60,100))
        crew_fatigue = py_float(np.random.uniform(0,100))
        incident_probability = (100-engine_health)*0.3/100 + (1-weather_score)*0.3 + (100-atc_score)*0.2/100 + (crew_fatigue*0.2)/100
        incident_probability *= (100 - airline_info['safety_score'])/100
        incident_probability = min(1, max(0, incident_probability))

        if incident_probability > 0.7: risk_level="CRITICAL"
        elif incident_probability > 0.5: risk_level="HIGH"
        elif incident_probability > 0.3: risk_level="MEDIUM"
        else: risk_level="LOW"

        # Delay reason
        delay_reason = None
        if status=="DELAYED":
            reason_pool = []
            if engine_health < 70: reason_pool.append("Technical Issue")
            if weather_score < 0.5: reason_pool.append("Weather (Fog/Storm)")
            if atc_score < 70: reason_pool.append("ATC Congestion")
            if crew_fatigue > 70: reason_pool.append("Crew Fatigue")
            delay_reason = np.random.choice(reason_pool) if reason_pool else "Operational Delay"
        elif risk_level in ["HIGH","CRITICAL"]:
            delay_reason = "⚠ Might lead to incident if not addressed"

        flights.append({
            "flight_id": f"{airline_info['code']}{1000+i}",
            "airline": airline_name,
            "aircraft_type": aircraft_type,
            "tail_number": tail_number,
            "origin": origin,
            "destination": destination,
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival,
            "actual_departure": actual_departure,
            "expected_arrival": expected_arrival,
            "status": status,
            "delay_minutes": delay_minutes,
            "risk_level": risk_level,
            "incident_probability": incident_probability,
            "delay_reason": delay_reason
        })
    return pd.DataFrame(flights)

# ============================================================
# Dashboard
# ============================================================
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye", page_icon="🛡️", layout="wide")

    # Load or generate dataset
    if not os.path.exists("aviation_dataset.csv"):
        df = generate_realistic_aviation_data(2000)
        df.to_csv("aviation_dataset.csv", index=False)
    else:
        df = pd.read_csv("aviation_dataset.csv", parse_dates=["scheduled_departure","scheduled_arrival","expected_arrival"])

    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    airlines = ["All Airlines"] + sorted(df["airline"].unique())
    selected_airline = st.sidebar.selectbox("Select Airline", airlines)
    filtered = df if selected_airline=="All Airlines" else df[df["airline"]==selected_airline]

    aircraft_types = ["All Types"] + sorted(filtered["aircraft_type"].unique())
    selected_type = st.sidebar.selectbox("Select Aircraft Type", aircraft_types)
    if selected_type!="All Types":
        filtered = filtered[filtered["aircraft_type"]==selected_type]

    # Overview
    st.title("🛡️ Guardian Eye - Aviation Operations Center")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Flights", len(filtered))
    c2.metric("Delayed", (filtered["status"]=="DELAYED").sum())
    c3.metric("On-Time", (filtered["status"]=="ON-TIME").sum())
    c4.metric("Early", (filtered["status"]=="EARLY").sum())

    # Flight monitor
    st.markdown("### ✈️ Flight Monitor")
    display = filtered[["flight_id","airline","aircraft_type","tail_number","origin","destination",
                        "status","delay_minutes","risk_level","delay_reason"]].copy()
    display["delay_minutes"] = display["delay_minutes"].astype(int)
    st.dataframe(display.head(50), use_container_width=True)

    # Risk breakdown chart
    st.markdown("### 📊 Risk Level Distribution")
    risk_counts = filtered["risk_level"].value_counts()
    if not risk_counts.empty:
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     color=risk_counts.index,
                     color_discrete_map={"CRITICAL":"#DC2626","HIGH":"#F59E0B","MEDIUM":"#3B82F6","LOW":"#10B981"})
        st.plotly_chart(fig, use_container_width=True)

    # Timeline View
    st.markdown("### ⏱ Flight Timeline (Scheduled vs Expected)")
    if len(filtered):
        timeline_df = pd.DataFrame({
            "Flight": filtered["flight_id"] + " (" + filtered["airline"] + ")",
            "Scheduled": filtered["scheduled_arrival"],
            "Expected": filtered["expected_arrival"],
            "Status": filtered["status"],
            "Tail": filtered["tail_number"],
            "Type": filtered["aircraft_type"],
            "Delay (min)": filtered["delay_minutes"],
            "Risk": filtered["risk_level"],
            "Reason": filtered["delay_reason"]
        })
        fig = px.timeline(
            timeline_df,
            x_start="Scheduled", x_end="Expected",
            y="Flight", color="Status",
            hover_data=["Tail","Type","Delay (min)","Risk","Reason"],
            color_discrete_map={"ON-TIME":"#10B981","DELAYED":"#F59E0B","EARLY":"#3B82F6"}
        )
        # Add border color by risk
        risk_colors = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue","LOW":"green"}
        for i, d in enumerate(fig.data):
            risks = timeline_df[timeline_df["Status"]==d.name]["Risk"].tolist()
            marker_colors = [risk_colors[r] for r in risks]
            d.marker.line = dict(width=2, color=marker_colors)
        fig.update_yaxes(autorange="reversed")  # Earliest at top
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
if __name__=="__main__":
    create_guardian_eye_streamlit()
