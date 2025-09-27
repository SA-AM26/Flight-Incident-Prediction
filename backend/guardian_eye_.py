import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# -------------------
# Helpers
# -------------------
def py_int(x): return int(np.asarray(x).item() if np.ndim(x) == 0 else x)
def py_float(x): return float(np.asarray(x).item() if np.ndim(x) == 0 else x)
def rand_tail(): return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

# -------------------
# Data Generation
# -------------------
def generate_realistic_aviation_data(n_days=30):
    airlines_routes = {
        "Air India": [("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL")],
        "IndiGo": [("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR")],
        "SpiceJet": [("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM")],
        "AirAsia India": [("BLR","DEL"),("DEL","BLR"),("MAA","CCU"),("CCU","MAA")]
    }

    airports_data = {
        'DEL': {'name': 'Delhi','lat': 28.5562,'lng': 77.1000},
        'BOM': {'name': 'Mumbai','lat': 19.0896,'lng': 72.8656},
        'BLR': {'name': 'Bangalore','lat': 13.1986,'lng': 77.7066},
        'MAA': {'name': 'Chennai','lat': 12.9941,'lng': 80.1709},
        'CCU': {'name': 'Kolkata','lat': 22.6547,'lng': 88.4467},
        'HYD': {'name': 'Hyderabad','lat': 17.2403,'lng': 78.4294},
        'COK': {'name': 'Kochi','lat': 10.1520,'lng': 76.4019}
    }

    flights_data, flight_counter = [], 100

    for airline, routes in airlines_routes.items():
        for origin, dest in routes:
            for d in range(n_days):
                flight_counter += 1
                flight_id = f"{airline[:2].upper()}{flight_counter}"

                sched_dep = datetime(2024,1,1) + timedelta(days=d, hours=np.random.randint(6,23))
                sched_arr = sched_dep + timedelta(hours=np.random.randint(1,3))

                r = np.random.rand()
                if r < 0.75: delay = np.random.randint(-5,15)
                elif r < 0.92: delay = np.random.randint(15,120)
                else: delay = np.random.randint(-30,-10)
                exp_arr = sched_arr + timedelta(minutes=delay)

                if delay > 15: status = "DELAYED"
                elif delay < -10: status = "EARLY"
                else: status = "ON-TIME"

                age = py_float(np.random.uniform(1,20))
                flight_hours = py_float(np.random.uniform(5000,80000))
                cycles = flight_hours / 1.5
                maint_days = py_float(np.random.uniform(1,180))

                engine = py_float(max(0, 100 - age*2 - np.random.uniform(0,20)))
                struct = py_float(max(0, 100 - age*1.5 - cycles/1000 - np.random.uniform(0,15)))
                avio = py_float(max(0, 100 - age*1 - np.random.uniform(0,10)))
                maint = py_float(max(0, 100 - maint_days/2 - np.random.uniform(0,20)))

                pilot_exp = py_float(np.random.uniform(500,15000)/150)
                crew_fatigue = py_float(np.random.uniform(40,100))

                tech_risk = (100-engine)*0.4 + (100-struct)*0.3 + (100-avio)*0.2 + (100-maint)*0.1
                human_risk = (100-pilot_exp)*0.7 + (100-crew_fatigue)*0.3
                env_risk = abs(delay)*0.5
                incident_prob = (tech_risk*0.5 + human_risk*0.3 + env_risk*0.2)/100
                incident_prob = min(1.0,max(0.0,incident_prob))

                if incident_prob > 0.7: risk = "CRITICAL"
                elif incident_prob > 0.5: risk = "HIGH"
                elif incident_prob > 0.3: risk = "MEDIUM"
                else: risk = "LOW"

                flights_data.append({
                    "flight_id": flight_id,
                    "airline": airline,
                    "aircraft_type": np.random.choice(["Airbus A320neo","A321neo","B737-800","ATR72"]),
                    "tail_number": rand_tail(),
                    "origin": origin,
                    "destination": dest,
                    "scheduled_departure": sched_dep,
                    "scheduled_arrival": sched_arr,
                    "expected_arrival": exp_arr,
                    "delay_minutes": delay,
                    "status": status,
                    "incident_probability": incident_prob,
                    "risk_level": risk
                })

    return pd.DataFrame(flights_data)

# -------------------
# Streamlit Dashboard
# -------------------
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye - Aviation Ops Center", layout="wide")
    st_autorefresh(interval=1000, key="refresh")

    now = datetime.now()
    st.markdown(f"<h4 style='text-align:right;color:#3B82F6;'>🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}</h4>", unsafe_allow_html=True)
    st.title("🛡️ Guardian Eye - Aviation Operations Center")
    st.caption("Real-time Flight Safety Monitoring & Risk Assessment")

    # ✅ Check if dataset exists and matches required schema
    required_cols = {"scheduled_departure","scheduled_arrival","expected_arrival"}
    if not os.path.exists("aviation_dataset.csv"):
        df = generate_realistic_aviation_data(30)
        df.to_csv("aviation_dataset.csv", index=False)
    else:
        df = pd.read_csv("aviation_dataset.csv")
        if not required_cols.issubset(df.columns):
            st.warning("⚠️ Old dataset detected. Regenerating with updated schema...")
            df = generate_realistic_aviation_data(30)
            df.to_csv("aviation_dataset.csv", index=False)
        else:
            df = pd.read_csv("aviation_dataset.csv", parse_dates=list(required_cols))

    # Sidebar filters
    airlines = ["All"] + sorted(df["airline"].unique())
    sel_airline = st.sidebar.selectbox("Airline", airlines)
    fdf = df if sel_airline=="All" else df[df["airline"]==sel_airline]

    types = ["All"] + sorted(fdf["aircraft_type"].unique())
    sel_type = st.sidebar.selectbox("Aircraft Type", types)
    if sel_type!="All": fdf = fdf[fdf["aircraft_type"]==sel_type]

    # Metrics
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Total Flights", len(fdf))
    with col2: st.metric("On-Time %", f"{(fdf['status'].eq('ON-TIME').mean()*100):.1f}%")
    with col3: st.metric("Delayed %", f"{(fdf['status'].eq('DELAYED').mean()*100):.1f}%")
    with col4: st.metric("Critical Risk", (fdf["risk_level"]=="CRITICAL").sum())

    # Flight table
    st.subheader("✈️ Flight Monitor")
    if len(fdf):
        upcoming_window = now + timedelta(hours=2)
        show_df = fdf.copy()
        show_df["Upcoming?"] = show_df["scheduled_departure"].apply(lambda t: now <= pd.to_datetime(t) <= upcoming_window)
        show_df = show_df.sort_values(by=["Upcoming?","scheduled_departure"], ascending=[False, True])
        show_df = show_df[[
            "flight_id","airline","aircraft_type","origin","destination",
            "scheduled_departure","scheduled_arrival","expected_arrival",
            "status","risk_level","delay_minutes","Upcoming?"
        ]]

        def highlight_row(row):
            if row["risk_level"] == "CRITICAL":
                return ["background-color: rgba(220,38,38,0.5); color:white; font-weight:bold;"] * len(row)
            elif row["Upcoming?"]:
                return ["background-color: rgba(16,185,129,0.3);"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(show_df.style.apply(highlight_row, axis=1), use_container_width=True)

# -------------------
if __name__ == "__main__":
    create_guardian_eye_streamlit()
