# guardian_eye_realglobe.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

# -----------------------------
# Reference Data
# -----------------------------
AIRLINES = {
    "Air India": {"code": "AI","types": ["Boeing 787-8","Boeing 777-300ER","Airbus A320neo","Boeing 737-800"],"safety_score": 85},
    "IndiGo": {"code": "6E","types": ["Airbus A320neo","Airbus A321neo","ATR 72-600"],"safety_score": 92},
    "SpiceJet": {"code": "SG","types": ["Boeing 737-800","Boeing 737 MAX 8","Bombardier Q400"],"safety_score": 78},
    "AirAsia India": {"code": "I5","types": ["Airbus A320neo"],"safety_score": 88}
}

AIRPORTS = {
    "DEL": {"name": "Delhi","lat": 28.5562,"lng": 77.1000},
    "BOM": {"name": "Mumbai","lat": 19.0896,"lng": 72.8656},
    "BLR": {"name": "Bangalore","lat": 13.1986,"lng": 77.7066},
    "MAA": {"name": "Chennai","lat": 12.9941,"lng": 80.1709},
    "CCU": {"name": "Kolkata","lat": 22.6547,"lng": 88.4467},
    "HYD": {"name": "Hyderabad","lat": 17.2403,"lng": 78.4294},
    "COK": {"name": "Kochi","lat": 10.1520,"lng": 76.4019},
    "AMD": {"name": "Ahmedabad","lat": 23.0726,"lng": 72.6263},
    "PNQ": {"name": "Pune","lat": 18.5822,"lng": 73.9197},
    "JAI": {"name": "Jaipur","lat": 26.8247,"lng": 75.8127},
}

# -----------------------------
# Data Generation
# -----------------------------
def generate_realistic_aviation_data(n_days: int = 7) -> pd.DataFrame:
    airlines_routes = {
        "Air India": [("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL")],
        "IndiGo": [("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR")],
        "SpiceJet": [("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM")],
        "AirAsia India": [("BLR","DEL"),("DEL","BLR"),("MAA","CCU"),("CCU","MAA")]
    }

    flights = []
    flight_counter = 1000

    for airline, routes in airlines_routes.items():
        info = AIRLINES[airline]
        code = info["code"]
        for origin, dest in routes:
            for d in range(n_days):
                flight_counter += 1
                sched_date = datetime(2024, 1, 1) + timedelta(days=d)
                dep_hour = np.random.randint(6, 23)
                scheduled_departure = sched_date.replace(hour=dep_hour, minute=0, second=0)
                scheduled_arrival = scheduled_departure + timedelta(hours=np.random.randint(1,3))

                # Delay distribution
                r = np.random.rand()
                if r < 0.7: delay = np.random.randint(-5,15)
                elif r < 0.9: delay = np.random.randint(15,60)
                else: delay = np.random.randint(-30,-10)

                expected_arrival = scheduled_arrival + timedelta(minutes=delay)
                status = "ON-TIME"
                if delay > 15: status = "DELAYED"
                elif delay < -10: status = "EARLY"

                # Aircraft
                aircraft_type = np.random.choice(info["types"])
                tail_number = rand_tail()

                # Position interpolation
                o, d_ = AIRPORTS[origin], AIRPORTS[dest]
                frac = np.random.rand()
                current_lat = o["lat"] + (d_["lat"] - o["lat"]) * frac
                current_lng = o["lng"] + (d_["lng"] - o["lng"]) * frac

                flights.append({
                    "flight_id": f"{code}{flight_counter}",
                    "airline": airline,
                    "aircraft_type": aircraft_type,
                    "tail_number": tail_number,
                    "origin": origin,
                    "destination": dest,
                    "scheduled_departure": scheduled_departure,
                    "scheduled_arrival": scheduled_arrival,
                    "expected_arrival": expected_arrival,
                    "delay_minutes": delay,
                    "status": status,
                    "origin_lat": o["lat"],
                    "origin_lng": o["lng"],
                    "dest_lat": d_["lat"],
                    "dest_lng": d_["lng"],
                    "current_lat": current_lat,
                    "current_lng": current_lng
                })
    return pd.DataFrame(flights)

# -----------------------------
# Streamlit Dashboard
# -----------------------------
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye - Real Globe", page_icon="🌍", layout="wide")
    st.title("🛡️ Guardian Eye - Aviation Monitoring with Real Globe 🌍")

    # Load dataset
    csv_path = "aviation_dataset.csv"
    if not os.path.exists(csv_path):
        df = generate_realistic_aviation_data(7)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path, parse_dates=["scheduled_departure","scheduled_arrival","expected_arrival"])

    # Sidebar filters
    airlines = ["All Airlines"] + sorted(df["airline"].unique())
    sel_airline = st.sidebar.selectbox("Select Airline", airlines)
    tmp = df if sel_airline == "All Airlines" else df[df["airline"]==sel_airline]

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Flights", len(tmp))
    with col2: st.metric("On-Time", (tmp["status"]=="ON-TIME").sum())
    with col3: st.metric("Delayed", (tmp["status"]=="DELAYED").sum())

    st.markdown("---")
    st.subheader("🌍 3D Real Globe View")
    fig = go.Figure()

    # Projection = orthographic (true globe)
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="rgb(20, 30, 40)",
        showocean=True, oceancolor="rgb(5, 12, 24)",
        showcountries=True, countrycolor="lightgray",
        center=dict(lat=20, lon=77), scope="asia"
    )

    # Flight routes
    for _, r in tmp.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[r["origin_lng"], r["dest_lng"]],
            lat=[r["origin_lat"], r["dest_lat"]],
            mode="lines",
            line=dict(width=1,color="rgba(100,150,255,0.5)"),
            hoverinfo="skip"
        ))

    # Aircraft positions
    hover_text = [f"✈️ {r['flight_id']} | {r['airline']}<br>{r['origin']} → {r['destination']}<br>⏱️ Delay: {r['delay_minutes']} min" for _, r in tmp.iterrows()]
    fig.add_trace(go.Scattergeo(
        lon=tmp["current_lng"], lat=tmp["current_lat"],
        text=hover_text, mode="markers",
        marker=dict(size=6, color="orange", line=dict(width=1,color="white")),
        hoverinfo="text"
    ))

    fig.update_layout(height=600, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ 2D Regional Map")
    st.map(tmp, latitude="current_lat", longitude="current_lng")

    st.markdown("---")
    st.subheader("📋 Flight Schedule")
    st.dataframe(tmp[["flight_id","airline","aircraft_type","origin","destination","scheduled_departure","expected_arrival","delay_minutes","status"]].head(50), use_container_width=True)

# -----------------------------
# Main
# -----------------------------
def main():
    create_guardian_eye_streamlit()

if __name__ == "__main__":
    main()
