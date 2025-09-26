# guardian_eye_.py
# Backend: Streamlit app + API endpoint for frontend fetch
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

# ---------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------
def generate_realistic_aviation_data(n_flights: int = 5000) -> pd.DataFrame:
    airlines = {
        "Air India": {"code": "AI", "aircraft_types": ["B787", "B777", "A320"], "safety_score": 85},
        "IndiGo": {"code": "6E", "aircraft_types": ["A320", "A321", "ATR72"], "safety_score": 92},
        "SpiceJet": {"code": "SG", "aircraft_types": ["B737", "Q400"], "safety_score": 78},
    }

    airports = {
        "DEL": {"name": "Delhi", "lat": 28.56, "lng": 77.10},
        "BOM": {"name": "Mumbai", "lat": 19.09, "lng": 72.86},
        "BLR": {"name": "Bengaluru", "lat": 13.20, "lng": 77.71},
    }

    def reg():
        return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

    flights = []
    for i in range(n_flights):
        airline = np.random.choice(list(airlines.keys()))
        info = airlines[airline]
        a_type = np.random.choice(info["aircraft_types"])
        tail = reg()

        origin, dest = np.random.choice(list(airports.keys()), 2, replace=False)
        sched = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        sched = sched + timedelta(hours=np.random.randint(5, 23))

        delay = np.random.poisson(10)
        status = "IN-FLIGHT" if np.random.rand() > 0.5 else "SCHEDULED"

        flights.append(
            {
                "flight_id": f"{info['code']}{1000+i}",
                "airline": airline,
                "aircraft_type": a_type,
                "tail_number": tail,
                "origin": origin,
                "destination": dest,
                "scheduled_departure": sched,
                "delay_minutes": delay,
                "status": status,
                "incident_probability": np.random.rand(),
                "risk_level": np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                "current_lat": airports[origin]["lat"] + np.random.uniform(-1, 1),
                "current_lng": airports[origin]["lng"] + np.random.uniform(-1, 1),
            }
        )
    return pd.DataFrame(flights)

# ---------------------------------------------------------------------
# API endpoint for frontend fetch
# ---------------------------------------------------------------------
if "data" in st.query_params and st.query_params["data"] == "flights":
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    else:
        df = generate_realistic_aviation_data(2000)
        df.to_csv(DATA_CSV, index=False)
    st.json(df.to_dict(orient="records"))
    st.stop()

# ---------------------------------------------------------------------
# Main Streamlit app (for debugging / dashboard)
# ---------------------------------------------------------------------
def app():
    st.title("🛡️ Guardian Eye — Backend")
    st.write("This is the backend API for Guardian Eye.")

    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    else:
        df = generate_realistic_aviation_data(2000)
        df.to_csv(DATA_CSV, index=False)

    st.write("### Sample Data")
    st.dataframe(df.head())

if __name__ == "__main__":
    app()
