import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px

# ---------------------------------------------------------
# Accident history reference database
# ---------------------------------------------------------
accident_history = {
    "ENGINE FAILURE": "Past incidents (Air India Express 2020, SpiceJet 2021) show engine issues often ignored during short-haul operations.",
    "POOR MAINTENANCE": "Go First 2019 & IndiGo 2022: Inadequate maintenance checks before departure caused diversions.",
    "STRUCTURAL FATIGUE": "Air India 2009: Fatigue cracks in fuselage contributed to major accidents.",
    "AVIONICS FAILURE": "SpiceJet 2015: Avionics system malfunction led to emergency landing.",
    "CREW FATIGUE": "Colgan Air 2009: Pilot fatigue directly contributed to crash in New York.",
    "WEATHER HAZARD": "Mangalore 2010: Poor visibility & wet runway led to runway excursion."
}

# ---------------------------------------------------------
# Generate synthetic aviation dataset
# ---------------------------------------------------------
def generate_realistic_aviation_data(n_flights=3000):
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]
    aircraft_types = ["Boeing 737-800", "Boeing 737 MAX 8", "Airbus A320neo", "Airbus A321", "ATR 72-600", "Boeing 787-8"]
    airports = ["DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "AMD", "PNQ", "JAI"]

    flights = []
    for i in range(n_flights):
        airline = np.random.choice(airlines)
        aircraft_type = np.random.choice(aircraft_types)
        origin, destination = np.random.choice(airports, 2, replace=False)

        dep_time = datetime(2024, 1, 1) + timedelta(
            days=np.random.randint(0, 60),
            hours=np.random.randint(0, 24),
            minutes=np.random.choice([0, 15, 30, 45])
        )
        delay = np.random.poisson(15)
        arr_time = dep_time + timedelta(hours=np.random.randint(1, 4), minutes=delay)

        # Risk factors
        engine = np.random.randint(50, 100)
        maintenance = np.random.randint(40, 100)
        fatigue = np.random.randint(30, 100)
        weather = np.random.randint(20, 100)

        incident_prob = (100 - engine) * 0.3 + (100 - maintenance) * 0.3 + (100 - fatigue) * 0.2 + (100 - weather) * 0.2
        incident_prob = incident_prob / 100
        if incident_prob > 0.7:
            risk = "CRITICAL"
        elif incident_prob > 0.5:
            risk = "HIGH"
        elif incident_prob > 0.3:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        reason = None
        if risk in ["CRITICAL", "HIGH"]:
            reason = np.random.choice(list(accident_history.keys()))

        flights.append({
            "Flight ID": f"{airline[:2].upper()}{1000+i}",
            "Airline": airline,
            "Aircraft Type": aircraft_type,
            "Origin": origin,
            "Destination": destination,
            "Departure Time": dep_time,
            "Arrival Time": arr_time,
            "Delay (min)": delay,
            "Risk Level": risk,
            "Incident Probability": round(incident_prob*100, 2),
            "Incident Reason": reason
        })

    return pd.DataFrame(flights)

# ---------------------------------------------------------
# Streamlit Dashboard
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="🛡️ Guardian Eye Dashboard", layout="wide")
    st.title("🛡️ Guardian Eye – Aviation Safety & Delay Monitoring")

    # Live clock
    now = datetime.now()
    st.markdown(f"⏱️ **{now.strftime('%H:%M:%S')}** | 📅 {now.strftime('%d %B %Y')}")

    # Load / Generate data
    if not os.path.exists("aviation_dataset.csv"):
        df = generate_realistic_aviation_data()
        df.to_csv("aviation_dataset.csv", index=False)
    else:
        df = pd.read_csv("aviation_dataset.csv", parse_dates=["Departure Time", "Arrival Time"])

    st.sidebar.header("🎛️ Filters")
    airlines = ["All"] + sorted(df["Airline"].unique())
    selected_airline = st.sidebar.selectbox("Airline", airlines)

    aircraft_types = ["All"] + sorted(df["Aircraft Type"].unique())
    selected_type = st.sidebar.selectbox("Aircraft Type", aircraft_types)

    origins = ["All"] + sorted(df["Origin"].unique())
    selected_origin = st.sidebar.selectbox("Origin", origins)

    destinations = ["All"] + sorted(df["Destination"].unique())
    selected_dest = st.sidebar.selectbox("Destination", destinations)

    # Apply filters
    filtered_df = df.copy()
    if selected_airline != "All":
        filtered_df = filtered_df[filtered_df["Airline"] == selected_airline]
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["Aircraft Type"] == selected_type]
    if selected_origin != "All":
        filtered_df = filtered_df[filtered_df["Origin"] == selected_origin]
    if selected_dest != "All":
        filtered_df = filtered_df[filtered_df["Destination"] == selected_dest]

    st.metric("Total Flights Loaded", len(filtered_df))

    # Risk level distribution
    st.subheader("📊 Fleet Risk Overview")
    risk_counts = filtered_df["Risk Level"].value_counts()
    st.bar_chart(risk_counts)

    # Flight table
    st.subheader("✈️ Flight Monitor")
    st.dataframe(filtered_df[[
        "Flight ID", "Airline", "Aircraft Type", "Origin", "Destination",
        "Departure Time", "Arrival Time", "Delay (min)", "Risk Level", "Incident Probability", "Incident Reason"
    ]].sort_values("Incident Probability", ascending=False).head(20))

    # High risk insights
    high_risk = filtered_df[filtered_df["Risk Level"].isin(["CRITICAL", "HIGH"])]
    if not high_risk.empty:
        st.subheader("⚠️ High-Risk Flights – Immediate Action Required")
        for _, row in high_risk.iterrows():
            reason = row["Incident Reason"]
            history = accident_history.get(reason, "No history found.")
            st.error(f"Flight {row['Flight ID']} ({row['Airline']}, {row['Aircraft Type']})\n"
                     f"- **Risk Level:** {row['Risk Level']} ({row['Incident Probability']}%)\n"
                     f"- **Reason:** {reason}\n"
                     f"- **Accident History:** {history}\n"
                     f"- **Action:** Inspect systems before next departure!")

if __name__ == "__main__":
    main()
