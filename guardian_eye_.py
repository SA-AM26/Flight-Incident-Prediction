import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time

np.random.seed(42)

# ----------------------------
# Generate Synthetic Aviation Data
# ----------------------------
def generate_data(n=10):  
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoFirst", "AirAsia India"]
    aircrafts = ["ATR72", "B737", "A350", "A320", "B787", "A321"]
    airports = {
        "DEL": (28.5562, 77.1000),
        "BOM": (19.0896, 72.8656),
        "BLR": (13.1986, 77.7066),
        "MAA": (12.9941, 80.1709),
        "CCU": (22.6547, 88.4467),
        "HYD": (17.2403, 78.4294),
        "COK": (10.1520, 76.4019),
        "AMD": (23.0726, 72.6263),
        "PNQ": (18.5822, 73.9197),
        "JAI": (26.8247, 75.8127),
    }

    rows = []
    for i in range(n):
        airline = np.random.choice(airlines)
        ac = np.random.choice(aircrafts)
        origin, dest = np.random.choice(list(airports.keys()), 2, replace=False)
        lat_o, lon_o = airports[origin]
        lat_d, lon_d = airports[dest]

        prob = np.random.rand()
        if prob > 0.7:
            risk = "CRITICAL"; delay = np.random.randint(60,180); reason="Engine anomaly + bad weather"
        elif prob > 0.5:
            risk = "HIGH"; delay = np.random.randint(30,90); reason="Weather + ATC congestion"
        elif prob > 0.3:
            risk = "MEDIUM"; delay = np.random.randint(10,40); reason="Moderate weather issues"
        else:
            risk = "LOW"; delay = np.random.randint(0,15); reason="Normal ops"

        rows.append({
            "flight_id": f"FL{i+1000}",
            "airline": airline,
            "aircraft_type": ac,
            "origin": origin, "dest": dest,
            "lat_o": lat_o, "lon_o": lon_o,
            "lat_d": lat_d, "lon_d": lon_d,
            "risk_level": risk, "incident_probability": prob,
            "delay_minutes": delay, "reason": reason,
        })
    return pd.DataFrame(rows)

# ----------------------------
# Great Circle Arc Generator
# ----------------------------
def great_circle_points(lat1, lon1, lat2, lon2, n_points=40):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = 2*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
    f = np.linspace(0,1,n_points)
    A = np.sin((1-f)*d)/np.sin(d)
    B = np.sin(f*d)/np.sin(d)
    x = A*np.cos(lat1)*np.cos(lon1)+B*np.cos(lat2)*np.cos(lon2)
    y = A*np.cos(lat1)*np.sin(lon1)+B*np.cos(lat2)*np.sin(lon2)
    z = A*np.sin(lat1)+B*np.sin(lat2)
    return x, y, z

# ----------------------------
# Dashboard
# ----------------------------
st.set_page_config(page_title="Guardian Eye 3D", layout="wide")
st.title("🛡️ Guardian Eye - 3D Animated Aviation Dashboard")

df = generate_data(8)

# Risk colors
color_map = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue","LOW":"green"}

# Precompute flight arcs
flight_arcs = {}
for _, r in df.iterrows():
    x_arc, y_arc, z_arc = great_circle_points(r["lat_o"], r["lon_o"], r["lat_d"], r["lon_d"], n_points=40)
    flight_arcs[r["flight_id"]] = (x_arc, y_arc, z_arc, r)

# Placeholders for animation
col1, col2 = st.columns([3,1])
with col1:
    chart_area = st.empty()
with col2:
    status_area = st.empty()

# Animation loop
n_steps=40
for step in range(n_steps):
    traces=[]
    # Draw Earth
    theta = np.linspace(0,2*np.pi,50)
    phi = np.linspace(0,np.pi,50)
    x = np.outer(np.sin(phi), np.cos(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.cos(phi), np.ones(50))
    traces.append(go.Surface(x=x,y=y,z=z,colorscale="Blues",opacity=0.3,showscale=False))

    # Track statuses
    status_list=[]
    for fid,(x_arc,y_arc,z_arc,r) in flight_arcs.items():
        # Arc path
        traces.append(go.Scatter3d(x=x_arc,y=y_arc,z=z_arc,mode="lines",
                                   line=dict(color=color_map[r["risk_level"]],width=2),opacity=0.2,showlegend=False))
        # Marker
        if step < n_steps-1:
            status = "IN-FLIGHT"; marker_color=color_map[r["risk_level"]]
        else:
            if r["delay_minutes"] > 30: status="DELAYED"; marker_color="red"
            else: status="COMPLETED"; marker_color="green"

        traces.append(go.Scatter3d(
            x=[x_arc[step]], y=[y_arc[step]], z=[z_arc[step]],
            mode="markers",
            marker=dict(size=5,color=marker_color),
            name=fid
        ))
        status_list.append(f"🛫 {fid} | {r['airline']} | {r['origin']}→{r['dest']} | {status} | Delay: {r['delay_minutes']} min")

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      paper_bgcolor="black",margin=dict(l=0,r=0,t=0,b=0))

    chart_area.plotly_chart(fig, use_container_width=True)
    with status_area:
        st.subheader("📋 Live Flight Status Panel")
        for s in status_list:
            st.write(s)

    time.sleep(0.3)  # controls animation speed

# Flight detail viewer
st.subheader("✈️ Flight Deep Dive")
sel_id = st.selectbox("Select Flight", ["None"]+list(df["flight_id"].unique()))
if sel_id != "None":
    r = df[df["flight_id"]==sel_id].iloc[0]
    st.write(f"**Airline:** {r['airline']}")
    st.write(f"**Type:** {r['aircraft_type']}")
    st.write(f"**Route:** {r['origin']} → {r['dest']}")
    st.write(f"**Risk Level:** {r['risk_level']} ({r['incident_probability']*100:.1f}%)")
    st.write(f"**Delay Minutes:** {r['delay_minutes']}")
    st.write(f"**Reason:** {r['reason']}")
