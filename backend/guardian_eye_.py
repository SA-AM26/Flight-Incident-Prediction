import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

np.random.seed(42)

# ----------------------------
# Generate Synthetic Aviation Data
# ----------------------------
def generate_data(n=12):  
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

df = generate_data(12)

# Globe Base
theta = np.linspace(0,2*np.pi,50)
phi = np.linspace(0,np.pi,50)
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones(50))
earth = go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.3, showscale=False)

# Risk colors
color_map = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue","LOW":"green"}

# Precompute flight arcs
flight_arcs = {}
for _, r in df.iterrows():
    x_arc, y_arc, z_arc = great_circle_points(r["lat_o"], r["lon_o"], r["lat_d"], r["lon_d"], n_points=40)
    flight_arcs[r["flight_id"]] = (x_arc, y_arc, z_arc, r)

# Create animation frames
frames=[]
n_steps=40
for step in range(n_steps):
    traces=[earth]
    for fid,(x_arc,y_arc,z_arc,r) in flight_arcs.items():
        # draw arc
        traces.append(go.Scatter3d(x=x_arc,y=y_arc,z=z_arc,mode="lines",
                                   line=dict(color=color_map[r["risk_level"]],width=2),opacity=0.3,showlegend=False))
        # Determine status dynamically
        if step < n_steps-1:
            status = "IN-FLIGHT"
            marker_color = color_map[r["risk_level"]]
        else:
            if r["delay_minutes"] > 30:
                status = "DELAYED"; marker_color = "red"
            else:
                status = "COMPLETED"; marker_color = "green"
        # moving aircraft marker
        traces.append(go.Scatter3d(
            x=[x_arc[step]],y=[y_arc[step]],z=[z_arc[step]],
            mode="markers+text",
            text=[f"{fid} ({status})"],
            textposition="top center",
            marker=dict(size=5,color=marker_color),
            showlegend=False))
    frames.append(go.Frame(data=traces, name=f"step{step}"))

# Build figure
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    paper_bgcolor="black",
    margin=dict(l=0,r=0,t=0,b=0),
    updatemenus=[dict(type="buttons",
        buttons=[dict(label="▶️ Play", method="animate", args=[None, {"frame":{"duration":200,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]),
                 dict(label="⏸️ Pause", method="animate", args=[[None], {"frame":{"duration":0,"redraw":False},"mode":"immediate"}])])]
)

# Layout with 2 columns
col1, col2 = st.columns([3,1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Live Flight Status Panel")
    for _, r in df.iterrows():
        if r["delay_minutes"] > 30:
            final_status = "DELAYED"
            color = "🔴"
        else:
            final_status = "COMPLETED"
            color = "🟢"
        st.write(f"{color} **{r['flight_id']}** | {r['airline']} | {r['origin']} → {r['dest']} | {final_status} ({r['delay_minutes']} min)")

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
