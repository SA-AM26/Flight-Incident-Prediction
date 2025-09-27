# guardian_eye.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Helpers
# -----------------------------
np.random.seed(42)

def py_int(x): return int(np.asarray(x).item() if np.ndim(x) == 0 else x)
def py_float(x): return float(np.asarray(x).item() if np.ndim(x) == 0 else x)
def rand_tail(): return "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

# -----------------------------
# Airlines & Airports
# -----------------------------
AIRLINES = {
    'Air India':     {'code':'AI','types':['Boeing 787-8','Boeing 777-300ER','Airbus A320neo','Boeing 737-800'],'safety_score':85},
    'IndiGo':        {'code':'6E','types':['Airbus A320neo','Airbus A321neo','ATR 72-600'],'safety_score':92},
    'SpiceJet':      {'code':'SG','types':['Boeing 737-800','Boeing 737 MAX 8','Bombardier Q400'],'safety_score':78},
    'AirAsia India': {'code':'I5','types':['Airbus A320neo'],'safety_score':88}
}

AIRPORTS = {
    'DEL': {'name':'Delhi','lat':28.5562,'lng':77.1000,'traffic':'Very High'},
    'BOM': {'name':'Mumbai','lat':19.0896,'lng':72.8656,'traffic':'Very High'},
    'BLR': {'name':'Bangalore','lat':13.1986,'lng':77.7066,'traffic':'High'},
    'MAA': {'name':'Chennai','lat':12.9941,'lng':80.1709,'traffic':'High'},
    'CCU': {'name':'Kolkata','lat':22.6547,'lng':88.4467,'traffic':'High'},
    'HYD': {'name':'Hyderabad','lat':17.2403,'lng':78.4294,'traffic':'High'},
    'COK': {'name':'Kochi','lat':10.1520,'lng':76.4019,'traffic':'Medium'},
    'AMD': {'name':'Ahmedabad','lat':23.0726,'lng':72.6263,'traffic':'Medium'},
    'PNQ': {'name':'Pune','lat':18.5822,'lng':73.9197,'traffic':'Medium'},
    'JAI': {'name':'Jaipur','lat':26.8247,'lng':75.8127,'traffic':'Medium'}
}

TRAFFIC_MULT = {'Very High':1.5,'High':1.2,'Medium':1.0,'Low':0.8}

# -----------------------------
# Data Generator
# -----------------------------
def generate_realistic_aviation_data(n_days:int=30)->pd.DataFrame:
    airlines_routes = {
        "Air India":[("DEL","BOM"),("BOM","DEL"),("DEL","BLR"),("BLR","DEL")],
        "IndiGo":[("DEL","MAA"),("MAA","DEL"),("BLR","DEL"),("DEL","BLR")],
        "SpiceJet":[("DEL","COK"),("COK","DEL"),("BOM","HYD"),("HYD","BOM")],
        "AirAsia India":[("BLR","DEL"),("DEL","BLR"),("MAA","CCU"),("CCU","MAA")]
    }
    flights=[]; fcount=1000

    for airline,routes in airlines_routes.items():
        info=AIRLINES[airline]; code=info['code']
        for (o,d) in routes:
            block=np.random.randint(90,180)
            for day in range(n_days):
                fcount+=1
                base=datetime(2024,1,1)+timedelta(days=day)
                dep=base.replace(hour=np.random.randint(6,23),minute=np.random.choice([0,15,30,45]))
                arr=dep+timedelta(minutes=block)

                r=np.random.rand()
                if r<0.70: delay=np.random.randint(-10,11)
                elif r<0.90: delay=np.random.randint(15,46)
                elif r<0.98: delay=np.random.randint(45,121)
                else: delay=np.random.randint(-30,-11)

                atype=np.random.choice(info['types']); tail=rand_tail()

                age=py_float(np.random.uniform(1,20)); fh=py_float(np.random.uniform(5000,80000))
                cycles=py_float(fh/1.6); lm=py_float(np.random.uniform(1,180))
                eng=py_float(max(0,100-age*2-np.random.uniform(0,20)))
                struc=py_float(max(0,100-age*1.3-cycles/1000-np.random.uniform(0,12)))
                avio=py_float(max(0,100-age*0.8-np.random.uniform(0,8)))
                maint=py_float(max(0,100-lm/2-np.random.uniform(0,15)))
                exp=py_float(min(100, np.random.uniform(800,12000)/120))
                fatigue=py_float(np.random.uniform(8,24)*4)
                monsoon=1.3 if dep.month in [6,7,8,9] else 1.0
                fog=1.2 if dep.month in [12,1,2] and o in ["DEL","JAI"] else 1.0
                wscore=py_float(min(1.0,np.random.uniform(0.2,1.0)*monsoon*fog))
                atc=py_float(max(0,100-(TRAFFIC_MULT[AIRPORTS[o]['traffic']]+TRAFFIC_MULT[AIRPORTS[d]['traffic']])/2*25-np.random.uniform(0,15)))

                tech=(100-eng)*0.4+(100-struc)*0.3+(100-avio)*0.2+(100-maint)*0.1
                human=(100-exp)*0.7+(100-fatigue)*0.3
                env=(wscore*100)*0.7+(100-atc)*0.3
                inc=((tech*0.5+human*0.3+env*0.2)/100)*(100-info['safety_score'])/100
                risk="LOW"
                if inc>0.7:risk="CRITICAL"
                elif inc>0.5:risk="HIGH"
                elif inc>0.3:risk="MEDIUM"

                expected=arr+timedelta(minutes=delay)
                status="ON-TIME"
                if delay>15: status="DELAYED"
                elif delay<-10: status="EARLY"

                wd=np.random.poisson(10) if wscore>0.7 else (np.random.poisson(5) if wscore>0.5 else 0)
                td=np.random.poisson(25) if (eng<70 or maint<60) else (np.random.poisson(10) if eng<85 else 0)
                ad=np.random.poisson(15) if atc<70 else (np.random.poisson(7) if atc<85 else 0)
                reasons={"Weather":wd,"Technical":td,"ATC":ad,"Other":max(0,delay-(wd+td+ad))}
                dom=max(reasons,key=reasons.get)

                frac=np.random.rand()
                ol,dl=AIRPORTS[o],AIRPORTS[d]
                lat=ol['lat']+(dl['lat']-ol['lat'])*frac
                lng=ol['lng']+(dl['lng']-ol['lng'])*frac

                flights.append({
                    "flight_id":f"{code}{fcount}","airline":airline,"aircraft_type":atype,"tail_number":tail,
                    "origin":o,"destination":d,"origin_name":ol['name'],"destination_name":dl['name'],
                    "scheduled_departure":dep,"scheduled_arrival":arr,"expected_arrival":expected,
                    "delay_minutes":int(delay),"status":status,
                    "incident_probability":inc,"risk_level":risk,
                    "origin_lat":ol['lat'],"origin_lng":ol['lng'],"dest_lat":dl['lat'],"dest_lng":dl['lng'],
                    "current_lat":lat,"current_lng":lng,
                    "dominant_delay_reason":dom,"dominant_delay_minutes":reasons[dom]
                })
    return pd.DataFrame(flights)

# -----------------------------
# Dashboard
# -----------------------------
def create_guardian_eye_streamlit():
    st.set_page_config(page_title="Guardian Eye", page_icon="🛡️", layout="wide")

    # Auto-refresh every 1 second
    st_autorefresh(interval=1000, key="clockref")

    # Header with live clock
    col1,col2=st.columns([3,1])
    with col1:
        st.title("🛡️ Guardian Eye - Aviation Operations Center")
    with col2:
        now=datetime.now()
        st.markdown(f"<h3 style='text-align:right;color:#3B82F6;'>{now.strftime('%H:%M:%S')}</h3>",unsafe_allow_html=True)
        st.caption(now.strftime("%A, %d %B %Y"))

    # Load data
    if not os.path.exists("aviation_dataset.csv"):
        df=generate_realistic_aviation_data(30)
        df.to_csv("aviation_dataset.csv",index=False)
    else:
        df=pd.read_csv("aviation_dataset.csv",parse_dates=["scheduled_departure","scheduled_arrival","expected_arrival"])

    # Filters
    st.sidebar.subheader("Filters")
    airline=st.sidebar.selectbox("Airline",["All"]+sorted(df["airline"].unique()))
    tmp=df if airline=="All" else df[df["airline"]==airline]

    # KPIs
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Flights",len(tmp))
    with c2: st.metric("On-Time",int((tmp["status"]=="ON-TIME").sum()))
    with c3: st.metric("Delayed",int((tmp["status"]=="DELAYED").sum()))
    with c4: st.metric("Critical Risk",int((tmp["risk_level"]=="CRITICAL").sum()))

    # Globe Map
    if len(tmp):
        fig=go.Figure()
        for _,r in tmp.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[r["origin_lng"],r["dest_lng"]],lat=[r["origin_lat"],r["dest_lat"]],
                mode="lines",line=dict(width=1,color="rgba(100,150,255,0.4)"),showlegend=False,hoverinfo="skip"))
        colors={"LOW":"#10B981","MEDIUM":"#3B82F6","HIGH":"#F59E0B","CRITICAL":"#DC2626"}
        fig.add_trace(go.Scattergeo(
            lon=tmp["current_lng"],lat=tmp["current_lat"],
            text=tmp.apply(lambda r:f"{r['flight_id']} | {r['airline']} ({r['status']})<br>Delay: {r['delay_minutes']} min | Risk: {r['risk_level']}",axis=1),
            mode="markers",marker=dict(size=6,color=tmp["risk_level"].map(colors)),hoverinfo="text"))
        fig.update_geos(projection_type="orthographic",showland=True,landcolor="rgb(20,30,40)",oceancolor="rgb(5,12,24)",showocean=True)
        st.plotly_chart(fig,use_container_width=True)

    # Table
    st.subheader("✈️ Flight Schedule vs Expected")
    show=tmp.sort_values("scheduled_departure")[["flight_id","airline","aircraft_type","origin","destination","scheduled_departure","scheduled_arrival","expected_arrival","status","delay_minutes","risk_level","dominant_delay_reason","dominant_delay_minutes"]].copy()
    st.dataframe(show,use_container_width=True)

# -----------------------------
# Entry
# -----------------------------
if __name__=="__main__":
    create_guardian_eye_streamlit()
