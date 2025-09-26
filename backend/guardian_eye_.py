# guardian_eye.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import streamlit.components.v1 as components

np.random.seed(42)

# -----------------------------
# 1) Synthetic Data Generator
# -----------------------------
def generate_realistic_aviation_data(n_flights=3000):
    airlines_data = {
        'Air India': {
            'code': 'AI',
            'aircraft_types': ['Boeing 787-8', 'Boeing 777-300ER', 'Airbus A320neo', 'Boeing 737-800'],
            'safety_score': 85
        },
        'IndiGo': {
            'code': '6E',
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo', 'ATR 72-600'],
            'safety_score': 92
        },
        'SpiceJet': {
            'code': 'SG',
            'aircraft_types': ['Boeing 737-800', 'Boeing 737 MAX 8', 'Bombardier Q400'],
            'safety_score': 78
        },
        'Vistara': {
            'code': 'UK',
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo', 'Boeing 787-9'],
            'safety_score': 95
        },
        'GoFirst': {
            'code': 'G8',
            'aircraft_types': ['Airbus A320neo', 'Airbus A321neo'],
            'safety_score': 82
        },
        'AirAsia India': {
            'code': 'I5',
            'aircraft_types': ['Airbus A320neo'],
            'safety_score': 88
        }
    }

    # Indian airports (+ extras you confirmed)
    airports_data = {
        'DEL': {'name': 'Delhi',     'lat': 28.5562, 'lng': 77.1000, 'traffic': 'Very High'},
        'BOM': {'name': 'Mumbai',    'lat': 19.0896, 'lng': 72.8656, 'traffic': 'Very High'},
        'BLR': {'name': 'Bangalore', 'lat': 13.1986, 'lng': 77.7066, 'traffic': 'High'},
        'MAA': {'name': 'Chennai',   'lat': 12.9941, 'lng': 80.1709, 'traffic': 'High'},
        'CCU': {'name': 'Kolkata',   'lat': 22.6547, 'lng': 88.4467, 'traffic': 'High'},
        'HYD': {'name': 'Hyderabad', 'lat': 17.2403, 'lng': 78.4294, 'traffic': 'High'},
        'COK': {'name': 'Kochi',     'lat': 10.1520, 'lng': 76.4019, 'traffic': 'Medium'},
        'AMD': {'name': 'Ahmedabad', 'lat': 23.0726, 'lng': 72.6263, 'traffic': 'Medium'},
        'PNQ': {'name': 'Pune',      'lat': 18.5822, 'lng': 73.9197, 'traffic': 'Medium'},
        'JAI': {'name': 'Jaipur',    'lat': 26.8247, 'lng': 75.8127, 'traffic': 'Medium'},
        'GOI': {'name': 'Goa',       'lat': 15.3800, 'lng': 73.8314, 'traffic': 'Medium'},
        'TRV': {'name': 'Trivandrum','lat': 8.4821,  'lng': 76.9200, 'traffic': 'Medium'},
        'IXB': {'name': 'Bagdogra',  'lat': 26.6812, 'lng': 88.3286, 'traffic': 'Low'},
        'IXC': {'name': 'Chandigarh','lat': 30.6735, 'lng': 76.7885, 'traffic': 'Low'},
        'VNS': {'name': 'Varanasi',  'lat': 25.4524, 'lng': 82.8593, 'traffic': 'Low'},
        'SXR': {'name': 'Srinagar',  'lat': 33.9871, 'lng': 74.7740, 'traffic': 'Low'}
    }

    flights = []
    for i in range(n_flights):
        airline = np.random.choice(list(airlines_data.keys()))
        ainfo = airlines_data[airline]
        a_type = np.random.choice(ainfo['aircraft_types'])
        tail = "VT-" + "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))

        origin = np.random.choice(list(airports_data.keys()))
        dest_choices = [k for k in airports_data.keys() if k != origin]
        dest = np.random.choice(dest_choices)

        base_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        sched_dep = base_date + timedelta(hours=np.random.randint(5, 24), minutes=int(np.random.choice([0, 15, 30, 45])))

        # Tech / maintenance / crew / environment
        age = np.random.uniform(1, 20)
        fhours = np.random.uniform(5_000, 80_000)
        cycles = fhours / 1.5
        last_maint_days = np.random.uniform(1, 180)

        month = sched_dep.month
        monsoon = 1.5 if month in [6, 7, 8, 9] else 1.0
        fog = 1.3 if (month in [12, 1, 2] and origin in ['DEL', 'JAI']) else 1.0
        weather_score = min(1.0, np.random.uniform(0.2, 1.0) * monsoon * fog)

        engine_health = max(0, 100 - age * 2 - np.random.uniform(0, 20))
        struct_int = max(0, 100 - age * 1.5 - cycles/1000 - np.random.uniform(0, 15))
        avionics = max(0, 100 - age * 1 - np.random.uniform(0, 10))
        maint_score = max(0, 100 - last_maint_days/2 - np.random.uniform(0, 20))

        pilot_exp_hours = np.random.uniform(500, 15000)
        pilot_exp = min(100, pilot_exp_hours / 150)
        crew_rest = np.random.uniform(8, 24)
        crew_fatigue = max(0, min(100, crew_rest * 4))

        traffic_mult = {'Very High': 1.5, 'High': 1.2, 'Medium': 1.0, 'Low': 0.8}
        atc_p = (traffic_mult[airports_data[origin]['traffic']] + traffic_mult[airports_data[dest]['traffic']]) / 2
        atc_score = max(0, 100 - atc_p * 30 - np.random.uniform(0, 20))

        # Risk components (lower health -> higher risk)
        tech_risk = (100 - engine_health)*0.4 + (100 - struct_int)*0.3 + (100 - avionics)*0.2 + (100 - maint_score)*0.1
        human_risk = (100 - pilot_exp)*0.7 + (100 - crew_fatigue)*0.3
        env_risk = (weather_score*100)*0.7 + (100 - atc_score)*0.3

        incident_p = (tech_risk*0.5 + human_risk*0.3 + env_risk*0.2) / 100
        incident_p *= (100 - ainfo['safety_score']) / 100
        incident_p = float(np.clip(incident_p, 0, 1))

        if incident_p > 0.7:
            risk_level = 'CRITICAL'
        elif incident_p > 0.5:
            risk_level = 'HIGH'
        elif incident_p > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # Delay components & total
        base_delay = np.random.poisson(8)
        weather_delay = np.random.poisson(30) if weather_score > 0.7 else (np.random.poisson(15) if weather_score > 0.5 else 0)
        tech_delay = np.random.poisson(45) if (engine_health < 70 or maint_score < 60) else (np.random.poisson(20) if engine_health < 85 else 0)
        atc_delay = np.random.poisson(25) if atc_score < 70 else (np.random.poisson(10) if atc_score < 85 else 0)
        total_delay = int(base_delay + weather_delay + tech_delay + atc_delay)

        # Status
        if total_delay > 60:
            status = 'DELAYED'
        elif np.random.random() > 0.7:
            status = 'IN-FLIGHT'
        elif np.random.random() > 0.5:
            status = 'COMPLETED'
        else:
            status = 'SCHEDULED'

        actual_dep = sched_dep + timedelta(minutes=total_delay)

        # Reasons (top contributor)
        delay_parts = {
            'Weather': weather_delay,
            'Technical': tech_delay,
            'ATC/Congestion': atc_delay,
            'Base Ops': base_delay
        }
        delay_reason = max(delay_parts, key=delay_parts.get)
        if total_delay == 0:
            delay_reason = 'On-time'

        risk_parts = {
            'Technical condition': tech_risk,
            'Human factors (crew)': human_risk,
            'Environment (weather/ATC)': env_risk
        }
        risk_reason = max(risk_parts, key=risk_parts.get)

        # Random geo positions near origin→dest arc (approx)
        o = airports_data[origin]; d = airports_data[dest]
        t = np.random.rand()
        lat = o['lat'] + (d['lat'] - o['lat']) * t + np.random.uniform(-0.5, 0.5)
        lng = o['lng'] + (d['lng'] - o['lng']) * t + np.random.uniform(-0.5, 0.5)

        flights.append({
            'flight_id': f"{ainfo['code']}{1000+i}",
            'airline': airline,
            'aircraft_type': a_type,
            'tail_number': tail,
            'origin': origin,
            'destination': dest,
            'scheduled_departure': sched_dep,
            'actual_departure': actual_dep,
            'delay_minutes': total_delay,
            'status': status,
            'aircraft_age_years': age,
            'flight_hours': fhours,
            'cycles': cycles,
            'last_maintenance_days': last_maint_days,
            'engine_health': engine_health,
            'structural_integrity': struct_int,
            'avionics_status': avionics,
            'maintenance_score': maint_score,
            'pilot_experience': pilot_exp,
            'crew_fatigue_factor': crew_fatigue,
            'weather_score': weather_score,
            'atc_score': atc_score,
            'technical_risk': tech_risk,
            'human_risk': human_risk,
            'environmental_risk': env_risk,
            'incident_probability': incident_p,
            'risk_level': risk_level,
            'weather_delay': weather_delay,
            'technical_delay': tech_delay,
            'atc_delay': atc_delay,
            'base_delay': base_delay,
            'delay_reason': delay_reason,
            'risk_reason': risk_reason,
            'lat': float(lat),
            'lng': float(lng)
        })

    return pd.DataFrame(flights)

# -----------------------------------
# 2) Minimal training (optional save)
# -----------------------------------
def train_and_cache_models(df):
    features = [
        'aircraft_age_years','flight_hours','cycles','last_maintenance_days',
        'engine_health','structural_integrity','avionics_status','maintenance_score',
        'pilot_experience','crew_fatigue_factor','weather_score','atc_score'
    ]
    X = df[features]
    y_inc = (df['incident_probability'] > 0.5).astype(int)
    y_delay = df['delay_minutes']

    Xtr, Xte, ytr, yte = train_test_split(X, y_inc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(Xtr, ytr)
    _ = clf.predict(Xte)  # not printed to keep UI clean

    _, _, ydtr, ydt = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=120, random_state=42)
    reg.fit(Xtr, ydtr)
    _ = reg.predict(Xte)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/incident_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("artifacts/delay_predictor.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open("artifacts/feature_columns.pkl", "wb") as f:
        pickle.dump(features, f)

# -----------------------------
# 3) Streamlit App
# -----------------------------
st.set_page_config(page_title="🛡️ Guardian Eye", layout="wide")
st.markdown(
    "<h1 style='text-align:center'>🛡️ GUARDIAN EYE – Aviation Operations Center</h1>",
    unsafe_allow_html=True
)

# Load or create data
if not os.path.exists("aviation_dataset.csv"):
    with st.spinner("Generating realistic aviation dataset..."):
        df = generate_realistic_aviation_data(3000)
        df.to_csv("aviation_dataset.csv", index=False)
        train_and_cache_models(df)
else:
    df = pd.read_csv("aviation_dataset.csv", parse_dates=['scheduled_departure','actual_departure'])

# --- Sidebar filters
st.sidebar.header("🎛️ Filters")
airlines = ["All Airlines"] + sorted(df['airline'].unique().tolist())
sel_airline = st.sidebar.selectbox("Airline", airlines)

df_f = df.copy()
if sel_airline != "All Airlines":
    df_f = df_f[df_f['airline'] == sel_airline]

types = ["All Types"] + sorted(df_f['aircraft_type'].unique().tolist())
sel_type = st.sidebar.selectbox("Aircraft Type", types)
if sel_type != "All Types":
    df_f = df_f[df_f['aircraft_type'] == sel_type]

tails = ["All Aircraft"] + sorted(df_f['tail_number'].unique().tolist())
sel_tail = st.sidebar.selectbox("Tail Number", tails)
if sel_tail != "All Aircraft":
    df_f = df_f[df_f['tail_number'] == sel_tail]

# --- Top metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Flights (filtered)", len(df_f))
with c2:
    st.metric("In Flight", int((df_f['status'] == 'IN-FLIGHT').sum()))
with c3:
    st.metric("Delayed > 60 min", int((df_f['delay_minutes'] > 60).sum()))
with c4:
    st.metric("Avg Incident Probability", f"{df_f['incident_probability'].mean()*100:.1f}%")

st.markdown("---")

# -----------------------------
# 3D Globe (Three.js via HTML)
# -----------------------------
st.subheader("🌍 Global Aircraft Tracking (3D)")

# Prepare JSON for globe markers (filtered subset to keep JS light)
df_draw = df_f.copy()
if len(df_draw) > 400:
    # sample for performance
    df_draw = df_draw.sample(400, random_state=42)

def risk_color(level):
    return {
        "CRITICAL": "#FF2D2E",
        "HIGH": "#FF9F1C",
        "MEDIUM": "#2EA8FF",
        "LOW": "#10B981"
    }.get(level, "#10B981")

markers = []
for _, r in df_draw.iterrows():
    markers.append({
        "id": str(r["flight_id"]),
        "tail": r["tail_number"],
        "airline": r["airline"],
        "type": r["aircraft_type"],
        "lat": float(r["lat"]),
        "lng": float(r["lng"]),
        "riskLevel": r["risk_level"],
        "riskColor": risk_color(r["risk_level"]),
        "status": r["status"],
        "delay": int(r["delay_minutes"]),
        "delayReason": r["delay_reason"],
        "riskReason": r["risk_reason"],
        "origin": r["origin"],
        "dest": r["destination"]
    })

globe_html = f"""
<div id="wrap" style="width:100%;height:520px;position:relative;border-radius:12px;overflow:hidden;background:radial-gradient(1200px 600px at 50% 50%, #0b1220 0%, #050a14 60%, #02060d 100%);">
  <div id="tooltip" style="position:absolute; top:12px; right:12px; max-width:360px; background:rgba(0,0,0,0.7); color:#fff; font-family:system-ui,-apple-system,Segoe UI,Roboto; font-size:13px; line-height:1.4; padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.15); display:none; white-space:normal;"></div>
  <canvas id="c"></canvas>
</div>

<script src="https://unpkg.com/three@0.157.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.157.0/examples/js/controls/OrbitControls.js"></script>
<script>
const FLIGHTS = {json.dumps(markers)};

const canvas = document.getElementById('c');
const wrap = document.getElementById('wrap');
const tooltip = document.getElementById('tooltip');

// Renderer
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(wrap.clientWidth, wrap.clientHeight);

// Scene & camera
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 0.1, 1000);
camera.position.set(0, 0, 5.5);

// Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;
controls.minDistance = 4.2;
controls.maxDistance = 9;

// Lights
const ambient = new THREE.AmbientLight(0x88aaff, 0.6);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(5,5,5);
scene.add(dir);

// Earth (textured + glow)
const loader = new THREE.TextureLoader();
const earthTex = loader.load("https://unpkg.com/three-globe/example/img/earth-dark.jpg");
const earthGeo = new THREE.SphereGeometry(2, 64, 64);
const earthMat = new THREE.MeshPhongMaterial({map: earthTex, shininess: 5});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

// Atmosphere glow
const glowGeo = new THREE.SphereGeometry(2.06, 64, 64);
const glowMat = new THREE.MeshBasicMaterial({color: 0x2e5cff, transparent:true, opacity:0.12, blending: THREE.AdditiveBlending});
const glow = new THREE.Mesh(glowGeo, glowMat);
scene.add(glow);

// Helpers
function latLngToXYZ(lat, lng, radius=2.02) {{
  const phi = (90 - lat) * (Math.PI/180);
  const theta = (lng + 180) * (Math.PI/180);
  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}}

// Markers
const markerGroup = new THREE.Group();
scene.add(markerGroup);

const colorHex = (c) => new THREE.Color(c);

FLIGHTS.forEach(f => {{
  const pos = latLngToXYZ(f.lat, f.lng, 2.02);
  const g = new THREE.SphereGeometry(0.035, 12, 12);
  const m = new THREE.MeshBasicMaterial({{color: colorHex(f.riskColor)}});
  const dot = new THREE.Mesh(g, m);
  dot.position.copy(pos);

  // a faint line to surface normal (subtle)
  const lineMat = new THREE.LineBasicMaterial({{color: 0x1f4fff, transparent:true, opacity:0.18}});
  const lineGeo = new THREE.BufferGeometry().setFromPoints([pos.clone().multiplyScalar(0.98), pos]);
  const line = new THREE.Line(lineGeo, lineMat);

  const group = new THREE.Group();
  group.add(dot);
  group.add(line);
  group.userData = f; // stash flight info
  markerGroup.add(group);
}});

// Raycaster for clicks
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onClick(e) {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(markerGroup.children.map(g => g.children[0])); // sphere child

  if (intersects.length > 0) {{
    const obj = intersects[0].object.parent; // group
    const f = obj.userData;
    tooltip.style.display = 'block';
    tooltip.innerHTML = `
      <div style="font-weight:700; margin-bottom:6px; font-size:14px;">
        ${'{'}f.tail{'}'} <span style="opacity:0.7;font-weight:500">(${ '{'}f.airline{'}'} • ${'{'}f.type{'}'})</span>
      </div>
      <div><b>Status:</b> ${'{'}f.status{'}'}</div>
      <div><b>Route:</b> ${'{'}f.origin{'}'} → ${'{'}f.dest{'}'}</div>
      <div><b>Delay (min):</b> ${'{'}f.delay{'}'} ${'{'}f.delay > 0 ? ' – ' + f.delayReason : ''{'}'}</div>
      <div><b>Risk:</b> <span style="color:${'{'}f.riskColor{'}'};font-weight:700">${'{'}f.riskLevel{'}'}</span> – ${'{'}f.riskReason{'}'}</div>
    `;
  }}
}}

function onMouseMove(e) {{
  if (tooltip.style.display !== 'none') {{
    const rect = wrap.getBoundingClientRect();
    tooltip.style.left = (e.clientX - rect.left - 10) + 'px';
    tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
  }}
}}

function onLeave() {{
  // keep tooltip until next click; comment line below if you want auto-hide on leave
  // tooltip.style.display = 'none';
}}

renderer.domElement.addEventListener('click', onClick);
renderer.domElement.addEventListener('mousemove', onMouseMove);
renderer.domElement.addEventListener('mouseleave', onLeave);

// Resize
function onResize() {{
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', onResize);

// Animate
function animate() {{
  requestAnimationFrame(animate);
  earth.rotation.y += 0.0009;    // gentle auto-rotate
  glow.rotation.y += 0.0009;
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
"""

components.html(globe_html, height=540, scrolling=False)

st.caption(
    "Click any marker to see details. Markers are colored by risk: "
    "🔴 Critical, 🟠 High, 🔵 Medium, 🟢 Low."
)

st.markdown("---")

# -----------------------------
# 4) Fleet table (filtered)
# -----------------------------
st.subheader("✈️ Active Aircraft (Filtered)")
show_cols = [
    'flight_id','tail_number','airline','aircraft_type','origin','destination',
    'status','delay_minutes','delay_reason','risk_level','risk_reason',
    'incident_probability'
]
df_show = df_f[show_cols].copy()
df_show['incident_probability'] = (df_show['incident_probability']*100).round(1).astype(str) + '%'
st.dataframe(df_show.head(200), use_container_width=True)

# -----------------------------
# 5) Helpful notes
# -----------------------------
with st.expander("ℹ️ What counts as the reason for delay & risk?"):
    st.write("""
- **Delay reason** = largest contributor among `Weather`, `Technical`, `ATC/Congestion`, `Base Ops`.
- **Risk reason** = largest component among `Technical condition`, `Human factors (crew)`, `Environment (weather/ATC)`.
- The globe uses a **dark earth texture + glow** for a clear, modern ops-center look.
""")
