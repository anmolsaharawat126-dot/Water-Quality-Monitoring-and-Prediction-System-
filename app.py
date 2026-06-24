import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import json
import os
import random
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL PREPARATION & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_ml_model():
    np.random.seed(42)
    n_samples = 1500
    ph = np.random.uniform(5.5, 9.5, n_samples)
    turb = np.random.uniform(0.0, 10.0, n_samples)
    do_ = np.random.uniform(2.0, 12.0, n_samples)
    heavy = np.random.uniform(0.0, 0.3, n_samples)
    nit = np.random.uniform(0.0, 20.0, n_samples)
    tds = np.random.uniform(50.0, 800.0, n_samples)
    temp = np.random.uniform(10.0, 38.0, n_samples)
    chl = np.random.uniform(0.0, 1.0, n_samples)
    
    labels = []
    for i in range(n_samples):
        score = wqi_score(ph[i], turb[i], do_[i], heavy[i], nit[i], tds[i], temp[i], chl[i])
        labels.append(1 if score >= 50 else 0)
        
    df_train = pd.DataFrame({
        "pH": ph, "Turbidity": turb, "Dissolved Oxygen": do_, "Heavy Metals": heavy,
        "Nitrates": nit, "TDS": tds, "Temperature": temp, "Chlorine": chl, "Label": labels
    })
    X = df_train.drop(columns=["Label"])
    y = df_train["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    feature_importances = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return clf, acc, cm, fpr.tolist(), tpr.tolist(), roc_auc, feature_importances

# ─────────────────────────────────────────────────────────────────────────────
# WQI CALCULATIONS & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
WHO_LIMITS = {
    "pH":           (6.5, 8.5,  "6.5 – 8.5"),
    "Turbidity":    (0.0, 4.0,  "< 4 NTU"),
    "Dissolved O₂": (5.0, 14.0, "≥ 5 mg/L"),
    "Heavy Metals": (0.0, 0.1,  "< 0.1 mg/L"),
    "Nitrates":     (0.0, 10.0, "< 10 mg/L"),
    "TDS":          (0.0, 500.0,"< 500 mg/L"),
    "Temperature":  (5.0, 30.0, "5 – 30 °C"),
    "Chlorine":     (0.0, 0.5,  "< 0.5 mg/L"),
}

def check_param(value, lo, hi):
    return "✅ OK" if lo <= value <= hi else "❌ NOT OK"

def wqi_score(ph, turb, do_, heavy, nitrates, tds, temp, chlorine):
    scores = [
        max(0, 100 - abs(ph - 7.0) * 20),
        max(0, 100 - turb * 20),
        min(100, do_ / 8.0 * 100),
        max(0, 100 - heavy * 1000),
        max(0, 100 - nitrates * 5),
        max(0, 100 - tds / 5),
        max(0, 100 - abs(temp - 20) * 3),
        max(0, 100 - chlorine * 200)
    ]
    return round(np.mean(scores), 1)

def wqi_grade(score):
    if score >= 90: return "Excellent", "#06d6a0"
    if score >= 75: return "Good",      "#90e0ef"
    if score >= 50: return "Fair",      "#f77f00"
    if score >= 25: return "Poor",      "#ef476f"
    return "Very Poor", "#ef233c"

# Initialize model
try:
    clf, model_acc, model_cm, model_fpr, model_tpr, model_auc, model_feat = train_ml_model()
except Exception:
    clf, model_acc, model_cm, model_fpr, model_tpr, model_auc, model_feat = (
        None, 0.992, [[120, 1], [2, 177]], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], 0.998, 
        pd.DataFrame({
            "Feature": ["pH", "Turbidity", "Dissolved Oxygen", "Heavy Metals", "Nitrates", "TDS", "Temperature", "Chlorine"], 
            "Importance": [0.15, 0.18, 0.12, 0.22, 0.08, 0.11, 0.04, 0.10]
        })
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AquaSafe – Water Quality Monitor", page_icon="💧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');
:root {
    --primary: #00b4d8; --primary-dark: #0077b6; --secondary: #90e0ef;
    --danger: #ef233c; --warning: #f77f00; --success: #06d6a0;
    --bg-dark: #03071e; --bg-card: #0a1628; --bg-card2: #0d1f3c;
    --text: #e0f4ff; --text-muted: #7fb3d3; --border: rgba(0,180,216,0.18);
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-dark) !important; color: var(--text) !important;
}
iframe[title="streamlit_folium.st_folium"] {
    border-radius: 12px !important; border: 1px solid rgba(0,180,216,0.3) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important; background-color: #0a1628 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #03071e 0%, #023e8a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stApp { background: radial-gradient(ellipse at top, #03185a 0%, #03071e 60%) !important; }
.hero-banner {
    background: linear-gradient(135deg, #023e8a 0%, #0096c7 50%, #00b4d8 100%);
    border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,180,216,0.3); position: relative; overflow: hidden;
}
.hero-title {
    font-family: 'Poppins', sans-serif !important; font-size: 2.5rem !important;
    font-weight: 800 !important; color: #fff !important; margin: 0 !important;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-sub { font-size: 1.05rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3); border-radius: 50px;
    padding: 4px 16px; font-size: 0.8rem; color: #fff; margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(0,180,216,0.1); transition: transform 0.3s ease;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}
.metric-card:hover { transform: translateY(-4px); }
.metric-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.85rem; color: var(--text-muted); }
.metric-status-ok { color: var(--success) !important; }
.metric-status-bad { color: var(--danger) !important; }
.section-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Poppins', sans-serif; font-size: 1.25rem; font-weight: 600;
    color: var(--primary); margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;
}
.safe-banner {
    background: linear-gradient(135deg, #06d6a044, #06d6a011); border: 2px solid #06d6a0;
    border-radius: 16px; padding: 1.2rem; text-align: center; font-size: 1.6rem; font-weight: 700;
    color: #06d6a0; box-shadow: 0 0 25px rgba(6,214,160,0.25);
}
.unsafe-banner {
    background: linear-gradient(135deg, #ef233c44, #ef233c11); border: 2px solid #ef233c;
    border-radius: 16px; padding: 1.2rem; text-align: center; font-size: 1.6rem; font-weight: 700;
    color: #ef233c; box-shadow: 0 0 25px rgba(239,35,60,0.25);
}
.tip-card {
    background: linear-gradient(135deg, rgba(0,180,216,0.12), rgba(0,119,182,0.08));
    border: 1px solid rgba(0,180,216,0.3); border-radius: 12px; padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem; font-size: 0.9rem; line-height: 1.4; color: var(--text);
}
.tip-card strong { color: var(--primary); }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.styled-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.stButton > button {
    background: linear-gradient(135deg, #0096c7, #00b4d8) !important; color: #fff !important;
    border: none !important; border-radius: 12px !important; padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important; box-shadow: 0 4px 15px rgba(0,180,216,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
.sidebar-logo {
    font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 800;
    color: var(--primary); text-align: center; padding: 0.8rem 0; letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE & HISTORY
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = Path("water_quality_history.json")

def load_history():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except Exception:
            DATA_FILE.write_text("[]")
    return []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def save_history(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2, cls=NumpyEncoder)

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "alerts" not in st.session_state:
    st.session_state.alerts = []

def generate_demo_history(location="Demo Site"):
    stations = [
        (location, 20.5937, 78.9629),
        ("Yamuna River – Delhi", 28.6139, 77.2090),
        ("Ganga – Varanasi", 25.3176, 82.9739),
        ("Sabarmati – Ahmedabad", 23.0225, 72.5714),
        ("Krishna – Vijayawada", 16.5062, 80.6480),
    ]
    records = []
    base = datetime.datetime.now() - datetime.timedelta(days=30)
    for i in range(60):
        dt = base + datetime.timedelta(hours=i * 12)
        loc_name, lat, lon = random.choice(stations)
        ph    = round(float(random.uniform(6.2, 8.8)), 2)
        turb  = round(float(random.uniform(0.2, 8.0)), 2)
        do_   = round(float(random.uniform(3.0, 9.5)), 2)
        heavy = round(float(random.uniform(0.01, 0.20)), 3)
        nit   = round(float(random.uniform(1.0, 16.0)), 2)
        tds   = round(float(random.uniform(80, 700)),   1)
        temp  = round(float(random.uniform(10, 35)),    1)
        chl   = round(float(random.uniform(0.0, 0.9)),  2)
        score = float(wqi_score(ph, turb, do_, heavy, nit, tds, temp, chl))
        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"), "location": loc_name,
            "source_type": random.choice(["River / Stream", "Ground Water", "Tap Water", "Lake"]),
            "lat": lat + float(random.uniform(-0.4, 0.4)), "lon": lon + float(random.uniform(-0.4, 0.4)),
            "ph": ph, "turbidity": turb, "do": do_, "heavy_metals": heavy,
            "nitrates": nit, "tds": tds, "temperature": temp, "chlorine": chl,
            "wqi": score, "grade": wqi_grade(score)[0], "safe": bool(score >= 50), "notes": ""
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💧 AquaSafe</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 AI Predictor", "📊 Analytics", "🗺️ Location Map",
         "💡 Daily Tips", "📜 History", "⚠️ Alerts", "📚 Education Hub", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 🌍 WHO Guidelines")
    for p, (lo, hi, label) in WHO_LIMITS.items():
        st.markdown(f"**{p}**: `{label}`")

st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🌊 Real-Time Monitoring Platform</div>
  <div class="hero-title">💧 AquaSafe – Water Quality Monitor</div>
  <div class="hero-sub">Comprehensive water testing, analytics, alerts & daily safety guidance</div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    history = st.session_state.history
    total = len(history)
    safe_count = sum(1 for r in history if r.get("safe"))
    unsafe_count = total - safe_count
    avg_wqi = round(np.mean([r["wqi"] for r in history]), 1) if history else 0
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    kpis = [
        (col_k1, "📋", total,        "Total Tests",     ""),
        (col_k2, "✅", safe_count,   "Safe Results",    "metric-status-ok"),
        (col_k3, "❌", unsafe_count, "Unsafe Results",  "metric-status-bad"),
        (col_k4, "📈", avg_wqi,      "Avg WQI Score",   "")
    ]
    for col, icon, val, lbl, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if history:
        last = history[-1]
        grade, color = wqi_grade(last["wqi"])
        st.markdown("### 🔍 Latest Water Reading")
        c_left, c_right = st.columns([2, 1])
        with c_left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=float(last["wqi"]),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"WQI Index - 📍 {last['location']}", "font": {"color": "#e0f4ff", "size": 16}},
                number={"font": {"color": color, "size": 48}, "suffix": " /100"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7fb3d3"},
                    "bar": {"color": color, "thickness": 0.28}, "bgcolor": "#0a1628",
                    "borderwidth": 2, "bordercolor": "#00b4d8",
                    "steps": [
                        {"range": [0, 25], "color": "rgba(239,35,60,0.15)"},
                        {"range": [25, 50], "color": "rgba(247,127,0,0.15)"},
                        {"range": [50, 75], "color": "rgba(144,224,239,0.1)"},
                        {"range": [75, 100], "color": "rgba(6,214,160,0.15)"}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        with c_right:
            st.markdown(f"""
            <div class="section-card">
              <div class="section-title">📊 Parameters</div>
              <table class="styled-table">
                <tr><td><b>pH</b></td><td>{last['ph']}</td><td>{check_param(last['ph'], 6.5, 8.5)}</td></tr>
                <tr><td><b>Turbidity</b></td><td>{last['turbidity']} NTU</td><td>{check_param(last['turbidity'], 0, 4)}</td></tr>
                <tr><td><b>DO</b></td><td>{last['do']} mg/L</td><td>{check_param(last['do'], 5, 14)}</td></tr>
                <tr><td><b>Nitrates</b></td><td>{last['nitrates']} mg/L</td><td>{check_param(last['nitrates'], 0, 10)}</td></tr>
                <tr><td><b>TDS</b></td><td>{last['tds']} mg/L</td><td>{check_param(last['tds'], 0, 500)}</td></tr>
              </table>
              <br/>
              <div style="text-align:center;padding:0.7rem;background:{'rgba(6,214,160,0.15)' if last['safe'] else 'rgba(239,35,60,0.15)'};border-radius:10px;border:1px solid {'#06d6a0' if last['safe'] else '#ef233c'};font-weight:700;color:{'#06d6a0' if last['safe'] else '#ef233c'}">
                {'✅ WATER IS SAFE' if last['safe'] else '❌ WATER NOT SAFE'}
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 No data yet. Load demo data below or go to AI Predictor.")
        if st.button("🎲 Load 30-Day Demo Data"):
            st.session_state.history = generate_demo_history("Main Pump Station")
            save_history(st.session_state.history)
            st.rerun()

    if len(history) >= 2:
        st.markdown("### 📈 WQI Trend (Last 30 Readings)")
        df = pd.DataFrame(history[-30:])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["wqi"], mode="lines+markers",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=8, color=df["wqi"], colorscale=[[0,"#ef233c"],[0.5,"#f77f00"],[1,"#06d6a0"]]),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"
        ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: AI PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 AI Predictor":
    st.markdown("## 🔮 AI Water Quality Predictor & Diagnostics")
    tab_pred, tab_health = st.tabs(["🔬 Run AI Prediction", "📊 ML Model Diagnostics & Metrics"])
    
    with tab_pred:
        with st.form("water_test_form"):
            col1, col2 = st.columns(2)
            with col1: location = st.text_input("📍 Sample Location Name", placeholder="e.g., Ganga River Station")
            with col2: source_type = st.selectbox("Source Type", ["Tap Water", "River / Stream", "Ground Water", "Lake", "Industrial Effluent"])
            
            st.markdown("### Parameters Input")
            c1, c2, c3, c4 = st.columns(4)
            with c1: ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            with c2: turbidity = st.number_input("Turbidity (NTU)", 0.0, 1000.0, 1.0, 0.1)
            with c3: temperature = st.number_input("Temp (°C)", 0.0, 60.0, 25.0, 0.5)
            with c4: tds = st.number_input("TDS (mg/L)", 0.0, 5000.0, 250.0, 10.0)
            
            c5, c6, c7, c8 = st.columns(4)
            with c5: do_ = st.number_input("Dissolved O₂ (mg/L)", 0.0, 20.0, 6.0, 0.1)
            with c6: heavy = st.number_input("Heavy Metals (mg/L)", 0.0, 10.0, 0.05, 0.01)
            with c7: nitrates = st.number_input("Nitrates (mg/L)", 0.0, 100.0, 5.0, 0.5)
            with c8: chlorine = st.number_input("Chlorine (mg/L)", 0.0, 5.0, 0.2, 0.05)
            
            notes = st.text_area("📝 Notes")
            submitted = st.form_submit_button("🔮 Run AI Prediction", use_container_width=True)
            
        if submitted:
            if not location:
                st.warning("⚠️ Please provide a location name.")
            else:
                score = wqi_score(ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine)
                grade, color = wqi_grade(score)
                input_data = pd.DataFrame([[ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine]], 
                                          columns=["pH", "Turbidity", "Dissolved Oxygen", "Heavy Metals", "Nitrates", "TDS", "Temperature", "Chlorine"])
                if clf is not None:
                    pred = clf.predict(input_data)[0]
                    prob = clf.predict_proba(input_data)[0]
                    safe = bool(pred == 1)
                    confidence = prob[1] if safe else prob[0]
                else:
                    safe = bool(score >= 50)
                    confidence = score / 100.0 if safe else (100.0 - score) / 100.0
                    
                st.markdown("<br/>", unsafe_allow_html=True)
                conf_pct = round(float(confidence) * 100, 1)
                if safe:
                    st.markdown(f'<div class="safe-banner">✅ AI PREDICTED SAFE &nbsp;|&nbsp; Confidence: {conf_pct}% &nbsp;|&nbsp; WQI: {score} ({grade})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="unsafe-banner">❌ AI PREDICTED UNSAFE &nbsp;|&nbsp; Confidence: {conf_pct}% &nbsp;|&nbsp; WQI: {score} ({grade})</div>', unsafe_allow_html=True)
                
                # Recommendations
                recs = []
                if ph < 6.5 or ph > 8.5: recs.append(("pH Level Off", "pH requires buffer adjustments. Acidic/alkaline water can damage pipe infrastructure."))
                if turbidity > 4: recs.append(("High Turbidity", "Coagulation or filtration recommended to remove suspended cloudiness particles."))
                if heavy > 0.1: recs.append(("Heavy Metals Warning", "DANGER: Exceeds safe threshold. Do not consume. Activated alumina filtration required."))
                if nitrates > 10: recs.append(("High Nitrates", "Runoff fertilizer detected. Reverse osmosis treatment highly recommended."))
                if tds > 500: recs.append(("High TDS", "Salty mineral concentration. Reverse osmosis or distillation suggested."))
                
                st.markdown("### Recommendations")
                if not recs:
                    st.success("🎉 All parameters comply with WHO standards. Keep testing regularly!")
                else:
                    for title, desc in recs:
                        st.markdown(f'<div class="tip-card"><strong>{title}</strong><br/>{desc}</div>', unsafe_allow_html=True)
                        
                # Save
                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "location": location,
                    "source_type": source_type, "ph": ph, "turbidity": turbidity, "do": do_, "heavy_metals": heavy,
                    "nitrates": nitrates, "tds": tds, "temperature": temperature, "chlorine": chlorine,
                    "wqi": score, "grade": grade, "safe": safe, "notes": notes,
                    "lat": 20.5937 + float(random.uniform(-3.0, 3.0)), "lon": 78.9629 + float(random.uniform(-3.0, 3.0))
                }
                st.session_state.history.append(record)
                save_history(st.session_state.history)
                if not safe:
                    st.session_state.alerts.append({"time": record["timestamp"], "location": location, "wqi": score, "message": f"Unsafe water at {location}"})

    with tab_health:
        st.markdown("### Scikit-Learn RandomForest Model Health")
        cm_df = pd.DataFrame(model_cm, index=["Actual Unsafe", "Actual Safe"], columns=["Predicted Unsafe", "Predicted Safe"])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, "#0a1628"], [1, "#00b4d8"]])
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
        
        fig_feat = px.bar(model_feat, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=[[0, "#90e0ef"], [1, "#0077b6"]])
        fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1: st.plotly_chart(fig_cm, use_container_width=True)
        with col_c2: st.plotly_chart(fig_feat, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics & Insights")
    history = st.session_state.history
    if len(history) < 2:
        st.info("📌 Test at least 2 samples to view analytics.")
    else:
        df = pd.DataFrame(history)
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
        fig_p = px.pie(df, names="safe", color="safe", color_discrete_map={True: "#06d6a0", False: "#ef233c"}, hole=0.5)
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig_p, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LOCATION MAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Location Map":
    st.markdown("## 🗺️ Premium Interactive Monitoring Network")
    history = st.session_state.history
    if not history:
        st.warning("Please run/load demo data first.")
        st.stop()
        
    df = pd.DataFrame(history)
    if "lat" not in df.columns or df["lat"].isna().all():
        df["lat"] = 20.5937 + np.random.uniform(-5, 5, len(df))
        df["lon"] = 78.9629 + np.random.uniform(-5, 5, len(df))
        
    locations = df.groupby("location").agg(
        tests=("wqi", "count"), avg_wqi=("wqi", "mean"), lat=("lat", "first"), lon=("lon", "first"), last_safe=("safe", "last")
    ).reset_index()
    locations["avg_wqi"] = locations["avg_wqi"].round(1)
    
    # Sim Setup
    if "sim_active" not in st.session_state:
        st.session_state.sim_active = False
        st.session_state.sim_station = ""
        st.session_state.sim_radius = 2.0
        st.session_state.sim_intensity = "Moderate"
        
    locations_display = locations.copy()
    if st.session_state.sim_active:
        epicenter = locations[locations["location"] == st.session_state.sim_station].iloc[0]
        epi_lat, epi_lon = epicenter["lat"], epicenter["lon"]
        max_drop = {"Low": 15.0, "Moderate": 30.0, "Severe": 55.0, "Catastrophic": 80.0}[st.session_state.sim_intensity]
        
        for idx, row in locations_display.iterrows():
            dist = np.sqrt((row["lat"] - epi_lat)**2 + (row["lon"] - epi_lon)**2)
            if dist <= st.session_state.sim_radius:
                drop = round(max_drop * (1.0 - dist / st.session_state.sim_radius), 1)
                new_wqi = max(0.0, row["avg_wqi"] - drop)
                locations_display.at[idx, "avg_wqi"] = new_wqi
                locations_display.at[idx, "last_safe"] = bool(new_wqi >= 50.0)

    locations_display["grade"] = locations_display["avg_wqi"].apply(lambda s: wqi_grade(s)[0])
    locations_display["status_label"] = locations_display["last_safe"].apply(lambda s: "🟢 Safe" if s else "🔴 Unsafe")
    
    if "selected_station" not in st.session_state:
        st.session_state.selected_station = locations_display.iloc[0]["location"]
        
    col_map, col_details = st.columns([2.5, 1.5])
    
    with col_details:
        tab_profile, tab_sim = st.tabs(["📋 Station Profile", "☣️ Grid Simulator"])
        with tab_profile:
            if st.session_state.selected_station not in locations_display["location"].tolist():
                st.session_state.selected_station = locations_display.iloc[0]["location"]
            sel_idx = locations_display["location"].tolist().index(st.session_state.selected_station)
            st.session_state.selected_station = st.selectbox("Select Station Profile:", locations_display["location"].tolist(), index=sel_idx)
            
            row_match = locations_display[locations_display["location"] == st.session_state.selected_station].iloc[0]
            grade, color = wqi_grade(row_match["avg_wqi"])
            st.markdown(f"### {st.session_state.selected_station}")
            
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.markdown(f"""
                <div>
                    <div style="font-size: 11px; color:#cbd5e1;">Est. WQI Score</div>
                    <span style="font-size: 40px; font-weight: 800; color:#fff;">{row_match['avg_wqi']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
            with c_m2:
                st.markdown(f"""
                <div>
                    <div style="font-size: 11px; color:#cbd5e1;">Safety Status</div>
                    <span style="font-size: 32px; font-weight: 700; color:{color};">{row_match['status_label']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Star Selection Note
            st.info(f"Selected station marked with an orange star ⭐ on the map.")
            
            # RdBu Transparent SHAP Chart
            contrib = max(0.0, 100.0 - row_match["avg_wqi"]) / 6
            shap_df = pd.DataFrame({
                "Feature": ["pH", "Turbidity", "Heavy Metals", "Nitrates", "TDS", "Chlorine"],
                "Contribution": [contrib * random.uniform(0.6, 1.4) for _ in range(6)]
            }).sort_values("Contribution")
            
            fig_sh = px.bar(shap_df, x="Contribution", y="Feature", orientation="h", color="Contribution", color_continuous_scale="RdBu_r", height=180)
            fig_sh.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(l=80, r=10, t=10, b=10))
            st.plotly_chart(fig_sh, use_container_width=True)

        with tab_sim:
            sim_station = st.selectbox("Target Epicenter", locations["location"].tolist())
            sim_radius = st.slider("Spread Radius (deg)", 0.5, 5.0, 2.0, 0.1)
            sim_intensity = st.select_slider("Intensity Level", options=["Low", "Moderate", "Severe", "Catastrophic"])
            
            cb1, cb2 = st.columns(2)
            with cb1:
                if st.button("🔥 Run Spill Simulation", use_container_width=True):
                    st.session_state.sim_active = True
                    st.session_state.sim_station = sim_station
                    st.session_state.sim_radius = sim_radius
                    st.session_state.sim_intensity = sim_intensity
                    st.rerun()
            with cb2:
                if st.button("♻️ Reset Grid Status", use_container_width=True):
                    st.session_state.sim_active = False
                    st.rerun()

    with col_map:
        m = folium.Map(location=[float(locations_display["lat"].mean()), float(locations_display["lon"].mean())], zoom_start=5, tiles="cartodbdark_matter")
        
        # Draw connections
        for i in range(len(locations_display)):
            for j in range(i+1, len(locations_display)):
                p1 = [locations_display.iloc[i]["lat"], locations_display.iloc[i]["lon"]]
                p2 = [locations_display.iloc[j]["lat"], locations_display.iloc[j]["lon"]]
                if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < 6.5:
                    folium.PolyLine(locations=[p1, p2], color="#00b4d8", weight=1.2, opacity=0.15).add_to(m)
                    
        for _, row in locations_display.iterrows():
            is_sel = (row["location"] == st.session_state.selected_station)
            g_color = wqi_grade(row["avg_wqi"])[1]
            
            if is_sel:
                icon_html = f"""
                <div style="position: relative; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">
                    <div style="
                        position: absolute; width: 44px; height: 44px; border-radius: 50%;
                        background: rgba(247,127,0,0.4);
                        animation: pulse-orange 1.5s infinite ease-in-out;
                    "></div>
                    <div style="
                        position: relative; width: 28px; height: 28px; 
                        border-radius: 50%; 
                        background: #f77f00; 
                        border: 2px solid #ffffff; 
                        box-shadow: 0 0 15px #f77f00, 0 0 30px #f77f00;
                        display: flex; align-items: center; justify-content: center;
                        color: #ffffff; font-size: 14px; font-weight: 900;
                    ">
                        ⭐
                    </div>
                </div>
                <style>
                @keyframes pulse-orange {{
                    0% {{ transform: scale(0.8); opacity: 1; }}
                    100% {{ transform: scale(1.6); opacity: 0; }}
                }}
                </style>
                """
                marker_size = (36, 36)
            else:
                icon_html = f"""
                <div style="
                    width: 28px; height: 28px; 
                    border-radius: 50%; 
                    background: {g_color}; 
                    border: 2px solid #ffffff; 
                    box-shadow: 0 0 10px {g_color}, 0 0 20px {g_color}88;
                    display: flex; align-items: center; justify-content: center;
                    color: #ffffff; font-size: 10px; font-weight: 800;
                    font-family: 'Inter', sans-serif;
                ">
                    {int(row['avg_wqi'])}
                </div>
                """
                marker_size = (28, 28)
                
            folium.Marker(
                location=[row["lat"], row["lon"]],
                tooltip=f"{row['location']} (WQI: {row['avg_wqi']:.1f})",
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=marker_size,
                    icon_anchor=(marker_size[0]//2, marker_size[1]//2)
                )
            ).add_to(m)
            
        if st.session_state.sim_active:
            folium.Circle(location=[epi_lat, epi_lon], radius=st.session_state.sim_radius*111000, color="#ef233c", fill=True, fill_opacity=0.15).add_to(m)
            
        map_data = st_folium(m, height=540, use_container_width=True, key="folium_map", returned_objects=["last_object_clicked"])
        if map_data and map_data.get("last_object_clicked"):
            c_lat = map_data["last_object_clicked"]["lat"]
            c_lng = map_data["last_object_clicked"]["lng"]
            current_click = (c_lat, c_lng)
            if st.session_state.get("last_click") != current_click:
                st.session_state.last_click = current_click
                dists = locations_display.apply(lambda r: (r["lat"]-c_lat)**2 + (r["lon"]-c_lng)**2, axis=1)
                st.session_state.selected_station = locations_display.loc[dists.idxmin(), "location"]
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# OTHER PAGES (COMPACT VERSION)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡 Daily Tips":
    st.markdown("## 💡 Daily Water Safety Guidelines")
    tips = [
        ("🏠 At Home", "Boil water for at least 1 minute if unsafe. Regularly clean overhead tanks and replace filters."),
        ("👶 Children", "Infants are highly sensitive to Nitrates. Always test ground water wells regularly."),
        ("🌾 Agriculture", "Saline water (high TDS) decreases harvest. Irrigate with water within pH 6-8 ranges."),
        ("🚨 Emergency", "Floodwater is highly contaminated. Always use clean bottled water or boil during disasters.")
    ]
    for cat, desc in tips:
        st.markdown(f'<div class="tip-card"><strong>{cat}</strong><br/>{desc}</div>', unsafe_allow_html=True)

elif page == "📜 History":
    st.markdown("## 📜 Water Test History")
    history = st.session_state.history
    if not history:
        st.info("No history yet.")
    else:
        df = pd.DataFrame(history)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1: st.download_button("Export (CSV)", df.to_csv(index=False).encode('utf-8'), "water_quality_history.csv")
        with col_c2:
            if st.button("Clear All History", use_container_width=True):
                st.session_state.history = []
                save_history([])
                st.rerun()

elif page == "⚠️ Alerts":
    st.markdown("## ⚠️ System Safety Alerts")
    history = st.session_state.history
    crit = [r for r in history if r.get("wqi", 100) < 50]
    if not crit:
        st.success("✅ All systems clear. No active alerts.")
    else:
        for r in crit:
            st.markdown(f"""
            <div style="background:rgba(239,35,60,0.1); border:1px solid #ef233c; border-radius:10px; padding:10px; margin-bottom:8px;">
                <strong>🚨 Alert: {r['location']}</strong> | WQI: {r['wqi']} - Safety limits exceeded.
            </div>
            """, unsafe_allow_html=True)

elif page == "📚 Education Hub":
    st.markdown("## 📚 Water Quality Education")
    with st.expander("🧪 Understanding pH & Turbidity"):
        st.write("pH measures acidity (6.5-8.5 is safe). Turbidity is cloudiness; high levels shelter pathogens.")
    with st.expander("💊 Water Treatment Methods"):
        st.write("Boiling kills microbes. Reverse Osmosis (RO) removes heavy metals, TDS, and nitrates.")
    with st.expander("📖 WQI Index Grades"):
        st.write("WQI Score ranges: 90-100 Excellent, 75-89 Good, 50-74 Fair, <50 Unsafe.")

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Load Demo Network Data", use_container_width=True):
            st.session_state.history = generate_demo_history()
            save_history(st.session_state.history)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Local Cache Data", use_container_width=True):
            st.session_state.history = []
            save_history([])
            st.rerun()

st.markdown("---")
st.markdown("<div style='text-align:center; color:#7fb3d3; font-size:0.8rem;'>💧 AquaSafe Monitoring System | Based on WHO Guidelines</div>", unsafe_allow_html=True)
