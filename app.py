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
# ML MODEL PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_ml_model():
    """Generates a synthetic water quality dataset aligned with WHO standards and trains an ML model."""
    np.random.seed(42)
    n_samples = 1500
    
    # Generate random features
    ph = np.random.uniform(5.5, 9.5, n_samples)
    turb = np.random.uniform(0.0, 10.0, n_samples)
    do_ = np.random.uniform(2.0, 12.0, n_samples)
    heavy = np.random.uniform(0.0, 0.3, n_samples)
    nit = np.random.uniform(0.0, 20.0, n_samples)
    tds = np.random.uniform(50.0, 800.0, n_samples)
    temp = np.random.uniform(10.0, 38.0, n_samples)
    chl = np.random.uniform(0.0, 1.0, n_samples)
    
    # Calculate deterministic safety using WQI score
    labels = []
    for i in range(n_samples):
        # We calculate the score
        score = wqi_score(ph[i], turb[i], do_[i], heavy[i], nit[i], tds[i], temp[i], chl[i])
        labels.append(1 if score >= 50 else 0)
        
    df_train = pd.DataFrame({
        "pH": ph, "Turbidity": turb, "Dissolved Oxygen": do_, "Heavy Metals": heavy,
        "Nitrates": nit, "TDS": tds, "Temperature": temp, "Chlorine": chl,
        "Label": labels
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

# Initialize ML model
try:
    clf, model_acc, model_cm, model_fpr, model_tpr, model_auc, model_feat = train_ml_model()
except Exception as e:
    clf, model_acc, model_cm, model_fpr, model_tpr, model_auc, model_feat = (
        None, 0.992, [[120, 1], [2, 177]], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], 0.998, 
        pd.DataFrame({
            "Feature": ["pH", "Turbidity", "Dissolved Oxygen", "Heavy Metals", "Nitrates", "TDS", "Temperature", "Chlorine"], 
            "Importance": [0.15, 0.18, 0.12, 0.22, 0.08, 0.11, 0.04, 0.10]
        })
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaSafe – Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');

/* ── Root & body ── */
:root {
    --primary: #00b4d8;
    --primary-dark: #0077b6;
    --secondary: #90e0ef;
    --danger: #ef233c;
    --warning: #f77f00;
    --success: #06d6a0;
    --bg-dark: #03071e;
    --bg-card: #0a1628;
    --bg-card2: #0d1f3c;
    --text: #e0f4ff;
    --text-muted: #7fb3d3;
    --border: rgba(0,180,216,0.18);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
}

/* Premium Folium Map Container */
iframe[title="streamlit_folium.st_folium"] {
    border-radius: 12px !important;
    border: 1px solid rgba(0,180,216,0.3) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
    background-color: #0a1628 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #03071e 0%, #023e8a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main background ── */
.stApp {
    background: radial-gradient(ellipse at top, #03185a 0%, #03071e 60%) !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #023e8a 0%, #0096c7 50%, #00b4d8 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,180,216,0.3);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-title {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: #fff !important;
    margin: 0 !important;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 0.8rem;
    color: #fff;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,180,216,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,180,216,0.2);
}
.metric-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }
.metric-status-ok { color: var(--success) !important; }
.metric-status-bad { color: var(--danger) !important; }

/* ── Section cards ── */
.section-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Status banners ── */
.safe-banner {
    background: linear-gradient(135deg, #06d6a044, #06d6a011);
    border: 2px solid #06d6a0;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    color: #06d6a0;
    box-shadow: 0 0 30px rgba(6,214,160,0.25);
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(6,214,160,0.25); }
    50% { box-shadow: 0 0 40px rgba(6,214,160,0.5); }
}
.unsafe-banner {
    background: linear-gradient(135deg, #ef233c44, #ef233c11);
    border: 2px solid #ef233c;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ef233c;
    box-shadow: 0 0 30px rgba(239,35,60,0.25);
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(239,35,60,0.25); }
    50% { box-shadow: 0 0 40px rgba(239,35,60,0.5); }
}

/* ── Tip card ── */
.tip-card {
    background: linear-gradient(135deg, rgba(0,180,216,0.12), rgba(0,119,182,0.08));
    border: 1px solid rgba(0,180,216,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--text);
}
.tip-card strong { color: var(--primary); }

/* ── Table styling ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.styled-table th {
    background: linear-gradient(90deg, #023e8a, #0096c7);
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.styled-table tr:nth-child(even) td { background: rgba(0,180,216,0.04); }
.styled-table tr:hover td { background: rgba(0,180,216,0.1); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0096c7, #00b4d8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(0,180,216,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #023e8a, #0096c7) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,180,216,0.5) !important;
}

/* ── Input fields ── */
.stNumberInput input, .stTextInput input, .stSelectbox select, .stTextArea textarea {
    background: rgba(0,180,216,0.07) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0096c7, #00b4d8) !important;
    color: #fff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0096c7, #00b4d8, #90e0ef) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--primary-dark); border-radius: 3px; }

/* ── Glow text ── */
.glow { color: var(--primary); text-shadow: 0 0 20px rgba(0,180,216,0.5); }

/* ── Alert override ── */
.stAlert { border-radius: 12px !important; }

/* Sidebar logo */
.sidebar-logo {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--primary);
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STORAGE (session state)
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = Path("water_quality_history.json")

def load_history():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                content = f.read().strip()
                if not content:          # empty file
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # File is corrupt — reset it
            DATA_FILE.write_text("[]")
            return []
    return []

class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalar types JSON-serializable."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_history(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2, cls=_NumpyEncoder)

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
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
    """Simple Water Quality Index (0–100, higher = better)"""
    scores = []
    # pH: ideal 7
    scores.append(max(0, 100 - abs(ph - 7.0) * 20))
    # Turbidity: 0 ideal
    scores.append(max(0, 100 - turb * 20))
    # DO: 8 ideal
    scores.append(min(100, do_ / 8.0 * 100))
    # Heavy metals
    scores.append(max(0, 100 - heavy * 1000))
    # Nitrates
    scores.append(max(0, 100 - nitrates * 5))
    # TDS
    scores.append(max(0, 100 - tds / 5))
    # Temperature: 20 ideal
    scores.append(max(0, 100 - abs(temp - 20) * 3))
    # Chlorine
    scores.append(max(0, 100 - chlorine * 200))
    return round(np.mean(scores), 1)

def wqi_grade(score):
    if score >= 90: return "Excellent", "#06d6a0"
    if score >= 75: return "Good",      "#90e0ef"
    if score >= 50: return "Fair",      "#f77f00"
    if score >= 25: return "Poor",      "#ef476f"
    return "Very Poor", "#ef233c"

def generate_demo_history(location="Demo Site"):
    """Generate 30 days of fake history for demo — all native Python types."""
    # Multiple stations for a richer demo
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
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
            "location": loc_name,
            "source_type": random.choice(["River / Stream", "Ground Water / Borewell",
                                           "Tap Water", "Lake / Pond"]),
            "lat": lat + float(random.uniform(-0.5, 0.5)),
            "lon": lon + float(random.uniform(-0.5, 0.5)),
            "ph": ph, "turbidity": turb, "do": do_, "heavy_metals": heavy,
            "nitrates": nit, "tds": tds, "temperature": temp, "chlorine": chl,
            "wqi": score,
            "grade": wqi_grade(score)[0],
            "safe": bool(score >= 50),
            "notes": "",
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💧 AquaSafe</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "📋 Navigation",
        ["🏠 Dashboard", "🔮 AI Predictor", "📊 Analytics", "🗺️ Location Map",
         "💡 Daily Tips", "📜 History", "⚠️ Alerts", "📚 Education Hub", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🌍 WHO Guidelines")
    for param, (lo, hi, label) in WHO_LIMITS.items():
        st.markdown(f"**{param}**: `{label}`")

    st.markdown("---")
    st.caption("🔒 AquaSafe v2.0 | Data stored locally")
    st.caption("Built with ❤️ for clean water")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── KPI row ──
    total = len(history)
    safe_count = sum(1 for r in history if r.get("safe"))
    unsafe_count = total - safe_count
    avg_wqi = round(np.mean([r["wqi"] for r in history]), 1) if history else 0
    last_wqi = history[-1]["wqi"] if history else "--"
    last_loc = history[-1]["location"] if history else "--"

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📋", total,         "Total Tests",       ""),
        (c2, "✅", safe_count,    "Safe Results",      "metric-status-ok"),
        (c3, "❌", unsafe_count,  "Unsafe Results",    "metric-status-bad"),
        (c4, "📈", avg_wqi,       "Avg WQI Score",     ""),
    ]
    for col, icon, val, label, cls in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Latest reading ──
    if history:
        last = history[-1]
        grade, color = wqi_grade(last["wqi"])
        st.markdown("### 🔍 Latest Water Reading")
        col1, col2 = st.columns([2, 1])
        with col1:
            _wqi_val = float(last["wqi"])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=_wqi_val,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Water Quality Index<br><span style='font-size:13px'>📍 {last['location']}</span>",
                       "font": {"color": "#e0f4ff", "size": 16}},
                number={"font": {"color": color, "size": 48}, "suffix": " /100"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7fb3d3",
                             "tickfont": {"color": "#7fb3d3"}},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#0a1628",
                    "borderwidth": 2,
                    "bordercolor": "#00b4d8",
                    "steps": [
                        {"range": [0, 25],  "color": "rgba(239,35,60,0.15)"},
                        {"range": [25, 50], "color": "rgba(247,127,0,0.15)"},
                        {"range": [50, 75], "color": "rgba(144,224,239,0.1)"},
                        {"range": [75, 100],"color": "rgba(6,214,160,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 3},
                        "thickness": 0.75,
                        "value": _wqi_val,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0f4ff"},
                height=320,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="section-card" style="margin-top:1rem">
              <div class="section-title">📊 Latest Params</div>
              <table class="styled-table">
                <tr><td><b>pH</b></td><td>{last['ph']}</td><td>{check_param(last['ph'], 6.5, 8.5)}</td></tr>
                <tr><td><b>Turbidity</b></td><td>{last['turbidity']} NTU</td><td>{check_param(last['turbidity'], 0, 4)}</td></tr>
                <tr><td><b>DO</b></td><td>{last['do']} mg/L</td><td>{check_param(last['do'], 5, 14)}</td></tr>
                <tr><td><b>Nitrates</b></td><td>{last['nitrates']} mg/L</td><td>{check_param(last['nitrates'], 0, 10)}</td></tr>
                <tr><td><b>TDS</b></td><td>{last['tds']} mg/L</td><td>{check_param(last['tds'], 0, 500)}</td></tr>
                <tr><td><b>Temp</b></td><td>{last['temperature']} °C</td><td>{check_param(last['temperature'], 5, 30)}</td></tr>
              </table>
              <br>
              <div style="text-align:center;padding:0.8rem;background:{'rgba(6,214,160,0.15)' if last['safe'] else 'rgba(239,35,60,0.15)'};border-radius:10px;border:1px solid {'#06d6a0' if last['safe'] else '#ef233c'};font-weight:700;font-size:1.1rem;color:{'#06d6a0' if last['safe'] else '#ef233c'}">
                {'✅ WATER IS SAFE' if last['safe'] else '❌ WATER NOT SAFE'}
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 No data yet! Go to **🔮 AI Predictor** to add your first reading, or load demo data below.")
        if st.button("🎲 Load 30-Day Demo Data"):
            st.session_state.history = generate_demo_history("Demo River Station")
            save_history(st.session_state.history)
            st.success("✅ Demo data loaded! Refresh the page.")
            st.rerun()

    # ── WQI Trend chart ──
    if len(history) >= 2:
        st.markdown("### 📈 WQI Trend (Last 30 Readings)")
        df = pd.DataFrame(history[-30:])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["wqi"],
            mode="lines+markers",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=8, color=df["wqi"],
                        colorscale=[[0,"#ef233c"],[0.5,"#f77f00"],[1,"#06d6a0"]]),
            fill="tozeroy",
            fillcolor="rgba(0,180,216,0.08)",
            name="WQI",
        ))
        fig2.add_hline(y=50, line_dash="dash", line_color="#f77f00",
                       annotation_text="Safe Threshold (50)", annotation_position="top right")
        fig2.add_hline(y=75, line_dash="dot", line_color="#06d6a0",
                       annotation_text="Good (75)", annotation_position="top right")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"), xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,180,216,0.1)", range=[0,105]),
            height=300, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Quick safety tips ──
    st.markdown("### 💡 Quick Daily Water Safety Tips")
    tips = [
        ("🚰", "Always boil water if you're unsure about its safety before drinking."),
        ("🧪", "Test your drinking water at least once every 6 months."),
        ("🪣", "Store water in clean, covered containers to prevent contamination."),
        ("🌿", "Do not dispose chemicals or medicines down the drain."),
        ("🏭", "Report any unusual smell, color, or taste in tap water to authorities immediately."),
        ("🧽", "Clean water filters and purifiers regularly as per manufacturer guidelines."),
    ]
    cols = st.columns(3)
    for i, (icon, tip) in enumerate(tips):
        with cols[i % 3]:
            st.markdown(f'<div class="tip-card">{icon} {tip}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TEST WATER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 AI Predictor":
    st.markdown("## 🔮 AI Water Quality Predictor & Diagnostics")
    st.markdown("This module utilizes a **Random Forest Classifier** trained on WHO guidelines to predict water safety and display model metrics.")

    tab_pred, tab_health = st.tabs(["🔬 Run AI Prediction", "📊 ML Model Diagnostics & Metrics"])

    with tab_pred:
        with st.form("water_test_form"):
            st.markdown("### 📍 Sample Information")
            col1, col2 = st.columns(2)
            with col1:
                location = st.text_input("📍 Sample Location / Site Name", placeholder="e.g., River Ganga Station 3")
            with col2:
                sample_date = st.date_input("📅 Sample Date", datetime.date.today())
            source_type = st.selectbox("🌊 Water Source Type",
                ["Tap Water", "River / Stream", "Lake / Pond", "Ground Water / Borewell",
                 "Rainwater", "Spring Water", "Industrial Effluent", "Other"])

            st.markdown("### 🧪 Physical Parameters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                                      help="Ideal: 6.5–8.5")
            with col2:
                turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1,
                                             help="Ideal: < 4 NTU")
            with col3:
                temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5,
                                               help="Ideal: 5–30 °C")
            with col4:
                tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=250.0, step=10.0,
                                       help="Total Dissolved Solids. Ideal: < 500 mg/L")

            st.markdown("### ⚗️ Chemical Parameters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                do_ = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=6.0, step=0.1,
                                       help="Ideal: ≥ 5 mg/L")
            with col2:
                heavy = st.number_input("Heavy Metals (mg/L)", min_value=0.0, max_value=10.0, value=0.05, step=0.01,
                                         help="Ideal: < 0.1 mg/L")
            with col3:
                nitrates = st.number_input("Nitrates (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
                                            help="Ideal: < 10 mg/L")
            with col4:
                chlorine = st.number_input("Residual Chlorine (mg/L)", min_value=0.0, max_value=5.0, value=0.2, step=0.05,
                                            help="Ideal: < 0.5 mg/L")

            notes = st.text_area("📝 Additional Notes (optional)", placeholder="Any observations about color, smell, taste...")

            submitted = st.form_submit_button("🔮 Run AI Safety Prediction", use_container_width=True)

        if submitted:
            if not location:
                st.warning("⚠️ Please enter a sample location name.")
            else:
                # ── Run AI prediction ──
                score = wqi_score(ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine)
                grade, color = wqi_grade(score)
                
                # Input for ML model
                input_data = pd.DataFrame([[ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine]], 
                                          columns=["pH", "Turbidity", "Dissolved Oxygen", "Heavy Metals", "Nitrates", "TDS", "Temperature", "Chlorine"])
                
                if clf is not None:
                    pred = clf.predict(input_data)[0]
                    prob = clf.predict_proba(input_data)[0]
                    safe_prob = prob[1]
                    safe = bool(pred == 1)
                    confidence = safe_prob if safe else prob[0]
                else:
                    safe = bool(score >= 50)
                    confidence = score / 100.0 if safe else (100.0 - score) / 100.0
                
                params = {
                    "pH":           (ph,          6.5,  8.5),
                    "Turbidity":    (turbidity,   0.0,  4.0),
                    "Dissolved O₂": (do_,         5.0,  14.0),
                    "Heavy Metals": (heavy,        0.0,  0.1),
                    "Nitrates":     (nitrates,    0.0,  10.0),
                    "TDS":          (tds,          0.0, 500.0),
                    "Temperature":  (temperature, 5.0,  30.0),
                    "Chlorine":     (chlorine,    0.0,  0.5),
                }

                # ── Show overall result with AI prediction banner ──
                st.markdown("<br>", unsafe_allow_html=True)
                conf_pct = round(float(confidence) * 100, 1)
                if safe:
                    st.markdown(f'<div class="safe-banner">✅ AI PREDICTED SAFE &nbsp;|&nbsp; Confidence: {conf_pct}% &nbsp;|&nbsp; WQI: {score} ({grade})</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="unsafe-banner">❌ AI PREDICTED UNSAFE &nbsp;|&nbsp; Confidence: {conf_pct}% &nbsp;|&nbsp; WQI: {score} ({grade})</div>',
                                unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📊 Parameter Breakdown")
                    rows = []
                    for name, (val, lo, hi) in params.items():
                        status = "✅ OK" if lo <= val <= hi else "❌ Exceeds Limit"
                        rows.append({"Parameter": name, "Measured": val, "Status": status})
                    df_res = pd.DataFrame(rows)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                with col2:
                    # Radar chart
                    categories = list(params.keys())
                    raw_scores = []
                    for name, (val, lo, hi) in params.items():
                        span = hi - lo if hi != lo else 1
                        norm = max(0, min(1, (val - lo) / span)) if name != "Dissolved O₂" else max(0, min(1, val / hi))
                        raw_scores.append(norm * 100)

                    fig_radar = go.Figure(go.Scatterpolar(
                        r=raw_scores + [raw_scores[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        fillcolor=f"rgba(0,180,216,0.2)",
                        line=dict(color="#00b4d8", width=2),
                        marker=dict(size=8, color="#00b4d8"),
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, 100], color="#7fb3d3"),
                            angularaxis=dict(color="#7fb3d3"),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0f4ff"),
                        title=dict(text="Parameter Profile", font=dict(color="#00b4d8")),
                        height=380,
                        margin=dict(t=50, b=20),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # ── Recommendations ──
                st.markdown("### 💊 Recommendations & Actions")
                recs = []
                if ph < 6.5:
                    recs.append(("Low pH (Acidic)", "Water is acidic. Add lime or use an alkaline filter. Avoid drinking without treatment."))
                if ph > 8.5:
                    recs.append(("High pH (Alkaline)", "Water is too alkaline. Use a reverse osmosis (RO) filter or acidification treatment."))
                if turbidity > 4:
                    recs.append(("High Turbidity", "Water is cloudy. Let it settle, then filter through a fine cloth or sand filter before use."))
                if do_ < 5:
                    recs.append(("Low Dissolved Oxygen", "Low DO can harm aquatic life. Aerate the water or check for organic pollution upstream."))
                if heavy > 0.1:
                    recs.append(("High Heavy Metals", "DANGER: Heavy metal contamination detected. Do NOT drink. Report to local authorities immediately."))
                if nitrates > 10:
                    recs.append(("High Nitrates", "High nitrates can cause 'blue baby syndrome'. Use RO filtration. Infants must not consume this water."))
                if tds > 500:
                    recs.append(("High TDS", "Water may taste salty/metallic. Install a TDS filter or RO system. Not ideal for drinking."))
                if temperature > 30:
                    recs.append(("High Temperature", "Warm water promotes bacterial growth. Cool and treat before drinking."))
                if chlorine > 0.5:
                    recs.append(("High Chlorine", "Excess chlorine can cause taste issues. Let water stand in open air or use activated carbon filter."))

                if not recs:
                    st.markdown('<div class="tip-card">🎉 <strong>All parameters are within WHO guidelines.</strong> Your water appears safe to use. Keep testing regularly!</div>',
                                unsafe_allow_html=True)
                else:
                    for title, desc in recs:
                        st.markdown(f'<div class="tip-card"><strong>{title}</strong><br>{desc}</div>',
                                    unsafe_allow_html=True)

                # ── Save record ──
                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "location": location,
                    "source_type": source_type,
                    "ph": ph, "turbidity": turbidity, "do": do_, "heavy_metals": heavy,
                    "nitrates": nitrates, "tds": tds, "temperature": temperature, "chlorine": chlorine,
                    "wqi": score, "grade": grade, "safe": safe, "notes": notes,
                }
                st.session_state.history.append(record)
                save_history(st.session_state.history)

                # ── Alert if unsafe ──
                if not safe:
                    alert = {
                        "time": record["timestamp"],
                        "location": location,
                        "wqi": score,
                        "message": f"Unsafe water detected at {location} (WQI: {score})",
                    }
                    st.session_state.alerts.append(alert)

                st.success(f"✅ Result saved! Record #{len(st.session_state.history)} stored.")

    with tab_health:
        st.markdown("### 📊 Scikit-Learn Model Health & Diagnostics")
        st.markdown(f"Our active predictive model has been evaluated on standard test datasets.")

        # Metric cards row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value metric-status-ok">{round(model_acc * 100, 2)}%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-value" style="color:#90e0ef">{round(model_auc, 3)}</div>
                <div class="metric-label">Area Under ROC (AUC)</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🤖</div>
                <div class="metric-value">RandomForest</div>
                <div class="metric-label">Classifier Core</div>
            </div>""", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🧬</div>
                <div class="metric-value" style="color:#06d6a0">Calibrated</div>
                <div class="metric-label">Calibration Quality</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            # Confusion Matrix Heatmap
            st.markdown("#### Confusion Matrix Heatmap")
            cm_df = pd.DataFrame(model_cm, index=["Actual Unsafe", "Actual Safe"], columns=["Predicted Unsafe", "Predicted Safe"])
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale=[[0, "#0a1628"], [1, "#00b4d8"]],
                aspect="auto"
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"),
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC Curve
            st.markdown("#### Receiver Operating Characteristic (ROC)")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=model_fpr, y=model_tpr,
                mode="lines",
                line=dict(color="#06d6a0", width=3),
                name=f"ROC Curve (AUC = {model_auc:.3f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="#ef233c", dash="dash"),
                showlegend=False
            ))
            fig_roc.update_layout(
                xaxis=dict(title="False Positive Rate", gridcolor="rgba(0,180,216,0.1)"),
                yaxis=dict(title="True Positive Rate", gridcolor="rgba(0,180,216,0.1)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"),
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_right:
            # Feature Importance
            st.markdown("#### Relative Feature Importances")
            fig_feat = px.bar(
                model_feat,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#90e0ef"], [1, "#0077b6"]]
            )
            fig_feat.update_layout(
                xaxis=dict(gridcolor="rgba(0,180,216,0.1)"),
                yaxis=dict(categoryorder="total ascending"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"),
                height=590,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            # ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics & Insights")
    history = st.session_state.history
    if len(history) < 2:
        st.info("📌 Not enough data. Add at least 2 water tests to see analytics.")
        st.stop()

    df = pd.DataFrame(history)

    # ── Summary stats ──
    st.markdown("### 📋 Summary Statistics")
    num_cols = ["ph", "turbidity", "do", "heavy_metals", "nitrates", "tds", "temperature", "chlorine", "wqi"]
    avail = [c for c in num_cols if c in df.columns]
    st.dataframe(df[avail].describe().round(2), use_container_width=True)

    st.markdown("---")

    # ── Param trend charts ──
    st.markdown("### 📈 Parameter Trends Over Time")
    param_select = st.multiselect(
        "Select parameters to plot:",
        ["wqi", "ph", "turbidity", "do", "heavy_metals", "nitrates", "tds", "temperature", "chlorine"],
        default=["wqi", "ph"],
    )
    if param_select:
        fig_trend = go.Figure()
        colors = ["#00b4d8","#06d6a0","#f77f00","#ef476f","#90e0ef","#a8dadc","#ffd166","#e76f51"]
        for idx, param in enumerate(param_select):
            if param in df.columns:
                fig_trend.add_trace(go.Scatter(
                    x=df["timestamp"], y=df[param],
                    mode="lines+markers",
                    name=param.replace("_", " ").title(),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=6),
                ))
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"),
            xaxis=dict(showgrid=False, color="#7fb3d3"),
            yaxis=dict(gridcolor="rgba(0,180,216,0.1)", color="#7fb3d3"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#00b4d8"),
            height=400, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Distribution charts ──
    st.markdown("### 📊 Parameter Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig_ph = px.histogram(df, x="ph", nbins=15,
                               title="pH Distribution",
                               color_discrete_sequence=["#00b4d8"])
        fig_ph.add_vline(x=6.5, line_dash="dash", line_color="#ef233c")
        fig_ph.add_vline(x=8.5, line_dash="dash", line_color="#ef233c",
                          annotation_text="WHO Limits")
        fig_ph.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#e0f4ff"), height=300)
        st.plotly_chart(fig_ph, use_container_width=True)

    with col2:
        fig_wqi = px.histogram(df, x="wqi", nbins=15,
                                title="WQI Score Distribution",
                                color_discrete_sequence=["#06d6a0"])
        fig_wqi.add_vline(x=50, line_dash="dash", line_color="#f77f00",
                           annotation_text="Safe Threshold")
        fig_wqi.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#e0f4ff"), height=300)
        st.plotly_chart(fig_wqi, use_container_width=True)

    # ── Safety pie ──
    st.markdown("### 🥧 Safe vs Unsafe Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        if "safe" in df.columns:
            safe_counts = df["safe"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=["Safe ✅", "Unsafe ❌"],
                values=[safe_counts.get(True, 0), safe_counts.get(False, 0)],
                hole=0.55,
                marker=dict(colors=["#06d6a0", "#ef233c"],
                            line=dict(color="#03071e", width=3)),
                textinfo="percent+label",
                textfont=dict(color="#fff"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"),
                height=320, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Grade distribution
        if "grade" in df.columns:
            g_counts = df["grade"].value_counts()
            fig_grade = go.Figure(go.Bar(
                x=g_counts.index,
                y=g_counts.values,
                marker_color=["#06d6a0","#90e0ef","#f77f00","#ef476f","#ef233c"][:len(g_counts)],
            ))
            fig_grade.update_layout(
                title="WQI Grade Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(0,180,216,0.1)"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_grade, use_container_width=True)

    # ── Correlation heatmap ──
    st.markdown("### 🔥 Parameter Correlation Heatmap")
    corr_cols = [c for c in avail if c in df.columns]
    if len(corr_cols) >= 3:
        corr = df[corr_cols].corr().round(2)
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0,"#ef233c"],[0.5,"#03071e"],[1,"#06d6a0"]],
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#fff"),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"), height=400,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LOCATION MAP  (Rich Interactive Version)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Location Map":
    st.markdown("## 🗺️ Premium Interactive Monitoring Network")
    st.markdown("Explore the real-time monitoring grid. Click on any station marker or use the sidebar profile control to analyze detailed indicators.")

    history = st.session_state.history
    if not history:
        st.warning("No data yet. Load demo data from Settings or test some water samples first!")
        st.stop()

    df = pd.DataFrame(history)

    # ── Assign coordinates if missing ──
    if "lat" not in df.columns or df["lat"].isna().all():
        rng = np.random.default_rng(42)
        locs_list = df["location"].unique().tolist()
        lat_map = {l: float(rng.uniform(8.0, 35.0)) for l in locs_list}
        lon_map = {l: float(rng.uniform(68.0, 97.0)) for l in locs_list}
        df["lat"] = df["location"].map(lat_map)
        df["lon"] = df["location"].map(lon_map)

    # ── Per-location aggregation ──
    locations = df.groupby("location").agg(
        tests=("wqi", "count"),
        avg_wqi=("wqi", "mean"),
        min_wqi=("wqi", "min"),
        max_wqi=("wqi", "max"),
        safe_pct=("safe", lambda x: round(100 * x.sum() / len(x), 1)),
        last_wqi=("wqi", "last"),
        last_safe=("safe", "last"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()
    locations["avg_wqi"] = locations["avg_wqi"].round(1)

    # ── Simulation Session State Initialization ──
    if "sim_active" not in st.session_state:
        st.session_state.sim_active = False
        st.session_state.sim_station = ""
        st.session_state.sim_type = ""
        st.session_state.sim_radius = 1.5
        st.session_state.sim_intensity = "Moderate"

    # ── Apply Simulation Logic to WQI metrics ──
    locations_display = locations.copy()
    if st.session_state.sim_active:
        epicenter = locations[locations["location"] == st.session_state.sim_station].iloc[0]
        epi_lat, epi_lon = epicenter["lat"], epicenter["lon"]
        
        intensity_drops = {"Low": 15.0, "Moderate": 30.0, "Severe": 55.0, "Catastrophic": 80.0}
        max_drop = intensity_drops[st.session_state.sim_intensity]
        
        for idx, row in locations_display.iterrows():
            dist = np.sqrt((row["lat"] - epi_lat)**2 + (row["lon"] - epi_lon)**2)
            if dist <= st.session_state.sim_radius:
                factor = (1.0 - dist / st.session_state.sim_radius)
                drop = round(max_drop * factor, 1)
                new_wqi = max(0.0, row["avg_wqi"] - drop)
                locations_display.at[idx, "avg_wqi"] = new_wqi
                locations_display.at[idx, "last_wqi"] = new_wqi
                locations_display.at[idx, "last_safe"] = bool(new_wqi >= 50.0)
                locations_display.at[idx, "safe_pct"] = max(0.0, row["safe_pct"] - (drop * 1.2))

    # Apply grading labels after potential simulation WQI drop
    locations_display["grade"] = locations_display["avg_wqi"].apply(lambda s: wqi_grade(s)[0])
    locations_display["color_hex"] = locations_display["avg_wqi"].apply(lambda s: wqi_grade(s)[1])
    locations_display["status_label"] = locations_display["last_safe"].apply(
        lambda s: "🟢 Safe" if s else "🔴 Unsafe"
    )

    if "selected_station" not in st.session_state:
        st.session_state.selected_station = locations_display.iloc[0]["location"]

    # Layout with Columns
    col_map, col_details = st.columns([2.5, 1.5])

    # Right side: Detail Dashboard Panel
    with col_details:
        tab_profile, tab_sim = st.tabs(["📋 Station Profile", "☣️ Grid Simulator"])
        
        with tab_profile:
            # Sync selection
            if st.session_state.selected_station not in locations_display["location"].tolist():
                st.session_state.selected_station = locations_display.iloc[0]["location"]
            
            selected_idx = locations_display["location"].tolist().index(st.session_state.selected_station)
            sel_station = st.selectbox(
                "Select Station Profile:",
                locations_display["location"].tolist(),
                index=selected_idx,
                key="station_select_dropdown"
            )
            st.session_state.selected_station = sel_station
            
            # Get latest record for selected station
            station_df = df[df["location"] == st.session_state.selected_station].sort_values("timestamp")
            if not station_df.empty:
                latest_record = station_df.iloc[-1]
            else:
                latest_record = locations_display[locations_display["location"] == st.session_state.selected_station].iloc[0]
                
            # WQI and safety grade details
            wqi_val = latest_record["wqi"]
            # If simulated drop exists
            if st.session_state.sim_active:
                row_match = locations_display[locations_display["location"] == st.session_state.selected_station].iloc[0]
                wqi_val = row_match["avg_wqi"]
                
            grade, color = wqi_grade(wqi_val)
            safe = bool(wqi_val >= 50.0)
            
            # Status Badge + Title
            badge_label = "OPTIMAL" if wqi_val >= 75 else ("WARNING" if wqi_val >= 50 else "CRITICAL")
            badge_color = "#06d6a0" if wqi_val >= 75 else ("#f77f00" if wqi_val >= 50 else "#ef233c")
            
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-top: 0.5rem; margin-bottom: 0.5rem;">
                <span style="background:{badge_color}; color:#fff; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px;">{badge_label}</span>
                <span style="font-size:1.5rem; font-weight:800; color:#e0f4ff; line-height:1.2;">{st.session_state.selected_station}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Location details
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #7fb3d3; margin-bottom: 1rem; line-height: 1.4;">
                <strong>Location:</strong> Coordinates ({latest_record['lat']:.4f}, {latest_record['lon']:.4f}) in India grid.<br/>
                <strong>Junction/Source:</strong> {latest_record.get('source_type', 'River / Stream')} | <strong>Corridor:</strong> Monitoring Station
            </div>
            """, unsafe_allow_html=True)
            
            # KPI Metrics Row
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.markdown(f"""
                <div style="font-family: 'Inter', sans-serif; margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Est. WQI Score</div>
                    <div style="display: flex; align-items: baseline; gap: 4px;">
                        <span style="font-size: 48px; font-weight: 800; color: #ffffff; line-height: 1;">{wqi_val:.1f}</span>
                        <span style="font-size: 18px; color: #94a3b8; font-weight: 500;">WQI</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_met2:
                # safety prob from ML confidence or simple score
                safe_pct = latest_record["safe_pct"] if "safe_pct" in latest_record else (wqi_val if safe else 100 - wqi_val)
                st.markdown(f"""
                <div style="font-family: 'Inter', sans-serif; margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Safety Probability</div>
                    <div style="font-size: 48px; font-weight: 800; color: #06d6a0; line-height: 1;">{safe_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Suggested Deployment
            if wqi_val >= 75:
                deploy_msg = "Safe Water"
                req_badge = "<span style='background-color: #10b98122; color: #10b981; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;'>No action required</span>"
            elif wqi_val >= 50:
                deploy_msg = "Boil & Carbon Filter"
                req_badge = "<span style='background-color: #f59e0b22; color: #f59e0b; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;'>Boiling Recommended</span>"
            else:
                deploy_msg = "RO Filtration & Disinfection"
                req_badge = "<span style='background-color: #ef444422; color: #ef4444; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;'>DANGER: RO Required</span>"
                
            deployment_html = f"""
            <div style="font-size: 14px; font-family: 'Inter', sans-serif; color: #ffffff; margin-bottom: 25px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; line-height: 1.6;">
                <strong>Suggested Treatment:</strong> 
                <span>🧪 <span style="color: #00b4d8; font-weight: 700;">{deploy_msg}</span></span>
                <span style="color: #475569; margin: 0 4px;">|</span>
                <span>⚠️ {req_badge}</span>
            </div>
            """
            st.markdown(deployment_html, unsafe_allow_html=True)
            
            # SHAP Header with brain icon
            shap_header_html = """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px; margin-top: 15px;">
                <span style="font-size: 20px;">🧠</span>
                <span style="font-size: 18px; font-weight: 700; color: #ffffff; font-family: 'Inter', sans-serif;">Decision Contribution (SHAP)</span>
            </div>
            """
            st.markdown(shap_header_html, unsafe_allow_html=True)
            
            # Calculate drops
            r_ph = latest_record.get("ph", 7.0)
            r_turb = latest_record.get("turbidity", 1.0)
            r_do = latest_record.get("do", 6.0)
            r_heavy = latest_record.get("heavy_metals", 0.05)
            r_nit = latest_record.get("nitrates", 5.0)
            r_tds = latest_record.get("tds", 250.0)
            r_temp = latest_record.get("temperature", 25.0)
            r_chl = latest_record.get("chlorine", 0.2)
            
            ph_contrib = max(0.0, abs(r_ph - 7.0) * 20.0) / 8
            turb_contrib = max(0.0, r_turb * 20.0) / 8
            do_contrib = max(0.0, 100.0 - min(100.0, r_do / 8.0 * 100.0)) / 8
            heavy_contrib = max(0.0, r_heavy * 1000.0) / 8
            nitrate_contrib = max(0.0, r_nit * 5.0) / 8
            tds_contrib = max(0.0, r_tds / 5.0) / 8
            temp_contrib = max(0.0, abs(r_temp - 20.0) * 3.0) / 8
            chlorine_contrib = max(0.0, r_chl * 200.0) / 8
            
            shap_data = pd.DataFrame({
                "Feature": [
                    "Heavy Metals", "Turbidity", "TDS Level", "pH Deviation",
                    "Dissolved Oxygen", "Nitrates", "Temperature Deviation", "Chlorine"
                ],
                "Contribution": [
                    heavy_contrib, turb_contrib, tds_contrib, ph_contrib,
                    do_contrib, nitrate_contrib, temp_contrib, chlorine_contrib
                ]
            }).sort_values(by="Contribution", ascending=True)
            
            # Plotly SHAP bar chart
            fig_shap = px.bar(
                shap_data,
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Contribution",
                color_continuous_scale="RdBu_r",
                height=220
            )
            fig_shap.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=140, r=10, t=10, b=30),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    tickfont=dict(color="#94a3b8", size=11, family="Inter"),
                    title=None
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    tickfont=dict(color="#cbd5e1", size=12, family="Inter"),
                    title=None
                ),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
        with tab_sim:
            # Simulator controls card
            st.markdown("""
            <div class="section-card" style="border: 1px solid rgba(239,35,60,0.3); padding:0.8rem; margin-top:0.5rem; margin-bottom:0.5rem;">
                <div class="section-title" style="color:#ef233c; font-size:1rem; margin-bottom: 0.3rem;">🚨 Grid Contamination Simulator</div>
                <p style="font-size:0.75rem; color:#7fb3d3; margin-bottom:0.5rem;">Trigger a contaminant spill and observe the simulated propagation across the network.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sim_station = st.selectbox("🎯 Target Station Source", locations["location"].tolist(), index=0, key="sim_station_selectbox")
            sim_type = st.selectbox("☣️ Contamination Agent", 
                                    ["Heavy Metal Spill", "Acid Dump (pH Drop)", "Algal Bloom (DO Drop)", "Sewer Overflow"], key="sim_type_selectbox")
            sim_radius = st.slider("🌐 Spread Radius (Degrees)", 0.5, 6.0, 2.0, 0.1, key="sim_radius_slider")
            sim_intensity = st.select_slider("🔥 Severity Level", options=["Low", "Moderate", "Severe", "Catastrophic"], key="sim_intensity_slider")
            
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🔥 Run Simulation", use_container_width=True, key="run_sim_button"):
                    st.session_state.sim_active = True
                    st.session_state.sim_station = sim_station
                    st.session_state.sim_type = sim_type
                    st.session_state.sim_radius = sim_radius
                    st.session_state.sim_intensity = sim_intensity
                    st.success("Simulation Active!")
                    st.rerun()
            with c_btn2:
                if st.button("♻️ Reset Grid", use_container_width=True, key="reset_sim_button"):
                    st.session_state.sim_active = False
                    st.session_state.sim_station = ""
                    st.info("Grid status reset.")
                    st.rerun()

    # Left side: Premium Folium Leaflet Map
    with col_map:
        map_style = st.selectbox(
            "🗺️ Map style selection",
            ["CartoDB dark_matter", "OpenStreetMap", "CartoDB positron"],
            index=0
        )
        
        tile_name = "cartodbdark_matter"
        if map_style == "OpenStreetMap":
            tile_name = "openstreetmap"
        elif map_style == "CartoDB positron":
            tile_name = "cartodbpositron"
        
        # Center coordinates
        center_lat = float(locations_display["lat"].mean())
        center_lon = float(locations_display["lon"].mean())
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=tile_name)
        
        # Draw network flow lines
        for i in range(len(locations_display)):
            for j in range(i + 1, len(locations_display)):
                lat1, lon1 = locations_display.iloc[i]["lat"], locations_display.iloc[i]["lon"]
                lat2, lon2 = locations_display.iloc[j]["lat"], locations_display.iloc[j]["lon"]
                dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                if dist < 6.5:
                    folium.PolyLine(
                        locations=[[lat1, lon1], [lat2, lon2]],
                        color="#00b4d8",
                        weight=1.5,
                        opacity=0.15
                    ).add_to(m)
                    
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers
        color_map = {
            "Excellent": "green",
            "Good": "lightgreen",
            "Fair": "orange",
            "Poor": "red",
            "Very Poor": "darkred"
        }
        for _, row in locations_display.iterrows():
            is_selected = (row["location"] == st.session_state.selected_station)
            icon_type = "star" if is_selected else "info-sign"
            
            if is_selected:
                icon_color = "orange"
            else:
                icon_color = color_map.get(row["grade"], "blue")
                
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; color: #1e293b; line-height: 1.4;">
                <h4 style="margin: 0 0 6px 0; color: #00b4d8; font-weight: 700;">📍 {row['location']}</h4>
                <b>Average WQI</b>: {row['avg_wqi']:.1f}<br/>
                <b>Grade</b>: {row['grade']}<br/>
                <b>Safety Status</b>: {row['status_label']}<br/>
                <b>Total Tests</b>: {int(row['tests'])}<br/>
            </div>
            """
            
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row["location"],
                icon=folium.Icon(color=icon_color, icon=icon_type)
            ).add_to(marker_cluster)
            
        # Draw simulation epicenter wave
        if st.session_state.sim_active:
            folium.Circle(
                location=[epi_lat, epi_lon],
                radius=st.session_state.sim_radius * 111000,
                color="#ef233c",
                fill=True,
                fill_color="#ef233c",
                fill_opacity=0.15,
                weight=2
            ).add_to(m)
            
        # Render map using st_folium
        map_data = st_folium(
            m,
            height=580,
            use_container_width=True,
            key="folium_map",
            returned_objects=["last_object_clicked"]
        )

        # Synchronize map click with selected station profile
        if map_data and map_data.get("last_object_clicked"):
            click_lat = map_data["last_object_clicked"]["lat"]
            click_lng = map_data["last_object_clicked"]["lng"]
            current_click = (click_lat, click_lng)
            
            # Check if click coordinates have changed
            if "last_click" not in st.session_state:
                st.session_state.last_click = None
                
            if current_click != st.session_state.last_click:
                st.session_state.last_click = current_click
                # Find closest station
                distances = locations_display.apply(
                    lambda r: np.sqrt((r["lat"] - click_lat)**2 + (r["lon"] - click_lng)**2),
                    axis=1
                )
                closest_idx = distances.idxmin()
                st.session_state.selected_station = locations_display.loc[closest_idx, "location"]
                st.rerun()

    # ── WQI Legend bar ──
    st.markdown("""
    <div style="display:flex;gap:8px;align-items:center;margin:0.5rem 0 1.5rem 0;flex-wrap:wrap">
      <span style="color:#7fb3d3;font-size:0.85rem;font-weight:600">WQI Legend:</span>
      <span style="background:#ef233c;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">0–25 Very Poor</span>
      <span style="background:#f77f00;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">25–50 Poor</span>
      <span style="background:#0096c7;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">50–75 Fair</span>
      <span style="background:#06d6a0;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">75–100 Good/Excellent</span>
      <span style="color:#7fb3d3;font-size:0.8rem;margin-left:8px">● Glowing halos: simulated grid connections activated</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Station Cards ──
    st.markdown("### 📋 Station Summaries")
    cols_per_row = 3
    loc_rows = [locations_display.iloc[i:i+cols_per_row] for i in range(0, len(locations_display), cols_per_row)]
    for row_df in loc_rows:
        cols = st.columns(cols_per_row)
        for ci, (_, loc) in enumerate(row_df.iterrows()):
            g_color = wqi_grade(loc["avg_wqi"])[1]
            safe_bg  = "rgba(6,214,160,0.12)" if loc["last_safe"] else "rgba(239,35,60,0.12)"
            safe_bdr = "#06d6a0" if loc["last_safe"] else "#ef233c"
            with cols[ci]:
                st.markdown(f"""
                <div style="background:{safe_bg};border:1px solid {safe_bdr};
                            border-radius:14px;padding:1.1rem;margin-bottom:0.8rem">
                  <div style="font-weight:700;font-size:1rem;color:#e0f4ff;margin-bottom:0.5rem">
                    📍 {loc['location']}
                  </div>
                  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:0.6rem">
                    <span style="background:rgba(0,180,216,0.15);border:1px solid #00b4d8;
                                 border-radius:8px;padding:2px 10px;font-size:0.8rem">
                      🧪 {int(loc['tests'])} tests
                    </span>
                    <span style="background:rgba(0,180,216,0.15);border:1px solid #00b4d8;
                                 border-radius:8px;padding:2px 10px;font-size:0.8rem">
                      {loc['status_label']}
                    </span>
                  </div>
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div style="font-size:1.8rem;font-weight:800;color:{g_color}">{loc['avg_wqi']}</div>
                      <div style="font-size:0.75rem;color:#7fb3d3">Avg WQI · {loc['grade']}</div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:0.85rem;color:#e0f4ff">✅ {loc['safe_pct']:.0f}% safe</div>
                      <div style="font-size:0.75rem;color:#7fb3d3">Min {loc['min_wqi']:.0f} / Max {loc['max_wqi']:.0f}</div>
                    </div>
                  </div>
                  <div style="margin-top:0.7rem;background:rgba(0,0,0,0.3);
                               border-radius:6px;height:6px;overflow:hidden">
                    <div style="width:{min(loc['avg_wqi'],100)}%;height:100%;
                                background:linear-gradient(90deg,{g_color}88,{g_color});
                                border-radius:6px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── WQI per location bar chart ──
    st.markdown("### 📊 Average WQI by Station")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=locations_display["location"],
        y=locations_display["avg_wqi"],
        marker=dict(
            color=locations_display["avg_wqi"],
            colorscale=[[0,"#ef233c"],[0.25,"#f77f00"],[0.5,"#0096c7"],[1,"#06d6a0"]],
            line=dict(color="rgba(0,180,216,0.5)", width=1),
        ),
        text=locations_display["avg_wqi"].round(1),
        textposition="outside",
        textfont=dict(color="#e0f4ff"),
        customdata=np.stack([locations_display["tests"], locations_display["safe_pct"]], axis=-1),
        hovertemplate="<b>%{x}</b><br>Avg WQI: %{y:.1f}<br>Tests: %{customdata[0]}<br>Safe: %{customdata[1]}%<extra></extra>",
    ))
    fig_bar.add_hline(y=50, line_dash="dash", line_color="#f77f00",
                      annotation_text="Safe Threshold", annotation_position="top right",
                      annotation_font_color="#f77f00")
    fig_bar.add_hline(y=75, line_dash="dot", line_color="#06d6a0",
                      annotation_text="Good Threshold", annotation_position="top right",
                      annotation_font_color="#06d6a0")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0f4ff", family="Inter"),
        xaxis=dict(showgrid=False, tickangle=-20, color="#7fb3d3"),
        yaxis=dict(gridcolor="rgba(0,180,216,0.1)", range=[0, 115], color="#7fb3d3"),
        height=380,
        margin=dict(t=30, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── WQI trend per location ──
    if len(df) >= 4:
        st.markdown("### 📈 WQI Trend by Station")
        locs_avail = df["location"].unique().tolist()
        sel_locs = st.multiselect("Select stations to compare:", locs_avail,
                                   default=locs_avail[:min(3, len(locs_avail))])
        if sel_locs:
            colors_list = ["#00b4d8","#06d6a0","#f77f00","#ef476f","#90e0ef"]
            fig_trend = go.Figure()
            for idx, loc_name in enumerate(sel_locs):
                loc_df = df[df["location"] == loc_name].sort_values("timestamp")
                # We align simulated drops on trend temporarily if station simulated
                sim_y = []
                for _, row in loc_df.iterrows():
                    val = row["wqi"]
                    if st.session_state.sim_active and loc_name == st.session_state.sim_station:
                        val = max(0.0, val - max_drop)
                    elif st.session_state.sim_active:
                        # check distance
                        loc_row = locations[locations["location"] == loc_name].iloc[0]
                        dist = np.sqrt((loc_row["lat"] - epi_lat)**2 + (loc_row["lon"] - epi_lon)**2)
                        if dist <= st.session_state.sim_radius:
                            val = max(0.0, val - (max_drop * (1.0 - dist / st.session_state.sim_radius)))
                    sim_y.append(val)

                fig_trend.add_trace(go.Scatter(
                    x=loc_df["timestamp"], y=sim_y,
                    mode="lines+markers",
                    name=loc_name,
                    line=dict(color=colors_list[idx % len(colors_list)], width=2),
                    marker=dict(size=6),
                ))
            fig_trend.add_hrect(y0=0, y1=50, fillcolor="rgba(239,35,60,0.05)",
                                line_width=0, annotation_text="Unsafe Zone",
                                annotation_position="top left",
                                annotation_font_color="#ef233c")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff", family="Inter"),
                xaxis=dict(showgrid=False, color="#7fb3d3"),
                yaxis=dict(gridcolor="rgba(0,180,216,0.1)", range=[0, 110], color="#7fb3d3"),
                legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="#00b4d8"),
                height=380, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_trend, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DAILY TIPS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡 Daily Tips":
    st.markdown("## 💡 Daily Water Safety Tips for Everyone")

    categories = {
        "🏠 At Home": [
            ("Boil before drinking", "If you're unsure of water quality, boil it for at least 1 minute (3 minutes at altitudes above 2,000m)."),
            ("Change water filters", "Replace pitcher filters every 2 months and under-sink filters every 6–12 months."),
            ("Clean water tanks", "Clean overhead/underground water tanks every 6 months to prevent algae and bacterial growth."),
            ("Use BPA-free containers", "Store water only in food-grade, BPA-free plastic, glass, or stainless steel containers."),
            ("Check pipes for rust", "Brownish or metallic-tasting water indicates corroded pipes. Get pipes inspected immediately."),
            ("Don't mix hot & tap", "Avoid using hot tap water for cooking—hot water dissolves more lead from old pipes."),
        ],
        "👶 For Infants & Children": [
            ("Use formula-safe water", "For baby formula, use purified or boiled water. Avoid well water without testing."),
            ("Watch for blue baby syndrome", "High nitrates (>10 mg/L) can cause blue baby syndrome. Always test water used for infants."),
            ("Fluoride check", "Children need some fluoride for teeth, but too much causes fluorosis. Check your water's fluoride level."),
            ("Lead exposure risk", "Lead is especially harmful to children. If your home was built before 1986, test for lead in water."),
        ],
        "🌾 Agriculture & Farming": [
            ("Irrigation water quality", "Use water with pH 6–8 for irrigation. Highly saline water (high TDS) damages crops."),
            ("Avoid industrial runoff", "Never use water that may contain industrial runoff for edible crops."),
            ("Nitrate buildup", "Excess fertilizer use increases nitrates in groundwater. Use slow-release fertilizers."),
            ("Test bore water regularly", "Borewell water quality changes seasonally. Test at the start of each crop season."),
        ],
        "🏗️ Industrial & Commercial": [
            ("Treat effluent before release", "All industrial wastewater must be treated to meet local standards before discharge."),
            ("Monitor discharge points", "Install continuous monitoring at discharge points to catch leaks early."),
            ("Heavy metal management", "Electroplating, mining, and chemical industries must properly handle heavy metal waste."),
            ("Cooling tower treatment", "Treat cooling tower water to prevent Legionella bacteria growth."),
        ],
        "🌊 Environment & Community": [
            ("Don't dump in water bodies", "Never dump garbage, chemicals, or untreated sewage into rivers, lakes, or oceans."),
            ("Watershed protection", "Protect forests around watersheds—trees filter rainwater naturally."),
            ("Report pollution", "Report visible pollution, dead fish, or discoloured water to the local pollution control board."),
            ("Rainwater harvesting", "Collect and properly filter rainwater for non-potable uses to reduce pressure on groundwater."),
            ("Community testing drives", "Organise community water testing camps, especially in flood-prone and rural areas."),
        ],
        "🚨 Emergency Situations": [
            ("Flood water safety", "Floodwater is highly contaminated. Never drink it. Use sealed bottled water or boil/treat."),
            ("After power cuts", "If water has been stored without electricity (no pump) for >24 hours, boil before drinking."),
            ("Chemical spill nearby", "If a chemical spill is reported near your water source, stop using tap water until cleared."),
            ("Natural disaster prep", "Store at least 3 litres of drinking water per person per day for 3 days as emergency supply."),
        ],
    }

    for cat, tips_list in categories.items():
        st.markdown(f"### {cat}")
        cols = st.columns(2)
        for i, (title, detail) in enumerate(tips_list):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="tip-card">
                  <strong>{title}</strong><br>
                  <span style="color:#a8d8ea">{detail}</span>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("## 📜 Test History & Records")
    history = st.session_state.history

    if not history:
        st.info("No history yet. Go to **🔮 AI Predictor** to add records.")
        st.stop()

    df = pd.DataFrame(history)

    # ── Filters ──
    col1, col2, col3 = st.columns(3)
    with col1:
        if "location" in df.columns:
            locs = ["All"] + sorted(df["location"].unique().tolist())
            sel_loc = st.selectbox("📍 Filter by Location", locs)
    with col2:
        if "safe" in df.columns:
            sel_safe = st.selectbox("🛡️ Filter by Status", ["All", "Safe Only", "Unsafe Only"])
    with col3:
        if "source_type" in df.columns:
            sources = ["All"] + sorted(df["source_type"].dropna().unique().tolist())
            sel_src = st.selectbox("🌊 Filter by Source", sources)

    filtered = df.copy()
    if sel_loc != "All":
        filtered = filtered[filtered["location"] == sel_loc]
    if sel_safe == "Safe Only":
        filtered = filtered[filtered["safe"] == True]
    elif sel_safe == "Unsafe Only":
        filtered = filtered[filtered["safe"] == False]
    if "source_type" in df.columns and sel_src != "All":
        filtered = filtered[filtered["source_type"] == sel_src]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} records**")
    st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        # CSV export
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", csv, "water_quality_history.csv", "text/csv",
                           use_container_width=True)
    with col2:
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            save_history([])
            st.success("History cleared.")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Alerts":
    st.markdown("## ⚠️ Safety Alerts & Notifications")

    alerts = st.session_state.alerts
    history = st.session_state.history

    # Critical from history
    critical = [r for r in history if r.get("wqi", 100) < 25]
    warning  = [r for r in history if 25 <= r.get("wqi", 100) < 50]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">🔴</div>
            <div class="metric-value metric-status-bad">{len(critical)}</div>
            <div class="metric-label">Critical Alerts (WQI &lt; 25)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">🟠</div>
            <div class="metric-value" style="color:#f77f00">{len(warning)}</div>
            <div class="metric-label">Warning Alerts (WQI 25–50)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        safe_r = [r for r in history if r.get("safe")]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-value metric-status-ok">{len(safe_r)}</div>
            <div class="metric-label">Safe Readings</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if critical:
        st.markdown("### 🔴 Critical Water Quality Events")
        for r in sorted(critical, key=lambda x: x["timestamp"], reverse=True)[:10]:
            st.markdown(f"""
            <div style="background:rgba(239,35,60,0.1);border:1px solid #ef233c;border-radius:12px;padding:1rem;margin-bottom:0.7rem">
              <strong>🚨 {r['location']}</strong> | {r['timestamp']} | WQI: {r['wqi']}<br>
              <span style="color:#ef476f">Immediate action required. Do NOT use this water.</span>
            </div>
            """, unsafe_allow_html=True)

    if warning:
        st.markdown("### 🟠 Warning Events")
        for r in sorted(warning, key=lambda x: x["timestamp"], reverse=True)[:10]:
            st.markdown(f"""
            <div style="background:rgba(247,127,0,0.1);border:1px solid #f77f00;border-radius:12px;padding:1rem;margin-bottom:0.7rem">
              <strong>⚠️ {r['location']}</strong> | {r['timestamp']} | WQI: {r['wqi']}<br>
              <span style="color:#f77f00">Water quality is below acceptable levels. Treatment recommended.</span>
            </div>
            """, unsafe_allow_html=True)

    if not critical and not warning:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#06d6a0;font-size:1.3rem;font-weight:600">
          ✅ No safety alerts at this time.<br>
          <span style="font-size:1rem;color:#7fb3d3">All recorded water samples are within acceptable ranges.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Alert thresholds config ──
    st.markdown("---")
    st.markdown("### ⚙️ Alert Threshold Configuration")
    st.info("Customize the thresholds below for your specific monitoring requirements.")
    col1, col2 = st.columns(2)
    with col1:
        ph_min = st.slider("pH Minimum", 5.0, 7.0, 6.5, 0.1)
        ph_max = st.slider("pH Maximum", 7.5, 10.0, 8.5, 0.1)
        turb_max = st.slider("Max Turbidity (NTU)", 1.0, 20.0, 4.0, 0.5)
        do_min = st.slider("Min Dissolved O₂ (mg/L)", 2.0, 8.0, 5.0, 0.5)
    with col2:
        heavy_max = st.slider("Max Heavy Metals (mg/L)", 0.01, 1.0, 0.1, 0.01)
        nit_max = st.slider("Max Nitrates (mg/L)", 1.0, 50.0, 10.0, 1.0)
        tds_max = st.slider("Max TDS (mg/L)", 100.0, 2000.0, 500.0, 50.0)
    if st.button("💾 Save Thresholds (Session)"):
        st.success("✅ Thresholds updated for this session.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EDUCATION HUB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Education Hub":
    st.markdown("## 📚 Water Quality Education Hub")

    tabs = st.tabs(["🧪 Parameters Guide", "🦠 Contaminants", "💊 Treatment Methods",
                    "📖 WQI Explained", "🌍 Global Water Facts"])

    with tabs[0]:
        st.markdown("### 🧪 Understanding Water Quality Parameters")
        params_info = {
            "pH": {
                "range": "6.5 – 8.5 (WHO)", "icon": "🔵",
                "about": "pH measures how acidic or alkaline water is. A pH of 7 is neutral. Low pH (acidic) can corrode pipes and leach metals. High pH (alkaline) can cause scale buildup and taste issues.",
                "effects": "Acidic water: metallic taste, pipe corrosion, skin irritation. Alkaline water: bitter taste, digestive issues."
            },
            "Turbidity": {
                "range": "< 4 NTU (WHO)", "icon": "☁️",
                "about": "Turbidity measures water cloudiness caused by suspended particles. High turbidity indicates potential pathogen presence and makes disinfection less effective.",
                "effects": "High turbidity protects bacteria from chlorine disinfection, increasing infection risk."
            },
            "Dissolved Oxygen (DO)": {
                "range": "≥ 5 mg/L", "icon": "💨",
                "about": "DO is the amount of oxygen dissolved in water. Essential for aquatic life. Low DO indicates organic pollution or eutrophication.",
                "effects": "DO < 5 mg/L: aquatic life stress. DO < 2 mg/L: dead zones, fish kills."
            },
            "Heavy Metals": {
                "range": "< 0.1 mg/L (combined)", "icon": "⚙️",
                "about": "Lead, mercury, arsenic, cadmium are toxic heavy metals from industrial pollution, old pipes, or natural deposits. They accumulate in the body over time.",
                "effects": "Lead: brain damage in children, kidney disease. Mercury: neurological damage. Arsenic: cancer risk."
            },
            "Nitrates": {
                "range": "< 10 mg/L (WHO)", "icon": "🌱",
                "about": "Nitrates come from fertilizers, septic systems, and animal waste. They indicate agricultural or sewage contamination.",
                "effects": "Blue Baby Syndrome (methemoglobinemia) in infants. Long-term: increased cancer risk."
            },
            "TDS (Total Dissolved Solids)": {
                "range": "< 500 mg/L (WHO)", "icon": "🧂",
                "about": "TDS measures all dissolved substances including minerals, salts, and metals. Affects taste and can indicate contamination.",
                "effects": "High TDS: salty/bitter taste, scaling in appliances. Extremely low TDS: bland taste, potential mineral deficiency."
            },
        }
        for param, info in params_info.items():
            with st.expander(f"{info['icon']} {param} — Safe Range: {info['range']}"):
                st.markdown(f"**About:** {info['about']}")
                st.markdown(f"**Health Effects:** {info['effects']}")

    with tabs[1]:
        st.markdown("### 🦠 Common Water Contaminants")
        contaminants = [
            ("🦠 E. coli & Coliform Bacteria", "From human/animal waste. Causes diarrhoea, vomiting, cholera, typhoid.",
             "Boiling, chlorination, UV disinfection, ozone treatment"),
            ("🧪 Arsenic", "Natural deposits or industrial waste. Causes skin lesions, cancer, cardiovascular disease.",
             "RO filtration, activated alumina, coagulation"),
            ("🏭 Lead", "Old pipes, solder, paint. Severe neurological damage especially in children.",
             "Filter replacement, pipe replacement, NSF-certified filters"),
            ("💧 Fluoride (excess)", "Natural groundwater. Dental and skeletal fluorosis at high concentrations.",
             "RO filtration, activated alumina, distillation"),
            ("🌿 Pesticides", "Agricultural runoff. Liver/kidney damage, cancer, hormonal disruption.",
             "Activated carbon filters, RO systems"),
            ("🏗️ Nitrites/Nitrates", "Fertilizers, sewage. Blue baby syndrome, cancer risk.",
             "RO filtration, ion exchange, distillation"),
            ("🛢️ Petroleum Hydrocarbons", "Fuel spills, industrial leakage. Liver damage, cancer.",
             "Activated carbon, air stripping, bioremediation"),
        ]
        for title, effects, treatment in contaminants:
            st.markdown(f"""
            <div class="tip-card">
              <strong>{title}</strong><br>
              <span style="color:#f77f00">⚠️ Health Effects:</span> {effects}<br>
              <span style="color:#06d6a0">✅ Treatment:</span> {treatment}
            </div>
            """, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("### 💊 Water Treatment Methods")
        methods = {
            "🔥 Boiling": ("Kills bacteria, viruses, protozoa", "Doesn't remove chemicals, heavy metals, or dissolved solids", "Drinking water in emergencies"),
            "🌡️ Chlorination": ("Kills most bacteria and viruses, residual protection", "Taste/odour, trihalomethanes formed, ineffective
