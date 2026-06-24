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

st.set_page_config(
    page_title="AquaSafe – Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');
:root {
    --primary: #00b4d8; --primary-dark: #0077b6; --secondary: #90e0ef;
    --danger: #ef233c; --warning: #f77f00; --success: #06d6a0;
    --bg-dark: #03071e; --bg-card: #0a1628; --bg-card2: #0d1f3c;
    --text: #e0f4ff; --text-muted: #7fb3d3; --border: rgba(0,180,216,0.18);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: var(--bg-dark) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #03071e 0%, #023e8a 100%) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stApp { background: radial-gradient(ellipse at top, #03185a 0%, #03071e 60%) !important; }
.hero-banner { background: linear-gradient(135deg, #023e8a 0%, #0096c7 50%, #00b4d8 100%); border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(0,180,216,0.3); position: relative; overflow: hidden; }
.hero-banner::before { content: ''; position: absolute; top: -50%; right: -10%; width: 400px; height: 400px; border-radius: 50%; background: rgba(255,255,255,0.05); }
.hero-title { font-family: 'Poppins', sans-serif !important; font-size: 2.8rem !important; font-weight: 800 !important; color: #fff !important; margin: 0 !important; text-shadow: 0 2px 20px rgba(0,0,0,0.3); }
.hero-sub { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; }
.hero-badge { display: inline-block; background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); border-radius: 50px; padding: 4px 16px; font-size: 0.8rem; color: #fff; margin-bottom: 1rem; backdrop-filter: blur(10px); }
.metric-card { background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: 0 8px 32px rgba(0,180,216,0.1); transition: transform 0.3s ease, box-shadow 0.3s ease; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--primary), var(--secondary)); }
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 16px 48px rgba(0,180,216,0.2); }
.metric-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }
.metric-status-ok { color: var(--success) !important; }
.metric-status-bad { color: var(--danger) !important; }
.section-card { background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%); border: 1px solid var(--border); border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
.section-title { font-family: 'Poppins', sans-serif; font-size: 1.3rem; font-weight: 600; color: var(--primary); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
.safe-banner { background: linear-gradient(135deg, #06d6a044, #06d6a011); border: 2px solid #06d6a0; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; font-size: 1.8rem; font-weight: 700; color: #06d6a0; box-shadow: 0 0 30px rgba(6,214,160,0.25); animation: pulse-green 2s infinite; }
@keyframes pulse-green { 0%, 100% { box-shadow: 0 0 20px rgba(6,214,160,0.25); } 50% { box-shadow: 0 0 40px rgba(6,214,160,0.5); } }
.unsafe-banner { background: linear-gradient(135deg, #ef233c44, #ef233c11); border: 2px solid #ef233c; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; font-size: 1.8rem; font-weight: 700; color: #ef233c; box-shadow: 0 0 30px rgba(239,35,60,0.25); animation: pulse-red 2s infinite; }
@keyframes pulse-red { 0%, 100% { box-shadow: 0 0 20px rgba(239,35,60,0.25); } 50% { box-shadow: 0 0 40px rgba(239,35,60,0.5); } }
.tip-card { background: linear-gradient(135deg, rgba(0,180,216,0.12), rgba(0,119,182,0.08)); border: 1px solid rgba(0,180,216,0.3); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; font-size: 0.92rem; line-height: 1.5; color: var(--text); }
.tip-card strong { color: var(--primary); }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.styled-table th { background: linear-gradient(90deg, #023e8a, #0096c7); color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
.styled-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text); }
.styled-table tr:nth-child(even) td { background: rgba(0,180,216,0.04); }
.styled-table tr:hover td { background: rgba(0,180,216,0.1); }
.stButton > button { background: linear-gradient(135deg, #0096c7, #00b4d8) !important; color: #fff !important; border: none !important; border-radius: 12px !important; padding: 0.7rem 2rem !important; font-weight: 600 !important; font-size: 1rem !important; transition: all 0.3s !important; box-shadow: 0 4px 20px rgba(0,180,216,0.3) !important; }
.stButton > button:hover { background: linear-gradient(135deg, #023e8a, #0096c7) !important; transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(0,180,216,0.5) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 12px !important; padding: 4px !important; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; border-radius: 8px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0096c7, #00b4d8) !important; color: #fff !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--primary-dark); border-radius: 3px; }
.stAlert { border-radius: 12px !important; }
.sidebar-logo { font-family: 'Poppins', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--primary); text-align: center; padding: 1rem 0 1.5rem 0; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = Path("water_quality_history.json")

def load_history():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            DATA_FILE.write_text("[]")
            return []
    return []

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def save_history(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2, cls=_NumpyEncoder)

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "alerts" not in st.session_state:
    st.session_state.alerts = []

WHO_LIMITS = {
    "pH": (6.5, 8.5, "6.5 – 8.5"), "Turbidity": (0.0, 4.0, "< 4 NTU"),
    "Dissolved O₂": (5.0, 14.0, "≥ 5 mg/L"), "Heavy Metals": (0.0, 0.1, "< 0.1 mg/L"),
    "Nitrates": (0.0, 10.0, "< 10 mg/L"), "TDS": (0.0, 500.0, "< 500 mg/L"),
    "Temperature": (5.0, 30.0, "5 – 30 °C"), "Chlorine": (0.0, 0.5, "< 0.5 mg/L"),
}

def check_param(value, lo, hi):
    return "✅ OK" if lo <= value <= hi else "❌ NOT OK"

def wqi_score(ph, turb, do_, heavy, nitrates, tds, temp, chlorine):
    scores = []
    scores.append(max(0, 100 - abs(ph - 7.0) * 20))
    scores.append(max(0, 100 - turb * 20))
    scores.append(min(100, do_ / 8.0 * 100))
    scores.append(max(0, 100 - heavy * 1000))
    scores.append(max(0, 100 - nitrates * 5))
    scores.append(max(0, 100 - tds / 5))
    scores.append(max(0, 100 - abs(temp - 20) * 3))
    scores.append(max(0, 100 - chlorine * 200))
    return round(float(np.mean(scores)), 1)

def wqi_grade(score):
    if score >= 90: return "Excellent", "#06d6a0"
    if score >= 75: return "Good", "#90e0ef"
    if score >= 50: return "Fair", "#f77f00"
    if score >= 25: return "Poor", "#ef476f"
    return "Very Poor", "#ef233c"

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
        tds   = round(float(random.uniform(80, 700)), 1)
        temp  = round(float(random.uniform(10, 35)), 1)
        chl   = round(float(random.uniform(0.0, 0.9)), 2)
        score = float(wqi_score(ph, turb, do_, heavy, nit, tds, temp, chl))
        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
            "location": loc_name,
            "source_type": random.choice(["River / Stream", "Ground Water / Borewell", "Tap Water", "Lake / Pond"]),
            "lat": lat + float(random.uniform(-0.5, 0.5)),
            "lon": lon + float(random.uniform(-0.5, 0.5)),
            "ph": ph, "turbidity": turb, "do": do_, "heavy_metals": heavy,
            "nitrates": nit, "tds": tds, "temperature": temp, "chlorine": chl,
            "wqi": score, "grade": wqi_grade(score)[0], "safe": bool(score >= 50), "notes": "",
        })
    return records

with st.sidebar:
    st.markdown('<div class="sidebar-logo">💧 AquaSafe</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("📋 Navigation",
        ["🏠 Dashboard", "🔬 Test Water", "📊 Analytics", "🗺️ Location Map",
         "💡 Daily Tips", "📜 History", "⚠️ Alerts", "📚 Education Hub", "⚙️ Settings"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🌍 WHO Guidelines")
    for param, (lo, hi, label) in WHO_LIMITS.items():
        st.markdown(f"**{param}**: `{label}`")
    st.markdown("---")
    st.caption("🔒 AquaSafe v2.0 | Data stored locally")

st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🌊 Real-Time Monitoring Platform</div>
  <div class="hero-title">💧 AquaSafe – Water Quality Monitor</div>
  <div class="hero-sub">Comprehensive water testing, analytics, alerts & daily safety guidance</div>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Dashboard":
    history = st.session_state.history
    total = len(history)
    safe_count = sum(1 for r in history if r.get("safe"))
    unsafe_count = total - safe_count
    avg_wqi = round(float(np.mean([r["wqi"] for r in history])), 1) if history else 0
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, cls in [
        (c1, "📋", total, "Total Tests", ""),
        (c2, "✅", safe_count, "Safe Results", "metric-status-ok"),
        (c3, "❌", unsafe_count, "Unsafe Results", "metric-status-bad"),
        (c4, "📈", avg_wqi, "Avg WQI Score", ""),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value {cls}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
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
                    "axis": {"range": [0, 100], "tickcolor": "#7fb3d3", "tickfont": {"color": "#7fb3d3"}},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#0a1628", "borderwidth": 2, "bordercolor": "#00b4d8",
                    "steps": [
                        {"range": [0, 25], "color": "rgba(239,35,60,0.15)"},
                        {"range": [25, 50], "color": "rgba(247,127,0,0.15)"},
                        {"range": [50, 75], "color": "rgba(144,224,239,0.1)"},
                        {"range": [75, 100], "color": "rgba(6,214,160,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.75, "value": _wqi_val},
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color": "#e0f4ff"}, height=320, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class="section-card" style="margin-top:1rem">
              <div class="section-title">📊 Latest Params</div>
              <table class="styled-table">
                <tr><td><b>pH</b></td><td>{last['ph']}</td><td>{check_param(last['ph'], 6.5, 8.5)}</td></tr>
                <tr><td><b>Turbidity</b></td><td>{last['turbidity']} NTU</td><td>{check_param(last['turbidity'], 0, 4)}</td></tr>
                <tr><td><b>DO</b></td><td>{last['do']} mg/L</td><td>{check_param(last['do'], 5, 14)}</td></tr>
                <tr><td><b>Nitrates</b></td><td>{last.get('nitrates','N/A')} mg/L</td><td>{check_param(last.get('nitrates',5), 0, 10)}</td></tr>
                <tr><td><b>TDS</b></td><td>{last.get('tds','N/A')} mg/L</td><td>{check_param(last.get('tds',250), 0, 500)}</td></tr>
                <tr><td><b>Temp</b></td><td>{last.get('temperature','N/A')} °C</td><td>{check_param(last.get('temperature',25), 5, 30)}</td></tr>
              </table>
              <br>
              <div style="text-align:center;padding:0.8rem;background:{'rgba(6,214,160,0.15)' if last['safe'] else 'rgba(239,35,60,0.15)'};border-radius:10px;border:1px solid {'#06d6a0' if last['safe'] else '#ef233c'};font-weight:700;font-size:1.1rem;color:{'#06d6a0' if last['safe'] else '#ef233c'}">
                {'✅ WATER IS SAFE' if last['safe'] else '❌ WATER NOT SAFE'}
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 No data yet! Go to **🔬 Test Water** to add your first reading, or load demo data below.")
        if st.button("🎲 Load 30-Day Demo Data"):
            st.session_state.history = generate_demo_history("Demo River Station")
            save_history(st.session_state.history)
            st.rerun()
    if len(history) >= 2:
        st.markdown("### 📈 WQI Trend (Last 30 Readings)")
        df = pd.DataFrame(history[-30:])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["wqi"], mode="lines+markers",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=8, color=df["wqi"], colorscale=[[0,"#ef233c"],[0.5,"#f77f00"],[1,"#06d6a0"]]),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)", name="WQI"))
        fig2.add_hline(y=50, line_dash="dash", line_color="#f77f00", annotation_text="Safe Threshold (50)", annotation_position="top right")
        fig2.add_hline(y=75, line_dash="dot", line_color="#06d6a0", annotation_text="Good (75)", annotation_position="top right")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"), xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,180,216,0.1)", range=[0,105]),
            height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
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

elif page == "🔬 Test Water":
    st.markdown("## 🔬 Water Quality Test")
    with st.form("water_test_form"):
        st.markdown("### 📍 Sample Information")
        col1, col2 = st.columns(2)
        with col1: location = st.text_input("📍 Sample Location / Site Name", placeholder="e.g., River Ganga Station 3")
        with col2: sample_date = st.date_input("📅 Sample Date", datetime.date.today())
        source_type = st.selectbox("🌊 Water Source Type",
            ["Tap Water", "River / Stream", "Lake / Pond", "Ground Water / Borewell", "Rainwater", "Spring Water", "Industrial Effluent", "Other"])
        st.markdown("### 🧪 Physical Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1: ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        with col2: turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
        with col3: temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
        with col4: tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=250.0, step=10.0)
        st.markdown("### ⚗️ Chemical Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1: do_ = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
        with col2: heavy = st.number_input("Heavy Metals (mg/L)", min_value=0.0, max_value=10.0, value=0.05, step=0.01)
        with col3: nitrates = st.number_input("Nitrates (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        with col4: chlorine = st.number_input("Residual Chlorine (mg/L)", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
        notes = st.text_area("📝 Additional Notes (optional)")
        submitted = st.form_submit_button("🔍 Analyse Water Quality", use_container_width=True)
    if submitted:
        if not location:
            st.warning("⚠️ Please enter a sample location name.")
        else:
            score = float(wqi_score(ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine))
            grade, color = wqi_grade(score)
            safe = bool(score >= 50)
            params = {
                "pH": (ph, 6.5, 8.5), "Turbidity": (turbidity, 0.0, 4.0),
                "Dissolved O₂": (do_, 5.0, 14.0), "Heavy Metals": (heavy, 0.0, 0.1),
                "Nitrates": (nitrates, 0.0, 10.0), "TDS": (tds, 0.0, 500.0),
                "Temperature": (temperature, 5.0, 30.0), "Chlorine": (chlorine, 0.0, 0.5),
            }
            st.markdown("<br>", unsafe_allow_html=True)
            if safe:
                st.markdown(f'<div class="safe-banner">✅ WATER IS SAFE &nbsp;|&nbsp; WQI: {score} / 100 &nbsp;|&nbsp; Grade: {grade}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="unsafe-banner">❌ WATER IS NOT SAFE &nbsp;|&nbsp; WQI: {score} / 100 &nbsp;|&nbsp; Grade: {grade}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Parameter Breakdown")
                rows = [{"Parameter": name, "Measured": val, "Status": "✅ OK" if lo <= val <= hi else "❌ Exceeds Limit"} for name, (val, lo, hi) in params.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with col2:
                categories = list(params.keys())
                raw_scores = []
                for name, (val, lo, hi) in params.items():
                    span = hi - lo if hi != lo else 1
                    norm = max(0, min(1, (val - lo) / span)) if name != "Dissolved O₂" else max(0, min(1, val / hi))
                    raw_scores.append(norm * 100)
                fig_radar = go.Figure(go.Scatterpolar(
                    r=raw_scores + [raw_scores[0]], theta=categories + [categories[0]],
                    fill="toself", fillcolor="rgba(0,180,216,0.2)",
                    line=dict(color="#00b4d8", width=2), marker=dict(size=8, color="#00b4d8")))
                fig_radar.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True, range=[0, 100], color="#7fb3d3"),
                               angularaxis=dict(color="#7fb3d3")),
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"),
                    title=dict(text="Parameter Profile", font=dict(color="#00b4d8")),
                    height=380, margin=dict(t=50, b=20))
                st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("### 💊 Recommendations & Actions")
            recs = []
            if ph < 6.5: recs.append(("🔴 Low pH (Acidic)", "Water is acidic. Add lime or use an alkaline filter."))
            if ph > 8.5: recs.append(("🔴 High pH (Alkaline)", "Use a reverse osmosis (RO) filter or acidification treatment."))
            if turbidity > 4: recs.append(("🟠 High Turbidity", "Water is cloudy. Filter through a fine cloth or sand filter before use."))
            if do_ < 5: recs.append(("🔴 Low Dissolved Oxygen", "Aerate the water or check for organic pollution upstream."))
            if heavy > 0.1: recs.append(("🔴 High Heavy Metals", "DANGER: Do NOT drink. Report to local authorities immediately."))
            if nitrates > 10: recs.append(("🟠 High Nitrates", "Use RO filtration. Infants must not consume this water."))
            if tds > 500: recs.append(("🟠 High TDS", "Install a TDS filter or RO system."))
            if temperature > 30: recs.append(("🟡 High Temperature", "Warm water promotes bacterial growth. Cool and treat before drinking."))
            if chlorine > 0.5: recs.append(("🟡 High Chlorine", "Let water stand in open air or use activated carbon filter."))
            if not recs:
                st.markdown('<div class="tip-card">🎉 <strong>All parameters are within WHO guidelines.</strong> Keep testing regularly!</div>', unsafe_allow_html=True)
            else:
                for title, desc in recs:
                    st.markdown(f'<div class="tip-card"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)
            record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "location": location, "source_type": source_type,
                "ph": float(ph), "turbidity": float(turbidity), "do": float(do_),
                "heavy_metals": float(heavy), "nitrates": float(nitrates), "tds": float(tds),
                "temperature": float(temperature), "chlorine": float(chlorine),
                "wqi": score, "grade": grade, "safe": safe, "notes": notes,
            }
            st.session_state.history.append(record)
            save_history(st.session_state.history)
            if not safe:
                st.session_state.alerts.append({"time": record["timestamp"], "location": location, "wqi": score})
            st.success(f"✅ Result saved! Record #{len(st.session_state.history)} stored.")

elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics & Insights")
    history = st.session_state.history
    if len(history) < 2:
        st.info("📌 Not enough data. Add at least 2 water tests to see analytics.")
        st.stop()
    df = pd.DataFrame(history)
    st.markdown("### 📋 Summary Statistics")
    num_cols = ["ph", "turbidity", "do", "heavy_metals", "nitrates", "tds", "temperature", "chlorine", "wqi"]
    avail = [c for c in num_cols if c in df.columns]
    st.dataframe(df[avail].describe().round(2), use_container_width=True)
    st.markdown("---")
    st.markdown("### 📈 Parameter Trends Over Time")
    param_select = st.multiselect("Select parameters to plot:",
        ["wqi", "ph", "turbidity", "do", "heavy_metals", "nitrates", "tds", "temperature", "chlorine"],
        default=["wqi", "ph"])
    if param_select:
        fig_trend = go.Figure()
        colors = ["#00b4d8","#06d6a0","#f77f00","#ef476f","#90e0ef","#a8dadc","#ffd166","#e76f51"]
        for idx, param in enumerate(param_select):
            if param in df.columns:
                fig_trend.add_trace(go.Scatter(x=df["timestamp"], y=df[param], mode="lines+markers",
                    name=param.replace("_", " ").title(), line=dict(color=colors[idx % len(colors)], width=2), marker=dict(size=6)))
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"), xaxis=dict(showgrid=False, color="#7fb3d3"),
            yaxis=dict(gridcolor="rgba(0,180,216,0.1)", color="#7fb3d3"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#00b4d8"), height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_ph = px.histogram(df, x="ph", nbins=15, title="pH Distribution", color_discrete_sequence=["#00b4d8"])
        fig_ph.add_vline(x=6.5, line_dash="dash", line_color="#ef233c")
        fig_ph.add_vline(x=8.5, line_dash="dash", line_color="#ef233c", annotation_text="WHO Limits")
        fig_ph.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=300)
        st.plotly_chart(fig_ph, use_container_width=True)
    with col2:
        fig_wqi = px.histogram(df, x="wqi", nbins=15, title="WQI Score Distribution", color_discrete_sequence=["#06d6a0"])
        fig_wqi.add_vline(x=50, line_dash="dash", line_color="#f77f00", annotation_text="Safe Threshold")
        fig_wqi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=300)
        st.plotly_chart(fig_wqi, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        if "safe" in df.columns:
            safe_counts = df["safe"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=["Safe ✅", "Unsafe ❌"],
                values=[safe_counts.get(True, 0), safe_counts.get(False, 0)],
                hole=0.55, marker=dict(colors=["#06d6a0", "#ef233c"], line=dict(color="#03071e", width=3)),
                textinfo="percent+label", textfont=dict(color="#fff")))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        if "grade" in df.columns:
            g_counts = df["grade"].value_counts()
            fig_grade = go.Figure(go.Bar(x=g_counts.index, y=g_counts.values,
                marker_color=["#06d6a0","#90e0ef","#f77f00","#ef476f","#ef233c"][:len(g_counts)]))
            fig_grade.update_layout(title="WQI Grade Distribution", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"), xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(0,180,216,0.1)"), height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_grade, use_container_width=True)
    st.markdown("### 🔥 Parameter Correlation Heatmap")
    corr_cols = [c for c in avail if c in df.columns]
    if len(corr_cols) >= 3:
        corr = df[corr_cols].corr().round(2)
        fig_heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#ef233c"],[0.5,"#03071e"],[1,"#06d6a0"]],
            text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=11, color="#fff")))
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

elif page == "🗺️ Location Map":
    st.markdown("## 🗺️ Interactive Monitoring Network")
    history = st.session_state.history
    if not history:
        st.warning("No data yet. Load demo data from Settings first!")
        st.stop()
    df = pd.DataFrame(history)
    if "lat" not in df.columns or df["lat"].isna().all():
        rng = np.random.default_rng(42)
        locs_list = df["location"].unique().tolist()
        lat_map = {l: float(rng.uniform(8.0, 35.0)) for l in locs_list}
        lon_map = {l: float(rng.uniform(68.0, 97.0)) for l in locs_list}
        df["lat"] = df["location"].map(lat_map)
        df["lon"] = df["location"].map(lon_map)
    locations = df.groupby("location").agg(
        tests=("wqi","count"), avg_wqi=("wqi","mean"), min_wqi=("wqi","min"), max_wqi=("wqi","max"),
        safe_pct=("safe", lambda x: round(100*x.sum()/len(x),1)),
        last_safe=("safe","last"), lat=("lat","first"), lon=("lon","first"),
    ).reset_index()
    locations["avg_wqi"] = locations["avg_wqi"].round(1)
    locations["grade"] = locations["avg_wqi"].apply(lambda s: wqi_grade(s)[0])
    locations["status_label"] = locations["last_safe"].apply(lambda s: "🟢 Safe" if s else "🔴 Unsafe")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,1])
    with col_ctrl1:
        map_style = st.selectbox("🗺️ Map Style", ["carto-darkmatter","open-street-map","carto-positron"], index=0)
    with col_ctrl2:
        color_by = st.selectbox("🎨 Colour By", ["Average WQI","Last Status (Safe/Unsafe)","Number of Tests"])
    with col_ctrl3:
        show_labels = st.toggle("Labels", value=True)
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(lat=locations["lat"], lon=locations["lon"], mode="markers",
        marker=dict(size=locations["tests"].clip(upper=40)*2.5+30, color=locations["avg_wqi"],
            colorscale=[[0.0,"rgba(239,35,60,0.25)"],[0.5,"rgba(144,224,239,0.2)"],[1.0,"rgba(6,214,160,0.2)"]],
            opacity=0.55, sizemode="diameter"),
        hoverinfo="skip", showlegend=False, name="heatring"))
    if color_by == "Average WQI":
        mc = locations["avg_wqi"]; mcs = [[0,"#ef233c"],[0.5,"#90e0ef"],[1,"#06d6a0"]]; ss = True
    elif color_by == "Last Status (Safe/Unsafe)":
        mc = locations["last_safe"].apply(lambda s: "#06d6a0" if s else "#ef233c"); mcs = None; ss = False
    else:
        mc = locations["tests"]; mcs = [[0,"#023e8a"],[1,"#00b4d8"]]; ss = True
    hover_text = [f"<b>📍 {row.location}</b><br>🧪 Tests: <b>{int(row.tests)}</b><br>📊 Avg WQI: <b>{row.avg_wqi:.1f}</b> ({row.grade})<br>✅ Safe Rate: <b>{row.safe_pct}%</b><br>🕐 Last: {row.status_label}" for row in locations.itertuples()]
    fig_map.add_trace(go.Scattermapbox(lat=locations["lat"], lon=locations["lon"],
        mode="markers+text" if show_labels else "markers",
        text=locations["location"].apply(lambda x: x[:14]+"…" if len(x)>14 else x) if show_labels else None,
        textposition="top center", textfont=dict(size=11, color="#e0f4ff"),
        marker=dict(size=locations["tests"].clip(upper=40)+14, color=mc, colorscale=mcs,
            colorbar=dict(title="WQI", thickness=12, bgcolor="rgba(10,22,40,0.85)",
                tickfont=dict(color="#e0f4ff"), titlefont=dict(color="#00b4d8")) if ss else None,
            showscale=ss, opacity=0.95, sizemode="diameter"),
        hovertemplate="%{hovertext}<extra></extra>", hovertext=hover_text, showlegend=False))
    unsafe_locs = locations[~locations["last_safe"]]
    if not unsafe_locs.empty:
        fig_map.add_trace(go.Scattermapbox(lat=unsafe_locs["lat"], lon=unsafe_locs["lon"], mode="markers",
            marker=dict(size=55, color="rgba(239,35,60,0.2)", sizemode="diameter"), hoverinfo="skip", showlegend=False))
    fig_map.update_layout(mapbox=dict(style=map_style, center=dict(lat=float(locations["lat"].mean()), lon=float(locations["lon"].mean())), zoom=4.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f4ff"), height=560, margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("""<div style="display:flex;gap:8px;align-items:center;margin:0.5rem 0 1.5rem 0;flex-wrap:wrap">
      <span style="color:#7fb3d3;font-size:0.85rem;font-weight:600">WQI Legend:</span>
      <span style="background:#ef233c;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">0–25 Very Poor</span>
      <span style="background:#f77f00;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">25–50 Poor</span>
      <span style="background:#0096c7;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">50–75 Fair</span>
      <span style="background:#06d6a0;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">75–100 Good/Excellent</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("### 📋 Station Summaries")
    loc_rows = [locations.iloc[i:i+3] for i in range(0, len(locations), 3)]
    for row_df in loc_rows:
        cols = st.columns(3)
        for ci, (_, loc) in enumerate(row_df.iterrows()):
            g_color = wqi_grade(loc["avg_wqi"])[1]
            safe_bg = "rgba(6,214,160,0.12)" if loc["last_safe"] else "rgba(239,35,60,0.12)"
            safe_bdr = "#06d6a0" if loc["last_safe"] else "#ef233c"
            with cols[ci]:
                st.markdown(f"""<div style="background:{safe_bg};border:1px solid {safe_bdr};border-radius:14px;padding:1.1rem;margin-bottom:0.8rem">
                  <div style="font-weight:700;color:#e0f4ff;margin-bottom:0.5rem">📍 {loc['location']}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:{g_color}">{loc['avg_wqi']}</div>
                  <div style="font-size:0.75rem;color:#7fb3d3">Avg WQI · {loc['grade']} · ✅ {loc['safe_pct']}% safe</div>
                  <div style="margin-top:0.5rem;background:rgba(0,0,0,0.3);border-radius:6px;height:6px">
                    <div style="width:{min(loc['avg_wqi'],100)}%;height:100%;background:linear-gradient(90deg,{g_color}88,{g_color});border-radius:6px"></div>
                  </div></div>""", unsafe_allow_html=True)

elif page == "💡 Daily Tips":
    st.markdown("## 💡 Daily Water Safety Tips for Everyone")
    categories = {
        "🏠 At Home": [
            ("Boil before drinking", "Boil water for at least 1 minute if unsure of its safety."),
            ("Change water filters", "Replace filters every 2–6 months as per manufacturer guidelines."),
            ("Clean water tanks", "Clean overhead tanks every 6 months to prevent algae and bacteria."),
            ("Use BPA-free containers", "Store water only in food-grade, BPA-free containers."),
            ("Check pipes for rust", "Brownish or metallic-tasting water indicates corroded pipes."),
            ("Don't use hot tap water for cooking", "Hot water dissolves more lead from old pipes."),
        ],
        "👶 For Infants & Children": [
            ("Use formula-safe water", "For baby formula, use purified or boiled water."),
            ("Watch for blue baby syndrome", "High nitrates (>10 mg/L) can harm infants — always test."),
            ("Fluoride check", "Too much fluoride causes fluorosis. Check your water level."),
            ("Lead exposure risk", "Lead is especially harmful to children. Test if home is pre-1986."),
        ],
        "🌾 Agriculture & Farming": [
            ("Irrigation water quality", "Use water with pH 6–8. High TDS damages crops."),
            ("Avoid industrial runoff", "Never use runoff water for edible crops."),
            ("Nitrate buildup", "Use slow-release fertilizers to reduce groundwater nitrates."),
            ("Test bore water regularly", "Borewell quality changes seasonally — test each season."),
        ],
        "🚨 Emergency Situations": [
            ("Flood water safety", "Floodwater is highly contaminated. Never drink without treatment."),
            ("After power cuts", "Boil water stored >24 hours without pump circulation."),
            ("Chemical spill nearby", "Stop using tap water until authorities declare it safe."),
            ("Natural disaster prep", "Store 3 litres per person per day for at least 3 days."),
        ],
    }
    for cat, tips_list in categories.items():
        st.markdown(f"### {cat}")
        cols = st.columns(2)
        for i, (title, detail) in enumerate(tips_list):
            with cols[i % 2]:
                st.markdown(f'<div class="tip-card"><strong>{title}</strong><br><span style="color:#a8d8ea">{detail}</span></div>', unsafe_allow_html=True)

elif page == "📜 History":
    st.markdown("## 📜 Test History & Records")
    history = st.session_state.history
    if not history:
        st.info("No history yet. Go to **🔬 Test Water** to add records.")
        st.stop()
    df = pd.DataFrame(history)
    col1, col2, col3 = st.columns(3)
    with col1:
        locs = ["All"] + sorted(df["location"].unique().tolist())
        sel_loc = st.selectbox("📍 Filter by Location", locs)
    with col2:
        sel_safe = st.selectbox("🛡️ Filter by Status", ["All", "Safe Only", "Unsafe Only"])
    with col3:
        if "source_type" in df.columns:
            sources = ["All"] + sorted(df["source_type"].dropna().unique().tolist())
            sel_src = st.selectbox("🌊 Filter by Source", sources)
        else:
            sel_src = "All"
    filtered = df.copy()
    if sel_loc != "All": filtered = filtered[filtered["location"] == sel_loc]
    if sel_safe == "Safe Only": filtered = filtered[filtered["safe"] == True]
    elif sel_safe == "Unsafe Only": filtered = filtered[filtered["safe"] == False]
    if sel_src != "All" and "source_type" in df.columns: filtered = filtered[filtered["source_type"] == sel_src]
    st.markdown(f"**Showing {len(filtered)} of {len(df)} records**")
    st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", csv, "water_quality_history.csv", "text/csv", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            save_history([])
            st.rerun()

elif page == "⚠️ Alerts":
    st.markdown("## ⚠️ Safety Alerts & Notifications")
    history = st.session_state.history
    critical = [r for r in history if r.get("wqi", 100) < 25]
    warning = [r for r in history if 25 <= r.get("wqi", 100) < 50]
    safe_r = [r for r in history if r.get("safe")]
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f'<div class="metric-card"><div class="metric-icon">🔴</div><div class="metric-value metric-status-bad">{len(critical)}</div><div class="metric-label">Critical (WQI &lt; 25)</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><div class="metric-icon">🟠</div><div class="metric-value" style="color:#f77f00">{len(warning)}</div><div class="metric-label">Warning (WQI 25–50)</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><div class="metric-icon">✅</div><div class="metric-value metric-status-ok">{len(safe_r)}</div><div class="metric-label">Safe Readings</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if critical:
        st.markdown("### 🔴 Critical Events")
        for r in sorted(critical, key=lambda x: x["timestamp"], reverse=True)[:10]:
            st.markdown(f'<div style="background:rgba(239,35,60,0.1);border:1px solid #ef233c;border-radius:12px;padding:1rem;margin-bottom:0.7rem"><strong>🚨 {r["location"]}</strong> | {r["timestamp"]} | WQI: {r["wqi"]}<br><span style="color:#ef476f">Immediate action required. Do NOT use this water.</span></div>', unsafe_allow_html=True)
    if warning:
        st.markdown("### 🟠 Warning Events")
        for r in sorted(warning, key=lambda x: x["timestamp"], reverse=True)[:10]:
            st.markdown(f'<div style="background:rgba(247,127,0,0.1);border:1px solid #f77f00;border-radius:12px;padding:1rem;margin-bottom:0.7rem"><strong>⚠️ {r["location"]}</strong> | {r["timestamp"]} | WQI: {r["wqi"]}<br><span style="color:#f77f00">Treatment recommended.</span></div>', unsafe_allow_html=True)
    if not critical and not warning:
        st.markdown('<div style="text-align:center;padding:3rem;color:#06d6a0;font-size:1.3rem;font-weight:600">✅ No safety alerts at this time.</div>', unsafe_allow_html=True)

elif page == "📚 Education Hub":
    st.markdown("## 📚 Water Quality Education Hub")
    tabs = st.tabs(["🧪 Parameters Guide", "🦠 Contaminants", "💊 Treatment Methods", "📖 WQI Explained", "🌍 Global Water Facts"])
    with tabs[0]:
        params_info = {
            "pH": {"range": "6.5–8.5 (WHO)", "icon": "🔵", "about": "Measures acidity/alkalinity. Low pH corrodes pipes; high pH causes scale.", "effects": "Acidic: metallic taste, pipe corrosion. Alkaline: bitter taste."},
            "Turbidity": {"range": "< 4 NTU (WHO)", "icon": "☁️", "about": "Measures cloudiness. High turbidity indicates potential pathogens.", "effects": "Protects bacteria from chlorine, increasing infection risk."},
            "Dissolved Oxygen": {"range": "≥ 5 mg/L", "icon": "💨", "about": "Essential for aquatic life. Low DO = organic pollution.", "effects": "< 5 mg/L: aquatic stress. < 2 mg/L: fish kills."},
            "Heavy Metals": {"range": "< 0.1 mg/L", "icon": "⚙️", "about": "Lead, mercury, arsenic accumulate in the body.", "effects": "Brain damage, kidney disease, cancer risk."},
            "Nitrates": {"range": "< 10 mg/L (WHO)", "icon": "🌱", "about": "From fertilizers and sewage.", "effects": "Blue Baby Syndrome in infants; long-term cancer risk."},
            "TDS": {"range": "< 500 mg/L (WHO)", "icon": "🧂", "about": "All dissolved substances.", "effects": "High TDS: salty/bitter taste, scaling."},
        }
        for param, info in params_info.items():
            with st.expander(f"{info['icon']} {param} — Safe Range: {info['range']}"):
                st.markdown(f"**About:** {info['about']}")
                st.markdown(f"**Health Effects:** {info['effects']}")
    with tabs[1]:
        contaminants = [
            ("🦠 E. coli & Coliform Bacteria", "Causes diarrhoea, cholera, typhoid.", "Boiling, chlorination, UV disinfection"),
            ("🧪 Arsenic", "Skin lesions, cancer, cardiovascular disease.", "RO filtration, activated alumina"),
            ("🏭 Lead", "Neurological damage especially in children.", "Filter replacement, pipe replacement"),
            ("🌿 Pesticides", "Liver/kidney damage, cancer.", "Activated carbon filters, RO systems"),
            ("🛢️ Petroleum Hydrocarbons", "Liver damage, cancer.", "Activated carbon, air stripping"),
        ]
        for title, effects, treatment in contaminants:
            st.markdown(f'<div class="tip-card"><strong>{title}</strong><br><span style="color:#f77f00">⚠️ Effects:</span> {effects}<br><span style="color:#06d6a0">✅ Treatment:</span> {treatment}</div>', unsafe_allow_html=True)
    with tabs[2]:
        methods = {
            "🔥 Boiling": ("Kills bacteria, viruses, protozoa", "Doesn't remove chemicals or metals", "Emergency drinking water"),
            "💡 UV Disinfection": ("Kills bacteria, viruses, protozoa without chemicals", "No residual protection", "Point-of-use disinfection"),
            "🔬 Reverse Osmosis (RO)": ("Removes 95–99% of contaminants", "Wastes water, removes minerals", "Drinking water purification"),
            "🪨 Activated Carbon": ("Removes chlorine, pesticides, odours", "Doesn't remove bacteria or heavy metals", "Pre-filter for taste/odour"),
        }
        col1, col2 = st.columns(2)
        for i, (method, (pros, cons, use)) in enumerate(methods.items()):
            with col1 if i % 2 == 0 else col2:
                with st.expander(method):
                    st.markdown(f"✅ **Pros:** {pros}")
                    st.markdown(f"❌ **Cons:** {cons}")
                    st.markdown(f"🎯 **Best for:** {use}")
    with tabs[3]:
        grade_data = [
            ("90–100", "Excellent", "#06d6a0", "Safe for all uses. Meets all WHO standards."),
            ("75–89", "Good", "#90e0ef", "Safe to drink."),
            ("50–74", "Fair", "#f77f00", "Some treatment recommended."),
            ("25–49", "Poor", "#ef476f", "Not safe without treatment."),
            ("0–24", "Very Poor", "#ef233c", "Severely contaminated. Do NOT use."),
        ]
        for score_range, grade, color, desc in grade_data:
            st.markdown(f'<div style="background:rgba(0,0,0,0.3);border-left:5px solid {color};border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:0.5rem"><strong style="color:{color}">{grade} ({score_range})</strong><br><span style="color:#e0f4ff">{desc}</span></div>', unsafe_allow_html=True)
    with tabs[4]:
        facts = [
            ("💧", "785 million people", "lack access to clean drinking water globally"),
            ("🦠", "2 billion people", "drink water contaminated with faeces"),
            ("☠️", "3.5 million deaths", "occur annually due to waterborne diseases"),
            ("👶", "1 child every 2 min", "dies from water-related diseases"),
            ("🌊", "70%", "of Earth's surface is water, but only 3% is freshwater"),
        ]
        for icon, stat, desc in facts:
            st.markdown(f'<div class="tip-card">{icon} <strong style="color:#00b4d8">{stat}</strong> — {desc}</div>', unsafe_allow_html=True)

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Application Settings")
    col1, col2 = st.columns(2)
    with col1:
        demo_loc = st.text_input("Demo Location Name", value="Demo Station")
        if st.button("🎲 Load Demo Data (60 records)", use_container_width=True):
            demo = generate_demo_history(demo_loc)
            st.session_state.history.extend(demo)
            save_history(st.session_state.history)
            st.success(f"✅ Demo records loaded for '{demo_loc}'!")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.history = []
            st.session_state.alerts = []
            save_history([])
            st.success("✅ All data cleared.")
            st.rerun()
    st.markdown("---")
    if st.session_state.history:
        df_exp = pd.DataFrame(st.session_state.history)
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export All Data (CSV)", csv, "aquasafe_export.csv", "text/csv", use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#7fb3d3;font-size:0.85rem;padding:1rem">💧 <strong>AquaSafe</strong> — Water Quality Monitoring System | Based on <a href="https://www.who.int" style="color:#00b4d8">WHO Guidelines</a> | For educational purposes only</div>', unsafe_allow_html=True)
