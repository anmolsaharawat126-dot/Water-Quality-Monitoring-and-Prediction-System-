import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import json
import random
from pathlib import Path

st.set_page_config(page_title="AquaSafe – Water Quality Monitor", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');
:root { --primary:#00b4d8; --primary-dark:#0077b6; --secondary:#90e0ef; --danger:#ef233c; --warning:#f77f00; --success:#06d6a0; --bg-dark:#03071e; --bg-card:#0a1628; --bg-card2:#0d1f3c; --text:#e0f4ff; --text-muted:#7fb3d3; --border:rgba(0,180,216,0.18); }
html,body,[class*="css"]{ font-family:'Inter',sans-serif!important; background-color:var(--bg-dark)!important; color:var(--text)!important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#03071e 0%,#023e8a 100%)!important; border-right:1px solid var(--border)!important; }
[data-testid="stSidebar"] *{ color:var(--text)!important; }
.stApp{ background:radial-gradient(ellipse at top,#03185a 0%,#03071e 60%)!important; }
.hero-banner{ background:linear-gradient(135deg,#023e8a 0%,#0096c7 50%,#00b4d8 100%); border-radius:20px; padding:2.5rem 3rem; margin-bottom:2rem; box-shadow:0 20px 60px rgba(0,180,216,0.3); position:relative; overflow:hidden; }
.hero-banner::before{ content:''; position:absolute; top:-50%; right:-10%; width:400px; height:400px; border-radius:50%; background:rgba(255,255,255,0.05); }
.hero-title{ font-family:'Poppins',sans-serif!important; font-size:2.8rem!important; font-weight:800!important; color:#fff!important; margin:0!important; text-shadow:0 2px 20px rgba(0,0,0,0.3); }
.hero-sub{ font-size:1.1rem; color:rgba(255,255,255,0.85); margin-top:0.5rem; }
.hero-badge{ display:inline-block; background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.3); border-radius:50px; padding:4px 16px; font-size:0.8rem; color:#fff; margin-bottom:1rem; }
.metric-card{ background:linear-gradient(135deg,var(--bg-card) 0%,var(--bg-card2) 100%); border:1px solid var(--border); border-radius:16px; padding:1.5rem; text-align:center; box-shadow:0 8px 32px rgba(0,180,216,0.1); transition:transform 0.3s ease,box-shadow 0.3s ease; position:relative; overflow:hidden; }
.metric-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,var(--primary),var(--secondary)); }
.metric-card:hover{ transform:translateY(-4px); box-shadow:0 16px 48px rgba(0,180,216,0.2); }
.metric-icon{ font-size:2.5rem; margin-bottom:0.5rem; }
.metric-value{ font-size:2rem; font-weight:700; color:var(--primary); }
.metric-label{ font-size:0.85rem; color:var(--text-muted); margin-top:0.25rem; }
.metric-status-ok{ color:var(--success)!important; }
.metric-status-bad{ color:var(--danger)!important; }
.section-card{ background:linear-gradient(135deg,var(--bg-card) 0%,var(--bg-card2) 100%); border:1px solid var(--border); border-radius:16px; padding:1.8rem; margin-bottom:1.5rem; box-shadow:0 8px 32px rgba(0,0,0,0.2); }
.section-title{ font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:600; color:var(--primary); margin-bottom:1rem; }
.safe-banner{ background:linear-gradient(135deg,#06d6a044,#06d6a011); border:2px solid #06d6a0; border-radius:16px; padding:1.5rem 2rem; text-align:center; font-size:1.8rem; font-weight:700; color:#06d6a0; box-shadow:0 0 30px rgba(6,214,160,0.25); animation:pulse-green 2s infinite; }
@keyframes pulse-green{ 0%,100%{ box-shadow:0 0 20px rgba(6,214,160,0.25); } 50%{ box-shadow:0 0 40px rgba(6,214,160,0.5); } }
.unsafe-banner{ background:linear-gradient(135deg,#ef233c44,#ef233c11); border:2px solid #ef233c; border-radius:16px; padding:1.5rem 2rem; text-align:center; font-size:1.8rem; font-weight:700; color:#ef233c; animation:pulse-red 2s infinite; }
@keyframes pulse-red{ 0%,100%{ box-shadow:0 0 20px rgba(239,35,60,0.25); } 50%{ box-shadow:0 0 40px rgba(239,35,60,0.5); } }
.tip-card{ background:linear-gradient(135deg,rgba(0,180,216,0.12),rgba(0,119,182,0.08)); border:1px solid rgba(0,180,216,0.3); border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem; font-size:0.92rem; line-height:1.5; color:var(--text); }
.tip-card strong{ color:var(--primary); }
.styled-table{ width:100%; border-collapse:collapse; font-size:0.9rem; }
.styled-table th{ background:linear-gradient(90deg,#023e8a,#0096c7); color:#fff; padding:10px 14px; text-align:left; font-weight:600; }
.styled-table td{ padding:10px 14px; border-bottom:1px solid var(--border); color:var(--text); }
.styled-table tr:nth-child(even) td{ background:rgba(0,180,216,0.04); }
.stButton>button{ background:linear-gradient(135deg,#0096c7,#00b4d8)!important; color:#fff!important; border:none!important; border-radius:12px!important; padding:0.7rem 2rem!important; font-weight:600!important; transition:all 0.3s!important; box-shadow:0 4px 20px rgba(0,180,216,0.3)!important; }
.stButton>button:hover{ background:linear-gradient(135deg,#023e8a,#0096c7)!important; transform:translateY(-2px)!important; }
.stTabs [data-baseweb="tab-list"]{ background:var(--bg-card)!important; border-radius:12px!important; padding:4px!important; border:1px solid var(--border)!important; }
.stTabs [data-baseweb="tab"]{ color:var(--text-muted)!important; border-radius:8px!important; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#0096c7,#00b4d8)!important; color:#fff!important; }
::-webkit-scrollbar{ width:6px; } ::-webkit-scrollbar-track{ background:var(--bg-dark); } ::-webkit-scrollbar-thumb{ background:var(--primary-dark); border-radius:3px; }
.stAlert{ border-radius:12px!important; }
.sidebar-logo{ font-family:'Poppins',sans-serif; font-size:1.6rem; font-weight:800; color:var(--primary); text-align:center; padding:1rem 0 1.5rem 0; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)

# ── Data Storage ──
DATA_FILE = Path("water_quality_history.json")

def load_history():
    if DATA_FILE.exists():
        try:
            content = DATA_FILE.read_text().strip()
            return json.loads(content) if content else []
        except (json.JSONDecodeError, ValueError):
            DATA_FILE.write_text("[]")
            return []
    return []

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def save_history(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2, cls=_NumpyEncoder)

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ── Constants ──
WHO_LIMITS = {
    "pH": (6.5, 8.5, "6.5–8.5"), "Turbidity": (0.0, 4.0, "< 4 NTU"),
    "Dissolved O₂": (5.0, 14.0, "≥ 5 mg/L"), "Heavy Metals": (0.0, 0.1, "< 0.1 mg/L"),
    "Nitrates": (0.0, 10.0, "< 10 mg/L"), "TDS": (0.0, 500.0, "< 500 mg/L"),
    "Temperature": (5.0, 30.0, "5–30 °C"), "Chlorine": (0.0, 0.5, "< 0.5 mg/L"),
}

def check_param(v, lo, hi): return "✅ OK" if lo <= v <= hi else "❌ NOT OK"

def wqi_score(ph, turb, do_, heavy, nit, tds, temp, chl):
    s = [max(0, 100-abs(ph-7)*20), max(0,100-turb*20), min(100,do_/8*100),
         max(0,100-heavy*1000), max(0,100-nit*5), max(0,100-tds/5),
         max(0,100-abs(temp-20)*3), max(0,100-chl*200)]
    return round(float(np.mean(s)), 1)

def wqi_grade(score):
    if score >= 90: return "Excellent", "#06d6a0"
    if score >= 75: return "Good", "#90e0ef"
    if score >= 50: return "Fair", "#f77f00"
    if score >= 25: return "Poor", "#ef476f"
    return "Very Poor", "#ef233c"

def generate_demo_history(location="Demo Site"):
    stations = [
        (location, 20.59, 78.96), ("Yamuna – Delhi", 28.61, 77.21),
        ("Ganga – Varanasi", 25.32, 82.97), ("Sabarmati – Ahmedabad", 23.02, 72.57),
        ("Krishna – Vijayawada", 16.51, 80.65),
    ]
    records, base = [], datetime.datetime.now() - datetime.timedelta(days=30)
    for i in range(60):
        dt = base + datetime.timedelta(hours=i*12)
        loc, lat, lon = random.choice(stations)
        ph = round(float(random.uniform(6.2, 8.8)), 2)
        turb = round(float(random.uniform(0.2, 8.0)), 2)
        do_ = round(float(random.uniform(3.0, 9.5)), 2)
        heavy = round(float(random.uniform(0.01, 0.20)), 3)
        nit = round(float(random.uniform(1.0, 16.0)), 2)
        tds = round(float(random.uniform(80, 700)), 1)
        temp = round(float(random.uniform(10, 35)), 1)
        chl = round(float(random.uniform(0.0, 0.9)), 2)
        score = float(wqi_score(ph, turb, do_, heavy, nit, tds, temp, chl))
        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"), "location": loc,
            "source_type": random.choice(["River / Stream", "Ground Water / Borewell", "Tap Water", "Lake / Pond"]),
            "lat": lat + float(random.uniform(-0.5, 0.5)),
            "lon": lon + float(random.uniform(-0.5, 0.5)),
            "ph": ph, "turbidity": turb, "do": do_, "heavy_metals": heavy,
            "nitrates": nit, "tds": tds, "temperature": temp, "chlorine": chl,
            "wqi": score, "grade": wqi_grade(score)[0], "safe": bool(score >= 50), "notes": "",
        })
    return records

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💧 AquaSafe</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Dashboard","🔬 Test Water","📊 Analytics","🗺️ Location Map",
                                   "💡 Daily Tips","📜 History","⚠️ Alerts","📚 Education Hub","⚙️ Settings"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🌍 WHO Guidelines")
    for param, (lo, hi, label) in WHO_LIMITS.items():
        st.markdown(f"**{param}**: `{label}`")
    st.markdown("---")
    st.caption("🔒 AquaSafe v2.0 | Built with ❤️")

# ── Hero ──
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🌊 Real-Time Monitoring Platform</div>
  <div class="hero-title">💧 AquaSafe – Water Quality Monitor</div>
  <div class="hero-sub">Comprehensive water testing, analytics, alerts & daily safety guidance</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════
if page == "🏠 Dashboard":
    history = st.session_state.history
    total = len(history)
    safe_count = sum(1 for r in history if r.get("safe"))
    avg_wqi = round(float(np.mean([r["wqi"] for r in history])), 1) if history else 0
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,val,label,cls in [
        (c1,"📋",total,"Total Tests",""),
        (c2,"✅",safe_count,"Safe Results","metric-status-ok"),
        (c3,"❌",total-safe_count,"Unsafe Results","metric-status-bad"),
        (c4,"📈",avg_wqi,"Avg WQI Score",""),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value {cls}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if history:
        last = history[-1]
        grade, color = wqi_grade(last["wqi"])
        st.markdown("### 🔍 Latest Water Reading")
        col1, col2 = st.columns([2,1])
        with col1:
            wv = float(last["wqi"])
            fig = go.Figure(go.Indicator(mode="gauge+number", value=wv,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":f"Water Quality Index<br><span style='font-size:13px'>📍 {last['location']}</span>","font":{"color":"#e0f4ff","size":16}},
                number={"font":{"color":color,"size":48},"suffix":" /100"},
                gauge={"axis":{"range":[0,100],"tickcolor":"#7fb3d3","tickfont":{"color":"#7fb3d3"}},
                       "bar":{"color":color,"thickness":0.28},"bgcolor":"#0a1628","borderwidth":2,"bordercolor":"#00b4d8",
                       "steps":[{"range":[0,25],"color":"rgba(239,35,60,0.15)"},{"range":[25,50],"color":"rgba(247,127,0,0.15)"},
                                {"range":[50,75],"color":"rgba(144,224,239,0.1)"},{"range":[75,100],"color":"rgba(6,214,160,0.15)"}],
                       "threshold":{"line":{"color":"#fff","width":3},"thickness":0.75,"value":wv}}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#e0f4ff"},height=320,margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            safe_col = "#06d6a0" if last["safe"] else "#ef233c"
            safe_bg = "rgba(6,214,160,0.15)" if last["safe"] else "rgba(239,35,60,0.15)"
            safe_txt = "✅ WATER IS SAFE" if last["safe"] else "❌ WATER NOT SAFE"
            st.markdown(f"""<div class="section-card" style="margin-top:1rem">
              <div class="section-title">📊 Latest Params</div>
              <table class="styled-table">
                <tr><td><b>pH</b></td><td>{last['ph']}</td><td>{check_param(last['ph'],6.5,8.5)}</td></tr>
                <tr><td><b>Turbidity</b></td><td>{last['turbidity']} NTU</td><td>{check_param(last['turbidity'],0,4)}</td></tr>
                <tr><td><b>DO</b></td><td>{last['do']} mg/L</td><td>{check_param(last['do'],5,14)}</td></tr>
                <tr><td><b>Nitrates</b></td><td>{last.get('nitrates','N/A')}</td><td>{check_param(last.get('nitrates',5),0,10)}</td></tr>
                <tr><td><b>TDS</b></td><td>{last.get('tds','N/A')}</td><td>{check_param(last.get('tds',250),0,500)}</td></tr>
                <tr><td><b>Temp</b></td><td>{last.get('temperature','N/A')} °C</td><td>{check_param(last.get('temperature',25),5,30)}</td></tr>
              </table><br>
              <div style="text-align:center;padding:0.8rem;background:{safe_bg};border-radius:10px;border:1px solid {safe_col};font-weight:700;color:{safe_col}">{safe_txt}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("📌 No data yet! Load demo data or go to **🔬 Test Water**.")
        if st.button("🎲 Load Demo Data"):
            st.session_state.history = generate_demo_history()
            save_history(st.session_state.history)
            st.rerun()

    if len(history) >= 2:
        st.markdown("### 📈 WQI Trend")
        df = pd.DataFrame(history[-30:])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["timestamp"],y=df["wqi"],mode="lines+markers",
            line=dict(color="#00b4d8",width=3),fill="tozeroy",fillcolor="rgba(0,180,216,0.08)",
            marker=dict(size=8,color=df["wqi"],colorscale=[[0,"#ef233c"],[0.5,"#f77f00"],[1,"#06d6a0"]])))
        fig2.add_hline(y=50,line_dash="dash",line_color="#f77f00",annotation_text="Safe Threshold")
        fig2.add_hline(y=75,line_dash="dot",line_color="#06d6a0",annotation_text="Good")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"),xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(0,180,216,0.1)",range=[0,105]),height=300,margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💡 Quick Daily Tips")
    tips = [("🚰","Always boil water if unsure before drinking."),("🧪","Test drinking water every 6 months."),
            ("🪣","Store water in clean, covered containers."),("🌿","Don't dispose chemicals down the drain."),
            ("🏭","Report unusual smell/taste in tap water."),("🧽","Clean water filters regularly.")]
    cols = st.columns(3)
    for i,(ic,tip) in enumerate(tips):
        with cols[i%3]: st.markdown(f'<div class="tip-card">{ic} {tip}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TEST WATER
# ═══════════════════════════════════════════
elif page == "🔬 Test Water":
    st.markdown("## 🔬 Water Quality Test")
    with st.form("water_test_form"):
        col1,col2 = st.columns(2)
        with col1: location = st.text_input("📍 Sample Location", placeholder="e.g., River Ganga Station 3")
        with col2: sample_date = st.date_input("📅 Sample Date", datetime.date.today())
        source_type = st.selectbox("🌊 Water Source Type",
            ["Tap Water","River / Stream","Lake / Pond","Ground Water / Borewell","Rainwater","Spring Water","Industrial Effluent","Other"])
        st.markdown("### 🧪 Physical Parameters")
        c1,c2,c3,c4 = st.columns(4)
        with c1: ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        with c2: turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
        with c3: temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
        with c4: tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=250.0, step=10.0)
        st.markdown("### ⚗️ Chemical Parameters")
        c1,c2,c3,c4 = st.columns(4)
        with c1: do_ = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
        with c2: heavy = st.number_input("Heavy Metals (mg/L)", min_value=0.0, max_value=10.0, value=0.05, step=0.01)
        with c3: nitrates = st.number_input("Nitrates (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        with c4: chlorine = st.number_input("Chlorine (mg/L)", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
        notes = st.text_area("📝 Notes (optional)")
        submitted = st.form_submit_button("🔍 Analyse Water Quality", use_container_width=True)

    if submitted:
        if not location:
            st.warning("⚠️ Please enter a sample location name.")
        else:
            score = float(wqi_score(ph, turbidity, do_, heavy, nitrates, tds, temperature, chlorine))
            grade, color = wqi_grade(score)
            safe = bool(score >= 50)
            params = {"pH":(ph,6.5,8.5),"Turbidity":(turbidity,0,4),"Dissolved O₂":(do_,5,14),
                      "Heavy Metals":(heavy,0,0.1),"Nitrates":(nitrates,0,10),"TDS":(tds,0,500),
                      "Temperature":(temperature,5,30),"Chlorine":(chlorine,0,0.5)}
            st.markdown("<br>", unsafe_allow_html=True)
            if safe:
                st.markdown(f'<div class="safe-banner">✅ WATER IS SAFE &nbsp;|&nbsp; WQI: {score}/100 &nbsp;|&nbsp; Grade: {grade}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="unsafe-banner">❌ WATER NOT SAFE &nbsp;|&nbsp; WQI: {score}/100 &nbsp;|&nbsp; Grade: {grade}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Parameter Breakdown")
                rows = [{"Parameter":n,"Measured":v,"Status":"✅ OK" if lo<=v<=hi else "❌ Exceeds"} for n,(v,lo,hi) in params.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with col2:
                cats = list(params.keys())
                raw = []
                for n,(v,lo,hi) in params.items():
                    span = hi-lo if hi!=lo else 1
                    norm = max(0,min(1,(v-lo)/span)) if n!="Dissolved O₂" else max(0,min(1,v/hi))
                    raw.append(norm*100)
                fig_r = go.Figure(go.Scatterpolar(r=raw+[raw[0]],theta=cats+[cats[0]],fill="toself",
                    fillcolor="rgba(0,180,216,0.2)",line=dict(color="#00b4d8",width=2),marker=dict(size=8,color="#00b4d8")))
                fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,100],color="#7fb3d3"),angularaxis=dict(color="#7fb3d3")),
                    paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f4ff"),title=dict(text="Parameter Profile",font=dict(color="#00b4d8")),height=380,margin=dict(t=50,b=20))
                st.plotly_chart(fig_r, use_container_width=True)
            st.markdown("### 💊 Recommendations")
            recs = []
            if ph<6.5: recs.append(("🔴 Low pH","Add lime or use alkaline filter."))
            if ph>8.5: recs.append(("🔴 High pH","Use RO filter or acidification treatment."))
            if turbidity>4: recs.append(("🟠 High Turbidity","Filter through sand/cloth filter before use."))
            if do_<5: recs.append(("🔴 Low Dissolved Oxygen","Aerate water; check for upstream pollution."))
            if heavy>0.1: recs.append(("🔴 High Heavy Metals","DANGER: Do NOT drink. Report to authorities!"))
            if nitrates>10: recs.append(("🟠 High Nitrates","Use RO. Not safe for infants."))
            if tds>500: recs.append(("🟠 High TDS","Install TDS filter or RO system."))
            if temperature>30: recs.append(("🟡 High Temperature","Cool and treat before drinking."))
            if chlorine>0.5: recs.append(("🟡 High Chlorine","Use activated carbon filter."))
            if not recs:
                st.markdown('<div class="tip-card">🎉 <strong>All parameters within WHO guidelines!</strong> Keep testing regularly.</div>', unsafe_allow_html=True)
            else:
                for t,d in recs: st.markdown(f'<div class="tip-card"><strong>{t}</strong><br>{d}</div>', unsafe_allow_html=True)
            record = {"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),"location":location,
                "source_type":source_type,"ph":float(ph),"turbidity":float(turbidity),"do":float(do_),
                "heavy_metals":float(heavy),"nitrates":float(nitrates),"tds":float(tds),
                "temperature":float(temperature),"chlorine":float(chlorine),"wqi":score,"grade":grade,"safe":safe,"notes":notes}
            st.session_state.history.append(record)
            save_history(st.session_state.history)
            if not safe: st.session_state.alerts.append({"time":record["timestamp"],"location":location,"wqi":score})
            st.success(f"✅ Result saved! Record #{len(st.session_state.history)} stored.")

# ═══════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics & Insights")
    history = st.session_state.history
    if len(history) < 2:
        st.info("📌 Add at least 2 water tests to see analytics.")
        st.stop()
    df = pd.DataFrame(history)
    num_cols = ["ph","turbidity","do","heavy_metals","nitrates","tds","temperature","chlorine","wqi"]
    avail = [c for c in num_cols if c in df.columns]
    st.markdown("### 📋 Summary Statistics")
    st.dataframe(df[avail].describe().round(2), use_container_width=True)
    st.markdown("---")
    st.markdown("### 📈 Parameter Trends")
    param_select = st.multiselect("Select parameters:", avail, default=["wqi","ph"])
    if param_select:
        fig_t = go.Figure()
        colors = ["#00b4d8","#06d6a0","#f77f00","#ef476f","#90e0ef","#ffd166","#a8dadc","#e76f51"]
        for idx,param in enumerate(param_select):
            if param in df.columns:
                fig_t.add_trace(go.Scatter(x=df["timestamp"],y=df[param],mode="lines+markers",
                    name=param.replace("_"," ").title(),line=dict(color=colors[idx%len(colors)],width=2),marker=dict(size=6)))
        fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f4ff"),xaxis=dict(showgrid=False,color="#7fb3d3"),
            yaxis=dict(gridcolor="rgba(0,180,216,0.1)",color="#7fb3d3"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)",bordercolor="#00b4d8"),height=400,margin=dict(t=20,b=20))
        st.plotly_chart(fig_t, use_container_width=True)
    col1,col2 = st.columns(2)
    with col1:
        fig_ph = go.Figure(go.Histogram(x=df["ph"],nbinsx=15,marker_color="#00b4d8"))
        fig_ph.add_vline(x=6.5,line_dash="dash",line_color="#ef233c")
        fig_ph.add_vline(x=8.5,line_dash="dash",line_color="#ef233c",annotation_text="WHO Limits")
        fig_ph.update_layout(title="pH Distribution",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f4ff"),height=300)
        st.plotly_chart(fig_ph, use_container_width=True)
    with col2:
        fig_w = go.Figure(go.Histogram(x=df["wqi"],nbinsx=15,marker_color="#06d6a0"))
        fig_w.add_vline(x=50,line_dash="dash",line_color="#f77f00",annotation_text="Safe Threshold")
        fig_w.update_layout(title="WQI Distribution",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f4ff"),height=300)
        st.plotly_chart(fig_w, use_container_width=True)
    col1,col2 = st.columns(2)
    with col1:
        if "safe" in df.columns:
            sc = df["safe"].value_counts()
            fig_pie = go.Figure(go.Pie(labels=["Safe ✅","Unsafe ❌"],values=[sc.get(True,0),sc.get(False,0)],
                hole=0.55,marker=dict(colors=["#06d6a0","#ef233c"],line=dict(color="#03071e",width=3)),
                textinfo="percent+label",textfont=dict(color="#fff")))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f4ff"),height=320,margin=dict(t=20,b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        if "grade" in df.columns:
            gc = df["grade"].value_counts()
            fig_g = go.Figure(go.Bar(x=gc.index,y=gc.values,
                marker_color=["#06d6a0","#90e0ef","#f77f00","#ef476f","#ef233c"][:len(gc)]))
            fig_g.update_layout(title="Grade Distribution",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f4ff"),xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(0,180,216,0.1)"),height=320,margin=dict(t=40,b=20))
            st.plotly_chart(fig_g, use_container_width=True)
    if len(avail) >= 3:
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[avail].corr().round(2)
        fig_h = go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
            colorscale=[[0,"#ef233c"],[0.5,"#03071e"],[1,"#06d6a0"]],
            text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=11,color="#fff")))
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f4ff"),height=400)
        st.plotly_chart(fig_h, use_container_width=True)

# ═══════════════════════════════════════════
# LOCATION MAP  ← FIXED
# ═══════════════════════════════════════════
elif page == "🗺️ Location Map":
    st.markdown("## 🗺️ Interactive Monitoring Network")
    history = st.session_state.history
    if not history:
        st.warning("No data yet. Load demo data from ⚙️ Settings first!")
        st.stop()
    df = pd.DataFrame(history)
    if "lat" not in df.columns or df["lat"].isna().all():
        rng = np.random.default_rng(42)
        locs_list = df["location"].unique().tolist()
        df["lat"] = df["location"].map({l: float(rng.uniform(8,35)) for l in locs_list})
        df["lon"] = df["location"].map({l: float(rng.uniform(68,97)) for l in locs_list})
    locations = df.groupby("location").agg(
        tests=("wqi","count"), avg_wqi=("wqi","mean"), min_wqi=("wqi","min"), max_wqi=("wqi","max"),
        safe_pct=("safe", lambda x: round(100*x.sum()/len(x),1)),
        last_safe=("safe","last"), lat=("lat","first"), lon=("lon","first"),
    ).reset_index()
    locations["avg_wqi"] = locations["avg_wqi"].round(1)
    locations["grade"] = locations["avg_wqi"].apply(lambda s: wqi_grade(s)[0])
    locations["status_label"] = locations["last_safe"].apply(lambda s: "🟢 Safe" if s else "🔴 Unsafe")

    c1,c2,c3 = st.columns([2,2,1])
    with c1: map_style = st.selectbox("🗺️ Map Style",["carto-darkmatter","open-street-map","carto-positron"],index=0)
    with c2: color_by = st.selectbox("🎨 Colour By",["Average WQI","Last Status (Safe/Unsafe)","Number of Tests"])
    with c3: show_labels = st.toggle("Labels", value=True)

    fig_map = go.Figure()
    # Glow rings
    fig_map.add_trace(go.Scattermapbox(lat=locations["lat"],lon=locations["lon"],mode="markers",
        marker=dict(size=locations["tests"].clip(upper=40)*2.5+30,color=locations["avg_wqi"],
            colorscale=[[0,"rgba(239,35,60,0.25)"],[0.5,"rgba(144,224,239,0.2)"],[1,"rgba(6,214,160,0.2)"]],
            opacity=0.55,sizemode="diameter"),hoverinfo="skip",showlegend=False))

    # ── Build marker dict safely ──
    if color_by == "Average WQI":
        _marker = dict(size=locations["tests"].clip(upper=40)+14, color=locations["avg_wqi"],
            colorscale=[[0,"#ef233c"],[0.25,"#f77f00"],[0.5,"#90e0ef"],[1,"#06d6a0"]],
            colorbar=dict(title="WQI",thickness=12,bgcolor="rgba(10,22,40,0.85)",
                tickfont=dict(color="#e0f4ff"),titlefont=dict(color="#00b4d8")),
            showscale=True, opacity=0.95, sizemode="diameter")
    elif color_by == "Number of Tests":
        _marker = dict(size=locations["tests"].clip(upper=40)+14, color=locations["tests"],
            colorscale=[[0,"#023e8a"],[1,"#00b4d8"]],
            colorbar=dict(title="Tests",thickness=12,bgcolor="rgba(10,22,40,0.85)",
                tickfont=dict(color="#e0f4ff"),titlefont=dict(color="#00b4d8")),
            showscale=True, opacity=0.95, sizemode="diameter")
    else:  # Last Status — hex strings, NO colorscale
        _marker = dict(size=locations["tests"].clip(upper=40)+14,
            color=list(locations["last_safe"].apply(lambda s: "#06d6a0" if s else "#ef233c")),
            opacity=0.95, sizemode="diameter")

    hover_text = [
        f"<b>📍 {r.location}</b><br>🧪 Tests: <b>{int(r.tests)}</b><br>"
        f"📊 Avg WQI: <b>{r.avg_wqi:.1f}</b> ({r.grade})<br>"
        f"✅ Safe Rate: <b>{r.safe_pct}%</b><br>🕐 Last: {r.status_label}"
        for r in locations.itertuples()
    ]
    fig_map.add_trace(go.Scattermapbox(
        lat=locations["lat"], lon=locations["lon"],
        mode="markers+text" if show_labels else "markers",
        text=locations["location"].apply(lambda x: x[:14]+"…" if len(x)>14 else x) if show_labels else None,
        textposition="top center", textfont=dict(size=11,color="#e0f4ff"),
        marker=_marker,
        hovertemplate="%{hovertext}<extra></extra>", hovertext=hover_text, showlegend=False))

    # Danger rings for unsafe
    unsafe_locs = locations[~locations["last_safe"]]
    if not unsafe_locs.empty:
        fig_map.add_trace(go.Scattermapbox(lat=unsafe_locs["lat"],lon=unsafe_locs["lon"],mode="markers",
            marker=dict(size=55,color="rgba(239,35,60,0.2)",sizemode="diameter"),hoverinfo="skip",showlegend=False))

    fig_map.update_layout(
        mapbox=dict(style=map_style,center=dict(lat=float(locations["lat"].mean()),lon=float(locations["lon"].mean())),zoom=4.5),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0f4ff"),height=560,margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""<div style="display:flex;gap:8px;align-items:center;margin:0.5rem 0 1.5rem 0;flex-wrap:wrap">
      <span style="color:#7fb3d3;font-size:0.85rem;font-weight:600">WQI Legend:</span>
      <span style="background:#ef233c;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">0–25 Very Poor</span>
      <span style="background:#f77f00;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">25–50 Poor</span>
      <span style="background:#0096c7;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">50–75 Fair</span>
      <span style="background:#06d6a0;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem">75–100 Good/Excellent</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 Station Summaries")
    for row_df in [locations.iloc[i:i+3] for i in range(0,len(locations),3)]:
        cols = st.columns(3)
        for ci,(_,loc) in enumerate(row_df.iterrows()):
            g_col = wqi_grade(loc["avg_wqi"])[1]
            sb = "rgba(6,214,160,0.12)" if loc["last_safe"] else "rgba(239,35,60,0.12)"
            sd = "#06d6a0" if loc["last_safe"] else "#ef233c"
            with cols[ci]:
                st.markdown(f"""<div style="background:{sb};border:1px solid {sd};border-radius:14px;padding:1.1rem;margin-bottom:0.8rem">
                  <div style="font-weight:700;color:#e0f4ff;margin-bottom:0.5rem">📍 {loc['location']}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:{g_col}">{loc['avg_wqi']}</div>
                  <div style="font-size:0.75rem;color:#7fb3d3">{loc['grade']} · ✅ {loc['safe_pct']}% safe · 🧪 {int(loc['tests'])} tests</div>
                  <div style="margin-top:0.5rem;background:rgba(0,0,0,0.3);border-radius:6px;height:6px">
                    <div style="width:{min(loc['avg_wqi'],100)}%;height:100%;background:linear-gradient(90deg,{g_col}88,{g_col});border-radius:6px"></div>
                  </div></div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Average WQI by Station")
    fig_bar = go.Figure(go.Bar(x=locations["location"],y=locations["avg_wqi"],
        marker=dict(color=locations["avg_wqi"],colorscale=[[0,"#ef233c"],[0.25,"#f77f00"],[0.5,"#0096c7"],[1,"#06d6a0"]]),
        text=locations["avg_wqi"].round(1),textposition="outside",textfont=dict(color="#e0f4ff")))
    fig_bar.add_hline(y=50,line_dash="dash",line_color="#f77f00",annotation_text="Safe Threshold",annotation_font_color="#f77f00")
    fig_bar.add_hline(y=75,line_dash="dot",line_color="#06d6a0",annotation_text="Good",annotation_font_color="#06d6a0")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0f4ff"),xaxis=dict(showgrid=False,tickangle=-20,color="#7fb3d3"),
        yaxis=dict(gridcolor="rgba(0,180,216,0.1)",range=[0,115],color="#7fb3d3"),height=380,margin=dict(t=30,b=60))
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════
# DAILY TIPS
# ═══════════════════════════════════════════
elif page == "💡 Daily Tips":
    st.markdown("## 💡 Daily Water Safety Tips")
    categories = {
        "🏠 At Home": [("Boil before drinking","Boil for at least 1 minute if unsure."),
            ("Change filters","Replace pitcher filters every 2 months."),
            ("Clean tanks","Clean water tanks every 6 months."),
            ("BPA-free containers","Store water in food-grade containers only."),
            ("Check for rust","Brownish water = corroded pipes. Get inspected."),
            ("No hot tap water for cooking","Hot water leaches lead from old pipes.")],
        "👶 For Infants": [("Formula-safe water","Use purified/boiled water for baby formula."),
            ("Blue baby syndrome","High nitrates (>10 mg/L) are dangerous for infants."),
            ("Fluoride check","Excess fluoride causes fluorosis — check levels."),
            ("Lead risk","Test for lead in homes built before 1986.")],
        "🌾 Agriculture": [("Irrigation pH","Use pH 6–8 water. High TDS damages crops."),
            ("Avoid runoff","Never use industrial runoff on edible crops."),
            ("Nitrate buildup","Use slow-release fertilizers."),
            ("Test seasonal","Borewell quality changes — test each season.")],
        "🚨 Emergency": [("Flood water","Never drink floodwater. Always treat/boil."),
            ("After power cuts","Boil water stored >24 hrs without circulation."),
            ("Chemical spill","Stop tap water use until authorities clear it."),
            ("Emergency prep","Store 3L per person per day for 3 days.")],
    }
    for cat,tips_list in categories.items():
        st.markdown(f"### {cat}")
        cols = st.columns(2)
        for i,(title,detail) in enumerate(tips_list):
            with cols[i%2]:
                st.markdown(f'<div class="tip-card"><strong>{title}</strong><br><span style="color:#a8d8ea">{detail}</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════
elif page == "📜 History":
    st.markdown("## 📜 Test History & Records")
    history = st.session_state.history
    if not history:
        st.info("No history yet. Go to **🔬 Test Water** to add records.")
        st.stop()
    df = pd.DataFrame(history)
    c1,c2,c3 = st.columns(3)
    with c1:
        locs = ["All"] + sorted(df["location"].unique().tolist())
        sel_loc = st.selectbox("📍 Filter Location", locs)
    with c2: sel_safe = st.selectbox("🛡️ Status", ["All","Safe Only","Unsafe Only"])
    with c3:
        if "source_type" in df.columns:
            sources = ["All"] + sorted(df["source_type"].dropna().unique().tolist())
            sel_src = st.selectbox("🌊 Source", sources)
        else: sel_src = "All"
    filtered = df.copy()
    if sel_loc != "All": filtered = filtered[filtered["location"]==sel_loc]
    if sel_safe == "Safe Only": filtered = filtered[filtered["safe"]==True]
    elif sel_safe == "Unsafe Only": filtered = filtered[filtered["safe"]==False]
    if sel_src != "All" and "source_type" in df.columns: filtered = filtered[filtered["source_type"]==sel_src]
    st.markdown(f"**Showing {len(filtered)} of {len(df)} records**")
    st.dataframe(filtered.sort_values("timestamp",ascending=False), use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "aquasafe_history.csv", "text/csv", use_container_width=True)
    with c2:
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []; save_history([]); st.rerun()

# ═══════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════
elif page == "⚠️ Alerts":
    st.markdown("## ⚠️ Safety Alerts")
    history = st.session_state.history
    critical = [r for r in history if r.get("wqi",100)<25]
    warning = [r for r in history if 25<=r.get("wqi",100)<50]
    safe_r = [r for r in history if r.get("safe")]
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-icon">🔴</div><div class="metric-value metric-status-bad">{len(critical)}</div><div class="metric-label">Critical (WQI &lt; 25)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-icon">🟠</div><div class="metric-value" style="color:#f77f00">{len(warning)}</div><div class="metric-label">Warning (WQI 25–50)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-icon">✅</div><div class="metric-value metric-status-ok">{len(safe_r)}</div><div class="metric-label">Safe Readings</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if critical:
        st.markdown("### 🔴 Critical Events")
        for r in sorted(critical,key=lambda x:x["timestamp"],reverse=True)[:10]:
            st.markdown(f'<div style="background:rgba(239,35,60,0.1);border:1px solid #ef233c;border-radius:12px;padding:1rem;margin-bottom:0.7rem"><strong>🚨 {r["location"]}</strong> | {r["timestamp"]} | WQI: {r["wqi"]}<br><span style="color:#ef476f">Do NOT use this water.</span></div>', unsafe_allow_html=True)
    if warning:
        st.markdown("### 🟠 Warnings")
        for r in sorted(warning,key=lambda x:x["timestamp"],reverse=True)[:10]:
            st.markdown(f'<div style="background:rgba(247,127,0,0.1);border:1px solid #f77f00;border-radius:12px;padding:1rem;margin-bottom:0.7rem"><strong>⚠️ {r["location"]}</strong> | {r["timestamp"]} | WQI: {r["wqi"]}<br><span style="color:#f77f00">Treatment recommended.</span></div>', unsafe_allow_html=True)
    if not critical and not warning:
        st.markdown('<div style="text-align:center;padding:3rem;color:#06d6a0;font-size:1.3rem;font-weight:600">✅ No safety alerts. All clear!</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# EDUCATION HUB
# ═══════════════════════════════════════════
elif page == "📚 Education Hub":
    st.markdown("## 📚 Water Quality Education Hub")
    tabs = st.tabs(["🧪 Parameters","🦠 Contaminants","💊 Treatment","📖 WQI Explained","🌍 Global Facts"])
    with tabs[0]:
        for param,icon,rng,about,effects in [
            ("pH","🔵","6.5–8.5","Measures acidity/alkalinity. Ideal=7.","Low: metallic taste, pipe corrosion. High: bitter taste."),
            ("Turbidity","☁️","< 4 NTU","Cloudiness from suspended particles.","Protects bacteria from disinfection."),
            ("Dissolved Oxygen","💨","≥ 5 mg/L","Essential for aquatic life.","< 5: aquatic stress. < 2: fish kills."),
            ("Heavy Metals","⚙️","< 0.1 mg/L","Lead, mercury, arsenic accumulate.","Brain damage, cancer, kidney disease."),
            ("Nitrates","🌱","< 10 mg/L","From fertilizers and sewage.","Blue baby syndrome in infants."),
            ("TDS","🧂","< 500 mg/L","All dissolved substances.","High: salty taste, scaling in appliances."),
        ]:
            with st.expander(f"{icon} {param} — Safe: {rng}"):
                st.markdown(f"**About:** {about}")
                st.markdown(f"**Health Effects:** {effects}")
    with tabs[1]:
        for t,e,tr in [
            ("🦠 E. coli & Bacteria","Diarrhoea, cholera, typhoid.","Boiling, chlorination, UV"),
            ("🧪 Arsenic","Skin lesions, cancer.","RO filtration, activated alumina"),
            ("🏭 Lead","Neurological damage in children.","Filter replacement, pipe replacement"),
            ("🌿 Pesticides","Liver/kidney damage, cancer.","Activated carbon, RO"),
            ("🛢️ Petroleum","Liver damage, cancer.","Activated carbon, air stripping"),
        ]:
            st.markdown(f'<div class="tip-card"><strong>{t}</strong><br><span style="color:#f77f00">⚠️ {e}</span><br><span style="color:#06d6a0">✅ {tr}</span></div>', unsafe_allow_html=True)
    with tabs[2]:
        for i,(m,p,c,u) in enumerate([
            ("🔥 Boiling","Kills bacteria, viruses","Doesn't remove chemicals","Emergency use"),
            ("💡 UV Disinfection","No chemicals needed","No residual protection","Point-of-use"),
            ("🔬 Reverse Osmosis","Removes 95–99% contaminants","Wastes water","Home drinking water"),
            ("🪨 Activated Carbon","Removes chlorine, odours","Doesn't kill bacteria","Taste/odour filter"),
        ]):
            with st.expander(m):
                st.markdown(f"✅ **Pros:** {p}")
                st.markdown(f"❌ **Cons:** {c}")
                st.markdown(f"🎯 **Best for:** {u}")
    with tabs[3]:
        for rng,grade,col,desc in [("90–100","Excellent","#06d6a0","Safe for all uses."),("75–89","Good","#90e0ef","Safe to drink."),
                                    ("50–74","Fair","#f77f00","Some treatment recommended."),("25–49","Poor","#ef476f","Not safe without treatment."),
                                    ("0–24","Very Poor","#ef233c","Severely contaminated. Do NOT use.")]:
            st.markdown(f'<div style="background:rgba(0,0,0,0.3);border-left:5px solid {col};border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:0.5rem"><strong style="color:{col}">{grade} ({rng})</strong><br><span style="color:#e0f4ff">{desc}</span></div>', unsafe_allow_html=True)
    with tabs[4]:
        for ic,stat,desc in [("💧","785 million people","lack access to clean drinking water"),
                              ("🦠","2 billion people","drink water contaminated with faeces"),
                              ("☠️","3.5 million deaths","occur annually from waterborne diseases"),
                              ("👶","1 child every 2 min","dies from water-related diseases"),
                              ("🌊","3%","of Earth's water is freshwater")]:
            st.markdown(f'<div class="tip-card">{ic} <strong style="color:#00b4d8">{stat}</strong> — {desc}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Application Settings")
    st.markdown("### 🎲 Demo Data")
    c1,c2 = st.columns(2)
    with c1:
        demo_loc = st.text_input("Demo Location Name", value="Demo Station")
        if st.button("🎲 Load Demo Data (60 records)", use_container_width=True):
            demo = generate_demo_history(demo_loc)
            st.session_state.history = demo
            save_history(st.session_state.history)
            st.success(f"✅ {len(demo)} records loaded! Go to 🏠 Dashboard.")
            st.balloons()
    with c2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.history = []; st.session_state.alerts = []
            save_history([]); st.success("✅ All data cleared."); st.rerun()
    st.markdown("---")
    if st.session_state.history:
        df_exp = pd.DataFrame(st.session_state.history)
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export All Data (CSV)", csv, "aquasafe_export.csv", "text/csv", use_container_width=True)
    st.markdown("---")
    st.markdown("### ℹ️ About AquaSafe")
    st.markdown('<div class="section-card"><div class="section-title">💧 AquaSafe v2.0</div><p>Comprehensive water quality monitoring platform. Built with Streamlit & WHO Guidelines.</p></div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown('<div style="text-align:center;color:#7fb3d3;font-size:0.85rem;padding:1rem">💧 <strong>AquaSafe</strong> — Water Quality Monitoring System | Based on <a href="https://www.who.int" style="color:#00b4d8">WHO Guidelines</a></div>', unsafe_allow_html=True)
