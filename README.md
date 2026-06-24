💧 AquaSafe – Water Quality Monitoring System
A comprehensive, beautiful water quality monitoring and analysis platform built with Streamlit.

PythonStreamlitPlotlyLicense

🌟 Features
Feature	Description
🏠 Dashboard	Live KPI cards, animated WQI gauge, 30-day trend chart
🔬 Test Water	8-parameter testing with WHO guidelines, radar chart, smart recommendations
📊 Analytics	Trend charts, histograms, pie charts, correlation heatmap
🗺️ Location Map	Interactive dark-mode map of all monitoring stations
💡 Daily Tips	Safety tips for Home, Children, Agriculture, Industry, Emergency
📜 History	Filterable records with CSV export
⚠️ Alerts	Critical & warning alerts with configurable thresholds
📚 Education Hub	Parameter guide, contaminants, treatment methods, WQI explained
⚙️ Settings	Demo data, export, and configuration
🧪 Parameters Tested
pH — Acidity/Alkalinity (WHO: 6.5–8.5)
Turbidity — Cloudiness (WHO: < 4 NTU)
Dissolved Oxygen — Aquatic health indicator (≥ 5 mg/L)
Heavy Metals — Lead, Mercury, Arsenic etc. (< 0.1 mg/L)
Nitrates — Fertilizer/sewage indicator (< 10 mg/L)
TDS — Total Dissolved Solids (< 500 mg/L)
Temperature — (5–30 °C)
Residual Chlorine — (< 0.5 mg/L)
🚀 Run Locally
bash

# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aquasafe-water-quality.git
cd aquasafe-water-quality
# 2. Install dependencies
pip install -r requirements.txt
# 3. Run the app
streamlit run app.py
Then open http://localhost:8501 in your browser.

📦 Tech Stack
Streamlit — Web framework
Plotly — Interactive charts
Pandas / NumPy — Data processing
JSON — Local data persistence
📖 Data Standards
All parameter limits are based on:

WHO Guidelines for Drinking-Water Quality (4th Edition, 2011)

👨‍💻 Author
Made with ❤️ by Anmol Saharawat

⚠️ For educational purposes only. Always consult a certified laboratory for official water quality testing.
