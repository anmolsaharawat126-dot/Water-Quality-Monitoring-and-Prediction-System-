import streamlit as st

st.set_page_config(page_title="Water Quality Monitoring System", page_icon="💧")

st.title("💧 Water Quality Monitoring & Prediction System")

name = st.text_input("Sample Location")

ph = st.number_input("pH Value", value=7.0)
turbidity = st.number_input("Turbidity", value=1.0)
do = st.number_input("Dissolved Oxygen", value=6.0)
heavy = st.number_input("Heavy Metal Concentration", value=0.05)

if st.button("Check Water Quality"):

    ph_status = "OK" if 6.5 <= ph <= 8.5 else "NOT OK"
    turb_status = "OK" if turbidity < 5 else "NOT OK"
    do_status = "OK" if do >= 5 else "NOT OK"
    heavy_status = "OK" if heavy < 0.1 else "NOT OK"

    st.subheader(f"Results for: {name}")

    st.write("pH Status:", ph_status)
    st.write("Turbidity Status:", turb_status)
    st.write("Dissolved Oxygen Status:", do_status)
    st.write("Heavy Metals Status:", heavy_status)

    if (
        ph_status == "OK"
        and turb_status == "OK"
        and do_status == "OK"
        and heavy_status == "OK"
    ):
        st.success("✅ WATER IS SAFE")
    else:
        st.error("❌ WATER IS NOT SAFE")
