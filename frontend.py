# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="AI Prescription Verifier", layout="wide")
st.title("🧪 AI Medical Prescription Verifier")

st.sidebar.header("Settings")
backend_url = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")

page = st.sidebar.radio("Page", ["Interaction Checker", "Dosage Checker"])

if page == "Interaction Checker":
    st.header("Drug–Drug Interaction Checker")
    text = st.text_area("Paste prescription text (e.g., 'Take ibuprofen 200 mg with warfarin')", height=160)
    if st.button("Check Interactions"):
        if not text.strip():
            st.warning("Enter prescription text.")
        else:
            try:
                resp = requests.post(f"{backend_url}/check_interactions", json={"prescription_text": text}, timeout=10)
                data = resp.json()
                st.subheader("Medicines detected")
                st.json(data.get("medicines_detected", []))
                interactions = data.get("interactions_found", [])
                if interactions:
                    st.error(f"Found {len(interactions)} potential interaction(s).")
                    for it in interactions:
                        with st.expander(f"{it['drug_a']} × {it['drug_b']} — {it.get('severity','unknown')}"):
                            st.write(it.get("interaction_description",""))
                else:
                    st.success("No interactions found.")
            except Exception as e:
                st.error(f"Request failed: {e}")

else:
    st.header("Dosage Checker (rules + ML)")
    drug = st.text_input("Medicine name (e.g., paracetamol)")
    age = st.number_input("Patient age", min_value=0, max_value=120, value=30)
    dose = st.number_input("Dose (mg)", min_value=0, max_value=10000, value=500)
    if st.button("Check Dosage"):
        if not drug.strip():
            st.warning("Enter medicine name")
        else:
            try:
                payload = {"prescription_text": f"{drug} {dose} mg", "age": int(age)}
                resp = requests.post(f"{backend_url}/check_dosage", json=payload, timeout=10)
                data = resp.json()
                st.json(data)
            except Exception as e:
                st.error(f"Request failed: {e}")
