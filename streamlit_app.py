import streamlit as st
import requests
from config import BACKEND_PORT
import os

st.set_page_config(page_title="Vehicle and Plate Number Detection", layout="wide")
st.title("🚦 Vehicle and Plate Number Detection")

default_backend = os.getenv("BACKEND_URL", f"http://127.0.0.1:8000")
backend_url = st.text_input("Backend URL", "http://127.0.0.1:8000")
uploaded_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

# --- Upload & Process ---
if uploaded_file and st.button("Process Video"):
    with st.spinner("Uploading and processing... please wait."):
        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        response = requests.post(f"{backend_url}/upload", files=files)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Processing complete!")
            st.video(result["output_video"])
            st.subheader("Detected Violations (Local Run):")
            if len(result["violations"]) == 0:
                st.write("No violations detected.")
            else:
                for v in result["violations"]:
                    st.image(v.get("local_path"), caption=f"Frame {v['frame']} — Plate: {v['plate_number']}")
        else:
            st.error(f"❌ Error: {response.text}")

# --- View Firebase Violations ---
st.divider()
st.subheader("📸 Violations from Firebase")

if st.button("Load Violations from Firebase"):
    try:
        response = requests.get(f"{backend_url}/violations")
        if response.status_code == 200:
            data = response.json()
            if len(data) == 0:
                st.info("No violation records found in Firebase yet.")
            else:
                # Custom CSS for better image display
                st.markdown("""
                <style>
                img {
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display violations in a 5-column grid
                cols_per_row = 5
                for i in range(0, len(data), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(data):
                            v = data[i + j]
                            with col:
                                st.image(
                                    v['image_url'], 
                                    caption=f"Plate: {v.get('plate_number', 'UNKNOWN')}",
                                    use_container_width=True
                                )
        else:
            st.error(f"Error fetching data: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")