import streamlit as st
import requests
import os

st.set_page_config(page_title="Traffic Violation Detection", layout="wide")
st.title("🚦 Traffic Light Violation Detection")

backend_url = st.text_input("Backend URL", "http://127.0.0.1:8000/upload")

uploaded_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

if uploaded_file and st.button("Process Video"):
    with st.spinner("Uploading and processing... please wait."):
        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Processing complete!")

            st.video(result["output_video"])

            st.subheader("Detected Violations:")
            if len(result["violations"]) == 0:
                st.write("No violations detected.")
            else:
                for v in result["violations"]:
                    st.image(v["local_path"], caption=f"Frame {v['frame']} — Plate: {v['plate_number']}")
        else:
            st.error(f"❌ Error: {response.text}")
