# config.py
import os
from dotenv import load_dotenv

# Load .env in local dev only; in production env vars will be used
load_dotenv()

# Firebase
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET", "traffic-violation-app-a5c5c.appspot.com")

# Gemini / LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models & OCR
MODEL_VEHICLE = os.getenv("MODEL_VEHICLE", "yolov9-c.pt")
MODEL_PLATE = os.getenv("MODEL_PLATE", "best.pt")
OCR_LANG = os.getenv("OCR_LANG", "en")

# Paths
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
SAVE_DIR = os.getenv("SAVE_DIR", "violations")

# App
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))