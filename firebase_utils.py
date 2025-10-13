import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
from datetime import datetime

# Initialize Firebase once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "YOUR_PROJECT_ID.appspot.com"
    })

db = firestore.client()
bucket = storage.bucket()

def upload_violation_image(local_path, plate_number, frame):
    blob = bucket.blob(f"violations/{os.path.basename(local_path)}")
    blob.upload_from_filename(local_path)
    blob.make_public()
    url = blob.public_url

    # Save record to Firestore
    data = {
        "frame": frame,
        "plate_number": plate_number,
        "image_url": url,
        "timestamp": datetime.utcnow()
    }
    db.collection("violations").add(data)
    print(f"🔥 Uploaded to Firebase: {url}")

    return url
