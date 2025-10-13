import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

# Path to your Firebase key file (make sure it's in .gitignore)
FIREBASE_KEY_PATH = "firebase_key.json"

# Initialize Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "storageBucket": "traffic-violation-75252.appspot.com"
    })

# Firestore DB reference
db = firestore.client()

# Upload violation image to Firebase Storage
def upload_violation_image(local_path, plate_number, frame_no):
    bucket = storage.bucket()
    blob = bucket.blob(f"violations/frame{frame_no}_{plate_number}.jpg")
    blob.upload_from_filename(local_path)
    blob.make_public()  # Optional: make image accessible by URL
    url = blob.public_url

    # Save metadata to Firestore
    db.collection("violations").add({
        "frame": frame_no,
        "plate_number": plate_number,
        "url": url
    })

    print(f"[UPLOAD] Saved {plate_number} to Firebase Storage: {url}")
    return url
