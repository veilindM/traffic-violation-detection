# firebase_utils.py
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
from datetime import datetime

FIREBASE_KEY_PATH = "firebase_key.json"  # path to the JSON you downloaded
BUCKET_NAME = "traffic-violation-app-a5c5c.firebasestorage.app"  # replace with your actual bucket

# Initialize Firebase app once
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "storageBucket": BUCKET_NAME
    })

db = firestore.client()
bucket = storage.bucket()

def upload_violation_image(local_path, plate_number, frame_no):
    """
    Upload the image at local_path to Firebase Storage and
    create a Firestore document in collection 'violations'.
    Returns public URL (or storage path).
    """
    blob_name = f"violations/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    # Optionally make public (easy for testing). Production: use signed URLs or secure rules.
    try:
        blob.make_public()
        url = blob.public_url
    except Exception:
        url = f"gs://{BUCKET_NAME}/{blob_name}"

    # store metadata in Firestore
    doc = {
        "frame": frame_no,
        "plate_number": plate_number,
        "image_url": url,
        "blob_path": blob_name,
        "timestamp": datetime.utcnow()
    }
    db.collection("violations").add(doc)
    print(f"[firebase_utils] uploaded {local_path} -> {url}")
    return url
