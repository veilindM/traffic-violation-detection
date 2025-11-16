from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from deteksi import process_video
from firebase_utils import db
from config import UPLOAD_DIR

app = FastAPI(title="Traffic Violation Detection API")

UPLOAD_DIR = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_video(file_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/violations")
def get_violations():
    docs = db.collection("violations").stream()
    return [doc.to_dict() for doc in docs]
