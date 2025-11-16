# deteksi.py
import os
import sys
import base64
import uuid
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential

# Ensure yolov9 local repo is available on path (if you clone it locally)
sys.path.append(os.path.join(os.getcwd(), "yolov9"))

# Config (centralized)
from config import MODEL_VEHICLE, MODEL_PLATE, GEMINI_API_KEY, SAVE_DIR

# PyTorch safe globals patch for custom model classes (keep this)
try:
    from models.yolo import DetectionModel
    add_safe_globals([DetectionModel, Sequential])
except Exception as e:
    print("⚠️ Warning: Could not register DetectionModel as safe global:", e)

# Monkey-patch torch.load to allow weights_only=False (older checkpoints)
_real_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# YOLOv9 imports (local repo expected)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Firebase uploader (your existing helper)
from firebase_utils import upload_violation_image

# Optional Gemini import & init (only if key is set)
gemini_available = False
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        gemini_available = True
    else:
        print("⚠️ GEMINI_API_KEY not set. Gemini disabled — will fallback to alternative OCR if available.")
except Exception as e:
    gemini_available = False
    print("⚠️ Could not initialize Gemini (google.generativeai). Exception:", e)

# Prepare save directory and CSV log
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "violations.csv")
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=["frame", "plate_number", "filename", "timestamp"])
    df.to_csv(LOG_PATH, index=False)

logged_plates = set()
tracked_vehicles = []

# ---------------- Helper functions ----------------
def log_violation(frame_no, plate_number, crop_img_path):
    df = pd.read_csv(LOG_PATH)
    new_entry = {
        "frame": frame_no,
        "plate_number": plate_number,
        "filename": crop_img_path,
        "timestamp": datetime.utcnow()
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[VIOLATION] Frame {frame_no}, Plate: {plate_number}, Saved: {crop_img_path}")

def is_left_turn_region(x, y, frame_w, frame_h):
    return (x < frame_w * 0.25) and (y > frame_h * 0.7)

def detect_plate_number_gemini(img):
    """
    Use Gemini Vision AI to read license plate.
    Returns plate string or 'UNKNOWN'.
    """
    if not gemini_available:
        return "UNKNOWN"
    try:
        # encode image as base64
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        prompt = """
Look at this image and read ONLY the license plate number.

Rules:
1. Return ONLY the alphanumeric characters you see on the plate
2. Remove all spaces
3. Convert to uppercase
4. If you cannot read it clearly, return "UNKNOWN"
5. Do not include any explanation, just the plate number

What is the license plate number?
"""
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_base64}
        ])
        # response.text may vary depending on SDK; handle safely
        plate_text = (getattr(response, "text", "") or str(response)).strip().upper()
        plate_text = "".join(c for c in plate_text if c.isalnum())

        if len(plate_text) < 4 or len(plate_text) > 15:
            print(f"  [GEMINI] Invalid length: '{plate_text}'")
            return "UNKNOWN"

        print(f"  [GEMINI] Detected: '{plate_text}'")
        return plate_text

    except Exception as e:
        print(f"  [GEMINI] Error: {e}")
        return "UNKNOWN"

def is_same_vehicle(cx, cy, tracked_vehicles, threshold=60):
    for (tx, ty) in tracked_vehicles:
        if np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) < threshold:
            return True
    return False

# ---------------- Main processing ----------------
def process_video(video_path,
                  vehicle_weights=MODEL_VEHICLE,
                  plate_weights=MODEL_PLATE,
                  conf_thres=0.25,
                  iou_thres=0.45,
                  save_dir=SAVE_DIR,
                  save_to_storage_fn=upload_violation_image,
                  use_gemini=True):
    """
    Process the video and detect violations.
    """
    device = select_device("")  # CPU by default or CUDA if available
    vehicle_model = DetectMultiBackend(vehicle_weights, device=device, dnn=False, fp16=False)
    vehicle_stride, vehicle_names = vehicle_model.stride, vehicle_model.names

    plate_model = DetectMultiBackend(plate_weights, device=device, dnn=False, fp16=False)
    plate_stride, plate_names = plate_model.stride, plate_model.names

    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out_path = os.path.join(save_dir, f"output_{uuid.uuid4().hex[:8]}.mp4")
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_no = 0
    results_list = []
    logged_plates.clear()
    tracked_vehicles.clear()

    print(f"\n🚀 Starting video processing...")
    print(f"   Video: {video_path}")
    print(f"   Vehicle model: {vehicle_weights}")
    print(f"   Plate model: {plate_weights}")
    print(f"   OCR Method: {'Gemini' if use_gemini and gemini_available else 'Fallback/OCR disabled'}")
    print(f"   Output: {out_path}\n")

    while True:
        ret, im0 = vid_cap.read()
        if not ret:
            break
        frame_no += 1

        # Vehicle detection
        im_vehicle = letterbox(im0, 640, stride=vehicle_stride, auto=True)[0]
        im_vehicle = im_vehicle[:, :, ::-1].transpose(2, 0, 1).copy()
        im_vehicle = torch.from_numpy(im_vehicle).to(device).float() / 255.0
        if im_vehicle.ndimension() == 3:
            im_vehicle = im_vehicle.unsqueeze(0)

        pred_vehicle = vehicle_model(im_vehicle, augment=False, visualize=False)
        pred_vehicle = pred_vehicle[0] if isinstance(pred_vehicle, list) else pred_vehicle
        pred_vehicle = non_max_suppression(pred_vehicle, conf_thres, iou_thres, max_det=1000)

        for det_vehicle in pred_vehicle:
            if len(det_vehicle):
                det_vehicle[:, :4] = scale_boxes(im_vehicle.shape[2:], det_vehicle[:, :4], im0.shape).round()
                for *xyxy, conf_vehicle, cls_vehicle in reversed(det_vehicle):
                    label_vehicle = vehicle_names[int(cls_vehicle)]
                    if label_vehicle not in ["car", "motorbike", "bus", "truck"]:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    if is_left_turn_region(cx, cy, width, height) or is_same_vehicle(cx, cy, tracked_vehicles):
                        continue

                    crop_vehicle = im0[y1:y2, x1:x2]
                    if crop_vehicle.size == 0:
                        continue

                    # Plate detection (run plate model inside vehicle crop)
                    im_plate = letterbox(crop_vehicle, 640, stride=plate_stride, auto=True)[0]
                    im_plate = im_plate[:, :, ::-1].transpose(2, 0, 1).copy()
                    im_plate = torch.from_numpy(im_plate).to(device).float() / 255.0
                    if im_plate.ndimension() == 3:
                        im_plate = im_plate.unsqueeze(0)

                    pred_plate = plate_model(im_plate, augment=False, visualize=False)
                    pred_plate = pred_plate[0] if isinstance(pred_plate, list) else pred_plate
                    pred_plate = non_max_suppression(pred_plate, conf_thres, iou_thres, max_det=1)

                    plate_number = "UNKNOWN"
                    if len(pred_plate) and len(pred_plate[0]):
                        det_plate = pred_plate[0]
                        det_plate[:, :4] = scale_boxes(im_plate.shape[2:], det_plate[:, :4], crop_vehicle.shape).round()
                        x1p, y1p, x2p, y2p = map(int, det_plate[0][:4])
                        crop_plate = crop_vehicle[y1p:y2p, x1p:x2p]

                        if crop_plate.size > 0:
                            if use_gemini and gemini_available:
                                plate_number = detect_plate_number_gemini(crop_plate)
                            else:
                                # You can add EasyOCR fallback here (if installed)
                                plate_number = "UNKNOWN"

                    # Save violation if unique
                    if plate_number != "UNKNOWN" and plate_number not in logged_plates:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        proof = crop_vehicle.copy()
                        cv2.putText(proof, f"PLATE: {plate_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"FRAME: {frame_no}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"TIME: {timestamp}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                        fname = f"frame{frame_no}_{plate_number}_{uuid.uuid4().hex[:6]}.jpg"
                        save_path = os.path.join(save_dir, fname)
                        cv2.imwrite(save_path, proof)
                        log_violation(frame_no, plate_number, save_path)
                        logged_plates.add(plate_number)
                        tracked_vehicles.append((cx, cy))

                        remote_url = None
                        if save_to_storage_fn:
                            try:
                                remote_url = save_to_storage_fn(save_path, plate_number, frame_no)
                            except Exception as e:
                                print("⚠️ Error uploading:", e)

                        results_list.append({
                            "frame": frame_no,
                            "plate_number": plate_number,
                            "local_path": save_path,
                            "remote_url": remote_url
                        })

                    # Visualize detection
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(im0, f"VIOLATION: {plate_number}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        out_video.write(im0)

        if frame_no % 30 == 0:
            print(f"  Processing frame {frame_no}... ({len(logged_plates)} violations found)")

    vid_cap.release()
    out_video.release()

    print(f"\n✅ Processing complete!")
    print(f"   Total frames: {frame_no}")
    print(f"   Violations detected: {len(results_list)}")
    print(f"   Output video: {out_path}")
    print(f"   CSV log: {LOG_PATH}\n")

    return {"output_video": out_path, "violations": results_list}