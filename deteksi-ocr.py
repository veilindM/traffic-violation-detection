import cv2
import torch
import os
import pandas as pd
import numpy as np
import easyocr
import uuid
from datetime import datetime
import sys, os
import torch
from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential

# Fix PyTorch 2.6+ weight loading restriction
try:
    from models.yolo import DetectionModel
    add_safe_globals([DetectionModel, Sequential])
except Exception as e:
    print("⚠️ Warning: Could not register DetectionModel as safe global:", e)

# Monkey-patch torch.load to always allow weights_only=False
_real_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = patched_torch_load

sys.path.append(os.path.join(os.getcwd(), "yolov9"))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from firebase_utils import upload_violation_image

# --- OCR ---
reader = easyocr.Reader(['en'])

SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_DIR, "violations.csv")
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=["frame", "plate_number", "filename"])
    df.to_csv(LOG_PATH, index=False)

logged_plates = set()
tracked_vehicles = []

# --- Helper functions ---
def log_violation(frame_no, plate_number, crop_img_path):
    df = pd.read_csv(LOG_PATH)
    new_entry = {"frame": frame_no, "plate_number": plate_number, "filename": crop_img_path}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[VIOLATION] Frame {frame_no}, Plate: {plate_number}, Saved: {crop_img_path}")

def is_left_turn_region(x, y, frame_w, frame_h):
    return (x < frame_w * 0.25) and (y > frame_h * 0.7)

def preprocess_plate_image(img):
    """
    Enhanced preprocessing for better OCR accuracy
    This fixes the DUMUFK → BJI7TFK issue
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    # Resize to standard height for better OCR
    height = 100
    if gray.shape[0] > 0:
        ratio = height / gray.shape[0]
        width = int(gray.shape[1] * ratio)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        resized = gray
    
    # Apply CLAHE with stronger parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    # Denoise to remove noise that confuses OCR
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpen the image to make text clearer
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Apply adaptive threshold for better text separation
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

def detect_plate_number_ocr(img):
    """
    Enhanced OCR with multiple preprocessing attempts
    Tries different preprocessing methods and picks the best result
    """
    # Prepare multiple preprocessing versions
    versions = []
    
    # Version 1: Original grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    versions.append(('original', gray))
    
    # Version 2: Enhanced version (from preprocess_plate_image)
    try:
        enhanced = preprocess_plate_image(img)
        versions.append(('enhanced', enhanced))
    except:
        pass
    
    # Version 3: Otsu threshold
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(('otsu', otsu))
    except:
        pass
    
    # Version 4: Inverted binary (for dark plates on light background)
    try:
        _, inv_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        versions.append(('inverted', inv_binary))
    except:
        pass
    
    best_text = "UNKNOWN"
    best_confidence = 0
    best_version = "none"
    
    # Try OCR on all versions, pick the best result
    for version_name, version_img in versions:
        try:
            results = reader.readtext(
                version_img,
                detail=1,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            if len(results) > 0:
                # Get text and confidence
                text = results[0][1].replace(" ", "").upper()
                text = "".join(c for c in text if c.isalnum())
                conf = results[0][2]
                
                # Update if this is better (higher confidence and reasonable length)
                if conf > best_confidence and len(text) >= 4:  # At least 4 characters
                    best_text = text
                    best_confidence = conf
                    best_version = version_name
                    
        except Exception as e:
            continue
    
    # Debug output (optional - comment out if too verbose)
    if best_text != "UNKNOWN":
        print(f"  [OCR] Best: '{best_text}' (conf: {best_confidence:.2f}, method: {best_version})")
    
    return best_text

def is_same_vehicle(cx, cy, tracked_vehicles, threshold=60):
    for (tx, ty) in tracked_vehicles:
        if np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) < threshold:
            return True
    return False


# --- Main function ---
def process_video(video_path, vehicle_weights="yolov9-c.pt", plate_weights="best.pt",
                  conf_thres=0.25, iou_thres=0.45, save_dir=SAVE_DIR, save_to_storage_fn=upload_violation_image):

    device = select_device("")
    vehicle_model = DetectMultiBackend(vehicle_weights, device=device, dnn=False, fp16=False)
    vehicle_stride, vehicle_names = vehicle_model.stride, vehicle_model.names

    plate_model = DetectMultiBackend(plate_weights, device=device, dnn=False, fp16=False)
    plate_stride, plate_names = plate_model.stride, plate_model.names

    vid_cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

                    # Plate detection
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
                            # Use enhanced OCR
                            plate_number = detect_plate_number_ocr(crop_plate)

                    # Save violation
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

                    # Draw on video frame
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(im0, f"VIOLATION: {plate_number}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        out_video.write(im0)
        
        # Progress indicator every 30 frames
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