# deteksi.py - Enhanced with Vehicle Type Classification & Color Detection
import os
import sys
import base64
import uuid
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
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
    df = pd.DataFrame(columns=["frame", "plate_number", "vehicle_type", "vehicle_color", "filename", "timestamp"])
    df.to_csv(LOG_PATH, index=False)

logged_plates = set()

# ============================================================================
# FEATURE 1: VEHICLE COLOR DETECTION WITH GEMINI
# ============================================================================

def detect_vehicle_color_gemini(img):
    """
    Use Gemini Vision AI to detect vehicle color.
    Returns color string (e.g., 'RED', 'BLUE', 'WHITE', etc.) or 'UNKNOWN'.
    """
    if not gemini_available:
        return detect_vehicle_color_opencv(img)  # Fallback to OpenCV
    
    try:
        # Encode image as base64
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        prompt = """
Look at this vehicle image and identify ONLY the primary color of the vehicle body.

Rules:
1. Return ONLY ONE color word (e.g., RED, BLUE, WHITE, BLACK, SILVER, GRAY, GREEN, YELLOW, BROWN, ORANGE)
2. Use the most dominant/visible color of the vehicle body
3. Ignore reflections, shadows, or background
4. If unclear, return "UNKNOWN"
5. Return ONLY the color word, nothing else

What color is this vehicle?
"""
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_base64}
        ])
        
        color_text = (getattr(response, "text", "") or str(response)).strip().upper()
        # Clean up response - take only first word if multiple words returned
        color_text = color_text.split()[0] if color_text else "UNKNOWN"
        
        # Validate color
        valid_colors = ['RED', 'BLUE', 'WHITE', 'BLACK', 'SILVER', 'GRAY', 'GREY', 
                       'GREEN', 'YELLOW', 'BROWN', 'ORANGE', 'PURPLE', 'PINK', 'BEIGE']
        
        if color_text in valid_colors:
            print(f"  [GEMINI-COLOR] Detected: '{color_text}'")
            return color_text
        else:
            print(f"  [GEMINI-COLOR] Invalid color '{color_text}', using fallback")
            return detect_vehicle_color_opencv(img)

    except Exception as e:
        print(f"  [GEMINI-COLOR] Error: {e}")
        return detect_vehicle_color_opencv(img)

def detect_vehicle_color_opencv(img):
    """
    Fallback: OpenCV-based vehicle color detection using HSV color space.
    Returns dominant color name.
    """
    try:
        # Resize for faster processing
        img_small = cv2.resize(img, (100, 100))
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            'RED': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'BLUE': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'GREEN': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'YELLOW': [(np.array([20, 50, 50]), np.array([40, 255, 255]))],
            'WHITE': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
            'BLACK': [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            'GRAY': [(np.array([0, 0, 50]), np.array([180, 30, 200]))]
        }
        
        # Count pixels for each color
        color_counts = {}
        for color_name, ranges in color_ranges.items():
            total_pixels = 0
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_pixels += cv2.countNonZero(mask)
            color_counts[color_name] = total_pixels
        
        # Find dominant color
        dominant_color = max(color_counts.keys(), key=lambda x: color_counts[x])
        
        # Require minimum threshold
        if color_counts[dominant_color] < 500:  # Minimum pixels
            return 'UNKNOWN'
        
        return dominant_color
        
    except Exception as e:
        print(f"  [OPENCV-COLOR] Error: {e}")
        return 'UNKNOWN'

# ============================================================================
# FEATURE 2: VEHICLE TYPE CLASSIFICATION & STATISTICS
# ============================================================================

# Vehicle type mapping (standardized names)
VEHICLE_TYPE_MAP = {
    'car': 'CAR',
    'truck': 'TRUCK',
    'bus': 'BUS',
    'motorbike': 'MOTORCYCLE',
    'motorcycle': 'MOTORCYCLE'
}

def get_vehicle_type(label_name):
    """
    Standardize vehicle type from YOLO detection label.
    Returns standardized type: CAR, TRUCK, BUS, MOTORCYCLE, or UNKNOWN
    """
    label_lower = label_name.lower()
    return VEHICLE_TYPE_MAP.get(label_lower, 'UNKNOWN')

class VehicleStatistics:
    """Track and analyze vehicle statistics"""
    def __init__(self):
        self.type_counts = defaultdict(int)
        self.color_counts = defaultdict(int)
        self.type_color_matrix = defaultdict(lambda: defaultdict(int))
        self.violations_by_type = defaultdict(int)
        self.violations_by_color = defaultdict(int)
        
    def add_vehicle(self, vehicle_type, vehicle_color, is_violation=False):
        """Record a vehicle detection"""
        self.type_counts[vehicle_type] += 1
        self.color_counts[vehicle_color] += 1
        self.type_color_matrix[vehicle_type][vehicle_color] += 1
        
        if is_violation:
            self.violations_by_type[vehicle_type] += 1
            self.violations_by_color[vehicle_color] += 1
    
    def get_summary(self):
        """Get statistics summary"""
        total_vehicles = sum(self.type_counts.values())
        total_violations = sum(self.violations_by_type.values())
        
        return {
            'total_vehicles': total_vehicles,
            'total_violations': total_violations,
            'type_distribution': dict(self.type_counts),
            'color_distribution': dict(self.color_counts),
            'violations_by_type': dict(self.violations_by_type),
            'violations_by_color': dict(self.violations_by_color),
            'type_color_matrix': {k: dict(v) for k, v in self.type_color_matrix.items()}
        }
    
    def print_summary(self):
        """Print formatted statistics"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 VEHICLE STATISTICS SUMMARY")
        print("="*60)
        
        print(f"\n🚗 Total Vehicles Detected: {summary['total_vehicles']}")
        print(f"🚨 Total Violations: {summary['total_violations']}")
        
        if summary['total_violations'] > 0:
            violation_rate = (summary['total_violations'] / summary['total_vehicles']) * 100
            print(f"📈 Violation Rate: {violation_rate:.1f}%")
        
        # Vehicle type distribution
        print(f"\n🚙 Vehicle Type Distribution:")
        for vtype, count in sorted(summary['type_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_vehicles']) * 100
            print(f"   {vtype:12s}: {count:3d} ({percentage:5.1f}%)")
        
        # Color distribution
        print(f"\n🎨 Vehicle Color Distribution:")
        for color, count in sorted(summary['color_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_vehicles']) * 100
            print(f"   {color:12s}: {count:3d} ({percentage:5.1f}%)")
        
        # Violations by type
        if summary['violations_by_type']:
            print(f"\n⚠️  Violations by Vehicle Type:")
            for vtype, count in sorted(summary['violations_by_type'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {vtype:12s}: {count:3d} violations")
        
        # Violations by color
        if summary['violations_by_color']:
            print(f"\n🚨 Violations by Vehicle Color:")
            for color, count in sorted(summary['violations_by_color'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {color:12s}: {count:3d} violations")
        
        # Most common combinations
        print(f"\n🔍 Most Common Vehicle Type + Color Combinations:")
        combinations = []
        for vtype, colors in summary['type_color_matrix'].items():
            for color, count in colors.items():
                combinations.append((f"{color} {vtype}", count))
        
        for combo, count in sorted(combinations, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {combo:20s}: {count:3d}")
        
        print("="*60 + "\n")

# Global statistics tracker
vehicle_stats = VehicleStatistics()

# ============================================================================
# ENHANCED LOGGING
# ============================================================================

def log_violation(frame_no, plate_number, crop_img_path, vehicle_type="UNKNOWN", vehicle_color="UNKNOWN"):
    """Enhanced log with vehicle type and color"""
    df = pd.read_csv(LOG_PATH)
    new_entry = {
        "frame": frame_no,
        "plate_number": plate_number,
        "vehicle_type": vehicle_type,
        "vehicle_color": vehicle_color,
        "filename": crop_img_path,
        "timestamp": datetime.utcnow()
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[VIOLATION] Frame {frame_no}, Plate: {plate_number}, Type: {vehicle_type}, Color: {vehicle_color}, Saved: {crop_img_path}")

# ============================================================================
# EXISTING HELPER FUNCTIONS
# ============================================================================

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

# ============================================================================
# MAIN PROCESSING (ENHANCED WITH TYPE & COLOR)
# ============================================================================

def process_video(video_path,
                  vehicle_weights=MODEL_VEHICLE,
                  plate_weights=MODEL_PLATE,
                  conf_thres=0.25,
                  iou_thres=0.45,
                  save_dir=SAVE_DIR,
                  save_to_storage_fn=upload_violation_image,
                  use_gemini=True):
    """
    Process the video and detect violations with vehicle type and color classification.
    """
    device = select_device("")
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
    tracked_vehicles = []
    
    # Reset statistics
    global vehicle_stats
    vehicle_stats = VehicleStatistics()

    print(f"\n🚀 Starting enhanced video processing...")
    print(f"   Video: {video_path}")
    print(f"   Vehicle model: {vehicle_weights}")
    print(f"   Plate model: {plate_weights}")
    print(f"   OCR Method: {'Gemini' if use_gemini and gemini_available else 'Fallback/OCR disabled'}")
    print(f"   🎨 Color Detection: {'Gemini' if gemini_available else 'OpenCV'}")
    print(f"   🚗 Vehicle Type Classification: Enabled")
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
                    
                    # ====== NEW: GET VEHICLE TYPE ======
                    vehicle_type = get_vehicle_type(label_vehicle)
                    
                    if is_left_turn_region(cx, cy, width, height) or is_same_vehicle(cx, cy, tracked_vehicles):
                        continue

                    crop_vehicle = im0[y1:y2, x1:x2]
                    if crop_vehicle.size == 0:
                        continue

                    # ====== COLOR DETECTION ======
                    if use_gemini and gemini_available:
                        vehicle_color = detect_vehicle_color_gemini(crop_vehicle)
                    else:
                        vehicle_color = detect_vehicle_color_opencv(crop_vehicle)

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
                            if use_gemini and gemini_available:
                                plate_number = detect_plate_number_gemini(crop_plate)
                            else:
                                plate_number = "UNKNOWN"

                    # ====== RECORD STATISTICS ======
                    is_violation = (plate_number != "UNKNOWN")
                    vehicle_stats.add_vehicle(vehicle_type, vehicle_color, is_violation)

                    # Save violation if unique
                    if plate_number != "UNKNOWN" and plate_number not in logged_plates:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        proof = crop_vehicle.copy()
                        
                        # ====== ENHANCED: Add type and color to proof image ======
                        cv2.putText(proof, f"PLATE: {plate_number}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"TYPE: {vehicle_type}", (10, 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"COLOR: {vehicle_color}", (10, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"FRAME: {frame_no}", (10, 105), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.putText(proof, f"TIME: {timestamp}", (10, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                        fname = f"frame{frame_no}_{plate_number}_{vehicle_type}_{vehicle_color}_{uuid.uuid4().hex[:6]}.jpg"
                        save_path = os.path.join(save_dir, fname)
                        cv2.imwrite(save_path, proof)
                        log_violation(frame_no, plate_number, save_path, vehicle_type, vehicle_color)
                        logged_plates.add(plate_number)
                        tracked_vehicles.append((cx, cy))

                        remote_url = None
                        if save_to_storage_fn:
                            try:
                                remote_url = save_to_storage_fn(save_path, plate_number, frame_no, vehicle_color, vehicle_type)
                            except Exception as e:
                                print("⚠️ Error uploading:", e)

                        results_list.append({
                            "frame": frame_no,
                            "plate_number": plate_number,
                            "vehicle_type": vehicle_type,
                            "vehicle_color": vehicle_color,
                            "local_path": save_path,
                            "remote_url": remote_url
                        })

                    # ====== ENHANCED: Visualize with type and color ======
                    color_bgr = (0, 0, 255) if plate_number != "UNKNOWN" else (0, 255, 0)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color_bgr, 2)
                    
                    # Multi-line info display
                    info_lines = [
                        f"{vehicle_type}",
                        f"{vehicle_color}",
                        f"{plate_number}" if plate_number != "UNKNOWN" else "Detected"
                    ]
                    
                    for i, line in enumerate(info_lines):
                        cv2.putText(im0, line, (x1, y1 - 10 - (i * 20)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

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
    
    # Print comprehensive statistics
    vehicle_stats.print_summary()

    return {
        "output_video": out_path, 
        "violations": results_list,
        "statistics": vehicle_stats.get_summary()
    }