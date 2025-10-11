import cv2
import torch
import os
import pandas as pd
import uuid
import sys
import easyocr

# Add YOLOv9 repo to Python path
sys.path.append(os.path.join(os.getcwd(), "yolov9"))

# --- YOLOv9 imports ---
from models.common import DetectMultiBackend
from models.yolo import DetectionModel
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# --- Fix for PyTorch 2.6+ weights loading ---
torch.serialization.add_safe_globals([DetectionModel])

# --- OCR setup ---
reader = easyocr.Reader(['en'])

SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_DIR, "violations.csv")
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=["frame", "plate_number", "filename"])
    df.to_csv(LOG_PATH, index=False)

logged_plates = set()

def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()[:100]

def log_violation(frame_no, plate_number, crop_img_path):
    df = pd.read_csv(LOG_PATH)
    new_entry = {"frame": frame_no, "plate_number": plate_number, "filename": crop_img_path}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[VIOLATION] Frame {frame_no}, Plate: {plate_number}, Saved: {crop_img_path}")

def is_left_turn_region(x, y, frame_w, frame_h):
    return (x < frame_w * 0.25) and (y > frame_h * 0.7)

def detect_plate_number(crop_img):
    try:
        results = reader.readtext(crop_img)
        if len(results) > 0:
            return results[0][1]
        return "UNKNOWN"
    except Exception as e:
        print("OCR error:", e)
        return "UNKNOWN"

def process_video(video_path, weights="yolov9-c.pt", conf_thres=0.25, iou_thres=0.45, save_dir=SAVE_DIR, save_to_storage_fn=None):
    """
    Process a traffic video and return:
    {
      "output_video": path_to_output_video,
      "violations": [
        {"frame": int, "plate_number": str, "local_path": str, "remote_url": optional str}
      ]
    }
    """
    device = select_device("")
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names = model.stride, model.names

    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out_path = os.path.join(save_dir, f"output_detected_{uuid.uuid4().hex[:8]}.mp4")
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_no = 0
    results_list = []
    global logged_plates
    logged_plates.clear()

    while True:
        ret, im0 = vid_cap.read()
        if not ret:
            break
        frame_no += 1

        im = letterbox(im0, 640, stride=stride, auto=True)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1).copy()
        im = torch.from_numpy(im).to(device).float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        if isinstance(pred, list):
            pred = pred[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    if label in ["car", "motorbike", "bus", "truck"]:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if is_left_turn_region(cx, cy, im0.shape[1], im0.shape[0]):
                            continue

                        traffic_light_red = True
                        if traffic_light_red:
                            crop_img = im0[y1:y2, x1:x2]
                            plate_number = detect_plate_number(crop_img)
                            plate_number_clean = sanitize_filename(plate_number or "UNKNOWN")

                            if plate_number_clean not in logged_plates:
                                fname = f"frame{frame_no}_{plate_number_clean}_{uuid.uuid4().hex[:6]}.jpg"
                                save_path = os.path.join(save_dir, fname)
                                cv2.imwrite(save_path, crop_img)
                                log_violation(frame_no, plate_number_clean, save_path)
                                logged_plates.add(plate_number_clean)

                                remote_url = None
                                if save_to_storage_fn:
                                    try:
                                        remote_url = save_to_storage_fn(save_path)
                                    except Exception as e:
                                        print("Error uploading:", e)
                                        remote_url = None

                                results_list.append({
                                    "frame": frame_no,
                                    "plate_number": plate_number_clean,
                                    "local_path": save_path,
                                    "remote_url": remote_url
                                })

                            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(im0, f"VIOLATION: {plate_number_clean}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_video.write(im0)

    vid_cap.release()
    out_video.release()
    return {"output_video": out_path, "violations": results_list}
