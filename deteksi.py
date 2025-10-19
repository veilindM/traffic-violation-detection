import cv2
import torch
import os
import argparse
import pandas as pd
import numpy as np
import easyocr

# Import dari paket YOLOv9 (asumsi sudah ada di lingkungan Colab)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Inisialisasi OCR
reader = easyocr.Reader(['en'])

# Folder penyimpanan
SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

# File log CSV
LOG_PATH = os.path.join(SAVE_DIR, "violations.csv")
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=["frame", "plate_number", "filename"])
    df.to_csv(LOG_PATH, index=False)

# Simpan kendaraan yang sudah dicatat
tracked_vehicles = []  # Menyimpan pusat koordinat kendaraan yang sudah dicatat
logged_plates = set()  # Menyimpan plat yang sudah dicatat


# ======================= Fungsi Pendukung ===========================
def log_violation(frame_no, plate_number, crop_img_path):
    """Catat pelanggaran ke CSV"""
    df = pd.read_csv(LOG_PATH)
    new_entry = {"frame": frame_no, "plate_number": plate_number, "filename": crop_img_path}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[VIOLATION] Frame {frame_no}, Plate: {plate_number}, Saved: {crop_img_path}")


def is_left_turn_region(x, y, frame_w, frame_h):
    """Cek apakah kendaraan ada di jalur belok kiri"""
    # Contoh: Kuadran kiri bawah (25% lebar dari kiri dan 30% tinggi dari bawah)
    return (x < frame_w * 0.25) and (y > frame_h * 0.7)


def preprocess_plate_image(crop_img):
    """
    Terapkan preprocessing (Grayscale + CLAHE) untuk meningkatkan kualitas OCR pada gambar plat.
    """
    if crop_img.ndim == 3:
        # Konversi ke Grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img
    
    # Adaptive Histogram Equalization (CLAHE) untuk meningkatkan kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def detect_plate_number_ocr(crop_img):
    """Deteksi teks plat nomor dengan normalisasi menggunakan OCR"""
    results = reader.readtext(crop_img)
    if len(results) > 0:
        # Asumsikan plat berada di urutan pertama (paling besar/jelas)
        text = results[0][1]
        # Normalisasi hasil OCR
        text = text.replace(" ", "").upper()
        # Filter karakter yang tidak mungkin (contoh sederhana, hanya alfanumerik)
        filtered_text = "".join(c for c in text if c.isalnum())
        return filtered_text
    return "UNKNOWN"


def is_same_vehicle(cx, cy, tracked_vehicles, threshold=60):
    """Cek apakah kendaraan sudah pernah terdeteksi sebelumnya berdasarkan jarak"""
    for (tx, ty) in tracked_vehicles:
        dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        if dist < threshold:
            return True
    return False


# ======================= Fungsi Utama ================================
def process_video(video_path, vehicle_weights, plate_weights, conf_thres=0.25, iou_thres=0.45):
    device = select_device("")
    
    # 1. Inisialisasi Model Deteksi Kendaraan (Vehicle Detector)
    print(f"Loading Vehicle Model: {vehicle_weights}...")
    vehicle_model = DetectMultiBackend(vehicle_weights, device=device, dnn=False, fp16=False)
    vehicle_stride, vehicle_names = vehicle_model.stride, vehicle_model.names

    # 2. Inisialisasi Model Deteksi Plat Nomor (License Plate Detector)
    print(f"Loading Plate Model: {plate_weights}...")
    plate_model = DetectMultiBackend(plate_weights, device=device, dnn=False, fp16=False)
    plate_stride, plate_names = plate_model.stride, plate_model.names
    
    vid_cap = cv2.VideoCapture(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(SAVE_DIR, "output_detected.mp4")
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_no = 0
    while True:
        ret, im0 = vid_cap.read()
        if not ret:
            break
        frame_no += 1
        
        # Preprocessing untuk Vehicle Model
        im_vehicle = letterbox(im0, 640, stride=vehicle_stride, auto=True)[0]
        im_vehicle = im_vehicle[:, :, ::-1].transpose(2, 0, 1).copy()
        im_vehicle = torch.from_numpy(im_vehicle).to(device).float() / 255.0
        if im_vehicle.ndimension() == 3:
            im_vehicle = im_vehicle.unsqueeze(0)

        # 3. Deteksi Kendaraan
        pred_vehicle = vehicle_model(im_vehicle, augment=False, visualize=False)
        if isinstance(pred_vehicle, list):
            pred_vehicle = pred_vehicle[0]
        pred_vehicle = non_max_suppression(pred_vehicle, conf_thres, iou_thres, max_det=1000)

        for det_vehicle in pred_vehicle:
            # Variabel untuk koordinat plat nomor yang terdeteksi
            x1_p, y1_p, x2_p, y2_p = -1, -1, -1, -1
            
            if len(det_vehicle):
                # Rescale BBox kendaraan ke ukuran frame asli
                det_vehicle[:, :4] = scale_boxes(im_vehicle.shape[2:], det_vehicle[:, :4], im0.shape).round()
                
                for *xyxy_vehicle, conf_vehicle, cls_vehicle in reversed(det_vehicle):
                    label_vehicle = vehicle_names[int(cls_vehicle)]
                    
                    if label_vehicle in ["car", "motorbike", "bus", "truck"]:
                        x1_v, y1_v, x2_v, y2_v = map(int, xyxy_vehicle)
                        cx_v, cy_v = (x1_v + x2_v) // 2, (y1_v + y2_v) // 2

                        # 🛑 Cek Jalur dan Tracking
                        if is_left_turn_region(cx_v, cy_v, im0.shape[1], im0.shape[0]):
                            continue
                        if is_same_vehicle(cx_v, cy_v, tracked_vehicles):
                            continue 

                        # 🚥 Simulasi Lampu Merah (Kondisi Pelanggaran)
                        traffic_light_red = True
                        if traffic_light_red:
                            crop_vehicle = im0[y1_v:y2_v, x1_v:x2_v]
                            plate_number = "UNKNOWN"
                            
                            if crop_vehicle.size > 0:
                                # Preprocessing untuk Plate Model
                                im_plate = letterbox(crop_vehicle, 640, stride=plate_stride, auto=True)[0]
                                im_plate = im_plate[:, :, ::-1].transpose(2, 0, 1).copy()
                                im_plate = torch.from_numpy(im_plate).to(device).float() / 255.0
                                if im_plate.ndimension() == 3:
                                    im_plate = im_plate.unsqueeze(0)
                                    
                                pred_plate = plate_model(im_plate, augment=False, visualize=False)
                                if isinstance(pred_plate, list):
                                    pred_plate = pred_plate[0]
                                    
                                pred_plate = non_max_suppression(pred_plate, conf_thres, iou_thres, max_det=1)
                                
                                # 4. Plat Terdeteksi oleh YOLO Plate Model
                                if len(pred_plate[0]):
                                    det_plate = pred_plate[0]
                                    det_plate[:, :4] = scale_boxes(im_plate.shape[2:], det_plate[:, :4], crop_vehicle.shape).round()
                                    
                                    # Simpan BBox plat (relatif terhadap crop_vehicle)
                                    x1_p, y1_p, x2_p, y2_p = map(int, det_plate[0][:4])
                                    
                                    # Crop area plat
                                    crop_plate = crop_vehicle[y1_p:y2_p, x1_p:x2_p]
                                    
                                    # 5. Preprocessing & OCR
                                    if crop_plate.size > 0:
                                        preprocessed_plate = preprocess_plate_image(crop_plate)
                                        plate_number = detect_plate_number_ocr(preprocessed_plate)
                                        
                                # 6. Fallback: OCR Langsung di Area Kendaraan (dengan Preprocessing)
                                else:
                                    preprocessed_vehicle = preprocess_plate_image(crop_vehicle)
                                    plate_number = detect_plate_number_ocr(preprocessed_vehicle)

                            # 📝 Logging dan Saving
                            if plate_number == "":
                                plate_number = "UNKNOWN"

                            if plate_number != "UNKNOWN" and plate_number not in logged_plates:
                                
                                # --- AUGMENTASI GAMBAR BUKTI (WATERMARKING) ---
                                proof_image = crop_vehicle.copy() 
                                
                                # Dapatkan waktu saat ini
                                import datetime
                                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                # Tambahkan teks informasi ke gambar bukti
                                info_line1 = f"PLAT: {plate_number}"
                                info_line2 = f"FRAME: {frame_no}"
                                info_line3 = f"TIME: {current_time}"
                                
                                cv2.putText(proof_image, info_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                cv2.putText(proof_image, info_line2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                cv2.putText(proof_image, info_line3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                # ---------------------------------------------
                                
                                # Simpan gambar kendaraan yang melanggar (gambar bukti yang sudah diaugmentasi)
                                save_path = os.path.join(SAVE_DIR, f"frame{frame_no}_{plate_number}.jpg")
                                cv2.imwrite(save_path, proof_image)
                                
                                log_violation(frame_no, plate_number, save_path)
                                logged_plates.add(plate_number)
                                tracked_vehicles.append((cx_v, cy_v))

                            # 🖼️ Visualisasi pada Video Output
                            color = (0, 0, 255)
                            cv2.rectangle(im0, (x1_v, y1_v), (x2_v, y2_v), color, 2)
                            cv2.putText(im0, f"VIOLATION: {plate_number}", (x1_v, y1_v - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Gambar bounding box plat nomor jika terdeteksi (hijau)
                            if x1_p != -1: # Cek jika koordinat plat nomor telah diperbarui
                                # Rescale koordinat plat ke frame asli
                                x1_p_abs = x1_v + x1_p
                                y1_p_abs = y1_v + y1_p
                                x2_p_abs = x1_v + x2_p
                                y2_p_abs = y1_v + y2_p
                                cv2.rectangle(im0, (x1_p_abs, y1_p_abs), (x2_p_abs, y2_p_abs), (0, 255, 0), 2)
                                cv2.putText(im0, f"LP: {plate_number}", (x1_p_abs, y1_p_abs - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Simpan frame
        out_video.write(im0)

    vid_cap.release()
    out_video.release()
    print(f"\n✅ Proses selesai! Hasil video: {out_path}")
    print(f"📂 Gambar pelanggar & log ada di folder: {SAVE_DIR}")


# ======================= Entry Point ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="traffic.mp4", help="Path ke file video")
    parser.add_argument("--vehicle-weights", type=str, default="yolov9-c.pt", help="YOLOv9 weights path untuk deteksi kendaraan")
    parser.add_argument("--plate-weights", type=str, default="yolov9_license.pt", help="YOLOv9 weights path untuk deteksi plat nomor")
    args = parser.parse_args()

    process_video(args.source, args.vehicle_weights, args.plate_weights)
