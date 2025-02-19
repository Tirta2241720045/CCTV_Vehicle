import os
import cv2
from ultralytics import YOLO
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Konfigurasi model YOLO
MODEL_PATH = "Model/People.pt"
model = YOLO(MODEL_PATH)
model.to('cpu')  # Gunakan CPU untuk inferensi

# Konfigurasi database
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Konfigurasi deteksi
DETECTION_THRESHOLD = 0.5  # Ambang kepercayaan deteksi
OVERTIME_THRESHOLD = 10  # Waktu maksimum (dalam detik) sebelum dianggap pelanggaran
RECORD_DURATION = 20  # Durasi rekaman (dalam detik)
RECORD_FPS = 30
PLAYBACK_FOLDER = "Playback"

def save_violation_to_db(cursor, id_cctv, overtime_duration, video_path):
    """
    Menyimpan data pelanggaran ke database.
    """
    cursor.execute("""
    INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback)
    VALUES (%s, NULL, FALSE, %s, %s)
    """, (id_cctv, overtime_duration, video_path))

def record_violation_video(cap, timestamp):
    """
    Merekam video pelanggaran.
    """
    try:
        # Buat nama file video
        video_filename = f"Pelanggaran_{timestamp}.mp4"
        video_path = os.path.join(PLAYBACK_FOLDER, video_filename)
        
        # Pastikan folder playback ada
        os.makedirs(PLAYBACK_FOLDER, exist_ok=True)
        
        # Ambil properti video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Inisialisasi video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, RECORD_FPS, (frame_width, frame_height))
        
        # Hitung jumlah frame yang perlu direkam
        frames_to_capture = RECORD_DURATION * RECORD_FPS
        frames_captured = 0
        
        # Mulai merekam
        while frames_captured < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Tambahkan teks pelanggaran ke frame
            cv2.putText(frame, f"Pelanggaran: Melebihi Waktu", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            frames_captured += 1
            
            # Tampilkan progress rekaman
            cv2.imshow('Recording Violation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        out.release()
        return video_path
        
    except Exception as e:
        logging.error(f"Error recording violation video: {str(e)}")
        return None

def process_video(input_path, id_cctv=1):
    """
    Memproses video untuk mendeteksi orang dan mencatat pelanggaran.
    """
    try:
        # Koneksi ke database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Buka video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        # Inisialisasi variabel
        start_time = datetime.now()
        detection_start_time = None
        overtime_duration = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi orang menggunakan YOLO
            results = model(frame, conf=DETECTION_THRESHOLD)

            # Jika orang terdeteksi
            if len(results[0].boxes) > 0:
                if detection_start_time is None:
                    detection_start_time = datetime.now()
                else:
                    # Hitung durasi deteksi
                    detection_duration = (datetime.now() - detection_start_time).total_seconds()
                    if detection_duration > OVERTIME_THRESHOLD:
                        overtime_duration = int(detection_duration)
                        # Rekam video pelanggaran
                        video_path = record_violation_video(cap, datetime.now().strftime('%Y%m%d_%H%M%S'))
                        # Simpan ke database
                        save_violation_to_db(cursor, id_cctv, overtime_duration, video_path)
                        conn.commit()
                        logging.info(f"Pelanggaran terdeteksi: Melebihi waktu {overtime_duration} detik")
                        detection_start_time = None  # Reset timer
            else:
                detection_start_time = None

            # Tampilkan frame
            cv2.imshow('People Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Bersihkan
        cap.release()
        cv2.destroyAllWindows()
        cursor.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    video_path = 'video-test/test.mp4'
    process_video(video_path)