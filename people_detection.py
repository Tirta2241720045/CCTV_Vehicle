import os
import cv2
from ultralytics import YOLO
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Konfigurasi model YOLO
MODEL_PATH = "Model/People.pt"
model = YOLO(MODEL_PATH)
model.to("cpu")  # Gunakan CPU untuk inferensi

# Inisialisasi DeepSORT tracker
tracker = DeepSort(max_age=30)  # max_age: berapa lama ID tetap aktif jika objek hilang

# Konfigurasi database
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Konfigurasi deteksi
DETECTION_THRESHOLD = 0.7  # Ambang kepercayaan deteksi
OVERTIME_THRESHOLD = 10  # Waktu maksimum (dalam detik) sebelum dianggap pelanggaran
RECORD_DURATION = 20  # Durasi rekaman (dalam detik)
RECORD_FPS = 30
PLAYBACK_FOLDER = "E:\\Magang\\Project\\Backend_CCTV\\Backend_CCTV\\Playback"


def save_violation_to_db(cursor, id_cctv, overtime_duration, video_path):
    """
    Menyimpan data pelanggaran ke database.
    """
    cursor.execute(
        """
    INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback)
    VALUES (%s, NULL, FALSE, %s, %s)
    """,
        (id_cctv, overtime_duration, video_path),
    )


def record_detection_video(cap, timestamp, initial_detection_duration=0):
    try:
        video_filename = f"Detection_{timestamp}.mp4"
        video_path = os.path.join(PLAYBACK_FOLDER, video_filename)
        os.makedirs(PLAYBACK_FOLDER, exist_ok=True)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Ambil FPS dari video input

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            video_path, fourcc, input_fps, (frame_width, frame_height)
        )

        frames_to_capture = (
            RECORD_DURATION * input_fps
        )  # Hitung frame berdasarkan FPS input
        frames_captured = 0

        while frames_captured < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break

            # Proses deteksi dan pelacakan
            results = model(frame, conf=DETECTION_THRESHOLD)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                bbox = (x1, y1, x2 - x1, x2 - y1)

                detection_duration = initial_detection_duration + (
                    frames_captured / input_fps
                )  # Hitung durasi berdasarkan FPS input
                color = (
                    (0, 255, 0)
                    if detection_duration <= OVERTIME_THRESHOLD
                    else (0, 0, 255)
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} - {str(timedelta(seconds=int(detection_duration)))}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            out.write(frame)
            frames_captured += 1

            cv2.imshow("Recording Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        return video_path

    except Exception as e:
        logging.error(f"Error recording detection video: {str(e)}")
        return None


def process_video(input_path, id_cctv=1):
    """
    Memproses video untuk mendeteksi orang dan menyimpan hasil deteksi ke video.
    """
    try:
        # Koneksi ke database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Buka video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        # Ambil frame rate video
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Inisialisasi variabel
        start_time = datetime.now()
        detection_start_times = (
            {}
        )  # Dictionary untuk menyimpan waktu mulai deteksi setiap ID
        overtime_duration = 0
        violation_bbox = None
        recording = False  # Flag untuk menandai apakah sedang merekam
        video_path = None  # Path video hasil rekaman
        frame_count = 0  # Hitung jumlah frame yang telah diproses

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Deteksi orang menggunakan YOLO
            results = model(frame, conf=DETECTION_THRESHOLD)

            # Konversi hasil deteksi ke format yang sesuai untuk DeepSORT
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, "person"))

            # Update tracker dengan deteksi terbaru
            tracks = tracker.update_tracks(detections, frame=frame)

            # Proses setiap objek yang dilacak
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # Jika ID belum ada dalam dictionary, tambahkan waktu mulai deteksi
                if track_id not in detection_start_times:
                    detection_start_times[track_id] = frame_count

                # Hitung durasi deteksi berdasarkan frame rate
                detection_duration = (
                    frame_count - detection_start_times[track_id]
                ) / input_fps

                # Jika durasi deteksi melebihi OVERTIME_THRESHOLD, ubah warna bounding box menjadi merah
                if detection_duration > OVERTIME_THRESHOLD:
                    overtime_duration = int(detection_duration)
                    violation_bbox = bbox
                    # Mulai merekam jika belum merekam
                    if not recording:
                        video_path = record_detection_video(
                            cap,
                            datetime.now().strftime("%Y%m%d_%H%M%S"),
                            detection_duration,
                        )
                        recording = True
                    # Simpan ke database
                    save_violation_to_db(cursor, id_cctv, overtime_duration, video_path)
                    conn.commit()
                    logging.info(
                        f"Pelanggaran terdeteksi: Melebihi waktu {overtime_duration} detik"
                    )
                    detection_start_times[track_id] = (
                        frame_count  # Reset timer untuk ID ini
                    )

                # Tentukan warna bounding box berdasarkan durasi deteksi
                color = (
                    (0, 255, 0)
                    if detection_duration <= OVERTIME_THRESHOLD
                    else (0, 0, 255)
                )

                # Gambar kotak, ID, dan timer
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} - {str(timedelta(seconds=int(detection_duration)))}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Tampilkan frame
            cv2.imshow("People Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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
    video_path = "video-test/test.mp4"
    process_video(video_path)
