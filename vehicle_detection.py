import time
import cv2
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from paddleocr import PaddleOCR
import os
from datetime import datetime


class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.logged_ids = set()
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def perform_ocr(self, image_array):
        if image_array is None:
            raise ValueError("Image is None")
        if isinstance(image_array, np.ndarray):
            results = self.ocr.ocr(image_array, rec=True)
        else:
            raise TypeError("Input image is not a valid numpy array")
        return " ".join([result[1][0] for result in results[0]] if results[0] else "")

    def estimate_speed(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = (
                f"{int(self.spd[track_id])} km/h"
                if track_id in self.spd
                else self.names[int(cls)]
            )
            label = f"ID: {track_id} {speed_label}"
            self.annotator.box_label(box, label=label, color=colors(track_id, True))

            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(
                self.r_s
            ):
                direction = "known"
            else:
                direction = "unknown"

            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = (
                        np.abs(
                            self.track_line[-1][1].item()
                            - self.trk_pp[track_id][1].item()
                        )
                        / time_difference
                    )
                    self.spd[track_id] = round(speed)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]
            x1, y1, x2, y2 = map(int, box)
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image)
            class_name = self.names[int(cls)]
            speed = self.spd.get(track_id)

            if (
                track_id not in self.logged_ids
                and ocr_text.strip()
                and speed is not None
            ):
                print(
                    f"Data to save: {current_time.strftime('%Y-%m-%d')}, {current_time.strftime('%H:%M:%S')}, {track_id}, {class_name}, {speed}, {ocr_text}"
                )
                self.logged_ids.add(track_id)
                # Tambahkan teks OCR ke frame
                cv2.putText(
                    im0,  # Frame yang akan ditambahi teks
                    f"OCR: {ocr_text}",  # Teks hasil OCR
                    (x1, y1 - 10),  # Posisi teks (di atas bounding box)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Jenis font
                    0.5,  # Ukuran font
                    (0, 255, 0),  # Warna teks (hijau)
                    1,  # Ketebalan garis
                )

        self.display_output(im0)
        return im0


def enhance_image(image):
    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    # Contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image


def compress_video(input_path, output_path, target_size_mb=10):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output_path = "temp_compressed.mp4"
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    # Compress video to target size
    os.system(
        f"ffmpeg -i {temp_output_path} -vcodec libx264 -crf 23 -preset medium -b:v 1000k -maxrate 1000k -bufsize 2000k -y {output_path}"
    )
    os.remove(temp_output_path)


def process_video(input_path, output_path, speed_obj):
    cap = cv2.VideoCapture(input_path)
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = enhance_image(frame)
        processed_frame = speed_obj.estimate_speed(frame)
        processed_frames.append(processed_frame)
        cv2.imshow("Processed Video", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save processed frames to a temporary video
    temp_output_path = "temp_processed.mp4"
    height, width, _ = processed_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, 30, (width, height))
    for frame in processed_frames:
        out.write(frame)
    out.release()

    # Compress the final video to ensure it's under 10 MB
    compress_video(temp_output_path, output_path, target_size_mb=10)
    os.remove(temp_output_path)


if __name__ == "__main__":
    input_video_path = r"E:\Magang\Project\CCTV_Vehicle\video-test\test_vehicle.mp4"
    output_video_path = r"E:\Magang\Project\CCTV_Vehicle\Playback\processed_video.mp4"
    region_points = [(0, 145), (1018, 145)]
    speed_obj = SpeedEstimator(
        region=region_points,
        model=r"E:\Magang\Project\CCTV_Vehicle\Model\best.pt",
        line_width=2,
    )

    process_video(input_video_path, output_video_path, speed_obj)
