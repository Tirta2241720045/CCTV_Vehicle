import time
import cv2
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from paddleocr import PaddleOCR
import torch
import os


# Create directory for license plate images
PLATE_OUTPUT_DIR = "Playback/plates"
os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)


class LicensePlateDetector(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize detection region
        self.logged_ids = set()  # Set to keep track of already logged IDs
        
        # Check and set CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize the OCR system with CUDA if available
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=torch.cuda.is_available())
        
        # Create visualization window
        cv2.namedWindow("License Plate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("License Plate", 400, 200)


    def perform_ocr(self, image_array):
        if image_array is None:
            raise ValueError("Image is None")
        if isinstance(image_array, np.ndarray):
            results = self.ocr.ocr(image_array, rec=True)
        else:
            raise TypeError("Input image is not a valid numpy array")
        return " ".join([result[1][0] for result in results[0]] if results[0] else "")


    def detect_plates(self, im0):
        """Detect license plates in the frame."""
        self.annotator = Annotator(
            im0, line_width=self.line_width
        )  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Get current date and time

        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)


            # Get class name
            class_name = self.names[int(cls)]
            
            # Draw the bounding box and track ID on it
            label = f"ID: {track_id} {class_name}"
            self.annotator.box_label(
                box, label=label, color=colors(track_id, True)
            )

            # Extract license plate
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image)

            # Process license plates with OCR text
            if track_id not in self.logged_ids and ocr_text.strip():

                print(
                    f"Plate detected: {current_time.strftime('%Y-%m-%d')}, {current_time.strftime('%H:%M:%S')}, {track_id}, {class_name}, {ocr_text}"
                )
                self.logged_ids.add(track_id)

                
                # Save the license plate image with overlay
                plate_filename = f"{PLATE_OUTPUT_DIR}/plate_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                
                # Create visualization of the license plate
                if cropped_image.size > 0:
                    # Resize for better visibility while maintaining aspect ratio
                    h, w = cropped_image.shape[:2]
                    max_width = 400
                    scale = min(max_width / w, 200 / h)
                    new_width, new_height = int(w * scale), int(h * scale)
                    plate_img = cv2.resize(cropped_image, (new_width, new_height))
                    
                    # Create info section
                    plate_vis = np.ones((new_height + 100, max(new_width, 400), 3), dtype=np.uint8) * 255
                    
                    # Add the plate image
                    plate_vis[:new_height, :new_width] = plate_img
                    
                    # Add text information
                    cv2.putText(plate_vis, f"ID: {track_id}", (10, new_height + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(plate_vis, f"Class: {class_name}", (10, new_height + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(plate_vis, f"OCR: {ocr_text}", (10, new_height + 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                    
                    # Display and save plate visualization
                    cv2.imshow("License Plate", plate_vis)
                    cv2.imwrite(plate_filename, plate_vis)


        self.display_output(im0)
        return im0

# Open the video file
cap = cv2.VideoCapture("video-test/tc.mp4")

# Define region points for detection
region_points = [(0, 145), (1018, 145)]

# Initialize the license plate detector with CUDA if available
detector = LicensePlateDetector(
    region=region_points,
    model="Model/best.pt",  # Using the license plate model
    line_width=2,
)

# Create video writer for output
output_path = "Playback/detection_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_width, output_height = 1020, 500
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Process every 3rd frame for better performance
        continue

    frame = cv2.resize(frame, (output_width, output_height))

    # Process the frame with the license plate detector
    result = detector.detect_plates(frame)
    
    # Write frame to output video
    out.write(result)

    # Show the frame
    cv2.imshow("License Plate Detection", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

