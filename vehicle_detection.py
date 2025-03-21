import cv2 # type: ignore
import numpy as np
from ultralytics.solutions.solutions import BaseSolution  # type: ignore
from ultralytics.utils.plotting import Annotator, colors  # type: ignore
from datetime import datetime
from paddleocr import PaddleOCR # type: ignore
import torch # type: ignore
import os
import time  # Add for potential delay
from database.database import get_db
import re
from database.database_operations import insert_detection

# Create directory for license plate images
PLATE_OUTPUT_DIR = "Playback/plates"
os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)

# Add for full vehicle context images 
VEHICLE_CONTEXT_DIR = "Playback/vehicles"
os.makedirs(VEHICLE_CONTEXT_DIR, exist_ok=True)

# For OCR results
OCR_OUTPUT_FILE = "Playback/ocr_results.txt"
os.makedirs(os.path.dirname(OCR_OUTPUT_FILE), exist_ok=True)


class LicensePlateDetector(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize detection region
        self.logged_ids = set() 
        
        # Force CUDA usage for PyTorch/detection
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for detection: {self.device}")
        
        # Force CPU usage for PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        print("Using CPU for OCR processing")
        
        # Create visualization window
        cv2.namedWindow("License Plate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("License Plate", 400, 200)

    def initialize_region(self):
        """Initialize region where detection should be performed."""
        if hasattr(self, 'region') and self.region:
            # Convert region points to numpy array for easier processing
            self.region_pts = np.array(self.region, dtype=np.int32)
            # Create a mask for the region
            self.has_region = True
            print(f"Detection region initialized with {len(self.region)} points")
        else:
            self.has_region = False
            print("No detection region specified, will process entire frame")

    def is_inside_region(self, box):
        """Check if a bounding box is inside or intersects with the defined region."""
        if not self.has_region:
            return True  # Process all boxes if no region defined
        
        # Get center point of bounding box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if center point is inside the polygon
        # For rectangle: Check if point is within bounds
        if len(self.region_pts) == 4:  # Rectangle
            min_x = min(pt[0] for pt in self.region_pts)
            max_x = max(pt[0] for pt in self.region_pts)
            min_y = min(pt[1] for pt in self.region_pts)
            max_y = max(pt[1] for pt in self.region_pts)
            
            return (min_x <= center_x <= max_x) and (min_y <= center_y <= max_y)
        else:  # Polygon
            return cv2.pointPolygonTest(self.region_pts, (center_x, center_y), False) >= 0

    def perform_ocr(self, image_array):
        """Performs OCR on the given image and returns the extracted text."""
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

        # Draw region on frame if defined
        if self.has_region and len(self.region_pts) > 2:
            cv2.polylines(im0, [self.region_pts], True, (128, 0, 128), 1)  # Purple color, thickness 1
    
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Skip if the object is not in the defined region
            if not self.is_inside_region(box):
                continue
                
            self.store_tracking_history(track_id, box)  

            # Get class name
            class_name = self.names[int(cls)]
            
            # Draw the bounding box and track ID on it
            label = f"ID: {track_id} {class_name}"
            self.annotator.box_label(
                box, label=label, color=colors(track_id, True)
            )

            # Extract license plate for OCR
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image)

            # Process license plates with OCR text
            if track_id not in self.logged_ids and ocr_text.strip():
                # Clean and standardize license plate text immediately
                cleaned_ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_text.strip().upper())
                
                # Only proceed if there's still text after cleaning
                if cleaned_ocr_text:
                    # Log detection
                    print(
                        f"Plate detected: {current_time.strftime('%Y-%m-%d')}, {current_time.strftime('%H:%M:%S')}, {track_id}, {class_name}, {cleaned_ocr_text}"
                    )
                    
                    # Write the OCR result to text file
                    with open(OCR_OUTPUT_FILE, 'a') as ocr_file:
                        ocr_file.write(f"{cleaned_ocr_text}\n")
                    
                    self.logged_ids.add(track_id)
                    
                    # Save the entire region context instead of just the license plate
                    # This captures the vehicle and surrounding area
                    if self.has_region:
                        # Get the region bounds
                        x_min = max(0, min(pt[0] for pt in self.region_pts))
                        y_min = max(0, min(pt[1] for pt in self.region_pts))
                        x_max = min(im0.shape[1], max(pt[0] for pt in self.region_pts))
                        y_max = min(im0.shape[0], max(pt[1] for pt in self.region_pts))
                        
                        # Extract the entire detection region
                        region_image = np.array(im0)[y_min:y_max, x_min:x_max]
                        
                        # Create filename for the region image
                        region_filename = f"{VEHICLE_CONTEXT_DIR}/vehicle_region_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        
                        # Save the region image
                        cv2.imwrite(region_filename, region_image)
                        
                        # Use the region image path for database
                        image_path = region_filename
                    else:
                        # If no region defined, create a larger context around the license plate
                        # Expand the crop area by 200% in each direction
                        expand_factor = 2.0
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        width, height = (x2 - x1), (y2 - y1)
                        
                        # Calculate expanded coordinates
                        ex1 = max(0, int(center_x - width * expand_factor / 2))
                        ey1 = max(0, int(center_y - height * expand_factor / 2))
                        ex2 = min(im0.shape[1], int(center_x + width * expand_factor / 2))
                        ey2 = min(im0.shape[0], int(center_y + height * expand_factor / 2))
                        
                        # Extract the expanded context
                        context_image = np.array(im0)[ey1:ey2, ex1:ex2]
                        
                        # Create filename for the context image
                        context_filename = f"{VEHICLE_CONTEXT_DIR}/vehicle_context_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        
                        # Save the context image
                        cv2.imwrite(context_filename, context_image)
                        
                        # Use the context image path for database
                        image_path = context_filename
                    
                    # Still save the original cropped license plate with overlay text
                    plate_filename = f"{PLATE_OUTPUT_DIR}/plate_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Insert into database with the region/context image path
                    person_type, owner_name = insert_detection(cleaned_ocr_text, class_name, image_path, current_time)
                    
                    # Create visualization of the license plate (still use this for display)
                    if cropped_image.size > 0:
                        # Resize for better visibility while maintaining aspect ratio
                        h, w = cropped_image.shape[:2]
                        max_width = 400
                        scale = min(max_width / w, 200 / h)
                        new_width, new_height = int(w * scale), int(h * scale)
                        plate_img = cv2.resize(cropped_image, (new_width, new_height))
                        
                        # Create info section
                        plate_vis = np.ones((new_height + 150, max(new_width, 400), 3), dtype=np.uint8) * 255
                        
                        # Add the plate image
                        plate_vis[:new_height, :new_width] = plate_img
                        
                        # Add text information (use cleaned_ocr_text here)
                        cv2.putText(plate_vis, f"ID: {track_id}", (10, new_height + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(plate_vis, f"Vehicle: {class_name}", (10, new_height + 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(plate_vis, f"Type: {person_type}", (10, new_height + 75), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                        cv2.putText(plate_vis, f"Owner: {owner_name}", (10, new_height + 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                        cv2.putText(plate_vis, f"OCR: {cleaned_ocr_text}", (10, new_height + 125), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Display and save plate visualization
                        cv2.imshow("License Plate", plate_vis)
                        cv2.imwrite(plate_filename, plate_vis)

        self.display_output(im0)  # Display output with base class function
        return im0


# Open the video file
cap = cv2.VideoCapture("video-test/plat.mp4")

region_points = [
    (450, 100),   # Top-left
    (1000, 100),   # Top-right
    (1000, 300),   # Bottom-right
    (450, 300)    # Bottom-left
]

detector = LicensePlateDetector(
    # region=region_points,  
    model="Model/platnomor.pt",
    line_width=2,
)

# Ensure model is on GPU
if torch.cuda.is_available():
    detector.model = detector.model.to('cuda:0')

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
