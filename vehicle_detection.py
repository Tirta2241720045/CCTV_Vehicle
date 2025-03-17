import cv2 # type: ignore
import numpy as np
from ultralytics.solutions.solutions import BaseSolution  # type: ignore
from ultralytics.utils.plotting import Annotator, colors  # type: ignore
from datetime import datetime
from paddleocr import PaddleOCR # type: ignore
import torch # type: ignore
import os
from database.database import get_db
import re 

# Create directory for license plate images
PLATE_OUTPUT_DIR = "Playback/plates"
os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)

# Add this near the top of your file with other constants
OCR_OUTPUT_FILE = "Playback/ocr_results.txt"

# Ensure directory exists
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
    
    def database_insert(self, license_plate, class_name, image_path, current_time):
        """
        Insert detected license plate into database with search algorithm:
        1. Search if license plate exists in Vehicle table
        2. If exists, check for Employee/Intership associations
        3. If no associations found, create a Guest entry
        4. If plate doesn't exist at all, create a new Vehicle and Guest
        Returns the person type for display purposes.
        """
        if not license_plate or not license_plate.strip():
            print("Empty license plate text, skipping database insertion")
            return "Unknown"
        
        # Map the detected class name to valid classification values
        class_mapping = {
            'car': 'CAR',
            'truck': 'TRUCK',
            'bus': 'BUS',
            'motorcycle': 'MOTORCYCLE',
            'license': 'CAR',  # If detecting just license plates, default to CAR
            'license plate': 'CAR',
            'numberplate': 'CAR'  # Additional mapping for 'numberplate' class
        }
        
        # Default to CAR if no mapping found
        classification = 'CAR'
        if class_name.lower() in class_mapping:
            classification = class_mapping[class_name.lower()]
        
        person_type = "Unknown"
        owner_name = "Unknown"
        conn = None
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # Clean and standardize license plate text
            license_plate = license_plate.strip().upper()
            
            # Remove all non-alphanumeric characters
            license_plate = re.sub(r'[^A-Z0-9]', '', license_plate)

            # Optionally, add validation to ensure the plate isn't empty after cleaning
            if not license_plate:
                print("License plate contains no valid characters after cleaning")
                return "Unknown", "Unknown"
            
            # Step 1: Search for license plate in Vehicle table
            search_query = 'SELECT id FROM "Vehicle" WHERE license_plate = %s'
            cursor.execute(search_query, (license_plate,))
            vehicle_result = cursor.fetchone()
            
            # Variables to track detection person type
            employee_id = None
            guest_id = None
            intership_id = None
            
            if vehicle_result:
                # License plate found - use existing vehicle record
                vehicle_id = vehicle_result[0]
                print(f"Found existing vehicle with ID: {vehicle_id}")
                
                # Check if vehicle is associated with an Employee
                employee_query = 'SELECT id, name FROM "Employee" WHERE id_vehicle = %s'
                cursor.execute(employee_query, (vehicle_id,))
                employee_result = cursor.fetchone()
                
                if employee_result:
                    # Vehicle belongs to an employee
                    employee_id = employee_result[0]
                    owner_name = employee_result[1]
                    person_type = "Employee"
                    print(f"Vehicle belongs to Employee ID: {employee_id}")
                else:
                    # Check if vehicle is associated with an Intern
                    intership_query = 'SELECT id, name FROM "Intership" WHERE id_vehicle = %s'
                    cursor.execute(intership_query, (vehicle_id,))
                    intership_result = cursor.fetchone()
                    
                    if intership_result:
                        # Vehicle belongs to an intern
                        intership_id = intership_result[0]
                        owner_name = intership_result[1]
                        person_type = "Intership"
                        print(f"Vehicle belongs to Intership ID: {intership_id}")
                    else:
                        # Check if vehicle is associated with a Guest
                        guest_query = 'SELECT id, name FROM "Guest" WHERE id_vehicle = %s'
                        cursor.execute(guest_query, (vehicle_id,))
                        guest_result = cursor.fetchone()
                        
                        if guest_result:
                            # Vehicle already belongs to a Guest
                            guest_id = guest_result[0]
                            owner_name = guest_result[1]
                            person_type = "Guest"
                            print(f"Vehicle belongs to Guest ID: {guest_id}")
                        else:
                            # Vehicle exists but no person association - create Guest
                            guest_insert = """
                            INSERT INTO "Guest" (name, id_vehicle)
                            VALUES (%s, %s) RETURNING id
                            """
                            cursor.execute(guest_insert, (f"Guest-{license_plate}", vehicle_id))
                            guest_id = cursor.fetchone()[0]
                            person_type = "Guest"
                            owner_name = f"Guest-{license_plate}"
                            print(f"Created new Guest with ID: {guest_id} for existing vehicle")
            else:
                # License plate not found - create new vehicle and guest records
                print(f"Creating new vehicle record for license plate: {license_plate}")
                
                # Insert new vehicle
                vehicle_insert = """
                INSERT INTO "Vehicle" (classification, brand, license_plate, color)
                VALUES (%s, %s, %s, %s) RETURNING id
                """
                cursor.execute(vehicle_insert, (classification, "Unknown", license_plate, "Unknown"))
                vehicle_id = cursor.fetchone()[0]
                
                # Insert new guest associated with this vehicle
                guest_insert = """
                INSERT INTO "Guest" (name, id_vehicle)
                VALUES (%s, %s) RETURNING id
                """
                cursor.execute(guest_insert, (f"Guest-{license_plate}", vehicle_id))
                guest_id = cursor.fetchone()[0]
                person_type = "Guest"
                owner_name = f"Guest-{license_plate}"
                print(f"Created new Guest with ID: {guest_id} for new vehicle")
            
            # Get camera ID (using first camera for demo, modify as needed)
            camera_query = 'SELECT id FROM "Camera" LIMIT 1'
            cursor.execute(camera_query)
            camera_result = cursor.fetchone()
            
            if camera_result:
                camera_id = camera_result[0]
            else:
                # Insert a default camera if none exists
                cursor.execute('INSERT INTO "Camera" (location, status) VALUES (%s, %s) RETURNING id', 
                              ("Main Entrance", "IN"))
                camera_id = cursor.fetchone()[0]
            
            # Insert detection record with appropriate person type
            detection_insert = """
            INSERT INTO "Detection" (id_employee, id_guest, id_intership, image_path, id_camera, time)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """
            cursor.execute(detection_insert, (employee_id, guest_id, intership_id, image_path, camera_id, current_time))
            detection_id = cursor.fetchone()[0]
            
            # Commit transaction
            conn.commit()
            print(f"Successfully inserted detection record ID: {detection_id}")
            
            # Return person information
            return person_type, owner_name
            
        except Exception as e:
            print(f"Database error: {e}")
            if conn:
                conn.rollback()
            return person_type, owner_name
        finally:
            if conn:
                conn.close()

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

            # Extract license plate
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
                    
                    # Save the license plate image with overlay
                    plate_filename = f"{PLATE_OUTPUT_DIR}/plate_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Insert into database and get person type (use the cleaned text)
                    person_type, owner_name = self.database_insert(cleaned_ocr_text, class_name, plate_filename, current_time)
                    
                    # Create visualization of the license plate
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
cap = cv2.VideoCapture("video-test/tc.mp4")

region_points = [
    (450, 100),   # Top-left
    (1000, 100),   # Top-right
    (1000, 300),   # Bottom-right
    (450, 300)    # Bottom-left
]

# Initialize the license plate detector with CUDA
detector = LicensePlateDetector(
    region=region_points,
    model="Model/best.pt",
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
