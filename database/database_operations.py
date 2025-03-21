import re
from database import get_db

def insert_detection(license_plate, class_name, image_path, current_time):
    """
    Insert detected license plate into database with search algorithm:
    1. Search if license plate exists in Vehicle table
    2. If exists, check for Employee/Intership associations
    3. If no associations found, create a Guest entry
    4. If plate doesn't exist at all, create a new Vehicle and Guest
    Returns the person type and owner name for display purposes.
    """
    if not license_plate or not license_plate.strip():
        print("Empty license plate text, skipping database insertion")
        return "Unknown", "Unknown"
    
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