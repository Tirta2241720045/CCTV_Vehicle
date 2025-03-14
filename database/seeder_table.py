import psycopg2 # type: ignore
from database import get_db
from datetime import datetime


def seed_data():
    """Seed dummy data into all tables."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # Updated vehicle data with the detected license plates
        vehicle_data = [
            ('CAR', 'Toyota', '2835BSY', 'Silver'),
            ('MOTORCYCLE', 'Honda', '0752GJR', 'Black'),
            ('TRUCK', 'Volvo', '8589BXT', 'Blue'),
            ('BUS', 'Mercedes', '6380CCS', 'White'),
            ('CAR', 'Nissan', '9079GCH', 'Red'),
            ('CAR', 'Toyota', 'MA4844CC', 'White'),
            ('MOTORCYCLE', 'Yamaha', '5553DNM', 'Green'),
            ('CAR', 'Hyundai', '3693FSG', 'Silver'),
            ('CAR', 'Ford', '3574BNW', 'Black'),
            ('TRUCK', 'Scania', '0262HFP', 'Blue'),
            ('CAR', 'Kia', '8934FMR', 'Red'),
            ('CAR', 'Mazda', '5280DLY', 'Gray'),
            ('BUS', 'Isuzu', '9916GHS', 'Yellow'),
            ('CAR', 'Suzuki', '7963JVJ', 'White'),
            ('CAR', 'Mitsubishi', '3092FRX', 'Green'),
            ('CAR', 'Daihatsu', 'MA3043DF', 'Silver'),
            ('MOTORCYCLE', 'Kawasaki', '1024DR', 'Red'),
            ('CAR', 'Tesla', '8174HGL', 'Black'),
            ('TRUCK', 'Hino', '4234DZK', 'White'),
            ('CAR', 'Lexus', '0526HGN', 'Blue'),
            ('CAR', 'Honda', '4557JMF', 'Silver'),
            ('CAR', 'Toyota', '5739JMC', 'Gray')
        ]
        
        vehicle_query = """
        INSERT INTO "Vehicle" (classification, brand, license_plate, color)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        vehicle_ids = []
        for data in vehicle_data:
            cursor.execute(vehicle_query, data)
            vehicle_id = cursor.fetchone()[0]
            vehicle_ids.append(vehicle_id)
        print(f"Added {len(vehicle_ids)} vehicles")
        
        # Continue with existing camera data
        camera_data = [
            ('Front Gate', 'IN'),
            ('Back Gate', 'OUT'),
            ('Parking Area', 'IN')
        ]
        
        camera_query = """
        INSERT INTO "Camera" (location, status)
        VALUES (%s, %s) RETURNING id
        """
        
        camera_ids = []
        for data in camera_data:
            cursor.execute(camera_query, data)
            camera_id = cursor.fetchone()[0]
            camera_ids.append(camera_id)
        print(f"Added {len(camera_ids)} cameras")
        
        # Update employees to use more vehicles (including some previously assigned to guests)
        employee_data = [
            ('John Doe', 'IT Department', 'EMP001', vehicle_ids[0]),
            ('Jane Smith', 'HR Department', 'EMP002', vehicle_ids[1]),
            ('Michael Johnson', 'Finance', 'EMP003', vehicle_ids[2]),
            ('Sarah Williams', 'Marketing', 'EMP004', vehicle_ids[3]),
            ('Robert Taylor', 'Operations', 'EMP005', vehicle_ids[4]),
            ('Emily Jones', 'Engineering', 'EMP006', vehicle_ids[5]),
            ('William Brown', 'Legal', 'EMP007', vehicle_ids[6])
        ]
        
        employee_query = """
        INSERT INTO "Employee" (name, division_vendor, nip_internship, id_vehicle)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        employee_ids = []
        for data in employee_data:
            cursor.execute(employee_query, data)
            employee_id = cursor.fetchone()[0]
            employee_ids.append(employee_id)
        print(f"Added {len(employee_ids)} employees")
        
        # Empty guest data - no guests will be created
        guest_data = []
        
        guest_query = """
        INSERT INTO "Guest" (name, id_vehicle)
        VALUES (%s, %s) RETURNING id
        """
        
        guest_ids = []
        # No guest data to insert
        print(f"Added {len(guest_ids)} guests")
     
        # Update interns to use more vehicles (including remaining previously assigned to guests)
        intership_data = [
            ('David Wilson', 'University A', 'Marketing', vehicle_ids[7]),
            ('Mary Johnson', 'College B', 'Engineering', vehicle_ids[8]),
            ('Thomas Brown', 'University C', 'IT Department', vehicle_ids[9]),
            ('Jennifer Davis', 'College D', 'Finance', vehicle_ids[10]),
            ('James Wilson', 'College E', 'Operations', vehicle_ids[11]),
            ('Alice Green', 'University F', 'Research', vehicle_ids[12]),
            ('Mark Anderson', 'College G', 'Development', vehicle_ids[13])
        ]
        
        intership_query = """
        INSERT INTO "Intership" (name, school_university, division_vendor, id_vehicle)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        intership_ids = []
        for data in intership_data:
            cursor.execute(intership_query, data)
            intership_id = cursor.fetchone()[0]
            intership_ids.append(intership_id)
        print(f"Added {len(intership_ids)} interns")
        
        # The remaining vehicles will not be associated with any person yet
        
        current_time = datetime.now()
        
        # Create detections for employees
        for i, employee_id in enumerate(employee_ids):
            detection_query = """
            INSERT INTO "Detection" (id_employee, id_guest, id_intership, image_path, id_camera, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(detection_query, 
                          (employee_id, None, None, f'images/employee{i+1}.jpg', camera_ids[0], current_time))
        
        # No detections for guests as there are no guests
        
        # Create detections for interns
        for i, intership_id in enumerate(intership_ids):
            detection_query = """
            INSERT INTO "Detection" (id_employee, id_guest, id_intership, image_path, id_camera, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(detection_query, 
                          (None, None, intership_id, f'images/intern{i+1}.jpg', camera_ids[2], current_time))

        conn.commit()
        print("All seed data inserted successfully")

    except Exception as err:
        print(f"Error seeding data: {err}")
        conn.rollback()  # Rollback in case of error
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    seed_data()
