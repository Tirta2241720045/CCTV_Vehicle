import psycopg2
from database import get_db
from datetime import datetime


def seed_data():
    """Seed dummy data into all tables."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        vehicle_data = [
            ('CAR', 'Toyota', 'ABC123', 'Silver'),
            ('MOTORCYCLE', 'Honda', 'XYZ789', 'Black'),
            ('TRUCK', 'Volvo', 'DEF456', 'Blue'),
            ('BUS', 'Mercedes', 'GHI789', 'White'),
            ('CAR', 'Nissan', 'JKL012', 'Red'),
            ('MOTORCYCLE', 'Yamaha', 'MNO345', 'Green')
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
        
        employee_data = [
            ('John Doe', 'IT Department', 'EMP001', vehicle_ids[0]),
            ('Jane Smith', 'HR Department', 'EMP002', vehicle_ids[1])
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
        
        guest_data = [
            ('Bob Johnson', vehicle_ids[2]),
            ('Alice Brown', vehicle_ids[3])
        ]
        
        guest_query = """
        INSERT INTO "Guest" (name, id_vehicle)
        VALUES (%s, %s) RETURNING id
        """
        
        guest_ids = []
        for data in guest_data:
            cursor.execute(guest_query, data)
            guest_id = cursor.fetchone()[0]
            guest_ids.append(guest_id)
        print(f"Added {len(guest_ids)} guests")
     
        intership_data = [
            ('David Wilson', 'University A', 'Marketing', vehicle_ids[4]),
            ('Mary Johnson', 'College B', 'Engineering', vehicle_ids[5])
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
        
      
        current_time = datetime.now()
        
        # Create detections for employees
        for i, employee_id in enumerate(employee_ids):
            detection_query = """
            INSERT INTO "Detection" (id_employee, id_guest, id_intership, image_path, id_camera, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(detection_query, 
                          (employee_id, None, None, f'images/employee{i+1}.jpg', camera_ids[0], current_time))
        
        # Create detections for guests
        for i, guest_id in enumerate(guest_ids):
            detection_query = """
            INSERT INTO "Detection" (id_employee, id_guest, id_intership, image_path, id_camera, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(detection_query, 
                          (None, guest_id, None, f'images/guest{i+1}.jpg', camera_ids[1], current_time))
        
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
