import psycopg2
from database import get_db
from datetime import datetime


def seed_data():
    """Seed dummy data into all tables."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # 1. Seed Vehicle table first (since it's referenced by Employee)
        vehicle_data = [
            ('CAR', 'Toyota', 'ABC123', 'Silver'),
            ('MOTORCYCLE', 'Honda', 'XYZ789', 'Black'),
            ('TRUCK', 'Volvo', 'DEF456', 'Blue'),
            ('BUS', 'Mercedes', 'GHI789', 'White')
        ]
        
        vehicle_query = """
        INSERT INTO Vehicle (classification, brand, license_plate, color)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        vehicle_ids = []
        for data in vehicle_data:
            cursor.execute(vehicle_query, data)
            vehicle_id = cursor.fetchone()[0]
            vehicle_ids.append(vehicle_id)
        
        # 2. Seed Camera table
        camera_data = [
            ('Front Gate', 'IN'),
            ('Back Gate', 'OUT'),
            ('Parking Area', 'IN')
        ]
        
        camera_query = """
        INSERT INTO Camera (location, status)
        VALUES (%s, %s) RETURNING id
        """
        
        camera_ids = []
        for data in camera_data:
            cursor.execute(camera_query, data)
            camera_id = cursor.fetchone()[0]
            camera_ids.append(camera_id)
        
        # 3. Seed Employee table
        employee_data = [
            ('John Doe', 'IT Department', 'EMP001', vehicle_ids[0]),
            ('Jane Smith', 'HR Department', 'EMP002', vehicle_ids[1]),
            ('Bob Johnson', 'Finance', 'EMP003', vehicle_ids[2]),
            ('Alice Brown', 'Marketing', 'EMP004', vehicle_ids[3])
        ]
        
        employee_query = """
        INSERT INTO Employee (name, division_vendor, nip_internship, id_vehicle)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        employee_ids = []
        for data in employee_data:
            cursor.execute(employee_query, data)
            employee_id = cursor.fetchone()[0]
            employee_ids.append(employee_id)
        
        # 4. Seed Detection table
        current_time = datetime.now()
        detection_data = [
            (employee_ids[0], 'images/detection1.jpg', camera_ids[0], current_time),
            (employee_ids[1], 'images/detection2.jpg', camera_ids[1], current_time),
            (employee_ids[2], 'images/detection3.jpg', camera_ids[2], current_time)
        ]
        
        detection_query = """
        INSERT INTO Detection (id_employee, image_path, id_camera, time)
        VALUES (%s, %s, %s, %s)
        """
        
        for data in detection_data:
            cursor.execute(detection_query, data)

        conn.commit()  # Commit the changes
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
