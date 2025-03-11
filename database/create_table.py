import psycopg2 
from database import get_db


def create_database_and_table():
    """Create database and table if they don't exist."""
    conn = get_db()
    try:
        cursor = conn.cursor()

        # Create tables in the correct order (dependencies first)
        create_vehicle_table = """
            CREATE TABLE IF NOT EXISTS Vehicle (
                id SERIAL PRIMARY KEY,
                classification VARCHAR(10) CHECK (classification IN ('CAR', 'MOTORCYCLE', 'TRUCK', 'BUS')) NOT NULL,
                brand VARCHAR(255) NOT NULL,
                license_plate VARCHAR(50) NOT NULL,
                color VARCHAR(50) NOT NULL
            );
        """
        cursor.execute(create_vehicle_table)
        
        create_camera_table = """
            CREATE TABLE IF NOT EXISTS Camera (
                id SERIAL PRIMARY KEY,
                location VARCHAR(255) NOT NULL,
                status VARCHAR(10) CHECK (status IN ('IN', 'OUT')) NOT NULL
            );
        """
        cursor.execute(create_camera_table)
        
        create_employee_table = """
            CREATE TABLE IF NOT EXISTS Employee (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                division_vendor VARCHAR(255) NOT NULL,
                nip_internship VARCHAR(50) NOT NULL,
                id_vehicle INT,
                FOREIGN KEY (id_vehicle) REFERENCES Vehicle(id)
            );
        """
        cursor.execute(create_employee_table)
        
        create_detection_table = """
            CREATE TABLE IF NOT EXISTS Detection (
                id SERIAL PRIMARY KEY,
                id_employee INT NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                id_camera INT NOT NULL,
                time TIMESTAMP NOT NULL,
                FOREIGN KEY (id_employee) REFERENCES Employee(id),
                FOREIGN KEY (id_camera) REFERENCES Camera(id)
            );
        """
        cursor.execute(create_detection_table)

        conn.commit()  # Commit the changes
        print("Tables created successfully.")
    except Exception as err:
        print(f"Error: {err}")
        conn.rollback()  # Rollback in case of error
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    create_database_and_table()
