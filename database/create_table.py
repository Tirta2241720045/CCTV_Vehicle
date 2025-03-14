import psycopg2 # type: ignore
from database import get_db


def drop_all_tables():
    """Drop all existing tables in the correct order to handle dependencies."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # Drop tables in reverse order of dependencies (CASCADE ensures all dependent objects are also dropped)
        print("Dropping existing tables...")
        tables = ["Detection", "Intership", "Guest", "Employee", "Camera", "Vehicle"]
        
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            print(f"Dropped table {table} if it existed")
        
        conn.commit()
        print("All tables dropped successfully.")
        return True
        
    except Exception as err:
        print(f"Error dropping tables: {err}")
        conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def create_database_and_table():
    """Create database and table if they don't exist."""
    conn = get_db()
    try:
        cursor = conn.cursor()

        # Create tables in the correct order (dependencies first)
        create_vehicle_table = """
            CREATE TABLE IF NOT EXISTS "Vehicle" (
                id SERIAL PRIMARY KEY,
                classification VARCHAR(10) CHECK (classification IN ('CAR', 'MOTORCYCLE', 'TRUCK', 'BUS')),
                brand VARCHAR(255),
                license_plate VARCHAR(50),
                color VARCHAR(50)
            );
        """
        cursor.execute(create_vehicle_table)
        
        create_camera_table = """
            CREATE TABLE IF NOT EXISTS "Camera" (
                id SERIAL PRIMARY KEY,
                location VARCHAR(255),
                status VARCHAR(10) CHECK (status IN ('IN', 'OUT'))
            );
        """
        cursor.execute(create_camera_table)
        
        create_employee_table = """
            CREATE TABLE IF NOT EXISTS "Employee" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                division_vendor VARCHAR(255),
                nip_internship VARCHAR(50),
                id_vehicle INT,
                FOREIGN KEY (id_vehicle) REFERENCES "Vehicle"(id)
            );
        """
        cursor.execute(create_employee_table)

        create_guest_table = """
            CREATE TABLE IF NOT EXISTS "Guest" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                id_vehicle INT,
                FOREIGN KEY (id_vehicle) REFERENCES "Vehicle"(id)
            );
        """
        cursor.execute(create_guest_table)
        
        create_intership_table = """
            CREATE TABLE IF NOT EXISTS "Intership" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                school_university VARCHAR(255),
                division_vendor VARCHAR(255),
                id_vehicle INT,
                FOREIGN KEY (id_vehicle) REFERENCES "Vehicle"(id)
            );
        """
        cursor.execute(create_intership_table)
        
        create_detection_table = """
            CREATE TABLE IF NOT EXISTS "Detection" (
                id SERIAL PRIMARY KEY,
                id_employee INT,
                id_guest INT,
                id_intership INT,
                image_path VARCHAR(255),
                id_camera INT,
                time TIMESTAMP,
                FOREIGN KEY (id_employee) REFERENCES "Employee"(id),
                FOREIGN KEY (id_guest) REFERENCES "Guest"(id),
                FOREIGN KEY (id_intership) REFERENCES "Intership"(id),
                FOREIGN KEY (id_camera) REFERENCES "Camera"(id)
            );
        """
        cursor.execute(create_detection_table)

        conn.commit()  # Commit the changes
        print("Tables created successfully.")
    except Exception as err:
        print(f"Error creating tables: {err}")
        conn.rollback()  # Rollback in case of error
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def reset_database():
    """Drop all tables and recreate them fresh."""
    if drop_all_tables():
        create_database_and_table()
        print("Database reset completed successfully.")
    else:
        print("Database reset failed during table dropping phase.")


if __name__ == "__main__":
    # Comment/uncomment as needed
    # drop_all_tables()  # Only drop tables
    # create_database_and_table()  # Only create tables
    reset_database()  # Drop and recreate tables
