import psycopg2
from database import get_db


def create_database_and_table():
    """Create database and table if they don't exist."""
    conn = get_db()
    try:
        cursor = conn.cursor()

        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS my_data (
            id SERIAL PRIMARY KEY,
            date DATE,
            time TIME,
            track_id INT,
            class_name VARCHAR(255),
            speed FLOAT,
            numberplate TEXT
        )
        """
        cursor.execute(create_table_query)
        conn.commit()  # Commit the changes
        print("Table 'my_data' checked/created.")
    except Exception as err:
        print(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    create_database_and_table()
