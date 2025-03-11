import psycopg2
from database import get_db
from datetime import datetime


def seed_data():
    """Seed dummy data into the my_data table."""
    conn = get_db()
    try:
        cursor = conn.cursor()

        # Dummy data to insert
        dummy_data = [
            (
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                1,
                "car",
                60.5,
                "ABC123",
            ),
            (
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                2,
                "truck",
                50.0,
                "XYZ789",
            ),
            (
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                3,
                "motorcycle",
                70.0,
                "DEF456",
            ),
        ]

        # Insert dummy data
        insert_query = """
        INSERT INTO my_data (date, time, track_id, class_name, speed, numberplate)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        for data in dummy_data:
            cursor.execute(insert_query, data)
            print(f"Inserted data: {data}")

        conn.commit()  # Commit the changes

    except Exception as err:
        print(f"Error seeding data: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    seed_data()
