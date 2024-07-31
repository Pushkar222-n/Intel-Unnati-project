import sqlite3
import time
from pathlib import Path

database_dir = Path(__file__).parent.absolute()


def preprocess_license_plate(text):
    text = "".join(text.split())
    possible_confusion_dict = {
        'S': '5',
        '|': '1',
        'I': '1',
        'O': '0',
        'Z': '2',
        'B': '8',
        'G': '6',
        'Q': '0',
        'L': '1',
        'T': '7',
    }
    text_list = list(text)
    if text_list[2] in possible_confusion_dict:
        text_list[2] = possible_confusion_dict[text_list[2]]

    if text_list[3] in possible_confusion_dict:
        text_list[3] = possible_confusion_dict[text_list[3]]

    return ''.join(text_list)


def create_approved_plates_database():
    conn = sqlite3.connect(database_dir / "approved_plates.db")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS approved_plates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT,
        approved_date TEXT
    )
    ''')
    conn.commit()
    conn.close()


def create_vehicle_tracker_database():
    conn = sqlite3.connect(database_dir / "vehicle_tracker.db")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_tracker(
        id INTEGER PRIMARY KEY,           
        track_id TEXT UNIQUE,
        vehicle_class TEXT,
        status TEXT,
        seen_entry TEXT,
        seen_exit TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS license_plates(
        id INTEGER PRIMARY KEY,
        vehicle_id INTEGER,
        plate_number TEXT,
        confidence FLOAT,
        detected_at TEXT,
        FOREIGN KEY(vehicle_id) REFERENCES vehicle_tracker(id)
    )
    ''')
    conn.commit()
    conn.close()


def create_parking_database():
    conn = sqlite3.connect(database_dir / "parking_management.db")
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS parking_occupancy(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parking_id TEXT,
        vehicle_id TEXT,
        timestamp TEXT,
        is_occupied INTEGER           
    )
    ''')
    conn.commit()
    conn.close()


def update_parking_occupancy(parking_id, vehicle_id, is_occupied):
    conn = sqlite3.connect(database_dir / 'parking_management.db')
    cursor = conn.cursor()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cursor.execute('''
        SELECT id FROM parking_occupancy
        WHERE parking_id = ? AND vehicle_id = ? AND is_occupied = 1
    ''', (parking_id, vehicle_id))

    exists = cursor.fetchone()
    if exists is None and is_occupied == 1 and vehicle_id:
        cursor.execute('''
        INSERT INTO parking_occupancy(parking_id, vehicle_id, timestamp, is_occupied)
        VALUES(?, ?, ?, ?)
        ''', (parking_id, vehicle_id, current_time, int(is_occupied)))
    elif exists is not None and is_occupied == 0 and vehicle_id:
        cursor.execute('''
        UPDATE parking_occupancy
        SET is_occupied = 0, timestamp = ?
        WHERE id = ?
        ''', (current_time, exists[0]))

    conn.commit()
    conn.close()


def add_approved_plates(plates_list):
    conn = sqlite3.connect(database_dir / "approved_plates.db")
    cursor = conn.cursor()
    current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for plate in plates_list:
        plate = preprocess_license_plate(plate)
        if not is_approved_plate(plate):
            cursor.execute("""
                INSERT INTO approved_plates(plate_number, approved_date) VALUES(?, ?)
            """, (plate, current_date))
            # print("Approved License plates added to database")
        else:
            print(f"{plate} already exists in the database")

    conn.commit()
    conn.close()


def remove_approved_plate(plate_number):
    conn = sqlite3.connect(database_dir / "approved_plates.db")
    cursor = conn.cursor()

    if is_approved_plate(plate_number):
        cursor.execute("""
        DELETE FROM approved_plates WHERE plate_number = ?
        """, (plate_number,))
    else:
        print(f"{plate_number} does not exist in the database")

    conn.commit()
    conn.close()


def is_approved_plate(text):
    plate_number = ""
    if text and len(text) == 7:
        plate_number = preprocess_license_plate(text)
    conn = sqlite3.connect(database_dir / "approved_plates.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM approved_plates WHERE plate_number = ?
    """, (plate_number,))

    result = cursor.fetchone()
    conn.close()
    return result is not None


if __name__ == "__main__":
    vehicle_tracker_path = database_dir / "vehicle_tracker.db"
    approved_plates_path = database_dir / "approved_plates.db"
    parking_path = database_dir / "parking_management.db"

    print(f"Checking for parking database at: {parking_path}")
    if not parking_path.exists():
        create_parking_database()
        print("Parking database created")
    else:
        print("Parking database already exists")

    print(f"Checking for tracker database at: {vehicle_tracker_path}")
    if not vehicle_tracker_path.exists():
        create_vehicle_tracker_database()
        print("Vehicle Tracker database created")
    else:
        print("Vehicle Tracker database already exists")

    print(f"Checking for approved plates database at: {approved_plates_path}")
    if not approved_plates_path.exists():
        create_approved_plates_database()
        print("Approved Plates database created")
    else:
        plates_list = ["NAI3NRU", "MWS1VSU", "GX15OGJ", "AP05JEO", "KH05ZZK",
                       "WR02FKd", "FJ14ZHY", "EY61NBG", "AV08HVF", "BG65USJ",
                       "NA54KGJ", "NL64OGX", "LM13VCV", "HD14CDX", "AK64DMV",
                       "AK64DMV", "EF10DZT", "AF65JKV", "KH06KSU", "GJ06EPD",
                       "LN15ZZC", "DA07CLX", "EY09VWS",  "BP63LYH", "WG65ZFX",
                       "NL64OGX", "LP14LJA", "HX52BPF", "CEE61WYL"]
        add_approved_plates(plates_list)
