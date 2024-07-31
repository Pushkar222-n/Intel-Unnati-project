import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

database_dir = Path(__file__).parent.absolute() / "database"

results_path = Path(__file__).parent.absolute() / "results"

parking_results_path = results_path / "parking"
road_results_path = results_path / "road"


def ensure_dir_exists(path):
    path.mkdir(parents=True, exist_ok=True)


ensure_dir_exists(results_path)
ensure_dir_exists(parking_results_path)
ensure_dir_exists(road_results_path)


def generate_parking_insights():
    conn = sqlite3.connect(database_dir / "parking_management.db")

    # For plotting occupancy over time
    occupancy_data = pd.read_sql_query('''
        SELECT parking_id, timestamp, is_occupied
        FROM parking_occupancy
        ORDER BY timestamp
    ''', conn)
    occupancy_data['timestamp'] = pd.to_datetime(occupancy_data['timestamp'])

    plt.figure(figsize=(15, 6))
    for parking_id in occupancy_data['parking_id'].unique():
        parking_data = occupancy_data[occupancy_data['parking_id'] == parking_id]
        plt.plot(parking_data['timestamp'], parking_data['is_occupied'], label=f'Parking{
                 parking_id}')

    plt.title('Parking Occupancy Over Time')
    plt.xlabel('Time')
    plt.ylabel('Occupied (1) / Vacant (0)')
    plt.legend()
    plt.savefig(parking_results_path / "time_occupancy_plot.jpg")
    plt.show()

    # For plotting frequently occupied parking lots
    frequently_occupied = pd.read_sql_query('''
        SELECT parking_id, AVG(is_occupied) as occupancy_rate
        FROM parking_occupancy
        GROUP BY parking_id
        ORDER BY occupancy_rate DESC
    ''', conn)
    frequently_occupied.plot(kind='bar', x='parking_id', y='occupancy_rate')
    plt.title('Occupancy Rate by Parking Lot')
    plt.xlabel('Parking ID')
    plt.ylabel('Occupancy Rate')
    plt.savefig(parking_results_path / "frequent_parking_plot.jpg")
    plt.show()

    # for plotting peak occupancy times
    occupancy_data["second"] = occupancy_data["timestamp"].dt.second
    peak_times = occupancy_data.groupby(
        'second')['is_occupied'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    peak_times.plot(kind='bar')
    plt.title('Average Occupancy by seconds')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Occupancy Rate')
    plt.show()

    conn.close()


def generate_road_insights():
    pass
