# Intel-Unnati-project

## Instructions and guidelines to run the code after cloning the repository

1. Create a virtual environment. You can refer to this [link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) based on your device.
2. Install necessary libraries.
   
   ```pip install -r requirements.txt```
4. We used SQLite3 for database purposes. Hence, you can install sqlite3 on your device according to this [tutorial](https://www.tutorialspoint.com/sqlite/sqlite_installation.htm).
5. The project is setup to take two video sources, assuming the parking area cctv footage would be differnt than that of vehicle movement/road cctv footage. You can specify same source for both tasks as well. You can run your code, using terminal. However, you have to specify few arguments to run the python script using terminal.
   - First navigate to the ```code_base``` folder using ```cd code_base``` command in terminal.
   - You can use the following command to take the video source for parking area and draw polygon around the parking spaces available in the first frame of the video. You may omit giving the flag ```--parking_coordinates``` if you have already specified parking coordinates and want to use the same coordinates again.
     
     ```python main.py --parking_source path/to/video/source/ --parking_coordinates --parking_occupancy```
   - You can use the following command to take the road video source and draw a line that would mark the area where vehicle entry and exit movement would happen. Our project assumes that vehicles enter and exit from a same area and that the movement is from North to South direction or vice versa.
     
     ```python main.py --road_source path/to/video/source --road_detection```
   - You can use the following command to run both operations on two different video source.
     
     ```python main.py --parking_source path/to/source1 --parking_coordinates --parking_occupancy --road_source path/to/source2 --road_detection```
6. The project is yet not able to generate insights from the data it saves.
