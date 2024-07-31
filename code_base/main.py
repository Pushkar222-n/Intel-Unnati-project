import argparse
import cv2
from pathlib import Path
import pickle
from utils import parking_draw, get_parking_coordinates, get_line_coordinates
from detection import Vehicle_MOT, LicensePlateDetection
from inference import run_parking_inference, run_road_inference
from insights_utils import generate_parking_insights, generate_road_insights
from multiprocessing import Pool
import multiprocessing
import time

parser = argparse.ArgumentParser()
parser.add_argument("--parking_source",
                    type=str,
                    default="parking1.mp4",
                    dest="source1",
                    help="Path to the parking video source")
parser.add_argument("--road_source",
                    type=str,
                    default="CarVideo.mp4",
                    dest="source2",
                    help="Path to the road video source")
parser.add_argument("-pc", "--parking_coordinates", action="store_true",
                    dest="parking_coordinates",
                    help="Get parking coordinates from the video source")

parser.add_argument("-po", "--parking_occupancy", action="store_true",
                    dest="parking",
                    help="Perform inference for parking occupancy")
parser.add_argument("-rd", "--road_detection", action="store_true",
                    dest="road",
                    help="Perform inference on moving vehicles")

args = parser.parse_args()

source_dir = Path(__file__).parent.absolute()
# source_dir = Path(source_dir)
print(source_dir)

path_to_parking_source = source_dir / args.source1
path_to_road_source = source_dir / args.source2
path_to_license_model = source_dir / "v2_license_plate_model.pt"
path_to_vehicle_model = source_dir / "yolov8n.pt"
path_to_parking_coordinates = source_dir / "parking_coordinates.pkl"

path_to_output_parking = source_dir / "parking_output.mp4"
path_to_road_video = source_dir / "road_output.mp4"

print("hello world")


def time_taken(start, end):
    time_format = time.strftime("%H:%M:%S", time.gmtime(end - start))
    return time_format


def vehicle_management(params):
    (parking_source, road_source, output_parking, output_road,
     parking_coordinates, parking, road) = params
    if parking_coordinates:
        spaces = get_parking_coordinates(parking_source)
        with open(path_to_parking_coordinates, "wb") as fp:
            pickle.dump(spaces, fp)
            print("Parking coordinates saved successfully")

    with open(path_to_parking_coordinates, 'rb') as fp:
        coordinates = pickle.load(fp)

    license_detect = LicensePlateDetection(
        path_to_license_model, conf=0.6, verbose=False, stream=True,)
    vehicle_detect = Vehicle_MOT(
        path_to_vehicle_model, conf=0.65, verbose=False, stream=True, tracker=True,
        tracker_type="bytetrack.yaml")

    cap1, cap2, out1, out2 = None, None, None, None
    if parking:
        cap1 = cv2.VideoCapture(parking_source)
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        frame_width1, frame_height1 = (1280, 720)
        count1 = 0
        out1 = cv2.VideoWriter(output_parking,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps1,
                               (frame_width1, frame_height1))
        while True:
            success1, frame1 = cap1.read()
            if not success1:
                break
            frame1 = cv2.resize(frame1, (1280, 720))
            frame1 = run_parking_inference(
                frame1, coordinates, license_detect, vehicle_detect)
            out1.write(frame1)
            count1 += 1
            print(f"Parking Frame {count1} processed...\n")
            cv2.imshow("Vehicle Management", frame1)
            key = cv2.waitKey(1)
            if key == 27:
                break

    if road:
        line_coordinates = get_line_coordinates(road_source)
        cap2 = cv2.VideoCapture(road_source)
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        # frame_width2, frame_height2 = (
        #     cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width2, frame_height2 = (1280, 720)
        out2 = cv2.VideoWriter(output_road,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps2,
                               (frame_width2, frame_height2))
        vehicle_centers = {}
        vehicles_count = {"in": [],
                          "out": []}
        count2 = 0
        vehicle_license_plate_status = dict()

        # Warm up
        for i in range(5):
            _, f = cap2.read()

        while True:
            frame2 = None
            success2, frame2 = cap2.read()
            if not success2:
                break
            frame2 = cv2.resize(frame2, (1280, 720))
            # if count2 % 2 == 0:
            frame2, vehicle_centers, vehicles_count, vehicle_license_plate_status = run_road_inference(
                frame2, line_coordinates=line_coordinates,
                license_detect=license_detect, vehicle_detect=vehicle_detect,
                vehicle_centers=vehicle_centers, vehicles_count=vehicles_count,
                vehicle_license_plate_status=vehicle_license_plate_status)
            out2.write(frame2)
            count2 += 1
            print(f"Road Frame {count2} processed...\n")

            cv2.imshow("Road Management", frame2)
            key = cv2.waitKey(1)
            if key == 27:
                break
    cv2.destroyAllWindows()
    if cap1:
        cap1.release()
        out1.release()
    if cap2:
        cap2.release()
        out2.release()


def generate_insights(parking, road):
    if parking:
        generate_parking_insights()
    if road:
        generate_road_insights()


if __name__ == "__main__":
    start1 = time.time()
    # vehicle_management(parking_source=path_to_parking_source,
    #                    road_source=path_to_road_source,
    #                    output_parking=path_to_output_parking,
    #                    output_road=path_to_road_video,
    #                    parking_coordinates=args.parking_coordinates,
    #                    parking=args.parking,
    #                    road=args.road)
    parameters1 = [(path_to_parking_source,
                   path_to_road_source,
                   path_to_output_parking,
                   path_to_road_video,
                   args.parking_coordinates,
                   args.parking, args.road)]
    parameters2 = [(args.parking, args.road)]
    with Pool(multiprocessing.cpu_count()) as p:
        p.map(vehicle_management, parameters1)
    end1 = time.time()
    print(f"Video process successful \nTime taken {time_taken(start1, end1)}")

    start2 = time.time()
    with Pool(multiprocessing.cpu_count()) as p2:
        p2.starmap(generate_insights, parameters2)
    end2 = time.time()
    print(f"Insights generation successful \nTime taken {
          time_taken(start2, end2)}")

    print("\n\n********EXPERIMENT SUCCESSFUL*********")
