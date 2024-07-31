import cv2
import numpy as np
from utils import get_direction, is_crossed_line, draw_text
from database.database_utils import is_approved_plate, preprocess_license_plate, update_parking_occupancy


def draw_vehicle(frame, x1, y1, x2, y2, vehicle_id, approved, text=""):
    color = (57, 255, 20) if approved else (203, 192, 255)
    text_prefix = "Approved: " if approved else "Not Approved"
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    draw_text(frame, f"{text_prefix}{text}", cv2.FONT_HERSHEY_SIMPLEX, (int(
        x1) - 10, int(y1 - 10)), 0.5, 2, (0, 0, 0), color)


def draw_license_plate(frame, x1, y1, x2, y2, text, approved):
    color = (57, 255, 20) if approved else (0, 0, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, f"{text}", (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_parking_inference(frame, coordinates, license_detect, vehicle_detect):
    """
    Perform inference on the parking video source
    """
    frame = cv2.resize(frame, (1280, 720))
    results = vehicle_detect.get_vehicle_predictions(source=frame)
    results_data = np.array([box.data.cpu().numpy()
                             for vehicle in results for box in vehicle.boxes]).squeeze()
    cars_in_parking = dict()
    for result in results_data:

        if not result.size == 0 and result.size == 7:
            x1, y1, x2, y2, vehicle_id, conf, class_id = result

            bbox_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            cars_in_parking[vehicle_id] = bbox_center
    for id, polyline in coordinates.items():
        # Check vehicles and get parking occupancy
        for vehicle_id, center in cars_in_parking.items():
            vehicle_id = int(vehicle_id)
            is_car_in_parking = cv2.pointPolygonTest(contour=polyline[0],
                                                     pt=(center),
                                                     measureDist=False)
            if is_car_in_parking >= 0:
                update_parking_occupancy(id, vehicle_id, True)
                cv2.polylines(img=frame,
                              pts=polyline,
                              isClosed=True,
                              color=(0, 255, 0),
                              thickness=2)
                cv2.circle(img=frame,
                           center=center,
                           radius=5,
                           color=(0, 255, 0),
                           thickness=-1)
                cv2.putText(frame,
                            text=f"{id}",
                            org=(polyline[0][0][0], polyline[0][0][1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=2)
            else:
                cv2.putText(frame,
                            text=f"{id}",
                            org=(polyline[0][0][0], polyline[0][0][1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255),
                            thickness=2)
                cv2.polylines(img=frame,
                              pts=polyline,
                              isClosed=True,
                              color=(0, 0, 255),
                              thickness=2)
                cv2.circle(img=frame,
                           center=center,
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
                update_parking_occupancy(id, vehicle_id, False)

    return frame


def run_road_inference(frame, line_coordinates, license_detect, vehicle_detect, vehicle_centers, vehicles_count, vehicle_license_plate_status):
    """
    Perform inference on the road video source
    """
    frame = cv2.resize(frame, (1280, 720))
    cv2.line(frame, line_coordinates[0],
             line_coordinates[1], (192, 192, 192), 2)

    # Getting predictions from the vehicle detection model
    vehicles = vehicle_detect.get_vehicle_predictions(source=frame)
    vehicles_data = np.array([box.data.cpu().numpy()
                             for vehicle in vehicles for box in vehicle.boxes]).squeeze()

    vehicles_index = [2, 3, 5, 7]

    vehicle_detections = []
    for info in vehicles_data:
        if len(info) == 7:
            x1, y1, x2, y2, vehicle_id, conf, class_id = info
            if int(class_id) in vehicles_index:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                vehicle_id = int(vehicle_id)
                vehicle_detections.append([x1, y1, x2, y2, vehicle_id])

                if vehicle_id not in vehicle_license_plate_status:
                    vehicle_license_plate_status[vehicle_id] = {
                        "status": False,
                        "text": ""}

                approved = vehicle_license_plate_status[vehicle_id]["status"]
                draw_vehicle(frame, x1, y1, x2, y2, vehicle_id, approved,
                             vehicle_license_plate_status[vehicle_id]["text"])

                # Object in and out counter
                if vehicle_id not in vehicle_centers:
                    vehicle_centers[vehicle_id] = [center]
                else:
                    vehicle_centers[vehicle_id].append(center)
                    # Only last two points are sufficient to get directions
                    vehicle_centers[vehicle_id] = vehicle_centers[vehicle_id][-2:]

                    if len(vehicle_centers[vehicle_id]) == 2:
                        prev_center, current_center = vehicle_centers[vehicle_id]
                        if is_crossed_line(line_coordinates, prev_center, current_center):
                            direction = get_direction(
                                prev_center, current_center)
                            if direction == "in" and vehicle_id not in vehicles_count["in"]:
                                vehicles_count["in"].append(vehicle_id)
                            elif direction == "out" and vehicle_id not in vehicles_count["out"]:
                                vehicles_count["out"].append(vehicle_id)

    # Getting predicitons from the license plate detection model
    license_plate_data = []
    license_plates = license_detect.get_license_predictions(source=frame)
    for license_plate in license_plates:
        license_plate_data.extend(license_plate.boxes.data.cpu().numpy())

    for data in license_plate_data:
        x1, y1, x2, y2, conf, class_id = data
        x_car1, y_car1, x_car2, y_car2, vehicle_id = license_detect.get_car_with_license_plate(
            data, vehicle_detections)

        plate_status = False
        if vehicle_id not in vehicle_license_plate_status:
            vehicle_license_plate_status[vehicle_id] = {"status": plate_status}
        if vehicle_id != -1:
            if vehicle_license_plate_status[vehicle_id]["status"] == True:
                text = vehicle_license_plate_status[vehicle_id]["text"]
                # draw_vehicle(frame, x_car1, y_car1, x_car2,
                #              y_car2, vehicle_id, True, text)
                continue

            draw_vehicle(frame, x_car1, y_car1, x_car2,
                         y_car2, vehicle_id, False, "")
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_text = license_detect.recognize_characters(
                license_plate_crop)
            if license_plate_text:
                draw_license_plate(frame, x1, y1, x2, y2,
                                   license_plate_text, False)
                # cv2.putText(frame, f"{license_plate_text}", (int(x1), int(
                #     y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if is_approved_plate(license_plate_text) == True:
                plate_status = True
                vehicle_license_plate_status[vehicle_id] = {
                    "status": plate_status,
                    "text": preprocess_license_plate(license_plate_text),
                }
            else:
                vehicle_license_plate_status[vehicle_id] = {
                    "status": plate_status,
                    "text": ""
                }

    draw_text(frame, f"Vehicles In: {len(vehicles_count['in'])}", cv2.FONT_HERSHEY_DUPLEX,
              pos=(10, 20), font_scale=0.9, text_color=(0, 0, 0), text_color_bg=(228, 136, 235))
    draw_text(frame, f"Vehicles Out: {len(vehicles_count['out'])}", cv2.FONT_HERSHEY_DUPLEX,
              pos=(frame.shape[0] - 100, 20), font_scale=0.9, text_color=(0, 0, 0), text_color_bg=(228, 136, 235))

    return frame, vehicle_centers, vehicles_count, vehicle_license_plate_status
