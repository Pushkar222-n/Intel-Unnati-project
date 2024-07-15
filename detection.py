from ultralytics import YOLO
# import easyocr
from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
from optimization import openvino_optimize


def get_optimized_model(model_path):
    return openvino_optimize(model_path)


class Vehicle_MOT:
    def __init__(self, path_to_vehicle_model, conf=0.5, verbose=False, stream=True, tracker=False, tracker_type=None):
        self.path_to_vehicle_model = path_to_vehicle_model
        self.optimized_model = get_optimized_model(self.path_to_vehicle_model)
        self.conf = conf
        self.verbose = verbose
        self.stream = stream
        self.tracker = tracker
        self.tracker_type = tracker_type

    def get_vehicle_predictions(self, source):
        results = None
        if not self.tracker:
            results = self.optimized_model(source=source,
                                           conf=self.conf,
                                           verbose=self.verbose,
                                           stream=self.stream)
        else:
            results = self.optimized_model.track(source=source,
                                                 tracker=self.tracker_type,
                                                 conf=self.conf,
                                                 verbose=self.verbose,
                                                 stream=self.stream,
                                                 persist=True)
        return results


class LicensePlateDetection:
    def __init__(self, path_to_license_model, conf, verbose=False, stream=True):
        self.path_to_license_model = path_to_license_model
        self.optimized_model = get_optimized_model(path_to_license_model)
        self.conf = conf
        self.verbose = verbose
        self.stream = stream
        self.license_conf = 0

    def get_license_predictions(self, source):
        results = self.optimized_model(source=source,
                                       conf=self.conf,
                                       verbose=self.verbose,
                                       stream=self.stream)
        return results

    def get_car_with_license_plate(self, license_plate, vehicle_detections):
        x1, y1, x2, y2, license_conf, class_id = license_plate
        found_licensePlate = False
        car_index = -1
        for i in range(len(vehicle_detections)):
            x1_car, y1_car, x2_car, y2_car, vehicle_id = vehicle_detections[i]
            if x1 > x1_car and y1 > y1_car and x2 < x2_car and y2 < y2_car:
                car_index = i
                found_licensePlate = True
                break

        if found_licensePlate:
            return vehicle_detections[car_index]
        return -1, -1, -1, -1, -1

    def recognize_characters(self, image):
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray Image", gray_image)
        # bilateral = cv2.bilateralFilter(image, 15, 75, 75)

        # resized = cv2.resize(bi, dsize=(0, 0), fx=2, fy=2,
        #                      interpolation=cv2.INTER_CUBIC)
        # image2 = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=50)

        # cv2.imshow("Resized Image", gray_image)
        # cv2.imshow("Thresholded Image", gray_image)
        # reader = easyocr.Reader(["en"])
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=50)
        ocr = PaddleOCR(use_angle_cls=True, lang='en',
                        gpu=False, show_log=False)
        result = ocr.ocr(image, cls=True)
        license_plate_text = ""
        if result[0] is not None:
            # license_plate_text = "".join([text for text in result])
            license_plate_text = result[0][0][1][0]
        # license_plate_text = ""
        # license_plate_result = reader.readtext(gray_image, workers=8)
        # if len(license_plate_result) > 0:
        #     license_plate_text = license_plate_result[0][-2]
        #     print(license_plate_text)
        return license_plate_text
