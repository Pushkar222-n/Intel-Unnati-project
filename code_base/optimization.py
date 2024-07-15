from ultralytics import YOLO
from pathlib import Path
import cv2


def openvino_export(model_path):
    model = YOLO(model_path, verbose=False)
    model_name = model_path.with_suffix('')
    print(model_name)
    model.export(format='openvino', half=True)


def openvino_optimize(model_path):
    model_name = model_path.with_suffix('')
    if Path(f'{model_name}_openvino_model/').exists():
        ov_model = YOLO(f'{model_name}_openvino_model/', verbose=False)
        return ov_model
    else:
        openvino_export(model_path)
        ov_model = YOLO(f'{model_name}_openvino_model/', verbose=False)
        return ov_model


if __name__ == '__main__':
    model_path = r"/home/arthur_canon/Documents/Vehicle_Management/Vehicle_Management/Pushkar_try/yolov8n.pt"
    original_model = YOLO(model_path)
    optimized_model = openvino_optimize(
        model_path, precision='FP16', device='CPU')

    image = cv2.imread(
        r"/home/arthur_canon/Documents/Vehicle_Management/Vehicle_Management/Pushkar_try/Body-Type-Header.jpg")
    results_1 = original_model(image, stream=True)
    results_2 = optimized_model(image, stream=True)

    print("original_model_results")
    for result in results_1:
        print(result.boxes.data.tolist())
    print("\n\n\n Openvino model results")
    for result_2 in results_2:
        print(result_2.boxes.data.tolist())
