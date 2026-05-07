import os
import cv2
from ultralytics import YOLO

model_path = os.path.join(os.path.dirname(__file__), "best.pt")

model = YOLO(model_path)

def detect_image(image_path, cnn_result):
    print("Running YOLO on:", image_path)

    results = model(image_path, conf=0.01)

    output_path = "static/uploads/detected.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if cnn_result == "NORMAL":
        print("CNN says NORMAL → Skipping YOLO boxes")

        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)

    else:
        print("CNN says PNEUMONIA → Showing YOLO boxes")

        annotated_frame = results[0].plot(labels=True, conf=True)
        cv2.imwrite(output_path, annotated_frame)

    print("Saved at:", output_path)
    return output_path