from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16
)