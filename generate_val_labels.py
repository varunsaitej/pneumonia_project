import os

image_folder = "yolo_dataset/images/val"
label_folder = "yolo_dataset/labels/val"

os.makedirs(label_folder, exist_ok=True)

for img in os.listdir(image_folder):
    if img.endswith(".jpeg") or img.endswith(".jpg"):
        label_file = os.path.join(label_folder, img.replace(".jpeg", ".txt").replace(".jpg", ".txt"))

        with open(label_file, "w") as f:
            f.write("0 0.5 0.5 1 1")

print("VAL labels created")