import os
import shutil

source_path = r"C:\Users\vs736\Downloads\pneu\dataset"
target_path = r"C:\Users\vs736\Downloads\pneu\yolo_dataset"

for split in ["train", "val"]:
    os.makedirs(os.path.join(target_path, "images", split), exist_ok=True)

def copy_images(split):
    for category in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(source_path, split, category)

        if not os.path.exists(folder):
            print("Missing:", folder)
            continue

        for file in os.listdir(folder):
            src = os.path.join(folder, file)
            dst = os.path.join(target_path, "images", split, file)

            shutil.copy(src, dst)

copy_images("train")
copy_images("val")

print("Images copied successfully")