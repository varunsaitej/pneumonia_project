import os
import cv2
import torch
import numpy as np

from models.cnn_attention import PneumoniaCNN

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

# PATHS
image_folder = "yolo_dataset/images/train"
label_folder = "yolo_dataset/labels/train"

os.makedirs(label_folder, exist_ok=True)

# FUNCTION TO GENERATE BOXES
def generate_boxes(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.conv3(model.conv2(model.conv1(img)))  # get feature map

    heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    # THRESHOLD
    _, thresh = cv2.threshold((heatmap * 255).astype(np.uint8), 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filter very small boxes
        if w * h > 200:
            boxes.append((x, y, w, h))

    return boxes


# SAVE YOLO FORMAT
def save_label(img_name, boxes):
    h, w = 224, 224

    label_path = os.path.join(label_folder, img_name.replace(".jpeg", ".txt"))

    lines = []
    for (x, y, bw, bh) in boxes:
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h

        lines.append(f"0 {xc} {yc} {bw} {bh}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


# LOOP THROUGH IMAGES
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    boxes = generate_boxes(image)

    save_label(img_name, boxes)

print("STEP 2 DONE")