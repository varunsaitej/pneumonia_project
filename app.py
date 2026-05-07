from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

from models.cnn_attention import PneumoniaCNN
from yolo.detect import detect_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Device
device = torch.device("cpu")

# Load CNN model
model = PneumoniaCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template("inde.html")   # keep your filename

@app.route('/predict', methods=["POST"])
def predict():
    file = request.files["image"]

    # Save file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ---------- CNN ----------
    image = Image.open(filepath).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

    classification = "PNEUMONIA" if predicted.item() == 1 else "NORMAL"

    # ---------- YOLO ----------
    detected_path = detect_image(filepath, classification)

    # If no detection → show original image
    if detected_path is None:
        detected_path = filepath

    return render_template(
        "result.html",
        prediction=classification,
        original=filepath.replace("\\", "/"),
        detected=detected_path.replace("\\", "/")
    )

if __name__ == "__main__":
    app.run(debug=True)