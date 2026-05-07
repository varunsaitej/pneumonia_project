import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn_attention import PneumoniaCNN

# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model --------
model = PneumoniaCNN()   # your custom CNN class
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# -------- Data Transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- Load Dataset --------
test_dataset = datasets.ImageFolder("dataset/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------- Predictions --------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# -------- Confusion Matrix --------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Pneumonia"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()