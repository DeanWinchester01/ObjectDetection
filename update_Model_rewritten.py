
import os
import sys
import pydicom
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder

print("Starting model update script", flush=True)

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def loadImage(path: str) -> torch.Tensor:
    try:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(path).convert("RGB").resize((512, 512))
            img = np.array(img).astype(np.float32) / 255.0
            return torch.tensor(img).permute(2, 0, 1)
        elif path.lower().endswith((".dcm", ".dicom")):
            imageData = pydicom.dcmread(path)
            img = imageData.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_tensor = torch.tensor(img).unsqueeze(0)
            img_tensor = transforms.Resize((512, 512))(img_tensor)
            return img_tensor.repeat(3, 1, 1)
        else:
            raise ValueError("Unsupported image format")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def GetImageData(folders, labels):
    data, dataLabels = [], []
    for folder, label in zip(folders, labels):
        for img_name in os.listdir(folder):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".dcm", ".dicom")):
                img_path = os.path.join(folder, img_name)
                img_tensor = loadImage(img_path)
                if img_tensor is not None:
                    data.append(img_tensor)
                    dataLabels.append(label)
    return data, dataLabels

class ImageDataset(Dataset):
    def __init__(self, data, labels, label_encoder):
        self.data = data
        self.labels = label_encoder.transform(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}", flush=True)

# Main process
while True:
    lines = sys.stdin.readline()
    if not lines.strip():
        continue

    parts = lines.strip().split('"')
    oldModel = parts[0].strip()
    folders = parts[1::2]
    labels = parts[2::2]

    print("Updating model:", oldModel, flush=True)
    print("Folders:", folders, flush=True)
    print("Labels:", labels, flush=True)

    # Load new training data
    data, dataLabels = GetImageData(folders, labels)
    print(f"Loaded {len(data)} images", flush=True)

    # Load previous label encoder
    modelPkl = os.path.join("Models", oldModel + ".pkl")
    modelPt = os.path.join("Models", oldModel + ".pt")

    with open(modelPkl, "rb") as f:
        old_label_encoder = pickle.load(f)

    # Merge new labels
    combined_classes = sorted(set(old_label_encoder.classes_).union(dataLabels))
    full_label_encoder = LabelEncoder()
    full_label_encoder.fit(combined_classes)

    # Save updated encoder
    with open(modelPkl, "wb") as f:
        pickle.dump(full_label_encoder, f)

    # Convert labels to indices
    dataset = ImageDataset(data, dataLabels, full_label_encoder)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Load old model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=len(full_label_encoder.classes_)).to(device)
    old_state = torch.load(modelPt, map_location=device)

    filtered_old_state = {
        k: v for k, v in old_state.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(filtered_old_state, strict=False)

    # Train with new data
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Save updated model
    torch.save(model.state_dict(), modelPt)
    print("Model and labels updated successfully.", flush=True)
