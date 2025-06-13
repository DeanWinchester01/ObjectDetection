print("Starting updating", flush=True)
import os
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
import sys

# Define CNN Model (same as in image_Detect.py)
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
#Load an image from a file path and convert it to a tensor.
    try:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(path).convert("RGB")
            img = img.resize((512, 512))
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).permute(2, 0, 1)  # CHW format
            return img_tensor
        elif( path.lower().endswith((".dcm", ".dicom"))):
            imageData = pydicom.dcmread(path)
            img = imageData.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_tensor = torch.tensor(img).unsqueeze(0)  # Add channel dimension
            transform = transforms.Resize((512, 512))
            img_tensor = transform(img_tensor)
            img = img_tensor.repeat(3,1,1)
            """img = np.stack((img,) * 3, axis=-1)
            img = img.resize((512, 512))  # Resize to 512x512
            img = np.array(img).astype(np.float32) / 255.0
            #img = tf.image.resize(img, (512, 512)).numpy()
            """
            return img
        else:
            raise ValueError("Unsupported image format")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    
def GetImageData(folders, labels):
    
    data = []
    dataLabels = []
    print(folders, flush=True)
    for i in range(len(folders)):
        folder = folders[i]
        label = labels[i]
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png",".dcm", ".dicom")):
                continue
            img_path = os.path.join(folder, img_name)
            img_tensor = loadImage(img_path)
            if img_tensor is not None:
                data.append(img_tensor)
                dataLabels.append(label)
    print("got labels and data", flush=True)
    
    return data, dataLabels

class ImageDataset(Dataset):
    def __init__(self, data, labels, label_encoder):
        self.data = data
        self.labels = labels
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        label_idx = self.label_encoder.transform([label])[0]
        return image, label_idx
    
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())  # BCEWithLogitsLoss expects float
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}", flush=True)

print("Starting update process", flush=True)
while True:
    lines = sys.stdin.readline()
    if not lines:
        continue
    if lines == "":
        continue
    path = os.path.dirname(os.path.abspath(__file__))
    address = []
    label = []
    #with open(path+"\\"+"Data.txt", "r") as data:
    parts = lines.split("\"")
    oldModel = parts[0]
    #parts.remove(modelName)
    parts.remove(oldModel)

    for i in range(0, len(parts)-1):
        if(lines[i] == "\n"):
            continue
        if(i%2) == 0:
            address.append(parts[i])
        else:
            label.append(parts[i])

    print("Model Name:", oldModel, flush=True)
    print("Addresses:\n", address, flush=True)
    print("Labels:\n", label, flush=True)

    data, labels = GetImageData(address, label)

    # Encode labels
    print("Encoding labels", flush=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)  # Fit on 
    print("encoded labels", flush=True)

    # Create dataset and dataloader
    dataset = ImageDataset(data, labels, label_encoder)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("Dataset and DataLoader created", flush=True)
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)  # num_classes=1 for binary classification with BCEWithLogitsLoss
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid and binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training", flush=True)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    print("Training complete", flush=True)

    #saveName = "Models\\temp.pt"
    #//torch.save(model.state_dict(), saveName)
    modelPkl = "Models\\" + oldModel + ".pkl"
    modelPt = "Models\\" + oldModel + ".pt"
    #print("Model name: "+ modelAddress, flush=True)
    #print("pkl name: "+ modelPkl, flush=True)
    print("Got all data, processing image", flush=True)

    with open(modelPkl, "rb") as label:
        old_label_encoder = pickle.load(label)

    newLabelEncoder = LabelEncoder()
    newLabelEncoder.fit(label_encoder.classes_ + old_label_encoder.classes_)  # Fit the new label encoder with the new classes
    #combinedLabels = old_label_encoder.classes_.tolist() + label_encoder.classes_.tolist()
    
    with open(modelPkl, "wb") as f:
        pickle.dump(newLabelEncoder, f)
    
    oldmodel = CNNModel(num_classes=len(old_label_encoder.classes_)+len(label_encoder.classes_)).to(device)
    old_state = torch.load(modelPt, map_location=device)
    oldmodel.load_state_dict(old_state)  # Load the old model state_dict

    new_state = model.state_dict()  # Get the new model state_dict
    merged_state = oldmodel.state_dict()

    merged_state.update(new_state)  # Update the old model with the new model's state_dict
    oldmodel.load_state_dict(merged_state)  # Load the merged state_dict into the old model
    torch.save(oldmodel.state_dict(), modelPt)  # Save the updated model state_dict
    """
    oldmodel.load_state_dict(torch.load(modelPt, weights_only=False))
    oldmodel.state_dict().update(model.state_dict())  # Update the old model with the new model's state_dict
    #oldModel.state_dict().update(model.state_dict())  # Update the old model with the new model's state_dict
    print(oldmodel, flush=True)
    """
    """
    oldLabelEncoder = "Models\\" + oldModel + ".pkl"
    oldModelPt = "Models\\" + oldModel + ".pt"
    newEncoder = ""
    with open(oldLabelEncoder, "rb") as f:
        labelEncoder = pickle.load(f)
        newEncoder = LabelEncoder()
        newEncoder.fit(labelEncoder.classes_+ label_encoder.classes_)  # Fit the new label encoder with the old classes
        #label_encoder.fit(labelEncoder.classes_)  # Ensure the label encoder is fitted with the same classes
        oldModel = CNNModel(num_classes=len(labelEncoder.classes_)).to(device)

    oldModel.load_state_dict(torch.load(oldModelPt, map_location=device, weights_only=False))
    #newModel = oldModel.state_dict() | (model.state_dict())
    oldModel.state_dict().update(model.state_dict())  # Update the old model with the new model's state_dict
    print(oldModel)
    #newModel = oldModel.state_dict() + model.state_dict()

    torch.save(oldModel, oldModelPt)
    with open(oldLabelEncoder, "wb") as f:
        pickle.dump(newEncoder, f)
    print("Model and label encoder saved", flush=True)
    #"""