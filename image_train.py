print("Starting image_train.py", flush=True)
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
print("Imported libraries", flush=True)

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

# Image loading function (adapted from loadImage in image_Detect.py)

def loadImage(path: str) -> torch.Tensor:
    #Load an image from a file path and convert it to a tensor.
    try:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(path).convert("RGB")
            img = img.resize((512, 512))
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).permute(2, 0, 1)  # CHW format
            return img_tensor
        elif path.lower().endswith((".dcm", ".dicom")): #dicom image stupport as scans typically output in this format
            imageData = pydicom.dcmread(path)
            img = imageData.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_tensor = torch.tensor(img).unsqueeze(0)  # Add channel dimension
            transform = transforms.Resize((512, 512))
            img_tensor = transform(img_tensor)
            img = img_tensor.repeat(3,1,1)
            return img
        else:
            raise ValueError("Unsupported image format")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


# Improved GetImageData function
def GetImageData(folders, labels):
    
    data = []
    dataLabels = []
    #iterate through each folder and give corresponding label to each image
    for i in range(len(folders)):
        folder = folders[i]
        label = labels[i]
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png",".dcm", ".dicom")):
                continue
            
            #get full path of the image
            img_path = os.path.join(folder, img_name)
            img_tensor = loadImage(img_path)
            if img_tensor is not None:
                data.append(img_tensor)
                dataLabels.append(label)
    print("got labels and data", flush=True)
    
    return data, dataLabels

# Custom Dataset class for PyTorch
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

# Training function
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Main execution
print("Starting main execution", flush=True)
while True:
    lines = sys.stdin.readline()
    if not lines:
        continue
    if(lines == ""): 
        continue
    
    path = os.path.dirname(os.path.abspath(__file__))# get the path of the current script
    address = []
    label = []

    #extract arguments from the input
    parts = lines.split(",")
    modelName = parts[0]
    args = parts[1].split("\"")
    for i in range(0, len(args)-1):
        if(lines[i] == "\n"):
            continue
        if(i%2) == 0:
            address.append(args[i])
        else:
            label.append(args[i])

    print("Received input from stdin and extracted data", flush=True)
                
    # Load data
    data, labels = GetImageData(address, label)
    print("Data and labels loaded", flush=True)
    
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
    model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)  #set number of classes to the number of unique labels
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training", flush=True)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    
    # Save model and label encoder
    saveName = path+"\\Models\\"+modelName
    print(saveName, flush=True)
    torch.save(model.state_dict(), saveName+".pt")
    with open(saveName+".pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("Model and label encoder saved.", flush=True)
    #"""