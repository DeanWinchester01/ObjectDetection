print("Starting image detection script", flush=True)
import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
import sys
import tensorflow as tf
import pydicom
import os
print("Imported libraries", flush=True)

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
    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((512, 512))
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # CHW format
            return img_tensor
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    elif path.lower().endswith((".dcm", ".dicom")):
        try:
            imageData = pydicom.dcmread(path)
            img = imageData.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.stack((img,) * 3, axis=-1)
            img = tf.image.resize(img, (512, 512)).numpy()
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # NCHW
            return img_tensor
        except Exception as e:
            raise ValueError(f"Error loading DICOM image: {e}")
        
    else:
        raise ValueError("Unsupported image format")
    

def Detect(modelPt: str, image: str) -> str:
    #model.load_state_dict(torch.load(modelPt, weights_only=True))
    with torch.no_grad():
        if '\n' in image:
            image = image[:-1]
        image = loadImage(image).to(device)
        output = model(image)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(prob, dim=1).item()
        probs = prob[0][predicted_class].item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label, probs
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ""


print("Waiting for input from stdin", flush=True)
while True:
    lines = sys.stdin.readline()
    if not lines:
        continue
    if(lines == ""): 
        continue
    print("Received input from stdin", flush=True)
    modelAddress = lines.split(",")[0]
    imageAddress = lines.split(",")[1]

    if imageAddress.endswith("\n"):
        imageAddress = lines.split(",")[1][:-1]  # Remove the trailing newline character

    modelPkl = "Models\\" + modelAddress + ".pkl"
    modelPt = "Models\\" + modelAddress + ".pt"
    print("Got all data, processing image", flush=True)

    with open(modelPkl, "rb") as label:
        label_encoder = pickle.load(label)
        model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)
    print("model loaded", flush=True)
        
    predicted_label, prob = Detect(modelPt, imageAddress)
    print(f"Detected: Predicted label: {predicted_label}, Probability: {prob:.4f}", flush=True)