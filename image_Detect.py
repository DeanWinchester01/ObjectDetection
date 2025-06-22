import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
import os
import sys
import tensorflow as tf
import pydicom
import pyperclip

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
    print(path[:-5], flush=True)
    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = 1
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((512, 512))
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # CHW format
            return img_tensor
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
        #return None
    elif path.lower().endswith((".dcm", ".dicom")):
        try:
            imageData = pydicom.dcmread(path)
            img = imageData.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.stack((img,) * 3, axis=-1)
            img = tf.image.resize(img, (512, 512)).numpy()
        except Exception as e:
            raise ValueError(f"Error loading DICOM image: {e}")
        
    else:
        raise ValueError("Unsupported image format")
    
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # NCHW
    return img_tensor

def Detect(modelPt: str, image: str) -> str:
    model.load_state_dict(torch.load(modelPt, weights_only=True))
    cancer = 0
    healthy = 0
    with torch.no_grad():
        for img in os.listdir(image):
            print(img, flush=True)
            if '\n' in img:
                img = img[:-1]
            img = loadImage(image+"\\"+img).to(device)
            output = model(img)
            prob = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(prob, dim=1).item()
            probs = prob[0][predicted_class].item()
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            if predicted_label == "healthy":
                healthy += 1
            elif predicted_label == "cancer":
                cancer += 1

    print(f"predicted: cancer {cancer} times", flush=True)
    print(f"predicted: healthy {healthy} times", flush=True)
    return cancer, healthy
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ""


while True:
    print("Waiting for input from stdin", flush=True)
    lines = sys.stdin.readline()
    if not lines:
        continue
    if(lines == ""): 
        continue
    print("Received input from stdin", flush=True)
    #os.path.dirname(os.path.abspath(__file__)) + "\\"+
    modelAddress = lines.split(",")[0]
    imageAddress = lines.split(",")[1]
    if imageAddress.endswith("\n"):
        imageAddress = lines.split(",")[1][:-1]  # Remove the trailing newline character
    #print(f"Received model address: {modelAddress}, image address: {imageAddress}", flush=True)
    modelPkl = "Models\\" + modelAddress + ".pkl"
    modelPt = "Models\\" + modelAddress + ".pt"
    #print("Model name: "+ modelAddress, flush=True)
    #print("pkl name: "+ modelPkl, flush=True)
    print("Got all data, processing image", flush=True)

    with open(modelPkl, "rb") as label:
        label_encoder = pickle.load(label)
        model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)
    print("model loaded", flush=True)
    #print("Label encoder loaded", flush=True)
    predicted_label, prob = Detect(modelPt, imageAddress)
    #print(f"Detected: Predicted label: {predicted_label}, Probability: {prob:.4f}", flush=True)
    #sys.stdin.close()
    #sys.exit(0)
#"""
"""
    print(f"Model Address: {modelAddress}, Image: {imageAddress}", flush=True)
"""
#sexyFolder = "C:\\Users\\Dean Winchester\\Desktop\\nsfw detection\\images"
#randomFolder = "C:\\Users\\Dean Winchester\\Pictures\\Screenshots"

"""
img = loadImage("C:\\Users\\Dean Winchester\\Pictures\\Screenshots\\Skärmbild 2025-02-13 025834.png").to(device)

with torch.no_grad():
    output = model(img)
    prob = torch.sigmoid(output).item()
    predicted_class = 1 if prob >= 0.5 else 0
    #_, predicted = torch.max(output.data, 1)
    print(f"Predicted class for happy-person: {prob}")
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    print(f"Image: Probability: {prob:.4f}, Predicted label: {predicted_label}")
"""


    

"""
sexyF = os.listdir(sexyFolder)
for sexy in sexyF: 
    img = loadImage(sexyFolder+"\\"+sexy).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()
        #_, predicted = torch.max(output.data, 1)
        print(f"Predicted class for sexyTest: {prob}")

randomF = os.listdir(randomFolder)
for random in randomF: 
    if not random.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = loadImage(randomFolder+"\\"+random).to(device)
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()
        predicted_class = 1 if prob >= 0.5 else 0
        #_, predicted = torch.max(output.data, 1)
        print(f"Predicted class for randomTest: {prob}")
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        print(f"Image: {random}, Probability: {prob:.4f}, Predicted label: {predicted_label}")

"""