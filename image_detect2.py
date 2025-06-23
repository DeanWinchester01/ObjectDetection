import torch
import torch.nn as nn

import numpy as np
import pickle
import os
import sys
import tensorflow as tf
import pydicom
import time

from PIL import Image, ImageDraw, ImageFont

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
        img = 1
        try:
            img = Image.open(path).convert("RGB")
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
        except Exception as e:
            raise ValueError(f"Error loading DICOM image: {e}")
        
    else:
        raise ValueError("Unsupported image format")
    
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # NCHW
    return img_tensor

def Detect(modelPt: str) -> str:
    images = os.listdir("testimages")
    data = []
    for image in images:
        model.load_state_dict(torch.load(modelPt, weights_only=True))

        with torch.no_grad():
            img = loadImage("testimages\\"+image).to(device)
            output = model(img)
            prob = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(prob, dim=1).item()
            probs = prob[0][predicted_class].item()
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            if predicted_label == "unknown" or predicted_label == "healthy":
                continue
            
            rounded = round(probs, 4)
            newName = predicted_label+"_"+str(rounded)+"_"+image
            Image.open("testimages\\"+image).save("testimages\\"+newName)
            
            newData = [predicted_label, probs]
            data.append(newData)

    for image in images:
        if not "_" in image:
            os.remove("testimages\\"+image)
    return data
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ""

def ScanImage(image: str):
    oldImage = Image.open(image).convert("RGB")
    iterations = 0
    startx = 0
    starty = 0
    round = 0
    for ogSize in range(256,oldImage.width, 256):
        sizeIncrease = ogSize
        iterations += 1
        for x in range(startx, oldImage.width* 1-ogSize, sizeIncrease):
            for y in range(starty, oldImage.height*1-ogSize, ogSize):
                newImage = Image.new("RGB", (sizeIncrease, sizeIncrease))
                image_position = (x, y, x+sizeIncrease, y+sizeIncrease)
                cropped_region = oldImage.crop(image_position)
                newImage.paste(cropped_region, (0,0))
                name = "testimages\\"+str(image_position)+".png"
                newImage.save(name)
                round+=1



def DisplayResults(originalImage: str, imageCoordinates: str):
    if(imageCoordinates == "originalImage.png"):
        return
    accuracy = float(imageCoordinates.split("_")[1])
    if accuracy < 0.7:
        return
    
    ogImg = Image.open(originalImage)
    if ogImg.mode != "RGB":
        ogImg = ogImg.convert("RGB")
    positions = imageCoordinates.split("_")[2][1:-5].split(",")
    x1, y1, x2, y2 = int(positions[0]), int(positions[1]), int(positions[2]), int(positions[3])

    ogImg.paste((255, 0, 0), (x1, y1, x2, y1+1), None)  # Draw rectangle
    ogImg.paste((255, 0, 0), (x1, y2, x2, y2+1), None)  # Draw rectangle
    ogImg.paste((255, 0, 0), (x1, y1, x1+1, y2), None)  # Draw rectangle
    ogImg.paste((255, 0, 0), (x2, y1, x2+1, y2), None)  # Draw rectangle
        
    label = imageCoordinates.split("_")[0]+" "+imageCoordinates.split("_")[1]+"%"
    imgfont = ImageFont.truetype("arial.ttf", 30)
    ImageDraw.Draw(ogImg).text((x1,y2), label, font=imgfont, fill=(255, 0, 0))
    
    ogImg.save(originalImage)  # Save the modified image
    ogImg.close()
    os.remove("testimages\\"+imageCoordinates)  # Remove the processed image

def CleanUp():
    # Clean up the testimages directory
    for file in os.listdir("testimages"):
        
        os.remove("testimages\\" + file)

while True:
    print("Waiting for input from stdin", flush=True)
    lines = sys.stdin.readline()
    if not lines:
        continue
    if(lines == ""): 
        continue
    print("Received input from stdin", flush=True)
    CleanUp()

    modelAddress = lines.split(",")[0]
    imageAddress = lines.split(",")[1]
    if imageAddress.endswith("\n"):
        imageAddress = lines.split(",")[1][:-1]  # Remove the trailing newline character
    modelPkl = "Models\\" + modelAddress + ".pkl"
    modelPt = "Models\\" + modelAddress + ".pt"
    print("Got all data, processing image", flush=True)
    nums = 0
    with open(modelPkl, "rb") as label:
        label_encoder = pickle.load(label)
        model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)

    print("model loaded", flush=True)
    ScanImage(imageAddress)
    data = Detect(modelPt)
    
    if len(data) == 0:
        print("Healthy", flush=True)
        continue

    # generate a new image from the original to draw the results on
    newImage = Image.open(imageAddress).save("testimages\\originalImage.png")
    for file in os.listdir("testimages"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            DisplayResults("testimages\\originalImage.png", file)
    
    time.sleep(1)  # Wait for the image to be saved before proceeding
    values = {}
    for label, prob in data:
        if label not in values:
            values[label] = prob
        else:
            values[label] += prob

    for label in values:
        values[label] /= len(data)

    returnString = ""
    for label, prob in values.items():
        returnString += f"{label} {prob:.4f}:"
    print(f"Detected:"+returnString, flush=True)