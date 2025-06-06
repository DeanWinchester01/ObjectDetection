import torch
import torch.nn as nn
#"""
import numpy as np
import pickle
import os
import sys
import tensorflow as tf
import pydicom
import pyperclip
#"""

from PIL import Image

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
    
def loadImage2(image: Image.Image) -> torch.Tensor:
    # Convert a PIL Image to a tensor
    img = image.convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # CHW format
    return img_tensor

def loadDicomImage(path: str) -> torch.Tensor:
    # Load a DICOM image and convert it to a tensor
    try:
        imageData = pydicom.dcmread(path)
        img = imageData.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.stack((img,) * 3, axis=-1)  # Convert to RGB
        img = tf.image.resize(img, (512, 512)).numpy()
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # NCHW format
        return img_tensor
    except Exception as e:
        raise ValueError(f"Error loading DICOM image: {e}")
def loadImage(path: str) -> torch.Tensor:
    
    #print(path[:-1], flush=True)
    #format = path.lower().endswith((".jpg", ".jpeg", ".png"))
    #format = path.lower().endswith((".jpg", ".jpeg", ".png"))
    #print(f"Loading image from {path}", flush=True)
    #print("image ends with correct format = " +str(format) , flush=True)
    #print(path.replace(" ",":"), flush=True)
    print(path, flush=True)
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
        """
        img = Image.open(path).convert("RGB")
        img = img.resize((512,512))
        img = np.array(img).astype(np.float32) / 255.0
        """
        
    else:
        raise ValueError("Unsupported image format")
    
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # NCHW
    return img_tensor

def Detect(modelPt: str) -> str:
    images = os.listdir("testimages")
    data = []
    for image in images:
        #print("Detecting image", flush=True)
        #print(f"Model address: {modelAddress}, Image address: {image}", flush=True)
        #print("Model loaded", flush=True)
        #print("Model loaded", flush=True)
        model.load_state_dict(torch.load(modelPt, weights_only=True))
        #print(f"Image size: {x}x{y}", flush=True)
        #intSize = i.read()
        #print(str(i), flush=True)

        with torch.no_grad():
            #print(f"Loading image {image}", flush=True)
            #print(f"Loading image: {image}", flush=True)
            img = loadImage("testimages\\"+image).to(device)
            #print("Image loaded", flush=True)
            output = model(img)
            #print("Model output computed", flush=True)
            prob = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(prob, dim=1).item()
            """
            prob = torch.sigmoid(output).item()
            predicted_class = 1 if prob >= 0.5 else 0
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            """
            probs = prob[0][predicted_class].item()
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            print(predicted_class, flush=True)
            print(prob, flush=True)
            if predicted_label == "unknown":
                continue
            newData = [predicted_label, probs]
            data.append(newData)
    return data
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ""
#model.load_state_dict(torch.load("image_classification_model.pt"))

#print("Model name: "+ modelAddress, flush=True)
#print("pkl name: "+ modelPkl, flush=True)
#print(f"Received image: {image}")

def GenerateImage(size, colors) -> Image:
    # Generate a new image with the specified size and color
    new_image = Image.new("RGB", (size,size), "white")  # Default to white background

    # Fill the image with the specified color
            #print(colors[x][y])
    """
    print(new_image.size)
    print(len(colors))
    print(colors[0], colors[1])
    """
    for x in range(len(colors[0])):
        for y in range(len(colors[1])):
            #print(x,y)
            new_image.putpixel((x, y), colors[x][y])

    return new_image

def ScanImage(image: str, #iterations: int = 1):
):
    ogSize = 512
    oldImage = Image.open(image).convert("RGB")
    goalSizex = oldImage.width
    goalSizey = oldImage.height
    #print("got all data")
    #print(ogSize, goalSizex, goalSizey)
    currentImage = 0
    startX = 0
    startY = 0
    for sizeIncrease in range(ogSize, goalSizex*2, ogSize):
        if sizeIncrease > goalSizex:
            sizeIncrease = goalSizex
        if sizeIncrease > goalSizey:
            sizeIncrease = goalSizey
        print("size increase: " + str(sizeIncrease))
        for currentY in range(ogSize, goalSizey*2, ogSize):
            for currentX in range(ogSize, goalSizex*2, ogSize):
                if(currentX > goalSizex):
                    currentX = goalSizex
                if(currentY > goalSizey):
                    currentY = goalSizey

                #print("currentX: " + str(currentX) + " currentY: " + str(currentY))
                #newImage = Image.new("RGB", (currentX, currentY))
                oldImageArray = []
                for currentXPixel in range(startX, currentX, 1):
                    yList = []
                    for currentYPixel in range(startY, currentY, 1):
                        if currentXPixel < goalSizex and currentYPixel < goalSizey:
                            #print(currentXPixel, currentYPixel)
                            oldpixel = oldImage.getpixel([currentXPixel, currentYPixel])
                            yList.append(oldpixel)
                            """
                            for x in range(1,currentX):
                                for y in range(1, currentY):
                                    newImage.putpixel((x, y), oldpixel)
                            """
                            #print("old image pixel color",oldpixel)
                            #print("size",newImage.size)
                            #print("putting pixel at: ", [currentXPixel, currentYPixel], "using old pixel: ", oldpixel)
                            #newImage.paste(oldImage, (oldImage[0],oldImage[1], currentX, currentY))
                            #newImage.putpixel((currentXPixel, currentYPixel), oldpixel)
                            #newImage.putpixel((currentX, currentY),oldpixel)
                            #newImage.putpixel((currentX, currentY), oldImage.getpixel((currentX, currentY)))
                    oldImageArray.append(yList)
                newImage = GenerateImage(ogSize, oldImageArray)
                #newImage.save("testimages\\newImage"+str(currentImage)+".png")
                #print("starting x: " + str(startX) + " starting y: " + str(startY))
                #print("ending x: " + str(currentX) + " ending y: " + str(currentY))
                startY = currentX
                currentImage += 1
                newImage.save("testimages\\newImage"+str(currentImage)+".png")
                print("saved new image: testimages\\newImage"+str(currentImage)+".png")
    ogSize = sizeIncrease
    """
    newImage = Image.new("RGB", (currentX, currentY))

    for xPixel in range(currentX, currentX*2, 1):
        for yPixel in range(currentY, currentY*2, 1):
            if xPixel < goalSizex and yPixel < goalSizey:
                print(xPixel, yPixel)
                oldpixel = oldImage.getpixel((xPixel, yPixel))
                newImage.putpixel([xPixel, yPixel], oldpixel)
                #newImage.putpixel((xPixel, yPixel), oldImage.getpixel((xPixel, yPixel)))
    print("saving new image")
    newImage.save("testimages\\newImage"+str(currentImage)+".png")
    currentImage += 1
    """
            #newImage.putpixel((currentX, currentY), oldImage.getpixel((currentX, currentY)))


def ScanImage2(model: str, image: str):
    oldImage = Image.open(image).convert("RGB")
    iterations = 0
    #ogSize = 256
    #newImage = Image.new("RGB", (oldImage.width, oldImage.height))
    #"""
    startx = 0
    starty = 0
    round = 0
    #print(nums)
    stopX = False
    stopY = False
    useAllPixels = False
    for ogSize in range(256,oldImage.width, 256):
        sizeIncrease = ogSize
        goalSizex = oldImage.width
        goalSizey = oldImage.height
        #for i in range(sizeIncrease, oldImage.width, sizeIncrease):
        """
        for sizeIncrease in range(ogSize, goalSizex*2, ogSize):
            if sizeIncrease > goalSizex:
                sizeIncrease = goalSizex
            if sizeIncrease > goalSizey:
                sizeIncrease = goalSizey
            startx = 0
            starty = 0
            print("size increase: " + str(sizeIncrease))
        """
        iterations += 1
        print(oldImage.mode)
        for x in range(startx, oldImage.width* (useAllPixels and 2 or 1)-ogSize, sizeIncrease):
            for y in range(starty, oldImage.height*(useAllPixels and 2 or 1)-ogSize, ogSize):
                #if i want to include every single pixel from old image (might include a lot of black pixels if at the very edge), i use this:
                """
                if x > oldImage.width:
                    x = oldImage.width - ogSize
                if y > oldImage.height:
                    y = oldImage.height - ogSize
                """
                #print("x going from",x ,"y going from",y)
                desinationX = x + ogSize
                desinationY = y + ogSize
                #print("x going to", desinationX, "y going to", desinationY)
                #working pixel extraction
                newImage = Image.new("RGB", (sizeIncrease, sizeIncrease))
                #"""
                image_position = (x, y, x+sizeIncrease, y+sizeIncrease)
                cropped_region = oldImage.crop(image_position)
                newImage.paste(cropped_region, (0,0))
                #"""
                #print(newImage. == oldImage.mode)
                #newImage.paste(image,(x, y, desinationX, desinationY))
                """
                for xPixel in range(x, desinationX,1):
                    for yPixel in range(y, desinationY,1):
                        if xPixel < oldImage.width and yPixel < oldImage.height:
                            #print(xPixel, yPixel)
                            oldpixel = oldImage.getpixel((xPixel, yPixel))
                            newImage.putpixel((xPixel - x, yPixel - y), oldpixel)
                            #newImage.paste(oldImage, (xPixel - x, yPixel - y, xPixel - x + 1, yPixel - y + 1))
                            #newImage.putpixel((xPixel, yPixel), oldImage.getpixel((xPixel, yPixel)))
                """
                #name = os.path.dirname(os.path.abspath(__file__)) + "\\test3\\newImage"+str(round)+".png"
                name = "test3\\newImage"+str(round)+".png"
                newImage.save(name)
                #print("saved new image: testimages\\newImage"+str(extranum)+str(num)+".png")
                round+=1
                #return Detect(model, name)
            #sizeIncrease += ogSize
                #if y > oldImage.height:
                #   break
            #if x > oldImage.width:
            #   break

            #ogSize = sizeIncrease

def Scanning():
    runs = 0
    #with open("Models\\many.pkl", "rb") as label:
        #label_encoder = pickle.load(label)
    for i in range(256, 1080, 256):
        runs += 1
        #num += 1
#ScanImage2("C:\\Users\\Dean Winchester\\Pictures\\brett\\street_brett.jpg")

#Scanning()


#ScanImage2("C:\\Users\\Dean Winchester\\Pictures\\brett\\street_brett.jpg")
"""
    for i in range(oldImage.width):
        newImage.height = ogSize * i
        newImage.width = ogSize * i
        
        for x in range(oldImage.width):
            for y in range(oldImage.height):
                if x < newImage.width and y < newImage.height:
                    newImage.putpixel((x, y), oldImage.getpixel((x, y)))
"""

#"""
while True:
    print("Waiting for input from stdin", flush=True)
    lines = sys.stdin.readline()
    if not lines:
        continue
    if(lines == ""): 
        continue
    print("Received input from stdin", flush=True)
    pyperclip.copy(lines)
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
    nums = 0
    with open(modelPkl, "rb") as label:
        label_encoder = pickle.load(label)
        model = CNNModel(num_classes=len(label_encoder.classes_)).to(device)
    print("model loaded", flush=True)
    #print("Label encoder loaded", flush=True)
    ScanImage2(model, imageAddress)
    data = Detect(modelPt)
    for label, prob in data:
        print(f"Detected: {label}, Probability: {prob:.4f}", flush=True)
    #print(data, flush=True)
    #predicted_label, prob = Detect(modelPt)
    #print(f"Detected: Predicted label: {predicted_label}, Probability: {prob:.4f}", flush=True)
    #sys.stdin.close()
    #sys.exit(0)
"""
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