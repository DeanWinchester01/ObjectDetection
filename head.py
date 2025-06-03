import image_Detect
import os

healthyValFolder = "C:\\Users\\Dean Winchester\\Desktop\\ultrasound breast classification\\train\\benign"
cancerValFolder = "C:\\Users\\Dean Winchester\\Desktop\\ultrasound breast classification\\train\\malignant"

correctHealthy = 0
correctCancer = 0
totalImages = 0

#image_Detect.model = "health"
print("Model set")

healthyDir = os.listdir(healthyValFolder)
for filename in healthyDir:
    label, probability = image_Detect.Detectf(healthyValFolder+"\\"+filename)
    if label == "healthy":
        correctHealthy += 1
print("Correct healthy:", correctHealthy, " out of ", len(healthyDir))
print(correctHealthy/len(healthyDir)*100, "%")

cancerDir = os.listdir(cancerValFolder)
for filename in cancerDir:
    label, probability = image_Detect.Detectf(cancerValFolder+"\\"+filename)
    if label == "cancer":
        correctCancer += 1

print("Correct cancer:", correctCancer," out of ", len(cancerDir))
print(correctCancer/len(cancerDir)*100, "%")

totalImages = len(healthyDir) + len(cancerDir)
print(f"Correctly classified: {correctCancer+correctHealthy} out of {totalImages}")