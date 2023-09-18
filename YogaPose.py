from ultralytics import YOLO

# Load the YOLO model
model = YOLO("models/best.pt")  #colab custom trained model

 #Perform object detection on the image
results = model(source='Yoga.jpg',save=True, conf=0.7)

#or

#For video

results = model(source='Yoga.mp4',save=True, conf=0.7)