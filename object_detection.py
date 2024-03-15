import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# Function to detect objects in an image
def detect_objects(image_path):
    prototxt_path = "C:/Users/Suryanarayan/OneDrive/Desktop/DSA/python/python_projects/MobileNetSSD_deploy.prototxt.txt"
    model_path = "C:/Users/Suryanarayan/OneDrive/Desktop/DSA/python/python_projects/MobileNetSSD_deploy.caffemodel"

    # Load pre-trained model for object detection
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    # Load input image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    # Preprocess image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # Set the blob as input to the network
    net.setInput(blob)
    # Perform a forward pass of the network
    detections = net.forward()

    # Initialize count of detected objects
    object_count = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Extract information about the detected object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box around the detected object
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Increment count of detected objects
            object_count += 1

    # Display the resulting image
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Update the label to display the count of detected objects
    result_label.config(text=f"Detected Objects: {object_count}")

# Function to handle button click event
def detect():
    image_path = entry.get()
    detect_objects(image_path)

# Create Tkinter window
root = Tk()
root.title("Object Detection")

# Create label and entry widget for image path
Label(root, text="Enter image path:", font=("Helvetica", 12)).pack(pady=10)
entry = Entry(root, width=50, font=("Helvetica", 12))
entry.pack()

# Create detect button
Button(root, text="Detect Objects", command=detect, font=("Helvetica", 12)).pack(pady=5)

# Create label to display the count of detected objects
result_label = Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
