import cv2
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainingImageLabel/trainner.yml')

# Load the Haar Cascade for face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set the font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

# Start video capture from the webcam
cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()  # Read a frame from the webcam
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        # Predict the ID of the detected face
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # Draw a rectangle around the detected face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the ID above the rectangle
        cv2.putText(im, str(Id), (x, y - 10), font, 1, (255, 255, 255), 2)

    # Show the video feed with recognized faces
    cv2.imshow('Face Recognition', im)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()