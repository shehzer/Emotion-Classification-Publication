import os
import cv2
from keras.models import model_from_json
import keras.utils as image
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# Reading the model json file
json_file = open('model.json','r')
json_model_read = json_file.read()
json_file.close()

# Importing the model and loading weights 
model = model_from_json(json_model_read)
model.load_weights('model.h5')

# Referencing Haar Cascade algorithm for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
emotion_list = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

while True:
    # Capturing the real-time camera img
    ret, current_img = capture.read()
    if not ret:
        continue

    # Converting the image to grayscale
    gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    # Detecting faces with openCV cascade classifier
    face_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)
    
    for (x, y, w, h) in face_detected:
        # Drawing ellipse with detected face
        center = (x + w//2, y + h//2)
        cv2.ellipse(current_img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        # Clipping face for model prediction
        clipped_gray = gray_img[y:y + w, x:x + h]

        # Preprocessing the image to match with expected activation layer input
        clipped_gray = cv2.resize(clipped_gray, (48, 48))
        img_input = np.expand_dims(clipped_gray, axis=0)

        # Prediction with model
        prediction = model.predict(img_input)
        
        # Determine the highest score from the prediction and get that index
        res = np.argmax(prediction[0])

        # Display the emotion from emotion_list with index
        cv2.putText(current_img, emotion_list[res], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    window = cv2.resize(current_img, (1280, 960))
    cv2.imshow('Facial Emotion Detection ', window)
    
    # If user presses ESC, close window
    if cv2.waitKey(10) == 27: 
        break
    
capture.release()
cv2.destroyAllWindows