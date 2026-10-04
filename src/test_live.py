import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import pyttsx3
import time
'''
import subprocess

def speak(text):
    subprocess.run(['say', text])
    '''

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speaking speed

def speak(text):
    engine.say(text)
    engine.runAndWait()



# ---- Load trained model ----
model = joblib.load('models/sign_classifier.pkl')

# ---- Setup hand detector ----
base_options = python.BaseOptions(
    model_asset_path='models/hand_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ---- Speech control state ----
last_spoken = None
last_spoken_time = 0
COOLDOWN = 2.0  # seconds before repeating the same sign
CONFIDENCE_THRESHOLD = 0.7  # only speak if model is fairly confident

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row = np.array(row).reshape(1, -1)

        prediction = model.predict(row)[0]
        confidence = model.predict_proba(row).max()

        for lm in landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        text = f"{prediction} ({confidence:.0%})"
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # ---- Speak logic ----
        current_time = time.time()
        if confidence >= CONFIDENCE_THRESHOLD:
            if prediction != last_spoken or (current_time - last_spoken_time) > COOLDOWN:
                speak(prediction)
                last_spoken = prediction
                last_spoken_time = current_time
    else:
        cv2.putText(frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Prediction', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()