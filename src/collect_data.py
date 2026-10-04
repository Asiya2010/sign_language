import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# ---- Setup ----
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

# ---- Config ----
SIGNS = ['hello', 'stop', 'yes', 'i_love_you', 'ok', 'same', 'thank_you', 'no' , 'help', 'what', 'more', 'goodbye', 'who']
SAMPLES_PER_SIGN = 150
CSV_PATH = 'dataset/landmarks.csv'

os.makedirs('dataset', exist_ok=True)

# writing header once if file doesn't exist
if not os.path.exists(CSV_PATH):
    header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(header)

cap = cv2.VideoCapture(0)

for sign in SIGNS:
    print(f"\n=== Get ready for sign: {sign.upper()} ===")
    print("Press SPACE when ready to start capturing, ESC to skip this sign.")

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: {sign} - Press SPACE to start", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            break
        elif key == 27:  # ESC
            break

    count = 0
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        while count < SAMPLES_PER_SIGN:
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
                row.append(sign)
                writer.writerow(row)
                count += 1

                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(frame, f"{sign}: {count}/{SAMPLES_PER_SIGN}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Data Collection', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
                break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\nData collection complete! Saved to dataset/landmarks.csv")