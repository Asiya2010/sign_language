import cv2
import mediapipe as mp

# -----------------------------
# 1. Load the MediaPipe model
# -----------------------------

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)


# -----------------------------
# 2. Open the Mac camera
# -----------------------------

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame into a MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # -----------------------------
    # 3. Detect the hand
    # -----------------------------

    result = detector.detect(mp_image)

    print(
        "Hands detected:",
        len(result.hand_landmarks)
    )

    # Show camera
    cv2.imshow(
        "Sign Language Camera",
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# 4. Clean up
# -----------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()
