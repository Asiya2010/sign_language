import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture one frame
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Show the camera
    cv2.imshow("Sign Language Camera", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()

# Close the window
cv2.destroyAllWindows()