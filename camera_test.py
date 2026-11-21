import cv2

# Open laptop's built-in camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Camera opened successfully!")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Display camera
    cv2.imshow("Driver Monitoring System - Camera Test", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()