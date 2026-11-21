import cv2
import mediapipe as mp
import time

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "models/face_landmarker.task"

CAMERA_INDEX = 0

# ==========================================================
# MEDIAPIPE INITIALIZATION
# ==========================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ==========================================================
# CAMERA INITIALIZATION
# ==========================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera started successfully")
print("Face Landmarker initialized")
print("Press Q to quit")


# ==========================================================
# FPS CALCULATION
# ==========================================================

previous_time = time.time()

fps = 0


# ==========================================================
# FACE LANDMARK DETECTION
# ==========================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Detect face landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ==================================================
        # PROCESS LANDMARKS
        # ==================================================

        if results.face_landmarks:

            for face_landmarks in results.face_landmarks:

                height, width, _ = frame.shape

                x_coordinates = []
                y_coordinates = []

                # Draw facial landmarks
                for landmark in face_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    x_coordinates.append(x)
                    y_coordinates.append(y)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

                # Calculate face bounding box
                x_min = min(x_coordinates)
                x_max = max(x_coordinates)

                y_min = min(y_coordinates)
                y_max = max(y_coordinates)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                # Display number of landmarks
                landmark_count = len(face_landmarks)

                cv2.putText(
                    frame,
                    f"Landmarks: {landmark_count}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # ==================================================
        # FPS
        # ==================================================

        current_time = time.time()

        elapsed_time = current_time - previous_time

        if elapsed_time > 0:

            fps = 1 / elapsed_time

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ==================================================
        # DISPLAY
        # ==================================================

        cv2.imshow(
            "AI Driver Monitoring - Face Landmarks",
            frame
        )

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("Driver monitoring application stopped")