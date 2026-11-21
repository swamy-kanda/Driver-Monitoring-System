
import os
import cv2
import csv
import json
import time
import threading
import numpy as np
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

ALERT_DIR = os.path.join(BASE_DIR, "alerts")
DROWSINESS_DIR = os.path.join(ALERT_DIR, "drowsiness")
YAWNING_DIR = os.path.join(ALERT_DIR, "yawning")

LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, "driver_events.csv")

STATE_PATH = os.path.join(BASE_DIR, "realtime_state.json")
FRAME_PATH = os.path.join(BASE_DIR, "live_frame.jpg")

EAR_THRESHOLD = 0.21
DROWSINESS_DURATION = 1.5

MAR_THRESHOLD = 0.60
YAWNING_DURATION = 1.0

EVENT_COOLDOWN = 5.0

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    DROWSINESS_DIR,
    YAWNING_DIR,
    LOG_DIR
]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# CSV INITIALIZATION
# ============================================================

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "event",
            "ear",
            "mar",
            "evidence"
        ])

# ============================================================
# REAL-TIME SHARED STATE
# ============================================================

shared_state = {
    "status": "STARTING",
    "ear": 0.0,
    "mar": 0.0,
    "fps": 0.0,
    "drowsiness": False,
    "yawning": False,
    "face_detected": False,
    "timestamp": "",
    "last_event": "NONE"
}


def update_state():

    temp_path = STATE_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(shared_state, f)

    os.replace(temp_path, STATE_PATH)


def save_live_frame(frame):

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    if not success:
        return

    temp_path = FRAME_PATH + ".tmp"

    with open(temp_path, "wb") as f:
        f.write(encoded.tobytes())

    os.replace(temp_path, FRAME_PATH)


# ============================================================
# ALARM
# ============================================================

def alarm():

    try:
        import winsound

        winsound.Beep(1500, 700)

    except Exception:
        print("\a")


# ============================================================
# LANDMARK FUNCTIONS
# ============================================================

def landmark_point(landmarks, index, width, height):

    point = landmarks[index]

    return np.array([
        point.x * width,
        point.y * height
    ])


def calculate_ear(landmarks, width, height):

    # Left eye
    left_eye = [
        33, 160, 158, 133, 153, 144
    ]

    # Right eye
    right_eye = [
        362, 385, 387, 263, 373, 380
    ]

    def eye_ratio(indices):

        p1 = landmark_point(landmarks, indices[0], width, height)
        p2 = landmark_point(landmarks, indices[1], width, height)
        p3 = landmark_point(landmarks, indices[2], width, height)
        p4 = landmark_point(landmarks, indices[3], width, height)
        p5 = landmark_point(landmarks, indices[4], width, height)
        p6 = landmark_point(landmarks, indices[5], width, height)

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)

        horizontal = np.linalg.norm(p1 - p4)

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    left_ear = eye_ratio(left_eye)
    right_ear = eye_ratio(right_eye)

    return (left_ear + right_ear) / 2.0


def calculate_mar(landmarks, width, height):

    upper_lip = landmark_point(
        landmarks, 13, width, height
    )

    lower_lip = landmark_point(
        landmarks, 14, width, height
    )

    left_mouth = landmark_point(
        landmarks, 78, width, height
    )

    right_mouth = landmark_point(
        landmarks, 308, width, height
    )

    vertical = np.linalg.norm(
        upper_lip - lower_lip
    )

    horizontal = np.linalg.norm(
        left_mouth - right_mouth
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# ============================================================
# EVIDENCE CAPTURE AND LOGGING
# ============================================================

def save_event(event, frame, ear, mar):

    timestamp = datetime.now()

    timestamp_string = timestamp.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    filename_timestamp = timestamp.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    if event == "DROWSINESS":

        folder = DROWSINESS_DIR

        filename = (
            f"drowsiness_{filename_timestamp}.jpg"
        )

    else:

        folder = YAWNING_DIR

        filename = (
            f"yawn_{filename_timestamp}.jpg"
        )

    evidence_path = os.path.join(
        folder, filename
    )

    cv2.imwrite(evidence_path, frame)

    with open(
        CSV_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp_string,
            event,
            round(ear, 4),
            round(mar, 4),
            evidence_path
        ])

    print(f"\n[ALARM] {event} DETECTED!")
    print(f"[SAVED] {evidence_path}")
    print(f"[LOG] {timestamp_string} | {event}")

    alarm()

    return timestamp_string


# ============================================================
# MEDIAPIPE FACE LANDMARK MODEL
# ============================================================

BaseOptions = python.BaseOptions

FaceLandmarker = vision.FaceLandmarker

FaceLandmarkerOptions = vision.FaceLandmarkerOptions

VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(
    options
)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():

    print("[ERROR] Cannot open webcam")

    exit()

print("=" * 60)
print("AI DRIVER MONITORING SYSTEM")
print("=" * 60)
print("[INFO] Camera started")
print("[INFO] Dashboard communication enabled")
print("[INFO] Press Q to quit")
print("=" * 60)

# ============================================================
# DETECTION TIMERS
# ============================================================

drowsiness_start = None
yawning_start = None

last_drowsiness_event = 0
last_yawning_event = 0

last_event = "NONE"

fps_start = time.time()
fps_counter = 0
fps = 0.0

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("[ERROR] Camera frame failed")

            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        current_time = time.time()

        fps_counter += 1

        # ====================================================
        # FPS CALCULATION
        # ====================================================

        if current_time - fps_start >= 1.0:

            fps = fps_counter / (
                current_time - fps_start
            )

            fps_counter = 0

            fps_start = current_time

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        ear = 0.0
        mar = 0.0

        drowsiness = False
        yawning = False

        face_detected = False

        status = "NORMAL"

        # ====================================================
        # FACE DETECTION
        # ====================================================

        if result.face_landmarks:

            face_detected = True

            landmarks = result.face_landmarks[0]

            ear = calculate_ear(
                landmarks,
                width,
                height
            )

            mar = calculate_mar(
                landmarks,
                width,
                height
            )

            # =================================================
            # DROWSINESS
            # =================================================

            if ear < EAR_THRESHOLD:

                if drowsiness_start is None:

                    drowsiness_start = current_time

                elapsed = (
                    current_time - drowsiness_start
                )

                if elapsed >= DROWSINESS_DURATION:

                    drowsiness = True

                    if (
                        current_time -
                        last_drowsiness_event
                    ) >= EVENT_COOLDOWN:

                        last_drowsiness_event = (
                            current_time
                        )

                        last_event = "DROWSINESS"

                        save_event(
                            "DROWSINESS",
                            frame,
                            ear,
                            mar
                        )

            else:

                drowsiness_start = None

            # =================================================
            # YAWNING
            # =================================================

            if mar > MAR_THRESHOLD:

                if yawning_start is None:

                    yawning_start = current_time

                elapsed = (
                    current_time - yawning_start
                )

                if elapsed >= YAWNING_DURATION:

                    yawning = True

                    if (
                        current_time -
                        last_yawning_event
                    ) >= EVENT_COOLDOWN:

                        last_yawning_event = (
                            current_time
                        )

                        last_event = "YAWNING"

                        save_event(
                            "YAWNING",
                            frame,
                            ear,
                            mar
                        )

            else:

                yawning_start = None

        else:

            drowsiness_start = None
            yawning_start = None

        # ====================================================
        # DRIVER STATUS
        # ====================================================

        if drowsiness:

            status = "DROWSY"

        elif yawning:

            status = "YAWNING"

        elif not face_detected:

            status = "NO FACE"

        else:

            status = "NORMAL"

        # ====================================================
        # DISPLAY OVERLAY
        # ====================================================

        if status == "DROWSY":

            color = (0, 0, 255)

        elif status == "YAWNING":

            color = (0, 165, 255)

        elif status == "NO FACE":

            color = (255, 0, 0)

        else:

            color = (0, 255, 0)

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # UPDATE SHARED STATE
        # ====================================================

        shared_state.update({

            "status": status,

            "ear": round(float(ear), 4),

            "mar": round(float(mar), 4),

            "fps": round(float(fps), 2),

            "drowsiness": bool(drowsiness),

            "yawning": bool(yawning),

            "face_detected": bool(face_detected),

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "last_event": last_event

        })

        update_state()

        # ====================================================
        # SAVE LIVE FRAME FOR DASHBOARD
        # ====================================================

        save_live_frame(frame)

        # ====================================================
        # OPENCV DISPLAY
        # ====================================================

        cv2.imshow(
            "AI Driver Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

except KeyboardInterrupt:

    print("\n[INFO] Monitoring stopped by user")

finally:

    shared_state["status"] = "STOPPED"

    update_state()

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print("[INFO] Camera released")
    print("[INFO] Driver monitoring stopped")
