import os
import cv2
import json
import time
import math
import csv
import threading
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import requests

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

STATE_FILE = os.path.join(BASE_DIR, "realtime_state.json")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "driver_events.csv")

DROWSINESS_DIR = os.path.join(BASE_DIR, "alerts", "drowsiness")
YAWN_DIR = os.path.join(BASE_DIR, "alerts", "yawning")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DROWSINESS_DIR, exist_ok=True)
os.makedirs(YAWN_DIR, exist_ok=True)

# Set your NEW Telegram bot token and chat ID here.
# IMPORTANT: Revoke any token previously exposed and use a fresh token.
BOT_TOKEN = "8826103607:AAGKIBUeaOjivnUAaqj2CSczTKBYlbuwLH8"
CHAT_ID = "1178667077"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
TELEGRAM_ENABLED = True
TELEGRAM_COOLDOWN = 30  # seconds per event type
telegram_last_sent = {"DROWSINESS": 0.0, "YAWNING": 0.0}

# ---------------- EYE DETECTION ----------------
EAR_THRESHOLD = 0.25
DROWSINESS_TIME = 1.5

# ---------------- YAWN DETECTION ----------------
MAR_THRESHOLD = 0.35
YAWN_TIME = 1.0

# ---------------- ALARM ----------------
ALARM_COOLDOWN = 3.0

# ---------------- COLORS (BGR) ----------------
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)

# ============================================================
# LANDMARKS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH = [13, 14, 61, 291]

# ============================================================
# GLOBAL VARIABLES
# ============================================================

eye_closed_start = None
yawn_start = None

drowsiness_count = 0
yawn_count = 0

last_alarm_time = 0

face_detected = False
drowsiness = False
yawning = False

current_status = "STARTING"

drowsiness_event_triggered = False
yawn_event_triggered = False

# ============================================================
# CSV INITIALIZATION
# ============================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "Image_Path", "EAR", "MAR"])

# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_ear(landmarks, eye_points):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks):
    upper = landmarks[MOUTH[0]]
    lower = landmarks[MOUTH[1]]
    left = landmarks[MOUTH[2]]
    right = landmarks[MOUTH[3]]

    vertical = distance(upper, lower)
    horizontal = distance(left, right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def draw_eye(frame, landmarks, eye_indices, color):
    height, width = frame.shape[:2]
    points = []

    for index in eye_indices:
        point = landmarks[index]
        x, y = int(point.x * width), int(point.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, color, -1)

    cv2.polylines(frame, [__import__("numpy").array(points, dtype="int32")], True, color, 2)


def draw_mouth(frame, landmarks, color):
    height, width = frame.shape[:2]
    points = []

    for index in MOUTH_CONTOUR:
        point = landmarks[index]
        x, y = int(point.x * width), int(point.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, color, -1)

    cv2.polylines(frame, [__import__("numpy").array(points, dtype="int32")], True, color, 2)

    for index in (13, 14):
        point = landmarks[index]
        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 6, color, -1)


def draw_landmarks(frame, landmarks, ear, mar):
    eye_color = RED if ear < EAR_THRESHOLD else GREEN
    mouth_color = RED if mar > MAR_THRESHOLD else GREEN

    draw_eye(frame, landmarks, LEFT_EYE, eye_color)
    draw_eye(frame, landmarks, RIGHT_EYE, eye_color)
    draw_mouth(frame, landmarks, mouth_color)


# ============================================================
# ALARM
# ============================================================

def alarm():
    global last_alarm_time
    now = time.time()

    if now - last_alarm_time < ALARM_COOLDOWN:
        return

    last_alarm_time = now
    threading.Thread(target=lambda: winsound.Beep(1500, 700), daemon=True).start()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_event(event, image_path, ear, mar):
    """Send event photo with caption to Telegram; runs in a background thread."""
    if not TELEGRAM_ENABLED:
        return

    now = time.time()
    previous = telegram_last_sent.get(event, 0.0)

    if now - previous < TELEGRAM_COOLDOWN:
        print(f"[TELEGRAM] {event} alert skipped (cooldown)")
        return

    # Set cooldown before starting the thread to avoid duplicate concurrent sends.
    telegram_last_sent[event] = now

    caption = (
        f"🚨 DRIVEGUARD AI ALERT\n"
        f"Event: {event}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"EAR: {ear:.3f}\n"
        f"MAR: {mar:.3f}\n"
        f"Please check the driver."
    )

    def worker():
        try:
            if not BOT_TOKEN or "PASTE_" in BOT_TOKEN:
                print("[TELEGRAM] Configure BOT_TOKEN in eye_monitor.py")
                return

            with open(image_path, "rb") as photo:
                response = requests.post(
                    f"{TELEGRAM_API}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": caption},
                    files={"photo": photo},
                    timeout=20
                )

            if response.ok:
                print(f"[TELEGRAM] {event} image/alert sent")
            else:
                print(f"[TELEGRAM ERROR] {response.status_code}: {response.text}")

        except Exception as error:
            print(f"[TELEGRAM ERROR] {error}")

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(event, frame, ear, mar):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = DROWSINESS_DIR if event == "DROWSINESS" else YAWN_DIR
    filename = f"{event}_{timestamp}.jpg"
    image_path = os.path.join(folder, filename)

    saved = cv2.imwrite(image_path, frame)
    if not saved:
        print(f"[ERROR] Could not save event image: {image_path}")
        return None

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event,
            image_path,
            round(ear, 4),
            round(mar, 4)
        ])

    print("\n" + "=" * 50)
    print(f"[ALERT] {event}")
    print(f"[EAR] {ear:.4f}")
    print(f"[MAR] {mar:.4f}")
    print(f"[IMAGE] {image_path}")
    print("=" * 50 + "\n")

    send_telegram_event(event, image_path, ear, mar)
    return image_path


# ============================================================
# DASHBOARD STATE
# ============================================================

def save_state(status, ear, mar, fps, detected, drowsy, yawn):
    state = {
        "status": status,
        "ear": float(ear),
        "mar": float(mar),
        "fps": float(fps),
        "face_detected": bool(detected),
        "drowsiness": bool(drowsy),
        "yawning": bool(yawn),
        "drowsiness_count": int(drowsiness_count),
        "yawn_count": int(yawn_count),
        "ear_threshold": EAR_THRESHOLD,
        "drowsiness_time": DROWSINESS_TIME,
        "mar_threshold": MAR_THRESHOLD,
        "yawn_time": YAWN_TIME,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_event": status
    }

    temporary_file = STATE_FILE + ".tmp"

    try:
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=4)
    except OSError as error:
        print(f"[WARNING] Failed to write temporary state: {error}")
        return False

    for _ in range(10):
        try:
            os.replace(temporary_file, STATE_FILE)
            return True
        except PermissionError:
            time.sleep(0.05)
        except OSError as error:
            print(f"[WARNING] State update failed: {error}")
            break

    try:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
    except OSError:
        pass

    return False


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("[ERROR] Cannot open webcam")
    landmarker.close()
    raise SystemExit(1)

print("=" * 60)
print("AI DRIVER MONITORING SYSTEM")
print("=" * 60)
print(f"[INFO] EAR Threshold: {EAR_THRESHOLD}")
print(f"[INFO] Drowsiness Duration: {DROWSINESS_TIME} seconds")
print(f"[INFO] MAR Threshold: {MAR_THRESHOLD}")
print(f"[INFO] Yawn Duration: {YAWN_TIME} seconds")
print("[INFO] Telegram alerts enabled:", TELEGRAM_ENABLED)
print("[INFO] Webcam started")
print("[INFO] Press Q to exit")

# ============================================================
# MAIN LOOP
# ============================================================

previous_time = time.time()
frame_number = 0

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("[ERROR] Camera frame unavailable")
            break

        frame = cv2.flip(frame, 1)
        frame_number += 1
        current_time = time.time()

        fps = 1.0 / max(current_time - previous_time, 0.001)
        previous_time = current_time

        ear = 0.0
        left_ear = 0.0
        right_ear = 0.0
        mar = 0.0
        face_detected = False
        drowsiness = False
        yawning = False
        current_status = "FACE NOT DETECTED"
        closed_duration = 0.0
        yawn_duration = 0.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, int(current_time * 1000))

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            face_detected = True

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks)

            draw_landmarks(frame, landmarks, ear, mar)

            # Eye closure tracking
            eyes_closed = ear < EAR_THRESHOLD

            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                closed_duration = time.time() - eye_closed_start
                current_status = "EYES CLOSED"
            else:
                eye_closed_start = None
                drowsiness_event_triggered = False
                closed_duration = 0.0
                current_status = "NORMAL"

            if eyes_closed and closed_duration >= DROWSINESS_TIME:
                drowsiness = True
                current_status = "DROWSY"

            # Yawn tracking
            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.time()
                yawn_duration = time.time() - yawn_start
            else:
                yawn_start = None
                yawn_event_triggered = False
                yawn_duration = 0.0

            if mar > MAR_THRESHOLD and yawn_duration >= YAWN_TIME:
                yawning = True
                if not drowsiness:
                    current_status = "YAWNING"

            # Drowsiness event
            if drowsiness:
                cv2.putText(frame, "DROWSINESS DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 3)
                cv2.putText(frame, "ALERT! WAKE UP!", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 3)

                if not drowsiness_event_triggered:
                    drowsiness_event_triggered = True
                    drowsiness_count += 1
                    save_event("DROWSINESS", frame, ear, mar)
                    alarm()

            # Yawning event
            if yawning:
                cv2.putText(frame, "YAWNING DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 3)

                if not yawn_event_triggered:
                    yawn_event_triggered = True
                    yawn_count += 1
                    save_event("YAWNING", frame, ear, mar)
                    alarm()

            # HUD
            cv2.putText(frame, "EYES CLOSED" if eyes_closed else "EYES OPEN",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        RED if eyes_closed else GREEN, 2)

            cv2.putText(frame, "MOUTH OPEN" if mar > MAR_THRESHOLD else "MOUTH NORMAL",
                        (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        RED if mar > MAR_THRESHOLD else GREEN, 2)

            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        RED if eyes_closed else GREEN, 2)

            cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (30, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            cv2.putText(frame, f"MAR: {mar:.3f}", (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        RED if mar > MAR_THRESHOLD else GREEN, 2)

            if eyes_closed:
                cv2.putText(frame, f"Eye Closure: {closed_duration:.1f}s", (30, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

            if mar > MAR_THRESHOLD:
                cv2.putText(frame, f"Yawn Duration: {yawn_duration:.1f}s", (30, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)

        else:
            eye_closed_start = None
            yawn_start = None
            drowsiness_event_triggered = False
            yawn_event_triggered = False

        # HUD counters
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.putText(frame, f"Drowsiness: {drowsiness_count}", (30, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        cv2.putText(frame, f"Yawning: {yawn_count}", (30, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)

        # Streamlit dashboard state only (no camera frame saved for dashboard)
        save_state(current_status, ear, mar, fps, face_detected, drowsiness, yawning)

        cv2.imshow("AI DRIVER MONITORING SYSTEM", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Eye monitoring stopped")
