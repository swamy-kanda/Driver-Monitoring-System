import os
import json

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE_DIR, "realtime_state.json")

CSV_PATH = os.path.join(BASE_DIR, "logs", "driver_events.csv")
DROWSINESS_DIR = os.path.join(BASE_DIR, "alerts", "drowsiness")
YAWNING_DIR = os.path.join(BASE_DIR, "alerts", "yawning")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Driver Monitoring System",
    page_icon="🚗",
    layout="wide"
)

st_autorefresh(interval=1000, key="dashboard_refresh")

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp { background-color: #0b1120; }
.block-container { padding-top: 1.5rem; }
.status-normal {
    background-color: #14532d; color: #bbf7d0; padding: 20px;
    text-align: center; border-radius: 12px; font-size: 25px; font-weight: bold;
}
.status-drowsy {
    background-color: #991b1b; color: #fecaca; padding: 20px;
    text-align: center; border-radius: 12px; font-size: 25px; font-weight: bold;
}
.status-yawning {
    background-color: #9a3412; color: #fed7aa; padding: 20px;
    text-align: center; border-radius: 12px; font-size: 25px; font-weight: bold;
}
.status-no-face {
    background-color: #1e3a8a; color: #bfdbfe; padding: 20px;
    text-align: center; border-radius: 12px; font-size: 25px; font-weight: bold;
}
.status-stopped {
    background-color: #374151; color: #d1d5db; padding: 20px;
    text-align: center; border-radius: 12px; font-size: 25px; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA FUNCTIONS
# ============================================================

def read_realtime_state():
    default_state = {
        "status": "NOT STARTED",
        "ear": 0.0,
        "mar": 0.0,
        "fps": 0.0,
        "drowsiness": False,
        "yawning": False,
        "face_detected": False,
        "timestamp": "",
        "last_event": "NONE",
        "drowsiness_count": 0,
        "yawn_count": 0,
        "ear_threshold": 0.25,
        "drowsiness_time": 1.5,
        "mar_threshold": 0.35,
        "yawn_time": 1.0
    }

    if not os.path.exists(STATE_PATH):
        return default_state

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        default_state.update(data)
        return default_state
    except Exception:
        return default_state


def load_events():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_PATH)
        rename_map = {
            "Timestamp": "timestamp",
            "Event": "event",
            "Image_Path": "image_path",
            "EAR": "ear",
            "MAR": "mar"
        }
        df.rename(columns=rename_map, inplace=True)
        return df
    except Exception as error:
        st.warning(f"Could not read event log: {error}")
        return pd.DataFrame()


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return sum(
        1 for filename in os.listdir(folder)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def get_latest_images(folder, limit=9):
    if not os.path.exists(folder):
        return []

    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                images.append(path)

    images.sort(key=os.path.getmtime, reverse=True)
    return images[:limit]


def show_image_gallery(folder, title, key_prefix):
    images = get_latest_images(folder)

    st.subheader(title)

    if not images:
        st.info("No images recorded for this event yet.")
        return

    columns = st.columns(3)

    for index, image_path in enumerate(images):
        with columns[index % 3]:
            st.image(
                image_path,
                caption=os.path.basename(image_path),
                width="stretch"
            )

            try:
                with open(image_path, "rb") as file:
                    image_bytes = file.read()

                st.download_button(
                    "Download Image",
                    image_bytes,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg",
                    key=f"{key_prefix}_{index}"
                )
            except OSError:
                st.warning("Could not read this image.")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">🚗 AI DRIVER MONITORING SYSTEM</h1>
    <p style="text-align:center;color:#94a3b8;">
    Driver Safety & Behavior Analytics Dashboard
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

state = read_realtime_state()
events = load_events()

# ============================================================
# LIVE STATUS
# ============================================================

st.subheader("🚦 Monitoring Status")

status = state.get("status", "UNKNOWN")

status_mapping = {
    "NORMAL": ("status-normal", "🟢 DRIVER NORMAL"),
    "EYES CLOSED": ("status-yawning", "🟠 EYES CLOSED"),
    "DROWSY": ("status-drowsy", "🔴 DROWSINESS DETECTED"),
    "YAWNING": ("status-yawning", "🟠 YAWNING DETECTED"),
    "NO FACE": ("status-no-face", "🔵 NO FACE DETECTED"),
    "FACE NOT DETECTED": ("status-no-face", "🔵 NO FACE DETECTED"),
    "STOPPED": ("status-stopped", "⚪ MONITORING STOPPED"),
    "NOT STARTED": ("status-stopped", "⚪ MONITORING NOT STARTED")
}

status_class, status_text = status_mapping.get(
    status, ("status-stopped", f"⚪ {status}")
)

st.markdown(
    f'<div class="{status_class}">{status_text}</div>',
    unsafe_allow_html=True
)

st.caption(f"Last update: {state.get('timestamp', 'N/A')}")

# ============================================================
# REAL-TIME METRICS
# ============================================================

st.subheader("📊 Real-Time Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Eye Aspect Ratio (EAR)", f"{state.get('ear', 0):.3f}")
col2.metric("Mouth Aspect Ratio (MAR)", f"{state.get('mar', 0):.3f}")
col3.metric("Processing FPS", f"{state.get('fps', 0):.1f}")
col4.metric(
    "Face Detection",
    "Detected" if state.get("face_detected", False) else "Not Detected"
)

# ============================================================
# DETECTION PARAMETERS / CONDITIONS
# ============================================================

st.divider()
st.subheader("⚙️ Detection Configuration")

param1, param2, param3, param4 = st.columns(4)

param1.metric("EAR Threshold", f"{state.get('ear_threshold', 0.25):.2f}")
param2.metric("Drowsiness Duration", f"{state.get('drowsiness_time', 1.5):.1f} s")
param3.metric("MAR Threshold", f"{state.get('mar_threshold', 0.35):.2f}")
param4.metric("Yawn Duration", f"{state.get('yawn_time', 1.0):.1f} s")

condition_col1, condition_col2 = st.columns(2)

with condition_col1:
    st.markdown("### 👁️ Eye Condition")
    if state.get("drowsiness", False):
        st.error("🔴 Drowsiness detected")
    elif state.get("face_detected", False):
        st.success("🟢 Eyes normal / no confirmed drowsiness")
    else:
        st.info("Waiting for face detection")

with condition_col2:
    st.markdown("### 👄 Mouth Condition")
    if state.get("yawning", False):
        st.warning("🟠 Yawning detected")
    elif state.get("face_detected", False):
        st.success("🟢 Mouth normal / no confirmed yawn")
    else:
        st.info("Waiting for face detection")

# ============================================================
# EVENT STATISTICS
# ============================================================

st.divider()
st.subheader("📈 Event Statistics")

total_events = len(events)
drowsiness_count = 0
yawning_count = 0

if not events.empty and "event" in events.columns:
    event_names = events["event"].astype(str).str.upper()
    drowsiness_count = int((event_names == "DROWSINESS").sum())
    yawning_count = int((event_names == "YAWNING").sum())

evidence_count = count_images(DROWSINESS_DIR) + count_images(YAWNING_DIR)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", total_events)
col2.metric("Drowsiness", drowsiness_count)
col3.metric("Yawning", yawning_count)
col4.metric("Evidence Images", evidence_count)

# ============================================================
# DEMONSTRATION SCORE
# ============================================================

st.subheader("🛡️ Session Event Score")

safety_score = max(0, 100 - drowsiness_count * 10 - yawning_count * 5)
st.progress(safety_score / 100)
st.metric("Session Score", f"{safety_score}/100")
st.caption("Demonstration score only. Not a validated safety assessment.")

# ============================================================
# EVENT ANALYTICS
# ============================================================

st.divider()
st.subheader("📊 Event Analytics")

if not events.empty and "event" in events.columns:
    event_counts = events["event"].astype(str).value_counts()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Event Distribution")
        st.bar_chart(event_counts)

    with chart_col2:
        st.markdown("### Event Timeline")

        if "timestamp" in events.columns:
            timeline_data = events.copy()
            timeline_data["timestamp"] = pd.to_datetime(
                timeline_data["timestamp"], errors="coerce"
            )
            timeline_data = timeline_data.dropna(subset=["timestamp"])

            if not timeline_data.empty:
                timeline_data["count"] = 1
                timeline = timeline_data.groupby(
                    [pd.Grouper(key="timestamp", freq="1min"), "event"]
                )["count"].sum().unstack(fill_value=0)
                st.line_chart(timeline)
            else:
                st.info("No valid timestamps for timeline.")
else:
    st.info("No driver events recorded yet.")

# ============================================================
# RECENT EVENTS
# ============================================================

st.divider()
st.subheader("📝 Recent Driver Events")

if not events.empty:
    st.dataframe(
        events.tail(20).iloc[::-1],
        width="stretch",
        hide_index=True
    )

    st.download_button(
        label="📥 Download Event Log",
        data=events.to_csv(index=False),
        file_name="driver_events.csv",
        mime="text/csv",
        width="stretch"
    )
else:
    st.info("Waiting for driver events...")

# ============================================================
# EVIDENCE IMAGE GALLERIES
# ============================================================

st.divider()
show_image_gallery(DROWSINESS_DIR, "📸 Drowsiness Evidence", "drowsy")
show_image_gallery(YAWNING_DIR, "📸 Yawning Evidence", "yawning")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
    AI Driver Monitoring System | OpenCV | MediaPipe | Streamlit | Telegram
    </div>
    """,
    unsafe_allow_html=True
)
