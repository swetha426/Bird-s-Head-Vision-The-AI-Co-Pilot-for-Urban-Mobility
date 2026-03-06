
import cv2
import threading
import time
import pyttsx3
import numpy as np
import pandas as pd
from ultralytics import YOLO
from queue import Queue

# ================= CONFIG =================
CAMERA_MAP = {
    1: "Front",
    2: "Left",
    3: "Right",
    4: "Back"
}

CAMERA_INDEXES = [1, 2, 3, 4]

TARGET_FPS = 7
FRAME_DELAY = 1 / TARGET_FPS

CSV_FILE = "detections.csv"

VOICE_INTERVAL = 6
EMERGENCY_INTERVAL = 10

ALERT_OBJECTS = ["person", "car", "bus", "truck", "motorbike"]
EMERGENCY_OBJECTS = ALERT_OBJECTS

running = True
frame_dict = {}
active_cams = []

# ================= YOLO (THREAD SAFE) =================
MODEL = YOLO("yolov8n.pt")
yolo_lock = threading.Lock()   # 🔐 IMPORTANT

# ================= SPEECH SYSTEM (SAFE) =================
speech_queue = Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    while running:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# ================= CSV =================
if not pd.io.common.file_exists(CSV_FILE):
    pd.DataFrame(columns=["Time", "Direction", "Object", "Position", "Emergency"])\
        .to_csv(CSV_FILE, index=False)

def save_csv(direction, obj, position, emergency):
    pd.DataFrame([{
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Direction": direction,
        "Object": obj,
        "Position": position,
        "Emergency": emergency
    }]).to_csv(CSV_FILE, mode="a", header=False, index=False)

# ================= HELPERS =================
def get_position(x, w):
    if x < w / 3:
        return "left"
    elif x > 2 * w / 3:
        return "right"
    return "center"

def is_close(box_h, frame_h):
    return (box_h / frame_h) > 0.55

# ================= CAMERA THREAD =================
def camera_worker(cam_id):
    direction = CAMERA_MAP[cam_id]
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ {direction} NOT AVAILABLE")
        return

    print(f"✅ {direction} ACTIVE")
    last_voice = 0
    last_emergency = 0

    while running:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        with yolo_lock:   # 🔐 CRITICAL FIX
            results = MODEL(frame, verbose=False)

        annotated = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            obj = results[0].names[cls]

            if obj in ALERT_OBJECTS:
                x1, y1, x2, y2 = box.xyxy[0]
                box_h = int(y2 - y1)
                x_center = int((x1 + x2) / 2)
                pos = get_position(x_center, w)

                emergency = False

                if is_close(box_h, h):
                    emergency = True
                    if time.time() - last_emergency > EMERGENCY_INTERVAL:
                        speech_queue.put(
                            f"Emergency. {obj} very close at {direction} {pos}"
                        )
                        last_emergency = time.time()

                elif time.time() - last_voice > VOICE_INTERVAL:
                    speech_queue.put(f"{obj} at {direction} {pos}")
                    last_voice = time.time()

                save_csv(direction, obj, pos, emergency)
                break

        frame_dict[cam_id] = annotated

        delay = FRAME_DELAY - (time.time() - start)
        if delay > 0:
            time.sleep(delay)

    cap.release()
    print(f"🛑 {direction} stopped")

# ================= MAIN =================
def main():
    global running

    print("🔍 Scanning cameras...\n")

    for cam_id in CAMERA_INDEXES:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            active_cams.append(cam_id)
            cap.release()

    if not active_cams:
        print("❌ No cameras found")
        return

    print("🎥 Active Directions:")
    for cam in active_cams:
        print(" -", CAMERA_MAP[cam])
    print()

    for cam_id in active_cams:
        threading.Thread(target=camera_worker, args=(cam_id,), daemon=True).start()

    cv2.namedWindow("Blind Assist – Direction View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blind Assist – Direction View", 900, 700)

    while True:
        frames = []
        for cam_id in CAMERA_INDEXES:
            direction = CAMERA_MAP[cam_id]
            frame = frame_dict.get(cam_id)

            if cam_id not in active_cams or frame is None:
                blank = np.zeros((240, 426, 3), np.uint8)
                cv2.putText(blank, f"{direction} - NO SIGNAL", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                frames.append(blank)
            else:
                cv2.putText(frame, direction, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(cv2.resize(frame, (426, 240)))

        grid = np.vstack((
            np.hstack((frames[0], frames[1])),
            np.hstack((frames[2], frames[3]))
        ))

        cv2.imshow("Blind Assist – Direction View", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            speech_queue.put(None)
            break

    cv2.destroyAllWindows()
    print("🎉 System exited cleanly")

# ================= RUN =================
if __name__ == "__main__":
    main()
