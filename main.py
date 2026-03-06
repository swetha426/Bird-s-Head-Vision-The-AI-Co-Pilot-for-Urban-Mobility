import cv2
import threading
import time
import os
import pyttsx3
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ================= CONFIG =================
MAX_CAMERAS = 5
TARGET_FPS = 7
FRAME_DELAY = 1 / TARGET_FPS

MODEL = YOLO("yolov8n.pt")

CAPTURE_DIR = "captures"
CSV_FILE = "detections.csv"

VOICE_INTERVAL = 6
EMERGENCY_INTERVAL = 10
BLIND_ASSIST = True

ALERT_OBJECTS = ["person", "car", "bus", "truck", "motorbike"]
EMERGENCY_OBJECTS = ["person", "car", "bus", "truck", "motorbike"]

os.makedirs(CAPTURE_DIR, exist_ok=True)

speech_locks = {}
last_voice_time = {}
last_emergency_time = {}
frame_dict = {}
running = True

# ================= CSV SETUP =================
if not os.path.exists(CSV_FILE):
    pd.DataFrame(
        columns=["Time", "Camera", "Object", "Position", "Emergency"]
    ).to_csv(CSV_FILE, index=False)

# ================= HELPERS =================
def get_position(x_center, width):
    if x_center < width / 3:
        return "left"
    elif x_center > 2 * width / 3:
        return "right"
    return "center"

def is_very_close(box_height, frame_height):
    return (box_height / frame_height) > 0.55

def speak(text, cam_id):
    if not speech_locks[cam_id].acquire(blocking=False):
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    finally:
        speech_locks[cam_id].release()

def save_csv(cam_id, obj, position, emergency):
    pd.DataFrame([{
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Camera": cam_id,
        "Object": obj,
        "Position": position,
        "Emergency": emergency
    }]).to_csv(CSV_FILE, mode="a", header=False, index=False)

# ================= CAMERA THREAD =================
def camera_worker(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Camera {cam_id} NOT AVAILABLE")
        return

    print(f"✅ Camera {cam_id} ACTIVE")

    speech_locks[cam_id] = threading.Lock()
    last_voice_time[cam_id] = 0
    last_emergency_time[cam_id] = 0

    while running:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        results = MODEL(frame, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            obj = results[0].names[cls_id]

            if obj in ALERT_OBJECTS:
                x1, y1, x2, y2 = box.xyxy[0]
                box_height = int(y2 - y1)
                x_center = int((x1 + x2) / 2)
                position = get_position(x_center, w)

                emergency = False

                if obj in EMERGENCY_OBJECTS and is_very_close(box_height, h):
                    emergency = True
                    if time.time() - last_emergency_time[cam_id] > EMERGENCY_INTERVAL:
                        threading.Thread(
                            target=speak,
                            args=(f"Camera {cam_id}. Emergency {obj} close on {position}", cam_id),
                            daemon=True
                        ).start()
                        last_emergency_time[cam_id] = time.time()

                elif BLIND_ASSIST and time.time() - last_voice_time[cam_id] > VOICE_INTERVAL:
                    threading.Thread(
                        target=speak,
                        args=(f"Camera {cam_id}. {obj} on {position}", cam_id),
                        daemon=True
                    ).start()
                    last_voice_time[cam_id] = time.time()

                save_csv(cam_id, obj, position, emergency)
                break

        frame_dict[cam_id] = annotated

        elapsed = time.time() - start_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

    cap.release()
    print(f"🛑 Camera {cam_id} stopped")

# ================= MAIN =================
def main():
    global running
    print("🔍 Scanning cameras...\n")

    active_cams = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            active_cams.append(i)
            cap.release()

    print(f"🎥 Active Cameras Found: {active_cams}\n")

    for cam_id in active_cams:
        threading.Thread(
            target=camera_worker,
            args=(cam_id,),
            daemon=True
        ).start()

    cv2.namedWindow("5-Camera Blind Assist", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("5-Camera Blind Assist", 1100, 900)

    while True:
        frames = []

        for i in range(MAX_CAMERAS):
            frame = frame_dict.get(i)
            if frame is None:
                blank = np.zeros((240, 426, 3), dtype=np.uint8)
                cv2.putText(blank, f"CAM {i} - NO SIGNAL", (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frames.append(blank)
            else:
                frames.append(cv2.resize(frame, (426, 240)))

        row1 = np.hstack((frames[0], frames[1]))
        row2 = np.hstack((frames[2], frames[3]))
        row3 = np.hstack((frames[4], np.zeros((240, 426, 3), dtype=np.uint8)))

        grid = np.vstack((row1, row2, row3))
        cv2.imshow("5-Camera Blind Assist", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()
    print("🎉 System exited cleanly")

# ================= RUN =================
if __name__ == "__main__":
    main()
