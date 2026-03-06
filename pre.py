import cv2
import time
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO

# ================= CONFIG ================= #

VIDEOS = {
    "front": r"C:\Users\yeluv\OneDrive\Desktop\front.mp4",
    "left":  r"C:\Users\yeluv\OneDrive\Desktop\left.mp4",
    "right": r"C:\Users\yeluv\OneDrive\Desktop\right2.mp4",
    "rear":  r"C:\Users\yeluv\OneDrive\Desktop\back.mp4"
}

GRID_W, GRID_H = 320, 240
CONF_THRESHOLD = 0.6
TARGET_FPS = 10  # Desired FPS
FRAME_DELAY = 1.0 / TARGET_FPS
REPEAT_INTERVAL = 5.0

# ========================================= #

model = YOLO("yolov8n.pt")

speech_lock = threading.Lock()
alert_state = {}
previous_sizes = {}

# ========================================= #

def get_depth(box_h, frame_h):
    ratio = box_h / frame_h
    if ratio > 0.55:
        return "very close"
    elif ratio > 0.3:
        return "nearby"
    return "far"

def motion_state(curr, prev):
    if prev is None:
        return "unknown"
    if curr > prev * 1.15:
        return "approaching"
    elif curr < prev * 0.85:
        return "receding"
    return "static"

def speak(text):
    if speech_lock.acquire(False):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()
            speech_lock.release()

def should_repeat(key):
    now = time.time()
    if now - alert_state.get(key, 0) >= REPEAT_INTERVAL:
        alert_state[key] = now
        return True
    return False

def detect(frame, direction):
    detections = []
    h, _, _ = frame.shape
    results = model(frame, verbose=False)

    for r in results:
        names = r.names
        for box in r.boxes:
            if box.conf >= CONF_THRESHOLD:
                cls = int(box.cls[0])
                name = names[cls]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                box_h = y2 - y1

                detections.append({
                    "name": name,
                    "size": box_h,
                    "depth": get_depth(box_h, h),
                    "direction": direction
                })

    return detections, results[0].plot()

# ========================================= #

def main():
    caps = {d: cv2.VideoCapture(p) for d, p in VIDEOS.items()}
    last_frame_time = 0

    while True:
        now = time.time()
        if now - last_frame_time < FRAME_DELAY:
            continue  # Skip frames to limit FPS
        last_frame_time = now

        frames = {}
        detections_all = []

        for direction, cap in caps.items():
            ret, frame = cap.read()

            if not ret:
                blank = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
                cv2.putText(blank, f"{direction.upper()} VIDEO END",
                            (20, GRID_H//2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,255), 2)
                frames[direction] = blank
                continue

            frame = cv2.resize(frame, (GRID_W, GRID_H))
            detections, annotated = detect(frame, direction)
            frames[direction] = annotated

            for d in detections:
                key = (direction, d["name"])
                prev = previous_sizes.get(key)
                d["motion"] = motion_state(d["size"], prev)
                previous_sizes[key] = d["size"]
                detections_all.append(d)

        # ===== DISPLAY GRID ===== #
        blank = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
        grid = np.vstack([
            np.hstack([frames.get("front", blank), frames.get("left", blank)]),
            np.hstack([frames.get("right", blank), frames.get("rear", blank)])
        ])
        cv2.imshow("360 View (Press Q to Exit)", grid)

        # ===== ALERT LOGIC ===== #
        for d in detections_all:
            obj = d["name"]
            direction = d["direction"]
            key = (direction, obj)

            # Skip non-approaching objects
            if d["motion"] != "approaching":
                continue

            # Alert only if object is close enough
            if d["depth"] not in ["nearby", "very close"]:
                continue

            # Front alert
            if direction == "front" and should_repeat(key):
                speak(f"A {obj} is approaching from the front.")

            # Side alerts
            elif direction in ["left", "right"] and should_repeat(key):
                speak(f"A {obj} is approaching from the {direction}.")

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

# ========================================= #

if __name__ == "__main__":
    main()
