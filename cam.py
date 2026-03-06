import cv2

CAM_INDEX = 3  # change this to 0, 1, 2, 3 ...

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"❌ Camera {CAM_INDEX} NOT working")
    exit()

print(f"✅ Camera {CAM_INDEX} is working")
print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow(f"Camera Test - Index {CAM_INDEX}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()