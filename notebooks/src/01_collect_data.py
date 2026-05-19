import cv2
import os
import time
import sys
from pathlib import Path

# باش Python يلقى config/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config.paths import RAW_DIR

CAMERA_INDEX = 1
TOTAL_IMAGES = 300
CAPTURE_DELAY = 0.10

class_name = input("Enter class name, example yassin: ").strip().lower()

if class_name == "":
    print("Class name cannot be empty.")
    exit()

save_dir = RAW_DIR / class_name
save_dir.mkdir(parents=True, exist_ok=True)

existing_images = [
    f for f in os.listdir(save_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_count = len(existing_images)
saved_this_session = 0
last_capture_time = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"Error: Camera index {CAMERA_INDEX} not opened.")
    print("Try changing CAMERA_INDEX to 0.")
    exit()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Camera opened.")
print(f"Saving raw images to: {save_dir}")
print(f"Target images this session: {TOTAL_IMAGES}")
print("Move your head slowly: front, left, right, up, down, close, far.")
print("Press q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    status = "Position your face"
    color = (255, 255, 255)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = time.time()

        if saved_this_session < TOTAL_IMAGES:
            if current_time - last_capture_time >= CAPTURE_DELAY:
                image_count += 1
                saved_this_session += 1

                image_name = f"{class_name}_{image_count:05d}.jpg"
                image_path = save_dir / image_name

                # raw frame كامل، preprocessing نعملوه بعد
                cv2.imwrite(str(image_path), frame)

                last_capture_time = current_time
                print(f"Saved: {image_path}")

            status = f"Collecting: {saved_this_session}/{TOTAL_IMAGES}"
            color = (0, 255, 255)
        else:
            status = "Collection completed"
            color = (0, 255, 0)

    cv2.putText(frame, f"Class: {class_name}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, status, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, "Raw Data Collection - saved to USB flash", (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if saved_this_session >= TOTAL_IMAGES:
        time.sleep(1)
        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Data collection finished.")
print(f"Class: {class_name}")
print(f"Saved this session: {saved_this_session}")
print(f"Total images in class folder: {image_count}")
print(f"Folder: {save_dir}")