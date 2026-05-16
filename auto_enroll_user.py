import cv2
import os
import time

CAMERA_INDEX = 1
# 🎯 التعديل الأول: التوجيه مباشرة للمجلد الكبير في الـ SSD 🎯
DATASET_DIR = "/Volumes/HIKSEMI/train"

# 🎯 التعديل الثاني: رفع العينات لـ 150 لضمان قوة ذاكرة الموديل 🎯
TOTAL_SAMPLES = 150  
CAPTURE_DELAY = 0.1  # تسريع الوقت بين اللقطات لأن العدد كبر

user_name = input("Enter user name (Write: yassin): ").strip().lower()

if user_name == "":
    print("Error: user name cannot be empty.")
    exit()

user_path = os.path.join(DATASET_DIR, user_name)
os.makedirs(user_path, exist_ok=True)

existing_images = [
    file for file in os.listdir(user_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_count = len(existing_images)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"Error: Camera index {CAMERA_INDEX} not opened.")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Camera opened successfully.")
print("Auto enrollment started.")
print("🔥 Look at the camera and move your head slowly (Left, Right, Up, Down) 🔥")
print("Press q to quit.")

last_capture_time = 0
saved_this_session = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    status_text = "Position your face"
    status_color = (255, 255, 255)

    if len(faces) > 0:
        # Choose biggest face
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
        x, y, w, h = faces[0]

        margin = 45
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_img = frame[y1:y2, x1:x2]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        current_time = time.time()

        if saved_this_session < TOTAL_SAMPLES:
            if current_time - last_capture_time >= CAPTURE_DELAY:
                image_count += 1
                saved_this_session += 1

                image_name = f"{user_name}_{image_count:03d}.jpg"
                image_path = os.path.join(user_path, image_name)

                # حفظ الصورة مقصوصة مباشرة داخل الـ SSD
                cv2.imwrite(image_path, face_img)
                last_capture_time = current_time

                print(f"Saved to SSD: {image_path}")

            status_text = f"Scanning... {saved_this_session}/{TOTAL_SAMPLES}"
            status_color = (0, 255, 255)
        else:
            status_text = "Enrollment completed"
            status_color = (0, 255, 0)

    cv2.putText(
        frame,
        f"User: {user_name}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "Move head slowly: left, right, up, down",
        (20, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Auto Enroll User", frame)

    key = cv2.waitKey(1) & 0xFF

    if saved_this_session >= TOTAL_SAMPLES:
        time.sleep(1)
        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Enrollment finished for user: {user_name}")
print(f"Total images now in SSD for this user: {image_count}")