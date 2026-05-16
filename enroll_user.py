import cv2
import os
import time
import importlib

CAMERA_INDEX = 1
DATASET_DIR = "dataset"  # المجلد المحلي للمصرح لهم

TOTAL_SAMPLES = 150  
CAPTURE_DELAY = 0.05  # تسريع دفق اللقطات

user_name = input("👤 Enter new user name: ").strip().lower()

if user_name == "":
    print("❌ Error: user name cannot be empty.")
    exit()

# 🎯 أتمتة 1: صنع المجلد تلقائياً بالاسم الجديد داخل dataset
user_path = os.path.join(DATASET_DIR, user_name)
os.makedirs(user_path, exist_ok=True)

existing_images = [
    file for file in os.listdir(user_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]
image_count = len(existing_images)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print(f"❌ Error: Camera index {CAMERA_INDEX} not opened.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("\n📸 Camera opened successfully. Look at the camera and move your head slowly...")
last_capture_time = 0
saved_this_session = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    status_text = "Position your face"
    status_color = (255, 255, 255)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
        x, y, w, h = faces[0]

        margin = 45
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_img = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        current_time = time.time()
        if saved_this_session < TOTAL_SAMPLES:
            if current_time - last_capture_time >= CAPTURE_DELAY:
                image_count += 1
                saved_this_session += 1

                image_name = f"{user_name}_{image_count:03d}.jpg"
                image_path = os.path.join(user_path, image_name)

                cv2.imwrite(image_path, face_img)
                last_capture_time = current_time

            status_text = f"Scanning... {saved_this_session}/{TOTAL_SAMPLES}"
            status_color = (0, 255, 255)
        else:
            status_text = "Enrollment completed"
            status_color = (0, 255, 0)

    cv2.putText(frame, f"User: {user_name}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.imshow("Auto Enroll User", frame)

    if saved_this_session >= TOTAL_SAMPLES or (cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Enrollment finished for user: {user_name} ({saved_this_session} images saved).")

# 🎯 أتمتة 2: تشغيل الـ Database Builder تلقائياً في الخلفية لتحديث الـ pkl فوراً 🎯
print("\n⚙️ Auto-triggering Database compilation... Please wait...")
try:
    build_db_module = importlib.import_module("src.03_build_database")
    build_db_module.build_database()
    print("\n🚀 All Done! New user is now live in the system.")
except Exception as e:
    print(f"⚠️ Could not auto-update database: {e}")