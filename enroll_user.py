import cv2
import os
import time
from pathlib import Path

# --- الإعدادات ---
# 1 غالبا Mac camera عندك | 0 غالبا iPhone camera (Continuity Camera)
CAMERA_INDEX = 1 

# المسار الجديد مباشرة للـ SSD (عوضا عن الفولدر المحلي)
DATASET_ROOT = "/Volumes/HIKSEMI/train_subset"

# --- البداية ---
user_name = input("Enter user name (e.g., yassin): ").strip().lower()

if user_name == "":
    print("❌ Error: user name cannot be empty.")
    exit()

# صنع مسار المستخدم داخل الـ SSD
user_path = os.path.join(DATASET_ROOT, user_name)
os.makedirs(user_path, exist_ok=True)

# فتح الكاميرا (Mac M4 Camera)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"❌ Error: Camera index {CAMERA_INDEX} not opened.")
    print("Try changing CAMERA_INDEX to 0 or 2 in the code.")
    exit()

# تحميل كاشف الوجوه (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("✅ Camera opened successfully.")
print(f"📂 Saving images to: {user_path}")
print("-" * 30)
print("Instructions:")
print("1. Press 's' to capture a face image.")
print("2. Move your head slightly (left, right, up, down) for better training.")
print("3. Aim for at least 30-50 images.")
print("4. Press 'q' to quit when finished.")
print("-" * 30)

# حساب عدد التصاور الموجودة حالياً (بناءً على الملفات في الـ SSD)
image_count = len([
    file for file in os.listdir(user_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
])

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Cannot read frame.")
        break

    # قلب الصورة (Mirror) لتسهيل الحركة
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # تحويل الصورة للرمادي للكشف عن الوجه
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # رسم مستطيل حول الوجه المكتشف
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, "Face Ready", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # معلومات على الشاشة
    cv2.putText(display_frame, f"User: {user_name}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Count: {image_count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Enrollment - Mac M4", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # الضغط على 's' للحفظ
    if key == ord("s"):
        if len(faces) == 0:
            print("⚠️ No face detected! Please center your face.")
            continue
        
        if len(faces) > 1:
            print("⚠️ Too many faces! Stay alone in the frame.")
            continue

        # قص الوجه مع مساحة صغيرة (Margin)
        x, y, w, h = faces[0]
        margin = 30
        y1, y2 = max(0, y-margin), min(frame.shape[0], y+h+margin)
        x1, x2 = max(0, x-margin), min(frame.shape[1], x+w+margin)
        face_crop = frame[y1:y2, x1:x2]

        # حفظ الصورة في الـ SSD
        image_count += 1
        img_filename = f"{user_name}_{image_count:03d}.jpg"
        save_path = os.path.join(user_path, img_filename)
        
        cv2.imwrite(save_path, face_crop)
        print(f"📸 Saved [{image_count}]: {save_path}")

    # الضغط على 'q' للخروج
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f"✅ Finished! Total images for {user_name}: {image_count}")
print(f"🚀 Now run 'python3 src/04_train_model.py' to update your model.")