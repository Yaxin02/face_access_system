import torch
import cv2
import pickle
import os
import time
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from project_paths import DATABASE_PATH, LOG_FILE_PATH, setup_sys_path

setup_sys_path()

# 1. إعدادات الجهاز والمسارات
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATABASE_PATH = str(DATABASE_PATH)
LOG_FILE_PATH = str(LOG_FILE_PATH)

# حد المسافة الإقليدية (Threshold) - كل ما تصغره يولي السيستيم صارم أكثر
THRESHOLD = 0.68  

if not os.path.exists(DATABASE_PATH):
    print(f"❌ Error: Database file not found at {DATABASE_PATH}.")
    print("👉 Please run: python3 notebooks/build_database.py first!")
    exit()

# 2. تحميل قاعدة أوزان المستخدمين المحليين
with open(DATABASE_PATH, 'rb') as f:
    database = pickle.load(f)

print(f"👥 Active Authorized Users in DB: {list(database.keys())}")

# 3. تحميل الـ Backbone الأصلي النظيف (بدون طبقة الـ Classifier المشوشة)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

last_granted_time = 0
cap = cv2.VideoCapture(1) # 1 لكاميرا الماك
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📸 Dynamic Face Verification System Online...")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

        # استخراج البصمة الرقمية الحالية لوجهك
        with torch.no_grad():
            embedding = model(input_tensor).squeeze().cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)

        best_match = "Unknown"
        min_dist = float("inf")

        # المقارنة الحقيقية مع الـ Database الديناميكية
        for user, data in database.items():
            db_embedding = data["embedding"]
            dist = np.linalg.norm(embedding - db_embedding)
            if dist < min_dist:
                min_dist = dist
                if dist < THRESHOLD:
                    best_match = user

        # تحديث الواجهة والنطق بالاسم
        if best_match != "Unknown":
            color = (0, 255, 0)  # أخضر لوجه معروف
            label = f"{best_match.upper()} (Dist: {min_dist:.2f})"

            current_time = time.time()
            if current_time - last_granted_time > 10:
                now = datetime.now()
                with open(LOG_FILE_PATH, "a") as f:
                    f.write(f"{best_match},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
                
                print(f"📝 Access Granted: {best_match}")
                os.system(f"say 'Welcome {best_match}' &")
                last_granted_time = current_time
        else:
            color = (0, 0, 255)  # أحمر للغرباء
            label = f"Unknown (Best Dist: {min_dist:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Dynamic Face Access Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()