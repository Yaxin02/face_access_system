import torch
import torch.nn as nn
import cv2
import pickle
import os
import time
from datetime import datetime
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# 1. إعدادات الجهاز والمسارات
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/face_classifier_subset.pth"
MAPPING_PATH = "models/class_mapping.pkl"
LOG_FILE_PATH = "database/access_log.csv"

# صنع مجلد الـ database إذا موش موجود
os.makedirs("database", exist_ok=True)

# 2. تحميل الـ Mapping (ربط الأرقام بالأسامي)
with open(MAPPING_PATH, 'rb') as f:
    class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 3. بناء وتحميل الموديل
base_model = InceptionResnetV1(pretrained='vggface2').to(DEVICE)
num_classes = len(idx_to_class)

class FaceClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        embeddings = self.base_model(x)
        return self.classifier(embeddings)

model = FaceClassifier(base_model, num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 4. تحضير الصورة (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 5. متغيرات الـ Logging لمنع التكرار
last_logged_time = 0
logging_cooldown = 10  # 10 ثواني مهلة بين كل تسجيل حضور

def log_access(name):
    """دالة لتسجيل الحضور في ملف CSV"""
    global last_logged_time
    current_time = time.time()
    
    if current_time - last_logged_time > logging_cooldown:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, "a") as f:
            if not file_exists:
                f.write("Name,Date,Time\n")
            f.write(f"{name},{date_str},{time_str}\n")
            
        print(f"📝 Logged: {name} at {time_str}")
        last_logged_time = current_time

# 6. تشغيل الكاميرا
cap = cv2.VideoCapture(1) # 1 لكاميرا الماك
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📸 System Online. Smooth Access Threshold active (75%)...")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # قص الوجه وتحضيره للموديل
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        # التوقع الحسابي (Prediction)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_name = idx_to_class[pred_idx.item()]
            confidence = conf.item() * 100

        # --- الفلترة الذكية المحدثة ---
        if predicted_name == "yassin" and confidence > 70:
            name_to_display = "yassin"
            color = (0, 255, 0) # أخضر ✅
        else:
            name_to_display = "Unknown"
            color = (0, 0, 255) # أحمر ❌

        # رسم النتائج على الشاشة
        label = f"{name_to_display} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # --- 🎯 التعديل الجديد: تم خفض نسبة الـ Access لـ 75% لمرونة كاملة 🎯 ---
        if name_to_display == "yassin" and confidence > 75:
            cv2.putText(frame, "ACCESS GRANTED ✅", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            log_access("yassin")

    cv2.imshow("Face Access System - M4", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()