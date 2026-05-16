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

# 2. تحميل الـ Mapping
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

# 5. متغيرات الـ Cooldown والـ Stabilization
last_granted_time = 0
last_denied_time = 0
cooldown_duration = 10  # 10 ثواني مهلة بين الإشعارات الصوتية

# --- 🎯 عدادات التثبيت لمنع التداخل عند التشغيل 🎯 ---
yassin_frames_counter = 0
unknown_frames_counter = 0
STABILIZATION_THRESHOLD = 5  # يجب أن يتكرر التوقع 5 مرات متتالية ليطلق الصوت

def process_access_granted():
    """دالة مستقلة للترحيب بياسين وتسجيله"""
    global last_granted_time
    current_time = time.time()
    
    if current_time - last_granted_time > cooldown_duration:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        file_exists = os.path.isfile(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, "a") as f:
            if not file_exists:
                f.write("Name,Date,Time\n")
            f.write(f"yassin,{date_str},{time_str}\n")
            
        print(f"📝 Logged: yassin at {time_str}")
        os.system("say 'Welcome Yassin, access granted' &")
        last_granted_time = current_time

def process_access_denied():
    """دالة مستقلة تماماً لرفض الغرباء"""
    global last_denied_time
    current_time = time.time()
    
    if current_time - last_denied_time > cooldown_duration:
        print("❌ Access Denied Logged")
        os.system("say 'Access denied' &")
        last_denied_time = current_time

# 6. تشغيل الكاميرا
cap = cv2.VideoCapture(1) # 1 لكاميرا الماك
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📸 System Online. Frame Stabilization Active (5 frames requirement)...")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    # إذا ما فماش وجوه في الكاميرا، صفر العدادات فوراً
    if len(faces) == 0:
        yassin_frames_counter = 0
        unknown_frames_counter = 0

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_name = idx_to_class[pred_idx.item()]
            confidence = conf.item() * 100

        # التثبت الشرطي وتحديث العدادات
        if predicted_name == "yassin" and confidence > 75:
            yassin_frames_counter += 1
            unknown_frames_counter = 0  # إلغاء الـ Unknown
            
            name_to_display = "yassin"
            color = (0, 255, 0)
            
            # ما يتكلم كان ما يتأكد 5 فريمات ورا بعضهم
            if yassin_frames_counter >= STABILIZATION_THRESHOLD:
                cv2.putText(frame, "ACCESS GRANTED ✅", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                process_access_granted()
        else:
            unknown_frames_counter += 1
            yassin_frames_counter = 0  # إلغاء الـ Yassin
            
            name_to_display = "Unknown"
            color = (0, 0, 255)
            
            # ما يتكلم كان ما يتأكد 5 فريمات ورا بعضهم
            if unknown_frames_counter >= STABILIZATION_THRESHOLD:
                process_access_denied()

        # رسم الإطار والنسبة
        label = f"{name_to_display} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Access System - M4", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()