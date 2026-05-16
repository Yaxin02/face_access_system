import torch
import torch.nn as nn
import cv2
import pickle
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# 1. إعدادات الجهاز والمسارات
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/face_classifier_subset.pth"
MAPPING_PATH = "models/class_mapping.pkl"

# 2. تحميل الـ Mapping (ربط الأرقام بالأسامي)
with open(MAPPING_PATH, 'rb') as f:
    class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 3. بناء وتحميل الموديل (FaceNet + Custom Head)
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

# 5. تشغيل الكاميرا (تأكد إنو الـ Index صحيح)
cap = cv2.VideoCapture(1) # بدلها لـ 0 إذا ما حلتش كاميرا الماك
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📸 System Online. Checking for access...")

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

        # --- 🔴 هوني الفلترة الذكية اللي حاشتك بيها 🔴 ---
        # إذا الموديل عرفك بنسبة أكبر من 70%، يطلع اسمك بالأخضر
        if predicted_name == "yassin" and confidence > 70:
            name_to_display = "yassin"
            color = (0, 255, 0) # أخضر مريغل ✅
        else:
            # أي حالة أخرى (شخص غريب، أو كود n*****) تولي فوراً Unknown بالأحمر
            name_to_display = "Unknown"
            color = (0, 0, 255) # أحمر للغرباء ❌

        # رسم المستطيل والكتيبة على الشاشة
        label = f"{name_to_display} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # إذا عرفك أنت بالذات بنسبة قوية، يعطيك الـ Access
        if name_to_display == "yassin" and confidence > 85:
            cv2.putText(frame, "ACCESS GRANTED ✅", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Face Access System - M4", frame)
    
    # اضغط على 'q' للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()