import sys
import cv2
import pickle
import os
import time
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox, QFrame)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# ==========================================
# ⚙️ 1. الإعدادات والمسارات
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATASET_DIR = "dataset"
DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "users_embeddings.pkl")
LOG_FILE_PATH = os.path.join(DATABASE_DIR, "access_log.csv")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

THRESHOLD = 0.65  
STABILIZATION_THRESHOLD = 5 

class FaceAccessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️ ULTRA PRO ACCESS CORE AI - PyQt Edition")
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #050508; color: #FFFFFF; font-family: Courier;")
        
        self.current_mode = "IDLE"  
        self.cap = None
        self.database = {}
        self.enroll_user_name = ""
        self.enroll_saved_count = 0
        self.last_capture_time = 0
        self.last_granted_time = 0
        self.last_denied_time = 0
        self.cooldown_duration = 10
        self.user_counters = {}
        
        self.init_ui()
        self.show()
        QApplication.processEvents()
        self.boot_ai()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #111118; border-right: 2px solid #272732;")
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(20, 40, 20, 20)

        title = QLabel("COMMAND CENTER")
        title.setStyleSheet("color: #00EEFF; font-size: 20px; font-weight: bold; border: none;")
        side_layout.addWidget(title)
        side_layout.addSpacing(30)

        btn_style = """
            QPushButton {
                background-color: #050508; color: #00EEFF; border: 2px solid #00EEFF;
                font-size: 15px; font-weight: bold; padding: 15px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #00EEFF; color: #050508; }
        """
        btn_verify_style = btn_style.replace("#00EEFF", "#00FF66")
        btn_stop_style = btn_style.replace("#00EEFF", "#FF0055")

        self.btn_verify = QPushButton("▶ START SCANNER")
        self.btn_verify.setStyleSheet(btn_verify_style)
        self.btn_verify.clicked.connect(self.start_verification_mode)
        side_layout.addWidget(self.btn_verify)
        side_layout.addSpacing(10)

        self.btn_enroll = QPushButton("➕ ADD IDENTITY")
        self.btn_enroll.setStyleSheet(btn_style)
        self.btn_enroll.clicked.connect(self.start_enrollment_mode)
        side_layout.addWidget(self.btn_enroll)
        side_layout.addSpacing(10)

        self.btn_stop = QPushButton("🛑 STOP ENGINE")
        self.btn_stop.setStyleSheet(btn_stop_style)
        self.btn_stop.clicked.connect(self.return_to_idle)
        side_layout.addWidget(self.btn_stop)
        
        side_layout.addStretch()

        self.lbl_stats = QLabel("SYSTEM LOGS")
        self.lbl_stats.setStyleSheet("color: #94A3B8; font-size: 16px; font-weight: bold; border: none;")
        side_layout.addWidget(self.lbl_stats)

        self.lbl_users = QLabel("Active IDs : 0")
        self.lbl_users.setStyleSheet("color: #FFFFFF; font-size: 14px; border: none;")
        side_layout.addWidget(self.lbl_users)

        self.lbl_log = QLabel("Status: BOOTING...")
        self.lbl_log.setStyleSheet("color: #F5A623; font-size: 14px; border: none; margin-top: 10px;")
        side_layout.addWidget(self.lbl_log)

        main_layout.addWidget(sidebar)

        # --- Main Video Area ---
        view_area = QFrame()
        view_area.setStyleSheet("background-color: #050508; border: none;")
        view_layout = QVBoxLayout(view_area)
        
        self.video_label = QLabel("SYSTEM STANDBY")
        self.video_label.setStyleSheet("color: #374151; font-size: 30px; font-weight: bold;")
        self.video_label.setAlignment(Qt.AlignCenter)
        view_layout.addWidget(self.video_label)
        
        main_layout.addWidget(view_area)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def boot_ai(self):
        print("🧠 Booting Neural Engine on GPU (MPS)...")
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.face_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.load_system_database()
        self.lbl_log.setText("Status: READY")
        self.lbl_log.setStyleSheet("color: #00FF66; font-size: 14px; border: none; margin-top: 10px;")
        print("✅ Engine Online.")

    def load_system_database(self):
        if os.path.exists(DATABASE_PATH) and os.path.getsize(DATABASE_PATH) > 0:
            with open(DATABASE_PATH, 'rb') as f:
                self.database = pickle.load(f)
            self.user_counters = {user: 0 for user in self.database.keys()}
            self.user_counters["Unknown"] = 0
            self.lbl_users.setText(f"Active IDs : {len(self.database)}")
        else:
            self.database = {}

    def return_to_idle(self):
        self.current_mode = "IDLE"
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.video_label.setText("SYSTEM STANDBY")
        self.lbl_log.setText("Status: READY")
        self.lbl_log.setStyleSheet("color: #94A3B8; font-size: 14px; border: none; margin-top: 10px;")

    def start_verification_mode(self):
        self.load_system_database()
        if not self.database:
            QMessageBox.warning(self, "Error", "Database Empty! Add an identity first.")
            return
        self.return_to_idle()
        self.cap = cv2.VideoCapture(1)
        self.current_mode = "VERIFYING"
        self.lbl_log.setText("Status: SCANNING...")
        self.lbl_log.setStyleSheet("color: #00FF66; font-size: 14px; border: none; margin-top: 10px;")
        self.timer.start(30)

    def start_enrollment_mode(self):
        name, ok = QInputDialog.getText(self, "Identity Registration", "Enter Target ID Name:")
        if not ok or not name.strip(): return
        self.return_to_idle()
        
        self.enroll_user_name = name.strip().lower()
        self.enroll_saved_count = 0
        self.user_path = os.path.join(DATASET_DIR, self.enroll_user_name)
        os.makedirs(self.user_path, exist_ok=True)
        
        self.cap = cv2.VideoCapture(1)
        self.current_mode = "ENROLLING"
        self.lbl_log.setText(f"Status: LOCKING {self.enroll_user_name.upper()}")
        self.lbl_log.setStyleSheet("color: #00EEFF; font-size: 14px; border: none; margin-top: 10px;")
        self.timer.start(30)

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (800, 600))
        
        # نسخة نظيفة للذكاء الاصطناعي
        clean_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

        if len(faces) == 0:
            for k in self.user_counters: self.user_counters[k] = 0

        margin = 45 

        if self.current_mode == "ENROLLING":
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                
                face_img = clean_frame[y1:y2, x1:x2]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 238, 0), 2)
                cv2.putText(frame, f"ACQUIRING: {self.enroll_saved_count}/150", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 238, 0), 2)
                
                if time.time() - self.last_capture_time >= 0.04 and self.enroll_saved_count < 150:
                    self.enroll_saved_count += 1
                    cv2.imwrite(os.path.join(self.user_path, f"{self.enroll_user_name}_{self.enroll_saved_count:03d}.jpg"), face_img)
                    self.last_capture_time = time.time()
                
                if self.enroll_saved_count >= 150:
                    self.return_to_idle()
                    self.background_db_update()

        elif self.current_mode == "VERIFYING":
            for (x, y, w, h) in faces:
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                
                face_img = clean_frame[y1:y2, x1:x2]
                
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(face_pil).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        embedding = self.face_model(input_tensor).squeeze().cpu().numpy()
                    embedding = embedding / np.linalg.norm(embedding)

                    best_match, min_dist = "Unknown", float("inf")
                    for user, data in self.database.items():
                        dist = np.linalg.norm(embedding - data["embedding"])
                        if dist < min_dist:
                            min_dist = dist
                            if dist < THRESHOLD: best_match = user

                    for k in self.user_counters:
                        if k == best_match: self.user_counters[k] += 1
                        else: self.user_counters[k] = 0

                    if best_match != "Unknown":
                        color = (102, 255, 0) # Green BGR
                        # 🎯 التعديل: إظهار الاسم فقط بدون Score
                        label = f"ID: {best_match.upper()}"
                        if self.user_counters[best_match] >= STABILIZATION_THRESHOLD:
                            cv2.putText(frame, "ACCESS GRANTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            if time.time() - self.last_granted_time > self.cooldown_duration:
                                self.lbl_log.setText(f"Status: GRANTED\nTarget: {best_match.upper()}")
                                self.lbl_log.setStyleSheet("color: #00FF66; font-size: 14px; border: none; margin-top: 10px;")
                                os.system(f"say 'Welcome {best_match}, access granted' &")
                                self.last_granted_time = time.time()
                    else:
                        color = (85, 0, 255) # Red BGR
                        # 🎯 التعديل: إظهار UNKNOWN فقط بدون Score
                        label = "UNKNOWN"
                        if self.user_counters["Unknown"] >= STABILIZATION_THRESHOLD:
                            cv2.putText(frame, "ACCESS DENIED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            if time.time() - self.last_denied_time > self.cooldown_duration:
                                self.lbl_log.setText("Status: DENIED\nTarget: UNKNOWN")
                                self.lbl_log.setStyleSheet("color: #FF0055; font-size: 14px; border: none; margin-top: 10px;")
                                os.system(f"say 'Access denied' &")
                                self.last_denied_time = time.time()

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception:
                    pass

        # تحويل الصورة للعرض في PyQt5
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def background_db_update(self):
        self.lbl_log.setText("Status: COMPILING DATA...\nPlease wait.")
        self.lbl_log.setStyleSheet("color: #00EEFF; font-size: 14px; border: none; margin-top: 10px;")
        QApplication.processEvents()

        user_folder = os.path.join(DATASET_DIR, self.enroll_user_name)
        images = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        user_embeddings = []
        for img_name in images:
            img_path = os.path.join(user_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32)
                img_array = (img_array - 127.5) / 128.0
                img_array = np.transpose(img_array, (2, 0, 1))
                tensor = torch.tensor(img_array).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    emb = self.face_model(tensor).squeeze().cpu().numpy()
                emb = emb / np.linalg.norm(emb)
                user_embeddings.append(emb)
            except Exception:
                continue

        if user_embeddings:
            avg_emb = np.mean(user_embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            
            if os.path.exists(DATABASE_PATH):
                with open(DATABASE_PATH, 'rb') as f:
                    self.database = pickle.load(f)
            
            self.database[self.enroll_user_name] = {"embedding": avg_emb, "num_images": len(user_embeddings)}
            with open(DATABASE_PATH, "wb") as f:
                pickle.dump(self.database, f)
            
            self.load_system_database()
            self.lbl_log.setText("Status: READY")
            self.lbl_log.setStyleSheet("color: #00FF66; font-size: 14px; border: none; margin-top: 10px;")
            QMessageBox.information(self, "System Update", f"Identity '{self.enroll_user_name.upper()}' integrated into core database.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceAccessApp()
    sys.exit(app.exec_())