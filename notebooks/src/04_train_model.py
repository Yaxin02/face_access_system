import torch
import torch.nn as nn
import cv2
import pickle
import os
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime
from PIL import Image, ImageTk
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# ==========================================
# ⚙️ 1. الإعدادات والمسارات العامة
# ==========================================
import sys
from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NOTEBOOKS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NOTEBOOKS_ROOT))

from project_paths import DATASET_DIR, DATABASE_DIR, DATABASE_PATH, LOG_FILE_PATH, ensure_data_dirs

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATASET_DIR = str(DATASET_DIR)
DATABASE_DIR = str(DATABASE_DIR)
DATABASE_PATH = str(DATABASE_PATH)
LOG_FILE_PATH = str(LOG_FILE_PATH)

ensure_data_dirs()

# ألوان الـ Dashboard الاحترافي
COLOR_BG = "#0F0F12"       
COLOR_CARD = "#16161D"     
COLOR_TEXT = "#E2E8F0"     
COLOR_ACCENT = "#00E676"   
COLOR_CYAN = "#00B0FF"     
COLOR_RED = "#FF1744"      

class FaceAccessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO ACCESS CORE AI v2.0")
        # 🎯 السطر المصلح بالظبط هوني 🎯
        self.root.geometry("1100x700")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # الحالات الداخلية للسيستيم
        self.current_mode = "IDLE"  
        self.cap = None
        self.database = {}
        self.enroll_user_name = ""
        self.enroll_saved_count = 0
        self.last_capture_time = 0
        self.last_granted_time = 0
        self.user_counters = {}
        self.STABILIZATION_THRESHOLD = 5
        self.THRESHOLD = 0.65

        # تحميل الـ AI Engine
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.face_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.load_system_database()
        self.build_ui_dashboard()
        self.update_video_stream()

    def load_system_database(self):
        if os.path.exists(DATABASE_PATH) and os.path.getsize(DATABASE_PATH) > 0:
            with open(DATABASE_PATH, 'rb') as f:
                self.database = pickle.load(f)
            self.user_counters = {user: 0 for user in self.database.keys()}
            self.user_counters["Unknown"] = 0
        else:
            self.database = {}

    def build_ui_dashboard(self):
        # --- Header ---
        header = tk.Frame(self.root, bg=COLOR_CARD, height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="🛡️ PRO ACCESS CONTROL CORE AI", font=("Helvetica", 16, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD)
        title_label.pack(side=tk.LEFT, padx=20)
        
        self.status_pill = tk.Label(header, text="● SYSTEM IDLE", font=("Helvetica", 11, "bold"), fg="#94A3B8", bg="#1E293B", padx=10, pady=4)
        self.status_pill.pack(side=tk.RIGHT, padx=20)

        # --- Side Panel ---
        side_panel = tk.Frame(self.root, bg=COLOR_CARD, width=300)
        side_panel.pack(fill=tk.Y, side=tk.LEFT, padx=15, pady=15)
        side_panel.pack_propagate(False)

        lbl_actions = tk.Label(side_panel, text="COMMAND CENTER", font=("Helvetica", 12, "bold"), fg=COLOR_CYAN, bg=COLOR_CARD)
        lbl_actions.pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.btn_verify = tk.Button(side_panel, text="🚀 START VERIFICATION", font=("Helvetica", 11, "bold"), bg=COLOR_BG, fg=COLOR_ACCENT, bd=1, relief=tk.FLAT, height=2, command=self.start_verification_mode)
        self.btn_verify.pack(fill=tk.X, padx=15, pady=6)

        self.btn_enroll = tk.Button(side_panel, text="➕ ENROLL NEW FACE", font=("Helvetica", 11, "bold"), bg=COLOR_BG, fg=COLOR_CYAN, bd=1, relief=tk.FLAT, height=2, command=self.start_enrollment_mode)
        self.btn_enroll.pack(fill=tk.X, padx=15, pady=6)

        self.btn_stop = tk.Button(side_panel, text="🛑 STOP CAMERA", font=("Helvetica", 11, "bold"), bg=COLOR_BG, fg=COLOR_RED, bd=1, relief=tk.FLAT, height=2, command=self.return_to_idle)
        self.btn_stop.pack(fill=tk.X, padx=15, pady=6)

        separator = tk.Frame(side_panel, bg="#272732", height=1)
        separator.pack(fill=tk.X, padx=15, pady=20)

        lbl_stats = tk.Label(side_panel, text="SYSTEM INTELLIGENCE", font=("Helvetica", 12, "bold"), fg="#94A3B8", bg=COLOR_CARD)
        lbl_stats.pack(anchor=tk.W, padx=15, pady=(0, 10))

        self.lbl_total_users = tk.Label(side_panel, text=f"👥 Total Identities: {len(self.database)}", font=("Helvetica", 11), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_total_users.pack(anchor=tk.W, padx=15, pady=4)

        self.lbl_last_log = tk.Label(side_panel, text="📝 Log Feed: Waiting scan...", font=("Helvetica", 10), fg="#94A3B8", bg=COLOR_CARD, justify=tk.LEFT)
        self.lbl_last_log.pack(anchor=tk.W, padx=15, pady=4)

        # --- Viewport ---
        self.view_frame = tk.Frame(self.root, bg="#000000", bd=2, highlightbackground="#272732", highlightthickness=1)
        self.view_frame.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT, padx=(0, 15), pady=15)

        self.video_canvas = tk.Canvas(self.view_frame, bg="#050508", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

    def start_verification_mode(self):
        self.load_system_database()
        if not self.database:
            messagebox.showwarning("Database Empty", "Please enroll at least one face first!")
            return
        self.return_to_idle()
        self.cap = cv2.VideoCapture(1)
        self.current_mode = "VERIFYING"
        self.status_pill.configure(text="● LIVE SCANNING", fg=COLOR_ACCENT, bg="#062F1D")
        self.lbl_total_users.configure(text=f"👥 Total Identities: {len(self.database)}")

    def start_enrollment_mode(self):
        name = simpledialog.askstring("Face Enrollment", "Enter name for new user:").strip().lower()
        if not name:
            return
        self.return_to_idle()
        
        self.enroll_user_name = name
        self.enroll_saved_count = 0
        self.user_path = os.path.join(DATASET_DIR, name)
        os.makedirs(self.user_path, exist_ok=True)
        
        self.cap = cv2.VideoCapture(1)
        self.current_mode = "ENROLLING"
        self.status_pill.configure(text=f"● ENROLLING: {name.upper()}", fg=COLOR_CYAN, bg="#0A2F44")

    def return_to_idle(self):
        self.current_mode = "IDLE"
        self.status_pill.configure(text="● SYSTEM IDLE", fg="#94A3B8", bg="#1E293B")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_canvas.delete("all")

    def update_video_stream(self):
        if self.current_mode != "IDLE" and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (760, 570)) 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

                if len(faces) == 0:
                    for k in self.user_counters: self.user_counters[k] = 0

                if self.current_mode == "ENROLLING":
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        x, y, w, h = faces[0]
                        
                        margin = 45
                        x1, y1 = max(0, x - margin), max(0, y - margin)
                        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                        
                        face_img = frame[y1:y2, x1:x2]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 176, 255), 2)
                        
                        if time.time() - self.last_capture_time >= 0.04 and self.enroll_saved_count < 150:
                            self.enroll_saved_count += 1
                            cv2.imwrite(os.path.join(self.user_path, f"{self.enroll_user_name}_{self.enroll_saved_count:03d}.jpg"), face_img)
                            self.last_capture_time = time.time()
                        
                        cv2.putText(frame, f"SCANNING: {self.enroll_saved_count}/150", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 176, 255), 2)
                        
                        if self.enroll_saved_count >= 150:
                            self.return_to_idle()
                            threading.Thread(target=self.background_db_update, args=(self.enroll_user_name,)).start()

                elif self.current_mode == "VERIFYING":
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        try:
                            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                            input_tensor = self.transform(face_pil).unsqueeze(0).to(DEVICE)
                            
                            with torch.no_grad():
                                embedding = self.face_model(input_tensor).squeeze().cpu().numpy()
                            embedding = embedding / np.linalg.norm(embedding)

                            best_match = "Unknown"
                            min_dist = float("inf")

                            for user, data in self.database.items():
                                dist = np.linalg.norm(embedding - data["embedding"])
                                if dist < min_dist:
                                    min_dist = dist
                                    if dist < self.THRESHOLD:
                                        best_match = user

                            for k in self.user_counters:
                                if k == best_match: self.user_counters[k] += 1
                                else: self.user_counters[k] = 0

                            if best_match != "Unknown":
                                color = (0, 230, 118) 
                                label = f"{best_match.upper()} | DIST: {min_dist:.2f}"
                                
                                if self.user_counters[best_match] >= self.STABILIZATION_THRESHOLD:
                                    cv2.putText(frame, "ACCESS GRANTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 118), 3)
                                    
                                    current_time = time.time()
                                    if current_time - self.last_granted_time > 10:
                                        now = datetime.now()
                                        with open(LOG_FILE_PATH, "a") as f:
                                            f.write(f"{best_match},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
                                        
                                        self.lbl_last_log.configure(text=f"📝 Log Feed:\nName: {best_match}\nTime: {now.strftime('%H:%M:%S')}", fg=COLOR_ACCENT)
                                        os.system(f"say 'Welcome {best_match}, access granted' &")
                                        self.last_granted_time = current_time
                            else:
                                color = (255, 23, 68) 
                                label = f"UNKNOWN | DIST: {min_dist:.2f}"
                                if self.user_counters["Unknown"] >= self.STABILIZATION_THRESHOLD:
                                    cv2.putText(frame, "ACCESS DENIED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 23, 68), 3)

                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, label, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        except Exception:
                            continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.video_canvas.image = img_tk  

        self.root.after(15, self.update_video_stream)

    def background_db_update(self, user_name):
        self.status_pill.configure(text="⚙️ COMPILING DATA...", fg=COLOR_CYAN, bg="#1E293B")
        user_folder = os.path.join(DATASET_DIR, user_name)
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
            
            database = {}
            if os.path.exists(DATABASE_PATH):
                with open(DATABASE_PATH, 'rb') as f:
                    database = pickle.load(f)
            
            database[user_name] = {"embedding": avg_emb, "num_images": len(user_embeddings)}
            with open(DATABASE_PATH, "wb") as f:
                pickle.dump(database, f)
            
            self.load_system_database()
            messagebox.showinfo("Core Update", f"User '{user_name.upper()}' has been seamlessly unified into the database!")
            self.lbl_total_users.configure(text=f"👥 Total Identities: {len(self.database)}")
            self.return_to_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAccessApp(root)
    root.mainloop()