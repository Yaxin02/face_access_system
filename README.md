# 🛡️ Ultra Pro Access Core AI — Safe Exit Pro

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-MPS%20%7C%20FaceNet-EE4C2C?style=flat&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Haar%20%2B%20Video-5C3EE8?style=flat&logo=opencv)
![PyQt5](https://img.shields.io/badge/PyQt5-Desktop%20UI-41CD52?style=flat&logo=qt)
![Arduino](https://img.shields.io/badge/Arduino-Serial%20I%2FO-00979D?style=flat&logo=arduino)
![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-000000?style=flat&logo=apple)

**Real-time facial recognition access control** combining **FaceNet embeddings** on Apple **MPS**, a **PyQt5 command-center UI**, and **Arduino-driven physical I/O** (LED grant/deny/off) over USB serial.

---

## 👥 Core Development Team

| Role | Name |
|------|------|
| **Lead Engineer & AI Pipeline** | **Yassine Mokni** |
| **Co-Developer & System Integration** | **Hadil Dhaya** |

---

## 🧠 System Architecture Overview

The platform is **decoupled** into two cooperating layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI INFERENCE NODE (macOS / Python)                             │
│  ┌──────────────┐   ┌─────────────┐   ┌──────────────────────┐  │
│  │ Camera Input │ → │ Haar Cascade│ → │ FaceNet (VGGFace2)   │  │
│  │ (OpenCV)     │   │ Face Detect │   │ 512-D Embedding      │  │
│  └──────────────┘   └─────────────┘   └──────────┬───────────┘  │
│                                                   │ L2 distance   │
│  ┌──────────────┐   ┌─────────────┐              ▼               │
│  │ PyQt5 UI     │ ← │ State Machine│ ← database/users_embeddings  │
│  │ + macOS TTS  │   │ (IDLE/ENROLL/│                              │
│  └──────────────┘   │  VERIFY)     │                              │
│         │           └──────┬───────┘                              │
│         │                  │ Serial 9600 baud (G / D / O)        │
└─────────┼──────────────────┼──────────────────────────────────┘
          │                  ▼
┌─────────┼──────────────────────────────────────────────────────┐
│         │  HARDWARE ACTUATOR NODE (Arduino)                     │
│         │  • Receives single-byte commands (no JSON overhead)   │
│         │  • Drives Green (GRANT) / Red (DENY) / OFF LEDs         │
│         │  • Firmware uploaded separately to the board            │
└─────────┴──────────────────────────────────────────────────────┘
```

### Operating modes (`main.py`)

| Mode | Description |
|------|-------------|
| **IDLE** | Camera released; standby UI; LEDs sent `O` (off) on stop |
| **ENROLLING** | Captures **150** face crops per user → builds mean embedding → updates `.pkl` DB |
| **VERIFYING** | Live 1:N match against `users_embeddings.pkl`; triggers hardware + voice on stable match |

### Client interfaces

| Layer | File | Role |
|-------|------|------|
| **Production** | `main.py` | PyQt5 app — live camera, FaceNet, Arduino I/O |
| **Web dashboard** | `streamlit_dashboard.py` | Browser control panel — launcher + read-only analytics |
| **Research / CLI** | `notebooks/*` | Jupyter experiments & offline scripts |

---

## 📊 Data Science & Research Methodology

The **`notebooks/`** directory houses our **academic research trail** and reproducible Deep Learning experimentation—separate from the production runtime in `main.py`, but fully aligned with the same FaceNet / VGGFace2 stack.

| Location | Purpose |
|----------|---------|
| **`notebooks/drafts/`** | Early-phase experiments documenting **rejected or superseded** approaches |
| **`notebooks/final/Notebook_Final_Soigne.ipynb`** | **Polished French notebook** — complete DL pipeline write-up for defense / submission |

### Draft experiments (`notebooks/drafts/`)

- **`01_Baseline_CNN.ipynb`** — Custom shallow CNN baseline (insufficient accuracy for access control).
- **`02_Data_Augmentation_LearningRates.ipynb`** — Learning-rate sweeps, dropout, and `torchvision` augmentation studies.
- **`03_Transfer_Learning_ResNet.ipynb`** — ResNet18 transfer learning vs. baseline CNN.

### Final notebook (`notebooks/final/`)

**`Notebook_Final_Soigne.ipynb`** consolidates the validated methodology:

1. **Prétraitement & augmentation** — 160×160 crops, FaceNet normalization.  
2. **Modèle** — InceptionResnetV1 (VGGFace2) embedding extractor.  
3. **Entraînement & validation** — Training loops, **loss / accuracy curves**.  
4. **Comparaison d'architectures** — Justification for FaceNet over CNN / ResNet.  
5. **Interprétation** — **Confusion matrix**, **ROC curves**, and metric discussion.

The pipeline also documents our **large-scale dataset strategy** (including external SSD collection via `config/paths.py` and VGGFace2-scale pretraining context — **~1.1M identities** in the source corpus used to pretrain the backbone). Local deployment uses per-user folders under `dataset/` and aggregated embeddings in `database/users_embeddings.pkl`.

Supporting CLI scripts (`build_database.py`, `train.py`, `enroll_user.py`, `src/*`) live under **`notebooks/`** and resolve paths to root-level `dataset/` and `database/` via `notebooks/project_paths.py`.

---

## ✨ Advanced Engineering Implementations

### 1. Apple Silicon MPS acceleration
- `DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`
- **InceptionResnetV1** (`facenet-pytorch`, weights: **vggface2**) runs in `.eval()` with `torch.no_grad()` for inference-only throughput.

### 2. Dual-frame vision pipeline
- **Display frame**: annotated with boxes, labels, and status text.
- **`clean_frame`**: unmodified copy used for cropping → prevents overlay pixels from polluting embeddings.

### 3. Dominant face targeting
When multiple faces appear, detections are sorted by **bounding-box area** (`width × height`). Only the **largest** (closest) face is enrolled or verified:

```python
faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
x, y, w, h = faces[0]
```

### 4. Temporal stabilization (anti-flicker)
- `STABILIZATION_THRESHOLD = 5` consecutive frames required before **ACCESS GRANTED**, **ACCESS DENIED**, Arduino I/O, or `say` audio fires.
- `last_triggered_state` ensures **instant-switch logic** without re-sending serial/audio for the same state.

### 5. L2 embedding matching
- Per-face embedding is **L2-normalized**.
- Enrollment stores the **mean** of up to 150 normalized embeddings, re-normalized.
- Match if Euclidean distance `< THRESHOLD` (`0.65` in production UI).

### 6. Serial buffer flushing protocol
Every hardware command calls `.flush()` immediately after `.write()` so macOS does not buffer bytes in the USB stack:

| Byte | Meaning | When sent |
|------|---------|-----------|
| `G` | **Grant** — green LED on | Stable recognized identity |
| `D` | **Deny** — red LED on | Stable unknown identity |
| `O` | **Off** — all LEDs off | Stop engine, no face, app close |

On application exit, `closeEvent()` waits **`0.1 s`** after sending `O` before `serial.close()` so the final off command reaches the MCU.

### 7. Graceful hardware degradation
If `serial.Serial(USB_PORT, 9600)` fails at startup, the app continues in **Screen-Only Mode** (UI + voice + vision still work).

### 8. USB port configuration
The serial port is set in `main.py`:

```python
USB_PORT = "/dev/cu.usbmodem11301"  # ← update to your Mac's port
```

Discover the correct device:

```bash
ls /dev/cu.usbmodem*
```

> **Note:** This repository ships the **Python serial client** only. Arduino **firmware is flashed separately** to your board (see [Hardware](#-hardware--arduino-integration)).

---

## 📂 Folder Structure

```text
face_access_system/
├── main.py                      # 🎯 Production app: PyQt5 UI + FaceNet + Arduino serial
├── streamlit_dashboard.py       # Web control panel (launcher + read-only analytics)
├── .streamlit/
│   └── config.toml              # Light theme for the Streamlit dashboard
├── requirements.txt             # Python dependencies (PyTorch, facenet-pytorch, PyQt5, streamlit, pyserial…)
├── config/
│   └── paths.py                 # SSD / external data paths (.env driven)
├── notebooks/                   # 📓 Data Science, research & offline tooling
│   ├── README.md
│   ├── project_paths.py         # Resolves ../dataset & ../database from any notebook script
│   ├── drafts/                  # Exploratory Jupyter notebooks (academic phase)
│   │   ├── 01_Baseline_CNN.ipynb
│   │   ├── 02_Data_Augmentation_LearningRates.ipynb
│   │   └── 03_Transfer_Learning_ResNet.ipynb
│   ├── final/
│   │   └── Notebook_Final_Soigne.ipynb   # Polished FR pipeline (ROC, confusion matrix, …)
│   ├── build_database.py        # Offline: rebuild users_embeddings.pkl from dataset/
│   ├── enroll_user.py           # CLI enrollment (OpenCV window, no GUI)
│   ├── auto_verify_face.py      # CLI verification loop (no Arduino)
│   ├── auto_enroll_user.py      # Automated enrollment helper
│   ├── face_detection_test.py   # Haar / camera diagnostics
│   ├── train.py                 # Entry point → src/04_train_model.py
│   ├── clean_corrupted_images.py
│   ├── expand_subset.py
│   └── src/
│       ├── 01_collect_data.py     # Large-scale raw capture → external SSD
│       ├── 02_clean_preprocess.py # Cleaning & preprocessing pipeline
│       ├── 03_build_database.py   # Embedding DB builder (pipeline variant)
│       └── 04_train_model.py      # Custom classifier training (tkinter UI)
├── database/                    # 🔒 Runtime identity store (project root)
│   ├── users_embeddings.pkl     # Local identity vectors (gitignored)
│   └── access_log.csv           # Optional access audit log
├── dataset/                     # Per-user raw face images (gitignored)
│   └── <username>/              # e.g. 150× .jpg per enrolled identity
├── models/                      # Trained classifier weights (gitignored)
├── reports/                     # Training reports (gitignored)
├── .env.example
└── README.md
```

---

## 🔌 Hardware & Arduino Integration

### Wiring table (recommended firmware)

Use **220 Ω** resistors in series with each LED.

| Component | Arduino Pin | Direction | Notes |
|-----------|-------------|-----------|--------|
| 🟢 Green LED (Access Granted) | **Digital 8** | OUTPUT | ON when Python sends `G` |
| 🔴 Red LED (Access Denied) | **Digital 9** | OUTPUT | ON when Python sends `D` |
| USB Serial | **Native USB** | — | 9600 baud, 8N1 |

### Expected firmware behavior

Your Arduino sketch should:

1. `Serial.begin(9600)` in `setup()`.
2. On received byte:
   - `'G'` / `'g'` → green HIGH, red LOW  
   - `'D'` / `'d'` → red HIGH, green LOW  
   - `'O'` / `'o'` (or default) → both LOW  
3. Avoid long `delay()` in `loop()` to keep the serial buffer responsive.

### Connection checklist

1. Flash firmware to the Arduino.
2. Plug USB into the Mac.
3. Run `ls /dev/cu.usbmodem*` and set `USB_PORT` in `main.py`.
4. Launch `main.py` — terminal should print: `✅ Arduino Hardware Connected via USB!`

---

## ⚙️ Setup & Installation

### Prerequisites

- **macOS** with Apple Silicon (MPS) recommended  
- **Python 3.9+**  
- Webcam and/or **iPhone Continuity Camera**  
- **Arduino** with grant/deny LED wiring (optional)

### 1. Clone & enter project

```bash
cd /path/to/face_access_system
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> First FaceNet run downloads **VGGFace2** pretrained weights automatically.

### 3. Configure Arduino port (if used)

Edit `main.py`:

```python
USB_PORT = "/dev/cu.usbmodemXXXXX"
```

### 4. (Optional) External SSD training data

```bash
cp .env.example .env
# Edit FACE_FLASH_ROOT / FACE_DATA_ROOT as needed
```

---

## 🚀 Execution

### Production GUI (recommended)

```bash
source venv/bin/activate
python3 main.py
```

| UI Control | Action |
|------------|--------|
| **➕ ADD IDENTITY** | Enroll new user (150 frames → embedding) |
| **▶ START SCANNER** | Live verification vs database |
| **🛑 STOP ENGINE** | Release camera; send `O` to Arduino |

**Camera picker:**

| Index | Label in UI |
|-------|-------------|
| `0` | iPhone Continuity Camera |
| `1` | Mac Built-in FaceTime HD Camera (default selection) |

---

## 🌐 Streamlit Web Control Panel

`streamlit_dashboard.py` is a **standalone browser dashboard** that wraps the production app. It does **not** reimplement face recognition—it **launches** `main.py` and surfaces **read-only** system metrics.

### Features

| Feature | Description |
|---------|-------------|
| **▶ Launch Safe Exit Pro Core** | Starts `main.py` in the background via `subprocess` (uses `venv/bin/python3` when available) |
| **PyQt status** | Sidebar shows whether the desktop app is **RUNNING** or **STOPPED** |
| **Total Enrolled Users** | Reads `database/users_embeddings.pkl` (read-only) |
| **Dataset analytics** | Scans `dataset/<identity>/` and counts images per user |
| **Identity table** | Merged view: embedding DB + folder stats |
| **Light corporate UI** | White theme via `.streamlit/config.toml` + custom CSS |

### Run the dashboard

```bash
cd /path/to/face_access_system
source venv/bin/activate

# Recommended (avoids broken venv script paths after moving the project)
./venv/bin/python3 -m streamlit run streamlit_dashboard.py
```

Browser opens at **http://localhost:8501**

| Dashboard control | Action |
|-------------------|--------|
| **▶ Launch Safe Exit Pro Core** | Opens the PyQt5 app (check Dock / Cmd+Tab) |
| **🔄 Refresh status** | Reloads metrics and PyQt running state |

> **Note:** Use the Streamlit panel for **monitoring and launching**. Enrollment, scanning, camera, and Arduino control remain in the **PyQt app** (`main.py`).

### Architecture (Streamlit layer)

```
Browser (Streamlit)  ──subprocess──▶  main.py (PyQt5 + FaceNet + Arduino)
       │                                      │
       └── read-only ──▶  dataset/  +  database/users_embeddings.pkl
```

---

### Offline / developer utilities

```bash
# Rebuild embedding database from dataset/ folders
python3 notebooks/build_database.py

# CLI enrollment (camera index 1 hardcoded in script)
python3 notebooks/enroll_user.py

# CLI verification (no Arduino)
python3 notebooks/auto_verify_face.py

# Full ML training pipeline (external SSD paths)
python3 notebooks/src/01_collect_data.py
python3 notebooks/src/02_clean_preprocess.py
python3 notebooks/train.py

# Jupyter — research notebooks
jupyter notebook notebooks/final/Notebook_Final_Soigne.ipynb
```

---

## 🧪 Key configuration constants (`main.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `THRESHOLD` | `0.65` | Max L2 distance for a positive match |
| `STABILIZATION_THRESHOLD` | `5` | Frames before grant/deny/I/O |
| Enrollment samples | `150` | Images per new identity |
| Capture interval | `0.04 s` | ~25 enroll snapshots/sec |
| UI timer | `30 ms` | ~33 FPS refresh |
| Face margin | `45 px` | Crop padding around Haar box |
| Haar `minSize` | `(120, 120)` | Ignore distant small faces |
| Serial baud | `9600` | Arduino link speed |

---

## 📊 Data model

Each entry in `database/users_embeddings.pkl`:

```python
{
  "username": {
    "embedding": np.ndarray,  # L2-normalized 512-D vector
    "num_images": int         # frames used for the mean
  }
}
```

---

## 🔒 Security & privacy notes

- All inference and storage are **local** — no cloud API in `main.py`.
- Biometric templates live in `users_embeddings.pkl` (excluded from git).
- `THRESHOLD` trades **security vs usability**; lower = stricter.
- Physical access should not rely on face recognition alone for high-security sites.

---

## 🛠️ Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Screen-Only Mode` in console | Wrong `USB_PORT` or Arduino unplugged | `ls /dev/cu.usbmodem*` → update `USB_PORT` |
| Black camera / no video | Wrong camera index or macOS privacy | System Settings → Privacy → Camera → allow Terminal/Cursor; try index `0` vs `1` |
| `Database Empty!` | No `.pkl` or empty DB | Run **ADD IDENTITY** or `notebooks/build_database.py` |
| LEDs stuck on | Missing `O` on exit | Press **STOP ENGINE** or restart app |
| Slow first launch | FaceNet weight download | Wait for one-time download to finish |
| `bad interpreter` on `streamlit` | venv created at old path | Use `./venv/bin/python3 -m streamlit run streamlit_dashboard.py` |
| Streamlit shows 0 users | Empty or missing `.pkl` | Enroll via PyQt **ADD IDENTITY** or run `notebooks/build_database.py` |

---

## 📜 License

MIT — see repository license terms.  
FaceNet weights subject to **VGGFace2** / `facenet-pytorch` usage policies.

---

<p align="center">
  <b>Ultra Pro Access Core AI — Safe Exit Pro</b><br>
  Developed by <b>Yassine Mokni</b> & <b>Hadil Dhaya</b>
</p>
