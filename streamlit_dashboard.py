"""
Ultra Pro Access Core AI — Web Control Panel (Streamlit)
Standalone launcher & read-only analytics for the Safe Exit Pro PyQt5 app.
Does NOT implement facial recognition — wraps the existing main.py black box.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Paths (project root = this file's directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
DATABASE_PATH = PROJECT_ROOT / "database" / "users_embeddings.pkl"
DATASET_DIR = PROJECT_ROOT / "dataset"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python3"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_python_executable() -> str:
    """Prefer project venv; fall back to current interpreter."""
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def is_main_running() -> bool:
    """Best-effort check whether main.py is already running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "Python main.py"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def launch_safe_exit_pro() -> tuple[bool, str]:
    """Start main.py detached in the background."""
    if not MAIN_SCRIPT.is_file():
        return False, f"Not found: {MAIN_SCRIPT}"

    if is_main_running():
        return True, "Safe Exit Pro is already running."

    python_exe = resolve_python_executable()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        subprocess.Popen(
            [python_exe, str(MAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, "Launched Safe Exit Pro Core — check your Dock for the PyQt5 window."
    except Exception as exc:
        return False, f"Launch failed: {exc}"


def load_embeddings_database() -> dict:
    """Read-only load of users_embeddings.pkl."""
    if not DATABASE_PATH.is_file() or DATABASE_PATH.stat().st_size == 0:
        return {}
    try:
        with open(DATABASE_PATH, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def scan_dataset_stats() -> list[dict]:
    """Count image files per user folder under dataset/."""
    rows: list[dict] = []
    if not DATASET_DIR.is_dir():
        return rows

    for user_dir in sorted(DATASET_DIR.iterdir()):
        if not user_dir.is_dir() or user_dir.name.startswith("."):
            continue
        count = sum(
            1
            for p in user_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows.append(
            {
                "Identity": user_dir.name,
                "Images in dataset/": count,
                "Folder": str(user_dir.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def build_identity_table(db: dict, dataset_rows: list[dict]) -> list[dict]:
    """Merge pickle metadata with dataset folder counts."""
    dataset_map = {r["Identity"]: r["Images in dataset/"] for r in dataset_rows}
    all_users = sorted(set(db.keys()) | set(dataset_map.keys()))

    table: list[dict] = []
    for user in all_users:
        entry = db.get(user, {})
        num_emb = entry.get("num_images", "—") if isinstance(entry, dict) else "—"
        in_db = "Yes" if user in db else "No"
        table.append(
            {
                "Identity": user,
                "In embeddings DB": in_db,
                "Embedding samples (pkl)": num_emb,
                "Images in dataset/": dataset_map.get(user, 0),
            }
        )
    return table


# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Safe Exit Pro — Control Panel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ——— Light / white corporate theme ——— */
    .stApp,
    [data-testid="stAppViewContainer"],
    .main,
    section.main > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    p, label, .stMarkdown, span {
        color: #334155;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0369a1 !important;
        letter-spacing: 0.02em;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #64748b !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #059669 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.05);
        border: 1px solid #0284c7 !important;
    }
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #0369a1 !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stExpander"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    .status-online { color: #059669 !important; font-weight: bold; }
    .status-offline { color: #dc2626 !important; font-weight: bold; }
    hr {
        border-color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ COMMAND CENTER")
    st.markdown("---")
    st.markdown("**Ultra Pro Access Core AI**")
    st.caption("Safe Exit Pro — Web Control Panel")
    st.markdown("---")
    st.markdown("##### Core Team")
    st.markdown("- **Yassine Mokni**")
    st.markdown("- **Hadil Dhaya**")
    st.markdown("---")
    st.markdown("##### System paths")
    st.code(str(PROJECT_ROOT), language=None)
    st.caption(f"Python: `{resolve_python_executable()}`")
    st.markdown("---")
    running = is_main_running()
    if running:
        st.markdown('<p class="status-online">● PyQt app: RUNNING</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">○ PyQt app: STOPPED</p>', unsafe_allow_html=True)
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">🛡️ ULTRA PRO ACCESS CORE AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Safe Exit Pro — Web Control Panel & Identity Analytics (read-only)</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Launcher section
# ---------------------------------------------------------------------------
st.markdown("### 🚀 Launch Safe Exit Pro Core")
st.markdown(
    "This panel **does not run face recognition**. It starts your existing "
    "**PyQt5** application (`main.py`) in a separate process."
)

launch_col1, launch_col2 = st.columns([2, 1])

with launch_col1:
    if st.button("▶ Launch Safe Exit Pro Core", type="primary", use_container_width=True):
        ok, message = launch_safe_exit_pro()
        if ok:
            st.success(message)
        else:
            st.error(message)

with launch_col2:
    if st.button("🔄 Refresh status", use_container_width=True):
        st.rerun()

st.info(
    "**Manual launch:** `cd` into the project folder, then run "
    "`./venv/bin/python3 main.py` — same as this button."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Analytics (read-only)
# ---------------------------------------------------------------------------
st.markdown("### 📊 Identity & Dataset Analytics")

db = load_embeddings_database()
dataset_rows = scan_dataset_stats()
identity_table = build_identity_table(db, dataset_rows)

total_users_db = len(db)
total_dataset_users = len(dataset_rows)
total_images = sum(r["Images in dataset/"] for r in dataset_rows)
total_emb_samples = sum(
    entry.get("num_images", 0)
    for entry in db.values()
    if isinstance(entry, dict) and isinstance(entry.get("num_images"), (int, float))
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Enrolled Users (DB)", total_users_db)
with m2:
    st.metric("Dataset folders", total_dataset_users)
with m3:
    st.metric("Total images (dataset/)", total_images)
with m4:
    st.metric("Embedding samples (pkl)", int(total_emb_samples))

st.markdown("#### Registered identities")

if identity_table:
    st.dataframe(
        identity_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning(
        "No identities found. Enroll users via **ADD IDENTITY** in the PyQt app "
        "or place folders under `dataset/`."
    )

if db:
    with st.expander("Raw embedding DB keys (read-only)"):
        for username, meta in sorted(db.items()):
            if isinstance(meta, dict):
                st.markdown(
                    f"- **{username}** — `num_images`: {meta.get('num_images', 'N/A')}, "
                    f"`embedding` shape: "
                    f"{getattr(meta.get('embedding'), 'shape', 'N/A')}"
                )
            else:
                st.markdown(f"- **{username}** — (legacy entry)")

st.markdown("---")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
foot1, foot2 = st.columns(2)
with foot1:
    st.markdown("##### 📁 Monitored resources")
    st.markdown(f"- Embeddings: `{DATABASE_PATH.relative_to(PROJECT_ROOT)}`")
    st.markdown(f"- Dataset root: `{DATASET_DIR.relative_to(PROJECT_ROOT)}/`")
with foot2:
    st.markdown("##### ⚙️ Run this dashboard")
    st.code("./venv/bin/python3 -m streamlit run streamlit_dashboard.py", language="bash")
    st.caption("Run from the project root with venv activated.")

st.markdown(
    "<p style='text-align:center;color:#94a3b8;margin-top:2rem;'>"
    "© Safe Exit Pro · Yassine Mokni & Hadil Dhaya · Local-first biometrics"
    "</p>",
    unsafe_allow_html=True,
)
