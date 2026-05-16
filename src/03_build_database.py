import cv2
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# Configuration mta3 l-paths kima scripts l-o5rin
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.paths import PROCESSED_DIR

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "users_embeddings.pkl"

# Na5tarou l-device (M4 fih MPS ama FaceNet CPU light w sraye3 barcha)
device = torch.device("cpu")

print("Loading FaceNet model for embedding extraction...")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
print("Model loaded.")


def preprocess_for_facenet(face_img):
    """N-preparou l-crop l-FaceNet kima fil auto_verify_face.py"""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_rgb).convert("RGB")
    img = img.resize((160, 160))

    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - 127.5) / 128.0
    img_array = np.transpose(img_array, (2, 0, 1))

    tensor = torch.tensor(img_array).unsqueeze(0).to(device)
    return tensor


def build_database():
    database = {}

    if not PROCESSED_DIR.exists():
        print(f"Error: Processed directory not found at {PROCESSED_DIR}")
        return

    # N-loopiou 3ala ga3 l-classes (users) fi وسط processed/
    user_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]

    if len(user_folders) == 0:
        print("No processed user folders found. Run 02_clean_preprocess.py first.")
        return

    for user_folder in user_folders:
        user_name = user_folder.name
        print(f"\nProcessing user: {user_name}")

        images = [
            img
            for img in os.listdir(user_folder)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) == 0:
            print(f"No images found for {user_name}, skipping.")
            continue

        user_embeddings = []

        for img_name in images:
            img_path = user_folder / img_name
            frame = cv2.imread(str(img_path))

            if frame is None:
                continue

            # El-tsawer déjà cropped w resized fi 02_clean_preprocess.py
            try:
                face_tensor = preprocess_for_facenet(frame)

                with torch.no_grad():
                    embedding = model(face_tensor)

                embedding = embedding.squeeze().cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)
                user_embeddings.append(embedding)

            except Exception as e:
                print(f"Error extracting embedding for {img_name}: {e}")

        if len(user_embeddings) > 0:
            # Na3mlou l-mean embedding mta3 l-user bech n-y9wio l-accuracy
            average_embedding = np.mean(user_embeddings, axis=0)
            average_embedding = average_embedding / np.linalg.norm(
                average_embedding
            )

            database[user_name] = {"embedding": average_embedding}
            print(
                f"Successfully calculated average embedding for {user_name} from {len(user_embeddings)} images."
            )

    # Save l-database fil pickle file
    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)

    print(f"\nDatabase built successfully and saved to: {DATABASE_PATH}")


if __name__ == "__main__":
    build_database()