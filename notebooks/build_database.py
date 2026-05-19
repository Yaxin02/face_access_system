import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

from project_paths import (
    DATASET_DIR,
    DATABASE_DIR,
    DATABASE_PATH,
    ensure_data_dirs,
    setup_sys_path,
)

setup_sys_path()
ensure_data_dirs()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading FaceNet backbone...")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
print("Model loaded successfully.")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - 127.5) / 128.0
    img_array = np.transpose(img_array, (2, 0, 1))
    tensor = torch.tensor(img_array).unsqueeze(0).to(device)
    return tensor


def get_embedding(image_path):
    try:
        face_tensor = preprocess_image(image_path)
        with torch.no_grad():
            embedding = model(face_tensor)
        embedding = embedding.squeeze().cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None


def build_database():
    database = {}
    dataset_dir = str(DATASET_DIR)

    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} folder not found.")
        return

    users = [
        user
        for user in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, user)) and not user.startswith(".")
    ]

    if len(users) == 0:
        print("ℹ️ No users found in local dataset folder.")
        return

    print(f"📊 Found {len(users)} local authorized users. Extracting signatures...")

    for user in users:
        user_folder = os.path.join(dataset_dir, user)
        user_embeddings = []
        images = [
            img
            for img in os.listdir(user_folder)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) == 0:
            continue

        for image_name in tqdm(images, desc=f"Embedding '{user}'"):
            image_path = os.path.join(user_folder, image_name)
            embedding = get_embedding(image_path)
            if embedding is not None:
                user_embeddings.append(embedding)

        if len(user_embeddings) == 0:
            continue

        average_embedding = np.mean(user_embeddings, axis=0)
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        database[user] = {
            "embedding": average_embedding,
            "num_images": len(user_embeddings),
        }

    db_path = str(DATABASE_PATH)
    with open(db_path, "wb") as file:
        pickle.dump(database, file)

    print("\n" + "=" * 40)
    print(f"✅ Success! Local features database created at: {db_path}")
    print(f"👥 Total registered identities: {len(database)}")
    print("=" * 40)


if __name__ == "__main__":
    build_database()
