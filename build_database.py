import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

DATASET_DIR = "dataset"
DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "users_embeddings.pkl")

os.makedirs(DATABASE_DIR, exist_ok=True)

device = torch.device("cpu")

print("Loading FaceNet model...")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
print("Model loaded successfully.")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((160, 160))

    img_array = np.array(img).astype(np.float32)

    # FaceNet normalization
    img_array = (img_array - 127.5) / 128.0

    # Convert HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))

    tensor = torch.tensor(img_array).unsqueeze(0).to(device)
    return tensor


def get_embedding(image_path):
    try:
        face_tensor = preprocess_image(image_path)

        with torch.no_grad():
            embedding = model(face_tensor)

        embedding = embedding.squeeze().cpu().numpy()

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def build_database():
    database = {}

    if not os.path.exists(DATASET_DIR):
        print("Error: dataset folder not found.")
        return

    users = [
        user for user in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, user))
    ]

    if len(users) == 0:
        print("No users found in dataset.")
        return

    for user in users:
        user_folder = os.path.join(DATASET_DIR, user)
        print(f"\nProcessing user: {user}")

        user_embeddings = []

        images = [
            img for img in os.listdir(user_folder)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) == 0:
            print(f"No images found for user: {user}")
            continue

        for image_name in images:
            image_path = os.path.join(user_folder, image_name)
            print(f"  Reading: {image_name}")

            embedding = get_embedding(image_path)

            if embedding is not None:
                user_embeddings.append(embedding)
                print("  OK")
            else:
                print("  Skipped")

        if len(user_embeddings) == 0:
            print(f"No valid embeddings for user: {user}")
            continue

        # Average embedding for this user
        average_embedding = np.mean(user_embeddings, axis=0)

        # Normalize average embedding
        average_embedding = average_embedding / np.linalg.norm(average_embedding)

        database[user] = {
            "embedding": average_embedding,
            "num_images": len(user_embeddings)
        }

        print(f"Saved user: {user} with {len(user_embeddings)} images")

    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)

    print("\nDatabase created successfully.")
    print(f"Saved at: {DATABASE_PATH}")
    print(f"Total users: {len(database)}")


if __name__ == "__main__":
    build_database()