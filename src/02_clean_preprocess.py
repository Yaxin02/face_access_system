import cv2
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.paths import RAW_DIR, PROCESSED_DIR

IMAGE_SIZE = 160
MARGIN = 45

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_class(class_name):
    raw_class_dir = RAW_DIR / class_name
    processed_class_dir = PROCESSED_DIR / class_name
    processed_class_dir.mkdir(parents=True, exist_ok=True)

    if not raw_class_dir.exists():
        print(f"Raw folder not found: {raw_class_dir}")
        return

    images = [
        file for file in os.listdir(raw_class_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nClass: {class_name}")
    print(f"Raw images found: {len(images)}")

    saved_count = 0
    skipped_no_face = 0
    skipped_multi_face = 0
    skipped_error = 0

    for image_name in images:
        image_path = raw_class_dir / image_name

        frame = cv2.imread(str(image_path))

        if frame is None:
            skipped_error += 1
            print(f"Error reading: {image_name}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            skipped_no_face += 1
            print(f"No face: {image_name}")
            continue

        if len(faces) > 1:
            skipped_multi_face += 1
            print(f"Multiple faces: {image_name}")
            continue

        x, y, w, h = faces[0]

        x1 = max(0, x - MARGIN)
        y1 = max(0, y - MARGIN)
        x2 = min(frame.shape[1], x + w + MARGIN)
        y2 = min(frame.shape[0], y + h + MARGIN)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            skipped_error += 1
            print(f"Empty crop: {image_name}")
            continue

        face_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))

        output_name = image_name
        output_path = processed_class_dir / output_name

        cv2.imwrite(str(output_path), face_resized)
        saved_count += 1

    print("\nPreprocessing summary")
    print(f"Saved clean images: {saved_count}")
    print(f"Skipped no face: {skipped_no_face}")
    print(f"Skipped multiple faces: {skipped_multi_face}")
    print(f"Skipped errors: {skipped_error}")
    print(f"Processed folder: {processed_class_dir}")


def main():
    class_name = input("Enter class name to preprocess, example yassin: ").strip().lower()

    if class_name == "":
        print("Class name cannot be empty.")
        return

    preprocess_class(class_name)


if __name__ == "__main__":
    main()