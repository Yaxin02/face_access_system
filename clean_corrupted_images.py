import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# مسار الـ Dataset الكبير في الـ SSD
DATASET_ROOT = "/Volumes/HIKSEMI/train"

def check_and_delete(file_path):
    """يتثبت من الصورة، إذا فاسدة يحذفها فوراً من الـ SSD"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # يتأكد إن الفايل سليم ومش مقطوع
        return False
    except Exception:
        try:
            os.remove(file_path)  # حذف الصورة الفاسدة نهائياً
            return True
        except Exception:
            return False

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Path not found: {DATASET_ROOT}")
        print("Please make sure your external SSD is connected.")
        return

    print("🔍 Gathering all image paths from SSD (This may take a minute)...")
    all_files = []
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    total_files = len(all_files)
    print(f"📸 Found {total_files} images. Starting Multi-threaded Deep Clean...")

    # تشغيل 8 مسارات متوازية لاستغلال كامل قوة معالج الـ M4 Air
    corrupted_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(check_and_delete, all_files), total=total_files, desc="Cleaning Dataset"))
        corrupted_count = sum(1 for r in results if r)

    print("=" * 40)
    print(f"✨ CLEANING COMPLETE!")
    print(f"🗑️ Total corrupted images permanently deleted: {corrupted_count}")
    print("=" * 40)

if __name__ == "__main__":
    main()