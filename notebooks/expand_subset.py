import os
import shutil
from tqdm import tqdm

# --- 🎯 تم تعديل المسار هنا إلى المجلد الصحيح train 🎯 ---
SOURCE_DATASET = "/Volumes/HIKSEMI/train" 
DEST_DATASET = "/Volumes/HIKSEMI/train_subset"

def expand_subset(num_new_classes_to_add):
    if not os.path.exists(SOURCE_DATASET):
        print(f"❌ Error: Source dataset not found at {SOURCE_DATASET}")
        print("Please check your external SSD path.")
        return

    os.makedirs(DEST_DATASET, exist_ok=True)

    # 1. معرفة الكلاصات الموجودة حالياً في الـ SSD
    existing_classes = set(os.listdir(DEST_DATASET))
    print(f"📊 Current classes in subset: {len(existing_classes)}")

    # 2. جلب الكلاصات المتاحة في المجلد الأصلي الكبير
    all_source_classes = [
        d for d in os.listdir(SOURCE_DATASET) 
        if os.path.isdir(os.path.join(SOURCE_DATASET, d)) and not d.startswith('.')
    ]
    
    # 3. تصفية الكلاصات (ناخذوا كان اللي موش موجودين ديجا)
    new_classes_available = [c for c in all_source_classes if c not in existing_classes]
    
    if not new_classes_available:
        print("🎉 All available classes are already in your subset!")
        return

    # تحديد العدد المطلوب نقله
    classes_to_copy = new_classes_available[:num_new_classes_to_add]
    print(f"🚀 Found {len(new_classes_available)} new classes. Copying {len(classes_to_copy)} to SSD...")

    # 4. عملية النقل الذكي باستخدام tqdm
    for class_name in tqdm(classes_to_copy, desc="Copying Classes"):
        if class_name.lower() in ["yassin", "unknown"]:
            continue
            
        source_class_dir = os.path.join(SOURCE_DATASET, class_name)
        dest_class_dir = os.path.join(DEST_DATASET, class_name)
        
        try:
            shutil.copytree(source_class_dir, dest_class_dir)
        except Exception as e:
            print(f"⚠️ Error copying class {class_name}: {e}")

    final_count = len(os.listdir(DEST_DATASET))
    print(f"=== ✅ Success! New total classes in train_subset: {final_count} ===")

if __name__ == "__main__":
    print("-" * 40)
    how_many = int(input("How many new classes do you want to add? (e.g., 400): "))
    print("-" * 40)
    expand_subset(how_many)