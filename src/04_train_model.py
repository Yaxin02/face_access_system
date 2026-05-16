import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
import os
import pickle
import shutil
from tqdm import tqdm

def train_model():
    # 1. إعدادات الجهاز والمسارات
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DATASET_ROOT = "/Volumes/HIKSEMI/train" 
    MODEL_SAVE_PATH = "models/face_classifier_full.pth"
    MAPPING_SAVE_PATH = "models/class_mapping_full.pkl"
    
    OLD_MODEL_PATH = "models/face_classifier_subset.pth"
    OLD_MAPPING_PATH = "models/class_mapping.pkl"

    os.makedirs("models", exist_ok=True)

    print(f"Using device: {DEVICE}")
    
    # --- 🎯 التعديل الذكي الجديد: حذف المجلدات التي أصبحت فارغة بعد التنظيف 🎯 ---
    print("🧹 Checking for empty class folders left by the purge...")
    if os.path.exists(DATASET_ROOT):
        for subdir in os.listdir(DATASET_ROOT):
            subdir_path = os.path.join(DATASET_ROOT, subdir)
            # نتأكدوا إنه مجلد حقيقي ومش ملف مخفي كيما .DS_Store
            if os.path.isdir(subdir_path) and not subdir.startswith('.'):
                valid_images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(valid_images) == 0:
                    print(f"🗑️ Dropping empty class folder from SSD: {subdir}")
                    shutil.rmtree(subdir_path)  # حذفه تماماً لسلامة الـ PyTorch

    print(f"🔥 Scanning the ENTIRE pristine dataset at {DATASET_ROOT}...")

    # 2. الـ Data Augmentation Pipeline
    train_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 3. تحميل الداتا الصارمة (توا مستحيل تكراش)
    dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transforms)
    num_classes = len(dataset.classes)
    new_class_to_idx = dataset.class_to_idx

    print(f"✅ STRICT Dataset Loaded: {len(dataset)} images across {num_classes} classes. Zero skips allowed.")

    # 4. بناء الـ Backbone وتجميده
    base_model = InceptionResnetV1(pretrained='vggface2').to(DEVICE)
    for param in base_model.parameters():
        param.requires_grad = False

    class FaceClassifier(nn.Module):
        def __init__(self, base_model, num_classes):
            super(FaceClassifier, self).__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(512, num_classes)
        
        def forward(self, x):
            embeddings = self.base_model(x)
            return self.classifier(embeddings)

    model = FaceClassifier(base_model, num_classes).to(DEVICE)

    # 5. جراحة الشبكة ونقل ذاكرة وجه ياسين
    if os.path.exists(OLD_MODEL_PATH) and os.path.exists(OLD_MAPPING_PATH):
        try:
            print("🧠 Injecting your face memory into the giant model...")
            with open(OLD_MAPPING_PATH, 'rb') as f:
                old_class_to_idx = pickle.load(f)
            old_state_dict = torch.load(OLD_MODEL_PATH, map_location=DEVICE)
            
            old_weight = old_state_dict['classifier.weight']
            old_bias = old_state_dict['classifier.bias']
            
            new_weight = model.classifier.weight.data
            new_bias = model.classifier.bias.data
            
            matched_count = 0
            for name, old_idx in old_class_to_idx.items():
                if name in new_class_to_idx:
                    new_idx = new_class_to_idx[name]
                    if old_weight.shape[0] > old_idx:
                        new_weight[new_idx] = old_weight[old_idx]
                        new_bias[new_idx] = old_bias[old_idx]
                        matched_count += 1
                        
            print(f"✅ Brain Surgery Complete! Transferred {matched_count} identities.")
        except Exception as e:
            print(f"⚠️ Memory transfer skipped ({e}). Training fresh full model...")
    else:
        print("🆕 Training full model from scratch...")

    # 6. الـ DataLoader والـ Optimizer (Batch Size 128 للسرعة القصوى)
    BATCH_SIZE = 128
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print(f"🚀 Launching Strict Giant Training (Batch Size: {BATCH_SIZE})...")
    
    # 7. حلقة التدريب مع الـ Checkpointing تلقائياً
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/5]")
        
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        # حفظ الـ Checkpoint بعد كل Epoch لحماية الملفات
        epoch_save_path = f"models/face_classifier_full_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"💾 Checkpoint saved: {epoch_save_path}")

    # 8. حفظ النسخة النهائية والـ Mapping
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(MAPPING_SAVE_PATH, 'wb') as f:
        pickle.dump(new_class_to_idx, f)
        
    print(f"🎉 EPIC SUCCESS! Giant Model saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()