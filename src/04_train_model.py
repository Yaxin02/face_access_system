import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import os
import pickle

# 1. إعدادات المسارات (Path للـ Subset الجديد)
DATA_PATH = "/Volumes/HIKSEMI/train_subset"
MODEL_SAVE_PATH = "models/face_classifier_subset.pth"
MAPPING_SAVE_PATH = "models/class_mapping.pkl"

# استعمال الـ GPU (MPS) للـ Mac M4 🔥
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. الـ Safe Loader (الحل النهائي للـ UnidentifiedImageError)
def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception:
        # إذا لقى تصويرة فاسدة، يطفي عليها الضو ويبعث تصويرة سوداء فارغة
        return Image.new('RGB', (160, 160), color=0)

# 3. تحضير البيانات
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# تحميل الداتا من الـ Subset
print(f"🔍 Scanning subset at {DATA_PATH}...")
train_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms, loader=safe_loader)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

num_classes = len(train_dataset.classes)
print(f"✅ Loaded {len(train_dataset)} images across {num_classes} classes.")

# 4. الموديل (FaceNet)
# ملاحظة: استعملنا FaceNet مدرب مسبقاً، ونحن فقط ندرب الطبقة الأخيرة
base_model = InceptionResnetV1(pretrained='vggface2').to(device)

# تجميد أوزان FaceNet (ما نلمسوهمش)
for param in base_model.parameters():
    param.requires_grad = False

# إضافة الطبقة التصنيفية الخاصة بالـ 100 شخص
class FaceClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FaceClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(512, num_classes) # تحويل الـ 512 embedding لعدد الكلاصات
    
    def forward(self, x):
        embeddings = self.base_model(x)
        return self.classifier(embeddings)

model = FaceClassifier(base_model, num_classes).to(device)

# 5. الـ Loss والـ Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 6. الـ Training Loop
def train():
    os.makedirs("models", exist_ok=True)
    epochs = 5
    print(f"🚀 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    # حفظ النتائج
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(MAPPING_SAVE_PATH, 'wb') as f:
        pickle.dump(train_dataset.class_to_idx, f)
    print(f"🎉 Done! Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()