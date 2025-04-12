import os
import shutil
import random

input_folders = ['ball', 'without_ball']
output_base = 'dataset'
split_ratio = 0.8  # 80% train, 20% val
random.seed(42)

for cls in input_folders:
    img_files = [f for f in os.listdir(cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(img_files)
    split_idx = int(len(img_files) * split_ratio)

    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    train_dir = os.path.join(output_base, 'train', cls)
    val_dir = os.path.join(output_base, 'val', cls)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(cls, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(cls, f), os.path.join(val_dir, f))

    print(f"✅ Classes '{cls}': {len(train_files)} train, {len(val_files)} val")

print("\n🎯 Dataset split completed and ready for classification.")
