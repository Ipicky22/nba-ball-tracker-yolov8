import os
import shutil
import random

images_folder = "/batches/batch_00/images"
labels_folder = "/batches/batch_00/labels"

output_base = "/batches/batch_00/split"
train_img = os.path.join(output_base, "images/train")
val_img = os.path.join(output_base, "images/val")
train_lbl = os.path.join(output_base, "labels/train")
val_lbl = os.path.join(output_base, "labels/val")

for folder in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(folder, exist_ok=True)

image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(file_list, dest_img, dest_lbl):
    for file in file_list:
        img_src = os.path.join(images_folder, file)
        lbl_src = os.path.join(labels_folder, file.replace('.jpg', '.txt').replace('.png', '.txt'))

        shutil.copy(img_src, os.path.join(dest_img, file))
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(dest_lbl, os.path.basename(lbl_src)))
        else:
            print(f"⚠️ Label manquant pour : {file}")

move_files(train_files, train_img, train_lbl)
move_files(val_files, val_img, val_lbl)

print(f"✅ Split : {len(train_files)} train / {len(val_files)} val")
