import cv2
import os
import shutil
from skimage.metrics import structural_similarity as ssim

batch_path = "/batch_00"
images_dir = os.path.join(batch_path, "images")
duplicates_dir = os.path.join(batch_path, "duplicates")
os.makedirs(duplicates_dir, exist_ok=True)

similarity_threshold = 0.89  
compare_range = 3           

image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])
checked = set()
moved = 0

for i in range(len(image_files)):
    if image_files[i] in checked:
        continue

    img1_path = os.path.join(images_dir, image_files[i])
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        continue

    for j in range(1, compare_range + 1):
        if i + j >= len(image_files):
            break

        img2_path = os.path.join(images_dir, image_files[i + j])
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img2 is None or image_files[i + j] in checked:
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        score = ssim(img1, img2)

        print(f"Comparing {image_files[i]} ↔ {image_files[i + j]} → SSIM: {score:.4f}")
        if score > similarity_threshold:
            shutil.move(img2_path, os.path.join(duplicates_dir, image_files[i + j]))
            checked.add(image_files[i + j])
            moved += 1
            print(f"🟡 {image_files[i + j]} considéré comme doublon de {image_files[i]} (SSIM = {score:.4f})")

print(f"\n📦 Moved to 'duplicates' : {moved}")
