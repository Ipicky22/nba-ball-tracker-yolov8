import cv2
import os
import shutil

images_dir = "/batch_00/images"
blur_dir = "/batch_00/blur"

os.makedirs(blur_dir, exist_ok=True)

blur_threshold = 150

flou_count = 0
total = 0

for i, filename in enumerate(sorted(os.listdir(images_dir))):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(images_dir, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    total += 1

    if lap_var < blur_threshold:
        flou_count += 1
        shutil.move(path, os.path.join(blur_dir, filename))
        print(f"[{i+1}] {filename} → ❌ Floue (score: {lap_var:.2f})")
    else:
        print(f"[{i+1}] {filename} → ✅ Nette (score: {lap_var:.2f})")

print("\n📋 Cleaning Summary :")
print(f"➡️  Total : {total}")
print(f"❌ Blurry, displaced images : {flou_count}")
print(f"✅ Images retained : {total - flou_count}")
