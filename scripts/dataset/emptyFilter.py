import cv2
import os
import shutil

batch_path = "/batch_00"
images_path = os.path.join(batch_path, "images")
empty_dir = os.path.join(batch_path, "empty")
os.makedirs(empty_dir, exist_ok=True)

variance_threshold = 5.0

total = 0
empty_count = 0
ok_count = 0

for i, filename in enumerate(sorted(os.listdir(images_path))):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(images_path, filename)
    if not os.path.isfile(path):
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    total += 1
    var = img.var()

    if var < variance_threshold:
        empty_count += 1
        shutil.move(path, os.path.join(empty_dir, filename))
        print(f"[{i+1}] {filename} → ⚠️ Vide / noire (var: {var:.2f})")
    else:
        ok_count += 1
        print(f"[{i+1}] {filename} → ✅ OK (var: {var:.2f})")

print("\n📋 Cleaning Summary :")
print(f"➡️  Total : {total}")
print(f"⚠️  Moved to 'empty' : {empty_count}")
print(f"✅ Images retained : {ok_count}")
