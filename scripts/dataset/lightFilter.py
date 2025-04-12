import cv2
import os
import shutil

batch_path = "/batch_00"
images_path = os.path.join(batch_path, "images")

dark_dir = os.path.join(batch_path, "dark")
bright_dir = os.path.join(batch_path, "bright")
os.makedirs(dark_dir, exist_ok=True)
os.makedirs(bright_dir, exist_ok=True)

dark_threshold = 20
bright_threshold = 230

total = 0
dark_count = 0
bright_count = 0
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

    mean_brightness = img.mean()
    total += 1

    if mean_brightness < dark_threshold:
        dark_count += 1
        shutil.move(path, os.path.join(dark_dir, filename))
        print(f"[{i+1}] {filename} → 🌑 Trop sombre ({mean_brightness:.2f})")
    elif mean_brightness > bright_threshold:
        bright_count += 1
        shutil.move(path, os.path.join(bright_dir, filename))
        print(f"[{i+1}] {filename} → 🌞 Trop clair ({mean_brightness:.2f})")
    else:
        ok_count += 1
        print(f"[{i+1}] {filename} → ✅ OK ({mean_brightness:.2f})")

print("\n📋 Cleaning Summary :")
print(f"➡️  Total : {total}")
print(f"🌑 Moved to 'dark' : {dark_count}")
print(f"🌞 Moved to 'bright' : {bright_count}")
print(f"✅ Images retained : {ok_count}")
