from ultralytics import YOLO
import os
import shutil

images_dir = "/batch_00/images"
without_ball_dir = os.path.join(os.path.dirname(images_dir), "without_ball")
os.makedirs(without_ball_dir, exist_ok=True)

model = YOLO("/models/classification/best.pt")

image_files = sorted([
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

kept, removed = 0, 0
for i, filename in enumerate(image_files):
    image_path = os.path.join(images_dir, filename)

    results = model.predict(source=image_path, imgsz=224, save=False, verbose=False)

    # (0 = ball, 1 = no_ball)
    predicted_label = results[0].names[results[0].probs.top1]

    if predicted_label == "no_ball":
        shutil.move(image_path, os.path.join(without_ball_dir, filename))
        removed += 1
        print(f"[{i+1}] ❌ {filename} → déplacé (no_ball)")
    else:
        kept += 1
        print(f"[{i+1}] ✅ {filename} → conservé (ball)")

print("\n📋 Cleaning Summary:")
print(f"✅ Images retained (ball)     : {kept}")
print(f"❌ Images displaced (no_ball)   : {removed}")
