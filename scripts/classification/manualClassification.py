import cv2
import os
import shutil

images_dir = "/batch_00/images"
no_ball_dir = os.path.join(images_dir, "without_ball")

os.makedirs(no_ball_dir, exist_ok=True)

yes_count = 0
no_count = 0

for filename in sorted(os.listdir(images_dir)):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(images_dir, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    # 📺 Affichage image
    cv2.imshow("Ballon présent ? (y = oui / n = non / q = quitter)", img)
    key = cv2.waitKey(0)

    if key == ord("y"):
        yes_count += 1
        print(f"✅ {filename})
    elif key == ord("n"):
        shutil.move(path, os.path.join(no_ball_dir, filename))
        no_count += 1
        print(f"❌ {filename} → déplacé dans 'without_ball'")
    elif key == ord("q"):
        print("⏹️  Interruption manuelle.")
        break

cv2.destroyAllWindows()

# 📊 Résumé
print("\n📋 Summary :")
print(f"✅ With ball    : {yes_count}")
print(f"❌ Without ball : {no_count}")
