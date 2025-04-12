import cv2
import os


clips_dir = "/clips"
output_base = "/batches"

os.makedirs(output_base, exist_ok=True)

global_frame_counter = 0

videos = sorted([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])

for index, video_name in enumerate(videos):
    batch_name = f"batch_{index:02d}"
    batch_path = os.path.join(output_base, batch_name)
    images_path = os.path.join(batch_path, "images")
    labels_path = os.path.join(batch_path, "labels")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    video_path = os.path.join(clips_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_name = f"{global_frame_counter:06d}.jpg"
        img_path = os.path.join(images_path, img_name)
        cv2.imwrite(img_path, frame)

        global_frame_counter += 1
        frame_id += 1

    cap.release()
    print(f"✅ End {video_name} → {frame_id} frames")

print("\n🎉 All batches have been created successfully !")
