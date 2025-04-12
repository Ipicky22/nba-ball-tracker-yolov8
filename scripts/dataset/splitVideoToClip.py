import cv2
import os

input_path = "video.mp4" 
output_dir = "clips"
segment_minutes = 5            

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(input_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_per_segment = fps * segment_minutes * 60
segment_index = 0
frame_count = 0

out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frames_per_segment == 0:
        if out:
            out.release()
        segment_path = os.path.join(output_dir, f"segment_{segment_index:03d}.mp4")
        out = cv2.VideoWriter(
            segment_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        print(f"▶️ New segment : {segment_path}")
        segment_index += 1

    out.write(frame)
    frame_count += 1

cap.release()
if out:
    out.release()

print("✅ Cutting completed !")
