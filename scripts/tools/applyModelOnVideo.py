from ultralytics import YOLO

model = YOLO("YOUR_MODEL.pt")

results = model.predict(
    source="YOUR_VIDEO.mp4",        
    save=True,                
    save_txt=True,            
    conf=0.4,                
    imgsz=720,                 
)

print("✅ Detection completed.")
