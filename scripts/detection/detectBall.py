from ultralytics import YOLO
import os

model_path = "/models/detection/yolo_n_1100/best.pt"  
images_dir = "/batches/batch_00/images"                 
output_labels_dir = "/batches/batch_00/labels"   
conf_threshold = 0.4                                    

os.makedirs(output_labels_dir, exist_ok=True)

model = YOLO(model_path)

results = model.predict(
    source=images_dir,
    conf=conf_threshold,
    save=False,
    save_txt=True,
    save_conf=False,
    project=output_labels_dir,
    name='',
    exist_ok=True,
)

print("✅ YOLOv8 pre-annotations generated.")
