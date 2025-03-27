from ultralytics import YOLO

# Load your existing YOLO model as the starting point
model = YOLO("old_model.pt")  

# Train on VOC-style dataset
model.train(
    data="voc_data.yaml",  # The YAML we just created
    epochs=50,             # Adjust how many epochs you want
    imgsz=640,            # Image resolution
    batch=8,               # Adjust for your hardware
    name="voc-training-run"
)
