from ultralytics import YOLO

# Use your own model file
model = YOLO("best.pt")  

# Inference on images
results = model.predict(
    source="images/",
    save=True,
    conf=0.25,
    imgsz=640,
)

print("Done running custom model.")
