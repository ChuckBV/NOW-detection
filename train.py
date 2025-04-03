from ultralytics import YOLO

def train_model():
    model = YOLO("your_old_model.pt")  # Start from an existing model (or 'yolov8n.pt' etc.)
    model.train(
        data="/path/to/data.yaml",
        epochs=50,
        imgsz=640,
        project="my_project",    # your main directory for runs
        name="my_experiment",    # your run subfolder
        batch=16,                # choose your batch size
        lr0=0.01                 # optional: initial learning rate
    )

if __name__ == "__main__":
    train_model()
    print("Training complete!")
