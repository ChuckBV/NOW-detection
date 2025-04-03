# train_new_model.py

from ultralytics import YOLO

# Change these to match your paths and preferred settings:
OLD_MODEL = "your_old_model.pt"     # Pre-trained model to fine-tune
DATA_CONFIG = "path/to/data.yaml"   # Data config with train/val info
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640

def main():
    model = YOLO(OLD_MODEL)
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE
    )
    model.val()  # Optional: run validation after training

if __name__ == "__main__":
    main()
