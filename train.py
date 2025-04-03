# train_yolo.py
from ultralytics import YOLO

def main():
    # Replace with your existing (older) model path if you want to fine-tune that:
    old_model_path = "your_old_model.pt"
    # Or if you prefer to start from a base model (e.g., yolov8n.pt), change above.

    # Point this to your dataset's data.yaml:
    data_file = "/project/ai-for-trap-processing-now/datasets/l1/data.yaml"

    # Create a YOLO object from your existing model:
    model = YOLO(old_model_path)

    # Train:
    model.train(
        data=data_file,
        epochs=50,          # Number of training epochs
        imgsz=640,          # Image size
        batch=16,           # Batch size
        project="runs_yolo",# Folder to save training results
        name="my_experiment" # Sub-folder name under 'runs_yolo'
    )

    # Optional: validate after training
    # model.val()

if __name__ == "__main__":
    main()
