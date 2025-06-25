from ultralytics import YOLO

def main():
 
    old_model_path = "test_venv/best1.pt"
    data_file = "test_venv/data-Copy1.yaml"

    model = YOLO(old_model_path)

    # Train:
    model.train(
        data=data_file,
        epochs=100,          # Number of training epochs
        imgsz=640,          # Image size
        batch=0.85,           # Batch size
        pretrained = True,
        optimizer = 'auto',
        project="runs_yolo",# Folder to save training results
        name="my_experiment" # Sub-folder name under 'runs_yolo'
    )

    # Optional: validate after training
    # model.val()

if __name__ == "__main__":
    main()
