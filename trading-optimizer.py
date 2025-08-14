from ultralytics import YOLO
import ultralytics

def main():
    print("ultralytics version:", ultralytics.__version__)

    old_model_path = "test_venv/best1.pt"
    data_file = "test_venv/data-Copy1.yaml"

    model = YOLO(old_model_path)

    model.train(
        data=data_file,
        epochs=300,
        imgsz=960,
        batch=8,
        cache=True,
        cls=1.8,
        box=7.0,
        mosaic=0.70,
        mixup=0.0,
        translate=0.10,
        fliplr=0.50,
        degrees=0.0,
        shear=0.0,
        scale=0.0,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        label_smoothing=0.05,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.001,
        close_mosaic=15,
        workers=8
    )

    model.val(
        data=data_file,
        imgsz=960,
        conf=0.60,
        iou=0.60,
        agnostic_nms=False
    )

if __name__ == "__main__":
    main()
