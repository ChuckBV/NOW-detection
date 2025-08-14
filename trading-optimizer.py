from ultralytics import YOLO

def main():
    old_model_path = "test_venv/best1.pt"          # or "yolov8m.pt" to start from base
    data_file = "test_venv/data-Copy1.yaml"

    model = YOLO(old_model_path)

    model.train(
        data=data_file,
        epochs=300,
        imgsz=960,
        batch=8,
        cache=True,

        # reduce false positives
        cls=1.8,
        fl_gamma=2.0,
        box=7.0,
        iou=0.70,              # NMS IoU during train/val

        # sane augmentations
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

        # regularization / optimizer
        label_smoothing=0.05,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.001,
        close_mosaic=15,

        # misc
        workers=8
    )

    # optional: eval on your val split after training
    # model.val(data=data_file)

if __name__ == "__main__":
    main()
