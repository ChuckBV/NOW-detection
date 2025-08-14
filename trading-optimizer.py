yolo detect train model=yolov8m.pt data=data.yaml epochs=300 imgsz=960 batch=8 cache=True \
  cls=1.8 fl_gamma=2.0 box=7.0 iou=0.70 \
  mosaic=0.70 mixup=0.0 translate=0.10 fliplr=0.50 degrees=0 shear=0 scale=0.0 \
  hsv_h=0.015 hsv_s=0.70 hsv_v=0.40 \
  label_smoothing=0.05 optimizer=AdamW lr0=0.001 lrf=0.1 weight_decay=0.001 close_mosaic=15
