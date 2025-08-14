from ultralytics import YOLO
import os, glob
import numpy as np
import pandas as pd

model_path = 'runs_yolo/my_experiment15/weights/best.pt'
images_dir = 'test_venv/USE-now-images'
output_csv = 'detections_summary.csv'
thr = 0.50
conf_all = 0.01
iou_nms = 0.50

model = YOLO(model_path)

#list images
exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
paths = []
for e in exts:
    paths += glob.glob(os.path.join(images_dir, e))
paths = sorted(paths)

#run
rows = []
for p in paths:
    r = model.predict(p, conf=conf_all, iou=iou_nms, verbose=False)
    b = r[0].boxes
    if b is None or len(b)==0:
        total = 0
        over = 0
    else:
        conf = b.conf.cpu().numpy()
        total = int(conf.size)
        over = int((conf >= thr).sum())
    rows.append({'file': os.path.basename(p), 'total_detections': total, f'detections_over_{thr:.2f}': over})

#save csv
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

#show table
df
#summary
avg_total = df['total_detections'].mean() if len(df) else 0.0
sum_total = df['total_detections'].sum() if len(df) else 0
col_over = f'detections_over_{thr:.2f}'
avg_over = df[col_over].mean() if len(df) else 0.0
sum_over = df[col_over].sum() if len(df) else 0

print(f'saved csv: {output_csv}')
print(f'images: {len(df)}')
print(f'avg total detections/image: {avg_total:.3f}')
print(f'total detections: {sum_total}')
print(f'avg detections ≥ {thr:.2f}/image: {avg_over:.3f}')
print(f'total detections ≥ {thr:.2f}: {sum_over}')
