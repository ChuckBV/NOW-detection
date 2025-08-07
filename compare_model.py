from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

model1 = YOLO('runs_yolo/my_experiment7/weights/best.pt')
model2 = YOLO('runs_yolo/my_experiment15/weights/best.pt')

def run_model(model, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img)
    boxes = results[0].boxes
    if boxes is not None:
        return boxes.conf.cpu().numpy()
    return []

# Scan images in current working directory
paths = [f for f in os.listdir() if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

conf1 = []
conf2 = []

for path in paths:
    img = cv2.imread(path)
    conf1.extend(run_model(model1, img))
    conf2.extend(run_model(model2, img))

plt.hist(conf1, bins=20, alpha=0.5, label='Model 1', color='blue', edgecolor='black')
plt.hist(conf2, bins=20, alpha=0.5, label='Model 2', color='red', edgecolor='black')
plt.axvline(np.mean(conf1), color='blue', linestyle='--', linewidth=2, label=f'Model 1 Mean: {np.mean(conf1):.2f}')
plt.axvline(np.mean(conf2), color='red', linestyle='--', linewidth=2, label=f'Model 2 Mean: {np.mean(conf2):.2f}')

plt.title('YOLO Model Confidence Score Comparison')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
