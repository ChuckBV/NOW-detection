import torch
import cv2
import os
import matplotlib.pyplot as plt

model1 = torch.load('runs_yolo/model1/weights/best.pt')
model2 = torch.load('runs_yolo/model2/weights/best.pt')
model1.eval()
model2.eval()

def run_model(model, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img)
    boxes = results[0].boxes
    if boxes is not None:
        return boxes.conf.cpu().numpy()
    return []

folder = 'test_venv/USE-now-images'
paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

conf1 = []
conf2 = []

for path in paths:
    img = cv2.imread(path)
    conf1.extend(run_model(model1, img))
    conf2.extend(run_model(model2, img))

plt.hist(conf1, bins=10, alpha=0.5, label='Model 1')
plt.hist(conf2, bins=10, alpha=0.5, label='Model 2')
plt.title('Confidence Score Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()
plt.show()
