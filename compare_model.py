from ultralytics import YOLO
import os, glob, cv2, numpy as np, matplotlib.pyplot as plt

model1 = YOLO('runs_yolo/my_experiment7/weights/best.pt')
model2 = YOLO('runs_yolo/my_experiment15/weights/best.pt')

def run_model(model, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = model(img)
    b = r[0].boxes
    return b.conf.cpu().numpy() if (b is not None and len(b) > 0) else []

print("wd:", os.getcwd())

patterns = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')  # opencv usually can't read HEIC
paths = []
for p in patterns:
    paths += glob.glob(p)

print(f"found {len(paths)} images")
if len(paths) > 0:
    print("first few:", paths[:5])

conf1, conf2 = [], []

for p in paths:
    img = cv2.imread(p)
    if img is None:
        print(f"skip (unreadable): {p}")
        continue
    conf1.extend(run_model(model1, img))
    conf2.extend(run_model(model2, img))

if len(conf1) == 0 and len(conf2) == 0:
    print("no detections collected. check that images are readable (not HEIC) and that models detect anything.")
else:
    plt.hist(conf1, bins=20, alpha=0.5, label='model 1')
    plt.hist(conf2, bins=20, alpha=0.5, label='model 2')
    if len(conf1): plt.axvline(np.mean(conf1), linestyle='--', linewidth=2, label=f'm1 mean {np.mean(conf1):.2f}')
    if len(conf2): plt.axvline(np.mean(conf2), linestyle='--', linewidth=2, label=f'm2 mean {np.mean(conf2):.2f}')
    plt.title('yolo confidence comparison')
    plt.xlabel('confidence')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("saved: confidence_comparison.png")
