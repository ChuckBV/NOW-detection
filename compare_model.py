from ultralytics import YOLO
import os, glob, cv2, numpy as np, matplotlib.pyplot as plt

# --- paths (edit these two if needed) ---
m1_path = 'runs_yolo/my_experiment7/weights/best.pt'
m2_path = 'runs_yolo/my_experiment15/weights/best.pt'
images_root = '.'  # current working dir

model1 = YOLO(m1_path)
model2 = YOLO(m2_path)

def list_images(root='.'):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, '**', e), recursive=True)
    return sorted(paths)

def run_conf(model, img):
    r = model.predict(img, conf=0.05, iou=0.45, verbose=False)
    b = r[0].boxes
    return b.conf.cpu().numpy() if (b is not None and len(b)>0) else []

print('wd:', os.getcwd())
paths = list_images(images_root)
print('found images:', len(paths))
if not paths:
    raise SystemExit('no readable images found (convert HEIC to jpg/png or point images_root to the right folder)')

conf1, conf2 = [], []
for p in paths:
    im = cv2.imread(p)
    if im is None:
        # unreadable (likely HEIC) -> skip
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    conf1.extend(run_conf(model1, im))
    conf2.extend(run_conf(model2, im))

if not conf1 and not conf2:
    raise SystemExit('no detections from either model. try lower conf, check model paths, or use images that contain targets.')

# --- histogram comparison ---
plt.figure()
plt.hist(conf1, bins=20, alpha=0.5, label='model 1')
plt.hist(conf2, bins=20, alpha=0.5, label='model 2')
if len(conf1): plt.axvline(np.mean(conf1), linestyle='--', linewidth=2, label=f'm1 mean {np.mean(conf1):.2f}')
if len(conf2): plt.axvline(np.mean(conf2), linestyle='--', linewidth=2, label=f'm2 mean {np.mean(conf2):.2f}')
plt.title('confidence distribution')
plt.xlabel('confidence'); plt.ylabel('frequency'); plt.legend(); plt.grid(True)
plt.savefig('confidence_comparison_hist.png', dpi=300, bbox_inches='tight')
plt.show()

# --- bar chart (avg conf + total detections) ---
avg1 = float(np.mean(conf1)) if conf1 else 0.0
avg2 = float(np.mean(conf2)) if conf2 else 0.0
tot1 = len(conf1)
tot2 = len(conf2)

plt.figure()
x = np.arange(2)
w = 0.35
plt.bar(x - w/2, [avg1, avg2], width=w, label='avg confidence')
plt.bar(x + w/2, [tot1, tot2], width=w, label='total detections')
plt.xticks(x, ['model 1','model 2'])
plt.title('model comparison')
plt.legend(); plt.grid(True, axis='y')
plt.savefig('confidence_comparison_bars.png', dpi=300, bbox_inches='tight')
plt.show()

print('saved: confidence_comparison_hist.png, confidence_comparison_bars.png')
