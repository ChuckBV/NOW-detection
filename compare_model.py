from ultralytics import YOLO
import os, glob, cv2, numpy as np, matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

m1_path = 'runs_yolo/my_experiment7/weights/best.pt'
m2_path = 'runs_yolo/my_experiment15/weights/best.pt'
images_dir = 'test_venv/USE-now-images'
iou_thr = 0.5

model1 = YOLO(m1_path)
model2 = YOLO(m2_path)

def voc_boxes(xml_path):
    if not os.path.exists(xml_path): return []
    r = ET.parse(xml_path).getroot()
    out = []
    for obj in r.findall('object'):
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text); ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text); ymax = float(bb.find('ymax').text)
        out.append([xmin, ymin, xmax, ymax])
    return np.array(out, dtype=np.float32)

def iou(a, b):
    # a: [N,4], b:[M,4] in xyxy
    if a.size == 0 or b.size == 0: return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0][:,None], a[:,1][:,None], a[:,2][:,None], a[:,3][:,None]
    bx1, by1, bx2, by2 = b[:,0][None,:], b[:,1][None,:], b[:,2][None,:], b[:,3][None,:]
    inter_w = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h
    area_a = (a[:,2]-a[:,0])[:,None]*(a[:,3]-a[:,1])[:,None]
    area_b = (b[:,2]-b[:,0])[None,:]*(b[:,3]-b[:,1])[None,:]
    union = area_a + area_b - inter + 1e-9
    return inter/union

def run(model, img_rgb):
    r = model.predict(img_rgb, conf=0.05, iou=0.45, verbose=False)
    b = r[0].boxes
    if b is None or len(b)==0:
        return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xyxy = b.xyxy.cpu().numpy().astype(np.float32)
    conf = b.conf.cpu().numpy().astype(np.float32)
    return xyxy, conf

exts = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')
paths = [p for p in glob.glob(os.path.join(images_dir, '*')) if os.path.splitext(p.lower())[1] in exts]

conf_all_1, conf_tp_1, conf_fp_1 = [], [], []
conf_all_2, conf_tp_2, conf_fp_2 = [], [], []

for p in paths:
    img = cv2.imread(p)
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xml = os.path.splitext(p)[0] + '.xml'
    gts = voc_boxes(xml)

    for model, conf_all, conf_tp, conf_fp in [
        (model1, conf_all_1, conf_tp_1, conf_fp_1),
        (model2, conf_all_2, conf_tp_2, conf_fp_2),
    ]:
        pred_xyxy, pred_conf = run(model, img)
        conf_all.extend(pred_conf.tolist())
        if len(pred_conf)==0:
            continue
        if len(gts)==0:
            conf_fp.extend(pred_conf.tolist())
        else:
            M = iou(pred_xyxy, gts)
            max_iou = M.max(axis=1) if M.size else np.zeros(len(pred_conf))
            for c, mi in zip(pred_conf, max_iou):
                (conf_tp if mi >= iou_thr else conf_fp).append(float(c))

def plot_and_save(c1, c2, title, outname):
    if not c1 and not c2:
        print(f"no data for {outname}")
        return
    plt.figure()
    if c1: plt.hist(c1, bins=20, alpha=0.5, label='model 1')
    if c2: plt.hist(c2, bins=20, alpha=0.5, label='model 2')
    if c1: plt.axvline(np.mean(c1), linestyle='--', linewidth=2, label=f'm1 mean {np.mean(c1):.2f}')
    if c2: plt.axvline(np.mean(c2), linestyle='--', linewidth=2, label=f'm2 mean {np.mean(c2):.2f}')
    plt.title(title); plt.xlabel('confidence'); plt.ylabel('frequency'); plt.legend(); plt.grid(True)
    plt.savefig(outname, dpi=300, bbox_inches='tight'); plt.show()
    print('saved:', outname)

print(f"images: {len(paths)}")
print(f"m1 total:{len(conf_all_1)} tp:{len(conf_tp_1)} fp:{len(conf_fp_1)}")
print(f"m2 total:{len(conf_all_2)} tp:{len(conf_tp_2)} fp:{len(conf_fp_2)}")

plot_and_save(conf_all_1, conf_all_2, 'all predictions', 'conf_all_hist.png')
plot_and_save(conf_tp_1,  conf_tp_2,  'true positives (iou≥0.5)', 'conf_tp_hist.png')
plot_and_save(conf_fp_1,  conf_fp_2,  'false positives (iou<0.5)', 'conf_fp_hist.png')
