import os
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model_path = "best.pt"
image_dir = "images"
iou_threshold = 0.5

model = YOLO(model_path)

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

tp_confidences = []

for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, filename)
    xml_path = os.path.splitext(image_path)[0] + ".xml"

    if not os.path.exists(xml_path):
        continue

    gt_boxes = parse_xml(xml_path)
    results = model(image_path, verbose=False)
    for result in results:
        for box in result.boxes:
            pred_box = box.xyxy[0].cpu().numpy()
            pred_conf = box.conf.item()
            for gt_box in gt_boxes:
                if iou(pred_box, gt_box) >= iou_threshold:
                    tp_confidences.append(pred_conf)
                    break

print("found", len(tp_confidences), "true positive confidence scores")

plt.hist(tp_confidences, bins=30, color='green', alpha=0.7)
plt.title("confidence score distribution")
plt.xlabel("confidence")
plt.ylabel("count")
plt.grid(True)
plt.show()
