import os
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import random

def test_model_on_folder(model_path, image_dir, iou_threshold=0.5, conf_threshold=0.4, num_samples=4):
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
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(image_files)
    sample_images = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        xml_path = os.path.splitext(image_path)[0] + ".xml"
        if not os.path.exists(xml_path):
            continue

        gt_boxes = parse_xml(xml_path)
        img = cv2.imread(image_path)
        results = model(image_path, verbose=False)

        total_detections = 0
        detections_over_threshold = 0

        for result in results:
            for box in result.boxes:
                pred_box = box.xyxy[0].cpu().numpy()
                pred_conf = box.conf.item()
                total_detections += 1
                if pred_conf >= conf_threshold:
                    detections_over_threshold += 1
                matched = False
                for gt_box in gt_boxes:
                    if iou(pred_box, gt_box) >= iou_threshold:
                        tp_confidences.append(pred_conf)
                        matched = True
                        break

                # draw prediction boxes in yellow
                x1, y1, x2, y2 = map(int, pred_box)
                color = (255, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{pred_conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # draw ground truth boxes in green
        for gt in gt_boxes:
            x1, y1, x2, y2 = gt
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # overlay counts
        summary_text = f"total: {total_detections}  over {int(conf_threshold*100)}%: {detections_over_threshold}"
        cv2.putText(img, summary_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        print(f"{filename} - total: {total_detections}, over {int(conf_threshold*100)}%: {detections_over_threshold}")

        if len(sample_images) < num_samples:
            sample_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print("found", len(tp_confidences), "true positive confidence scores")

    # plot
    plt.figure(figsize=(8, 6))
    plt.hist(tp_confidences, bins=30, color='green', alpha=0.7)
    plt.title("confidence score distribution")
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.grid(True)
    plt.show()

    # plot sample images
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
    for ax, img in zip(axes, sample_images):
        ax.imshow(img)
        ax.axis("off")
    plt.show()
