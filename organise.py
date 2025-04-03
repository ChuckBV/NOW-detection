import os, glob, random, shutil
import xml.etree.ElementTree as ET

# Change these as needed:
INPUT_DIR = "path_to_unsorted"  # Folder with mixed .jpg and .xml
OUTPUT_DIR = "path_to_output"   # Will create images/ and labels/ subfolders
TRAIN_RATIO = 0.8               # How much data goes to 'train' vs 'val'
RANDOM_SEED = 42                # For reproducible shuffle if needed

def parse_xml(xml_path):
    t = ET.parse(xml_path).getroot()
    w = float(t.find("size/width").text)
    h = float(t.find("size/height").text)
    boxes = []
    for obj in t.iter("object"):
        cls_name = obj.find("name").text
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        boxes.append((cls_name, xmin, ymin, xmax, ymax))
    return boxes, w, h

def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    x_center = ((xmin + xmax) / 2.0) / w
    y_center = ((ymin + ymax) / 2.0) / h
    box_w = (xmax - xmin) / w
    box_h = (ymax - ymin) / h
    return x_center, y_center, box_w, box_h

def main():
    random.seed(RANDOM_SEED)
    all_jpgs = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    data_pairs = []
    for jpg in all_jpgs:
        base = os.path.splitext(os.path.basename(jpg))[0]
        xml = os.path.join(INPUT_DIR, base + ".xml")
        if os.path.isfile(xml):
            data_pairs.append((jpg, xml))

    random.shuffle(data_pairs)
    split_index = int(len(data_pairs) * TRAIN_RATIO)
    train_pairs = data_pairs[:split_index]
    val_pairs   = data_pairs[split_index:]

    # Create output folders
    img_train = os.path.join(OUTPUT_DIR, "images", "train")
    img_val   = os.path.join(OUTPUT_DIR, "images", "val")
    lbl_train = os.path.join(OUTPUT_DIR, "labels", "train")
    lbl_val   = os.path.join(OUTPUT_DIR, "labels", "val")
    for d in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(d, exist_ok=True)

    # Collect all classes so we can assign them numeric IDs
    all_classes = set()
    for (jpg, xml) in data_pairs:
        boxes, _, _ = parse_xml(xml)
        for (c, *_rest) in boxes:
            all_classes.add(c)
    all_classes = sorted(list(all_classes))
    class_to_id = {c: i for i, c in enumerate(all_classes)}

    def process_split(pairs, img_out, lbl_out):
        for (jpg_path, xml_path) in pairs:
            base = os.path.splitext(os.path.basename(jpg_path))[0]
            shutil.copy2(jpg_path, os.path.join(img_out, base + ".jpg"))
            boxes, w, h = parse_xml(xml_path)
            lines = []
            for (cls_name, xmin, ymin, xmax, ymax) in boxes:
                x_c, y_c, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
                lines.append(f"{class_to_id[cls_name]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
            with open(os.path.join(lbl_out, base + ".txt"), "w") as f:
                f.writelines(lines)

    process_split(train_pairs, img_train, lbl_train)
    process_split(val_pairs,   img_val,   lbl_val)

    print("Done sorting images and labels!")
    print(f"Images and labels organized under: {OUTPUT_DIR}")
    print(f"Classes found: {all_classes}")

if __name__ == "__main__":
    main()
