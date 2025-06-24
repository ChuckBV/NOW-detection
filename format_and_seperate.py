import os
import shutil
import random
import xmltodict
import cv2

# folder stuff 
source_folder = "/path/to/mixed_folder"         # Input folder with .jpg and .xml
output_folder = "/path/to/output_yolo_dataset"  # Output folder

# ratio stuff
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15
# working stuff

def voc_to_yolo(xml_path, img_width, img_height):
    with open(xml_path) as f:
        data = xmltodict.parse(f.read())
    yolo_labels = []
    objects = data['annotation'].get('object', [])
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        cls = obj['name']
        bbox = obj['bndbox']
        xmin = int(bbox['xmin'])
        xmax = int(bbox['xmax'])
        ymin = int(bbox['ymin'])
        ymax = int(bbox['ymax'])

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")  # class 0
    return yolo_labels

def prepare_yolo_split_dataset(src_folder, out_folder, ratios):
    if not os.path.isdir(src_folder):
        print("Invalid source folder.")
        return

    all_pairs = []
    for filename in os.listdir(src_folder):
        if filename.lower().endswith('.jpg'):
            name_no_ext = os.path.splitext(filename)[0]
            jpg_path = os.path.join(src_folder, filename)
            xml_path = os.path.join(src_folder, name_no_ext + '.xml')
            if os.path.exists(xml_path):
                all_pairs.append((jpg_path, xml_path))

    print(f"Found {len(all_pairs)} valid image+xml pairs")

    # Shuffle/ split
    random.shuffle(all_pairs)
    total = len(all_pairs)
    n_train = int(ratios[0] * total)
    n_val   = int(ratios[1] * total)
    n_test  = total - n_train - n_val

    train_set = all_pairs[:n_train]
    val_set   = all_pairs[n_train:n_train + n_val]
    test_set  = all_pairs[n_train + n_val:]

    splits = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    for split in splits:
        os.makedirs(os.path.join(out_folder, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_folder, 'labels', split), exist_ok=True)

    for split, items in splits.items():
        for jpg_path, xml_path in items:
            img = cv2.imread(jpg_path)
            if img is None:
                print(f"Could not read {jpg_path}, skipping.")
                continue
            h, w = img.shape[:2]
            yolo_lines = voc_to_yolo(xml_path, w, h)

            fname = os.path.basename(jpg_path)
            name_no_ext = os.path.splitext(fname)[0]

            shutil.copy2(jpg_path, os.path.join(out_folder, 'images', split, fname))
            with open(os.path.join(out_folder, 'labels', split, name_no_ext + '.txt'), 'w') as f:
                f.write('\n'.join(yolo_lines))

    print("✅ YOLOv8 dataset ready with splits:")
    print(f" - Train: {len(train_set)}")
    print(f" - Val:   {len(val_set)}")
    print(f" - Test:  {len(test_set)}")

# ====== RUN SCRIPT ======
prepare_yolo_split_dataset(source_folder, output_folder, (train_ratio, val_ratio, test_ratio))
