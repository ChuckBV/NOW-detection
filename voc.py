import os
import random
import shutil

SOURCE_DIR = "source"  
VOC_DIR = "VOCdataset"      # <-- Will create/overwrite this folder
TRAIN_RATIO = 0.8           # 80% for training, 20% for validation

# Create VOC subfolders
jpeg_dir = os.path.join(VOC_DIR, "JPEGImages")
ann_dir = os.path.join(VOC_DIR, "Annotations")
main_dir = os.path.join(VOC_DIR, "ImageSets", "Main")

os.makedirs(jpeg_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)

# Find all .jpg files in SOURCE_DIR
all_jpg = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]
random.shuffle(all_jpg)

total_images = len(all_jpg)
train_count = int(TRAIN_RATIO * total_images)

print(f"Found {total_images} .jpg images in '{SOURCE_DIR}'.")
print(f"Splitting {train_count} for train, {total_images - train_count} for val.")

# Copy each image + matching .xml to VOCdataset folders
#    We'll also record each image's "ID" (filename without extension) in train/val lists.
train_ids = []
val_ids = []

def copy_to_voc(jpg_file, is_train=True):
    # Source paths
    src_jpg_path = os.path.join(SOURCE_DIR, jpg_file)
    xml_file = os.path.splitext(jpg_file)[0] + ".xml"
    src_xml_path = os.path.join(SOURCE_DIR, xml_file)

    # Destination paths
    dst_jpg_path = os.path.join(jpeg_dir, jpg_file)
    dst_xml_path = os.path.join(ann_dir, xml_file)

    # Copy image
    shutil.copy2(src_jpg_path, dst_jpg_path)

    # Copy xml if exists
    if os.path.exists(src_xml_path):
        shutil.copy2(src_xml_path, dst_xml_path)
    else:
        print(f"Warning: No .xml found for {jpg_file}")

    # Return the "ID" (filename without extension)
    return os.path.splitext(jpg_file)[0]

for i, jpg in enumerate(all_jpg):
    if i < train_count:
        img_id = copy_to_voc(jpg, is_train=True)
        train_ids.append(img_id)
    else:
        img_id = copy_to_voc(jpg, is_train=False)
        val_ids.append(img_id)

# Write train/val IDs into ImageSets/Main/*.txt
train_txt_path = os.path.join(main_dir, "train.txt")
val_txt_path = os.path.join(main_dir, "val.txt")

with open(train_txt_path, "w") as f:
    for tid in train_ids:
        f.write(tid + "\n")

with open(val_txt_path, "w") as f:
    for vid in val_ids:
        f.write(vid + "\n")

print("\n✅ Pascal VOC-format dataset created at:", os.path.abspath(VOC_DIR))
print(f"  JPEGImages: {len(os.listdir(jpeg_dir))} files")
print(f"  Annotations: {len(os.listdir(ann_dir))} files")
print(f"  train.txt: {len(train_ids)} IDs")
print(f"  val.txt: {len(val_ids)} IDs")
