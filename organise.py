import os
import random
import shutil

# ---- EDIT THESE PATHS BEFORE RUNNING ----
SOURCE_DIR = r"path\to\all_jpg_and_xml"  # Contains .jpg images + .xml labels
DEST_DIR = "dataset"                     # Name (or path) of the organized output folder
TRAIN_RATIO = 0.8                        # 80% train, 20% val
# -----------------------------------------

def main():
    # 1. Make the output folders
    os.makedirs(os.path.join(DEST_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "labels", "val"), exist_ok=True)

    # 2. Gather all .jpg files in SOURCE_DIR
    all_jpg = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]
    random.shuffle(all_jpg)

    total_images = len(all_jpg)
    train_count = int(TRAIN_RATIO * total_images)

    print(f"Found {total_images} .jpg files in '{SOURCE_DIR}'.")
    print(f"Splitting {train_count} for train, {total_images - train_count} for val.")

    # 3. Copy function: moves one image + its XML label (if exists)
    def copy_img_and_xml(jpg_name, subset):
        # Copy the image
        src_img = os.path.join(SOURCE_DIR, jpg_name)
        dst_img = os.path.join(DEST_DIR, "images", subset, jpg_name)
        shutil.copy2(src_img, dst_img)

        # Copy the matching XML (if present)
        xml_name = os.path.splitext(jpg_name)[0] + ".xml"
        src_xml = os.path.join(SOURCE_DIR, xml_name)
        dst_xml = os.path.join(DEST_DIR, "labels", subset, xml_name)
        if os.path.exists(src_xml):
            shutil.copy2(src_xml, dst_xml)
        else:
            print(f"Warning: No XML file found for {jpg_name}")

    # 4. Distribute the files between train and val
    for i, jpg_file in enumerate(all_jpg):
        if i < train_count:
            copy_img_and_xml(jpg_file, "train")
        else:
            copy_img_and_xml(jpg_file, "val")

    print("✅ Done! Organized dataset is in:")
    print(f"   {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    main()
