import os
import random
import shutil

# Set these paths before running
SOURCE_DIR = "path/to/all_images_and_labels"
DEST_DIR = "dataset"
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Create folder structure:
# dataset/images/train, dataset/images/val
# dataset/labels/train, dataset/labels/val
os.makedirs(os.path.join(DEST_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "labels", "val"), exist_ok=True)

# Gather all image files (assuming .jpg, .png, .jpeg). Adjust if needed.
valid_exts = {".jpg", ".jpeg", ".png"}
image_files = []
for file_name in os.listdir(SOURCE_DIR):
    ext = os.path.splitext(file_name)[1].lower()
    if ext in valid_exts:
        image_files.append(file_name)

# Shuffle for random train/val split
random.shuffle(image_files)

# Determine how many go to training
num_images = len(image_files)
train_count = int(TRAIN_SPLIT * num_images)

print(f"Found {num_images} images. Splitting {train_count} for train, {num_images-train_count} for val.")

# Helper function to copy one image + its label
def copy_data(img_filename, subset):
    # e.g., subset is "train" or "val"
    src_img_path = os.path.join(SOURCE_DIR, img_filename)
    dst_img_path = os.path.join(DEST_DIR, "images", subset, img_filename)
    shutil.copy2(src_img_path, dst_img_path)

    # Label filename has the same base name, but .txt
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    src_label_path = os.path.join(SOURCE_DIR, label_filename)
    dst_label_path = os.path.join(DEST_DIR, "labels", subset, label_filename)

    # Only copy if label exists
    if os.path.exists(src_label_path):
        shutil.copy2(src_label_path, dst_label_path)
    else:
        print(f"Warning: Label file not found for {img_filename}")

# Copy images and labels to train/val folders
for i, img_file in enumerate(image_files):
    if i < train_count:
        copy_data(img_file, "train")
    else:
        copy_data(img_file, "val")

print("✅ Dataset organized successfully!")
print(f"Check the '{DEST_DIR}' folder for the 'train' and 'val' splits.")
