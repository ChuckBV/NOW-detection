import os
import shutil

def duplicate_to_new_folder(source_folder, target_folder):
    if not os.path.isdir(source_folder):
        print("Invalid source folder.")
        return

    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.xml'):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            shutil.copy2(source_path, target_path)
            print(f"Copied: {filename} -> {target_folder}")

# Example usage
source = "/path/to/source"    # <- Replace with actual source folder path
target = "/path/to/target"    # <- Replace with actual destination folder path
duplicate_to_new_folder(source, target)
