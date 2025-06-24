import os
import shutil

def duplicate_files(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and (filename.endswith('.jpg') or filename.endswith('.xml')):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_copy{ext}"
            new_path = os.path.join(folder_path, new_filename)
            shutil.copy2(full_path, new_path)
            print(f"Duplicated: {filename} -> {new_filename}")

# Example usage
folder = "/path/to/your/folder"  # <- Replace with the actual path
duplicate_files(folder)
