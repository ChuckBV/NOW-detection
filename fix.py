import os

def delete_copies(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    deleted = 0
    for filename in os.listdir(folder_path):
        if "copy" in filename:
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"Deleted: {filename}")
                deleted += 1
    print(f"Total deleted: {deleted}")

# Example usage
folder = "/path/to/your/folder"  # <- Replace this with the correct folder
delete_copies(folder)
