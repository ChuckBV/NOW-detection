import cv2
import os

def rotate_all_jpegs_90ccw_in_place(folder='.'):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"Could not read {path}, skipping.")
                continue
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path, rotated)
            print(f"Rotated {filename}")

# Run it on the current folder
rotate_all_jpegs_90ccw_in_place()
