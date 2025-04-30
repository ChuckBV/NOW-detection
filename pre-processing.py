import cv2
import numpy as np
import os

def preprocess_insect_image(image_path, save_path=None, extract_bbox=False):
    img = cv2.imread(image_path)
    original = img.copy()

    # to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # separate foreground
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 7)

    # remove noise
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # filter small blobs
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # apply mask
    result = cv2.bitwise_and(original, original, mask=mask)

    # save output
    if save_path:
        cv2.imwrite(save_path, result)

    # return cropped insect
    if extract_bbox:
        x, y, w, h = cv2.boundingRect(mask)
        cropped = result[y:y+h, x:x+w]
        return cropped, (x, y, w, h)

    return result

if __name__ == "__main__":
    input_folder = "raw_images"
    output_folder = "processed_images"
    os.makedirs(output_folder, exist_ok=True)

    # batch process
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            preprocess_insect_image(input_path, save_path=output_path)
