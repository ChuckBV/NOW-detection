from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import os

# ---- Step 1: Load the YOLOv8 Model ----
model = YOLO("yolov8n.pt")  # Using the 'n' (nano) version, you can switch to 's', 'm', etc.

# ---- Step 2: Load an Image ----
image_path = "your_image.jpg"  # Replace with your actual image path
image = cv2.imread(image_path)

# Ensure the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

# Convert image to RGB (since OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- Step 3: Run YOLOv8 Inference ----
results = model(image)

# Ensure results is not a list
if isinstance(results, list):
    results = results[0]

# ---- Step 4: Draw Bounding Boxes ----
# Extract detections
for result in results.boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
    confidence = result.conf[0]  # Confidence score
    class_id = int(result.cls[0])  # Class ID
    label = model.names[class_id]  # Get class name

    # Draw the bounding box
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image_rgb, f"{label} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# ---- Step 5: Save & Display the Image ----
output_path = "output.jpg"
cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

# Check if the file was saved
if os.path.exists(output_path):
    print(f"✅ Results saved successfully: {output_path}")
else:
    print("❌ Error: Output image was not saved.")

# ---- Step 6: Display Image in Jupyter Notebook (if applicable) ----
output_image = Image.open(output_path)
display(output_image)

# ---- Step 7: Allow Downloading the Image (for web-based Jupyter) ----
import shutil
shutil.move(output_path, "/mnt/data/output.jpg")  # Move to a directory where it can be downloaded

print("📥 Download your image from: /mnt/data/output.jpg")
