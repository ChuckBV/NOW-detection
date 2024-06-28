import os
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

def draw_bounding_boxes(image_path, annotation_path, output_path):
    try:
        # Open the image
        print(f"Opening image: {image_path}")
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
    except Exception as e:
        print(f"Error opening image: {image_path}. Error: {e}")
        return
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    try:
        # Parse the annotation file
        print(f"Parsing annotation: {annotation_path}")
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        print(f"Root tag: {root.tag}, attributes: {root.attrib}")
        
        # Debugging: Print the structure of the XML file
        for elem in root.iter():
            print(f"Element: {elem.tag}, attributes: {elem.attrib}")
            if elem.text:
                print(f"Text: {elem.text.strip()}")
    except Exception as e:
        print(f"Error parsing annotation file: {annotation_path}. Error: {e}")
        return

    # Loop through each object in the annotation file
    for obj in root.findall('object'):
        try:
            # Get the bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            print(f"Drawing box: ({xmin}, {ymin}), ({xmax}, {ymax})")

            # Draw a rectangle around the object
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="green", width=2)

            # Optionally, put label on the bounding box
            label = obj.find('name').text
            draw.text((xmin, ymin - 10), label, fill="green", font=font)
        except Exception as e:
            print(f"Error processing object in annotation: {e}")
            continue
    
    try:
        # Save the output image
        print(f"Saving output to: {output_path}")
        image.save(output_path)
    except Exception as e:
        print(f"Error saving image: {output_path}. Error: {e}")

# Paths
base_path = '/Users/zacharydawson/Desktop/NOW-Obj-Detct'
images_path = os.path.join(base_path, 'Train')
annotations_path = os.path.join(base_path, 'Val')
output_path = os.path.join(base_path, 'Test')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Loop through each image and its corresponding annotation
for image_name in os.listdir(images_path):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(images_path, image_name)
        annotation_path = os.path.join(annotations_path, image_name.replace('.jpg', '.xml'))
        
        if os.path.exists(annotation_path):
            output_image_path = os.path.join(output_path, image_name)
            draw_bounding_boxes(image_path, annotation_path, output_image_path)
        else:
            print(f"Annotation file not found for image: {image_name}")
