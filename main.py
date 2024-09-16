from grounded import grounded
from robust_sam import robust_sam

# image_path = "instance_folder/instance_18.jpg"
# detections = grounded(image_path, "shirt")
# bbox = detections.xyxy[0]

# mask = robust_sam(image_path, bbox)

import os
import subprocess
from PIL import Image
import json

# Paths
instance_folder_path = "instance_folder/"
mask_folder_path = "mask_folder/"
output_folder_path = "garment_folder/"
image_urls_file = "image_urls.txt"

# Create output folders if they don't exist
os.makedirs(instance_folder_path, exist_ok=True)
os.makedirs(mask_folder_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)

# List to store the results
results = []

# Step 1: Download images using wget from image_urls.txt and save as instance_xx.jpg
with open(image_urls_file, 'r') as file:
    image_urls = file.readlines()

for idx, url in enumerate(image_urls, start=1):
    url = url.strip()  # Clean up the URL
    if not url:
        continue  # Skip if empty line

    # Generate filename
    filename = f"instance_{idx:02d}.jpg"
    image_path = os.path.join(instance_folder_path, filename)
    abs_image_path = os.path.abspath(image_path)  # Get absolute path

    # Download the image using wget
    try:
        subprocess.run(["wget", "-O", abs_image_path, url], check=True)
        print(f"Downloaded {filename} from {url}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        continue

    # Step 2: Process each downloaded image
    detections = grounded(abs_image_path, "shirt")

    if len(detections.xyxy) > 0:
        bbox = detections.xyxy[0]

        # Run the robust_sam function with the detected bounding box
        mask = robust_sam(abs_image_path, bbox)

        # Generate the corresponding mask filename
        mask_filename = filename.replace("instance_", "mask_").rsplit(".", 1)[0] + ".jpg"
        mask_path = os.path.join(mask_folder_path, mask_filename)
        abs_mask_path = os.path.abspath(mask_path)  # Get absolute path for mask
        mask.save(mask_path)

        # Assuming the mask is already saved as a binary image
        # Open the mask and the instance image
        instance_img = Image.open(abs_image_path).convert("RGBA")
        mask = mask.convert("L")  # Convert to grayscale (L mode)

        # Create a white background the same size as the instance image
        white_background = Image.new("RGB", instance_img.size, (255, 255, 255))

        # Composite the instance image and the mask onto the white background
        composite_img = Image.composite(instance_img, white_background, mask)

        # Save the final composite image
        output_filename = filename.replace("instance_", "garment_").rsplit(".", 1)[0] + ".jpg"
        output_path = os.path.join(output_folder_path, output_filename)
        abs_output_path = os.path.abspath(output_path)  # Get absolute path for composite
        composite_img.save(output_path)

        # Save the result details with absolute paths
        results.append(
            {
                "url": url,
                "instance_file_name": abs_image_path,
                "mask_file_name": abs_mask_path,
                "garment_file_name": abs_output_path,
            }
        )
        print(f"Processed {filename} and saved composite as {output_filename}")
    else:
        print(f"No detections for {filename}")

# Step 3: Save the results to a JSON file
results_file_path = "results.json"
abs_results_file_path = os.path.abspath(results_file_path)  # Get absolute path for results.json
with open(abs_results_file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {abs_results_file_path}")
