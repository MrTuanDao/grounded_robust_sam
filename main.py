# from grounded import grounded
# from robust_sam import robust_sam

# image_path = "instance_folder/instance_18.jpg"
# detections = grounded(image_path, "shirt")
# bbox = detections.xyxy[0]

# mask = robust_sam(image_path, bbox)

from grounded import grounded, create_grounded_model
from robust_sam import robust_sam, create_sam_model
import os
from PIL import Image
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

warnings.filterwarnings("ignore")

# Global variables for models
grounding_dino_model = None
sam_model = None
sam_transform = None

def initializer():
    """
    Initializes the models for each worker process.
    This function is called once per worker when the pool starts.
    """
    global grounding_dino_model, sam_model, sam_transform
    grounding_dino_model = create_grounded_model()
    sam_model, sam_transform = create_sam_model()
    print(f"Worker {multiprocessing.current_process().name} initialized models.")

def process_line(line, line_number, mask_folder_path, output_folder_path):
    """
    Processes a single line from the metadata file.
    Returns the updated metadata dictionary.
    """
    try:
        data = json.loads(line.strip())
        image_url = data.get("url")
        abs_image_path = data.get("image_file_path")
        
        if not abs_image_path or not os.path.exists(abs_image_path):
            print(f"[Line {line_number}] Image path does not exist: {abs_image_path}. Skipping.")
            # Update metadata with None values
            updated_data = data.copy()
            updated_data["mask_file_path"] = None
            updated_data["segment_file_path"] = None
            updated_data["shirt_detected"] = False
            return updated_data

        filename = os.path.basename(abs_image_path)
        print(f"[Line {line_number}] Processing {filename}...")

        # Generate expected mask and segment filenames
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"  # Mask filename
        mask_path = os.path.join(mask_folder_path, mask_filename)
        abs_mask_path = os.path.abspath(mask_path)

        segment_filename = f"{os.path.splitext(filename)[0]}_segmented.png"  # Segment filename
        segment_path = os.path.join(output_folder_path, segment_filename)
        abs_segment_path = os.path.abspath(segment_path)

        # Check if mask (and optionally segment) already exists
        if os.path.exists(mask_path) and os.path.exists(segment_path):
            print(f"[Line {line_number}] Mask and segmented images already exist for {filename}. Skipping processing.")
            # Update metadata with existing paths
            updated_data = data.copy()
            updated_data["mask_file_path"] = abs_mask_path
            updated_data["segment_file_path"] = abs_segment_path
            updated_data["shirt_detected"] = True  # Assuming existing files are valid
            return updated_data

        # Step 1: Detect "shirt" in the image
        detections = grounded(abs_image_path, "shirt", grounding_dino_model)

        if len(detections.xyxy) > 0:
            bbox = detections.xyxy[0]

            # Step 2: Generate mask using robust_sam
            mask = robust_sam(abs_image_path, bbox, sam_model, sam_transform)

            # Save the mask image
            mask.save(mask_path)

            # Step 3: Create segmented image
            instance_img = Image.open(abs_image_path).convert("RGBA")
            mask = mask.convert("L")  # Convert to grayscale

            # Create a white background
            white_background = Image.new("RGB", instance_img.size, (255, 255, 255))

            # Composite the instance image and the mask onto the white background
            composite_img = Image.composite(instance_img, white_background, mask)

            # Save the segmented image
            composite_img.save(segment_path)

            # Update the metadata with new paths
            updated_data = data.copy()
            updated_data["mask_file_path"] = abs_mask_path
            updated_data["segment_file_path"] = abs_segment_path
            updated_data["shirt_detected"] = True

            print(f"[Line {line_number}] Processed {filename}: Mask saved as {mask_filename}, Segment saved as {segment_filename}")
        else:
            print(f"[Line {line_number}] No 'shirt' detected in {filename}. Updating metadata accordingly.")
            # Update the metadata to reflect no detection
            updated_data = data.copy()
            updated_data["mask_file_path"] = None
            updated_data["segment_file_path"] = None
            updated_data["shirt_detected"] = False

        return updated_data

    except json.JSONDecodeError as e:
        print(f"[Line {line_number}] Error decoding JSON: {e}. Line content: {line}")
        # Optionally, handle the error by skipping or logging
        return None
    except Exception as e:
        print(f"[Line {line_number}] An error occurred while processing: {e}. Line content: {line}")
        # Optionally, handle the error by skipping or logging
        return None

def main():
    # Paths
    input_metadata_file_path = "./metadata.jsonl"
    mask_folder_path = "./test_mask/"
    output_folder_path = "./test_segment/"
    new_metadata_file_path = "./test.jsonl"

    # Create output folders if they don't exist
    os.makedirs(mask_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)

    # List to store the results
    results = []

    # Read all lines from the metadata file
    with open(input_metadata_file_path, 'r') as file:
        lines = file.readlines()  # Read all lines into a list

    start_time = time.time()

    # Use ProcessPoolExecutor with 3 workers
    with ProcessPoolExecutor(max_workers=3, initializer=initializer) as executor:
        # Submit all tasks to the executor
        future_to_line = {
            executor.submit(process_line, line, idx + 1, mask_folder_path, output_folder_path): idx + 1
            for idx, line in enumerate(lines)
        }

        # Iterate over the completed futures as they finish
        for future in as_completed(future_to_line):
            line_number = future_to_line[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"[Line {line_number}] Generated an exception: {exc}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

    # Step 4: Save the updated metadata to a new JSONL file
    with open(new_metadata_file_path, "w") as jsonl_file:
        print(results)
        for entry in results:
            jsonl_file.write(json.dumps(entry) + "\n")

    print(f"Updated metadata saved to {new_metadata_file_path}")

if __name__ == "__main__":
    start = time.time()

    while True:
        try:
            main()  # Run the main function
            print("Main function finished successfully. Exiting the loop.")
            break  # Exit the loop if the main function finishes successfully
        except Exception as e:
            print(f"Main function crashed or was killed with error: {e}. Restarting...")
            time.sleep(1)  # Optional: wait before restarting

    end = time.time()
    print("Finish process in:", end - start)
