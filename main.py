# from grounded import grounded
# from robust_sam import robust_sam

# image_path = "instance_folder/instance_18.jpg"
# detections = grounded(image_path, "shirt")
# bbox = detections.xyxy[0]

# mask = robust_sam(image_path, bbox)

# Required Imports
from grounded import grounded, create_grounded_model
from robust_sam import robust_sam, create_sam_model
import os
from PIL import Image
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import requests

warnings.filterwarnings("ignore")

# Global variables for models (to be initialized in each worker)
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

def download_image(url, save_path):
    """
    Downloads an image from a URL and saves it to the specified path.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded image from {url} to {save_path}")
            return save_path
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def process_entry(entry, entry_number, mask_folder_path, output_folder_path):
    """
    Processes a single metadata entry.
    Returns the updated metadata dictionary.
    """
    try:
        updated_entry = entry.copy()
        
        # Process Human Image if flag is True
        if entry.get("process_user_image", False):
            human_img_url = entry.get("human_img")
            if human_img_url:
                human_img_filename = f"human_{entry_number}.jpg"
                human_img_path = os.path.join("downloaded_images", human_img_filename)
                downloaded_human_img = download_image(human_img_url, human_img_path)
                
                if downloaded_human_img and os.path.exists(downloaded_human_img):
                    # Process the human image
                    detections = grounded(downloaded_human_img, "shirt", grounding_dino_model)
                    
                    if len(detections.xyxy) > 0:
                        bbox = detections.xyxy[0]
                        mask = robust_sam(downloaded_human_img, bbox, sam_model, sam_transform)
                        
                        # Save the mask
                        mask_filename = f"human_{entry_number}_mask.png"
                        mask_path = os.path.join(mask_folder_path, mask_filename)
                        mask.save(mask_path)
                        
                        # Create segmented image
                        instance_img = Image.open(downloaded_human_img).convert("RGBA")
                        mask = mask.convert("L")
                        white_background = Image.new("RGB", instance_img.size, (255, 255, 255))
                        composite_img = Image.composite(instance_img, white_background, mask)
                        segment_filename = f"human_{entry_number}_segmented.png"
                        segment_path = os.path.join(output_folder_path, segment_filename)
                        composite_img.save(segment_path)
                        
                        # Update metadata
                        updated_entry["human_mask_file_path"] = os.path.abspath(mask_path)
                        updated_entry["human_segment_file_path"] = os.path.abspath(segment_path)
                        updated_entry["human_shirt_detected"] = True
                        
                        print(f"[Entry {entry_number}] Processed human image: Mask and segmented images saved.")
                    else:
                        print(f"[Entry {entry_number}] No 'shirt' detected in human image.")
                        updated_entry["human_mask_file_path"] = None
                        updated_entry["human_segment_file_path"] = None
                        updated_entry["human_shirt_detected"] = False
                else:
                    print(f"[Entry {entry_number}] Failed to download human image. Skipping processing.")
                    updated_entry["human_mask_file_path"] = None
                    updated_entry["human_segment_file_path"] = None
                    updated_entry["human_shirt_detected"] = False
            else:
                print(f"[Entry {entry_number}] No 'human_img' URL found.")
        
        # Process Garment Images if flag is True
        if entry.get("process_garment_image", False):
            garments = entry.get("garment_data", [])
            for idx, garment in enumerate(garments):
                garment_img_path = garment.get("image")
                if garment_img_path and os.path.exists(garment_img_path):
                    filename = os.path.basename(garment_img_path)
                    print(f"[Entry {entry_number}, Garment {idx + 1}] Processing {filename}...")
                    
                    # Generate expected mask and segment filenames
                    mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
                    mask_path = os.path.join(mask_folder_path, mask_filename)
                    
                    segment_filename = f"{os.path.splitext(filename)[0]}_segmented.png"
                    segment_path = os.path.join(output_folder_path, segment_filename)
                    
                    # Check if mask and segment already exist
                    if os.path.exists(mask_path) and os.path.exists(segment_path):
                        print(f"[Entry {entry_number}, Garment {idx + 1}] Mask and segmented images already exist. Skipping.")
                        updated_entry["garment_data"][idx]["mask_file_path"] = os.path.abspath(mask_path)
                        updated_entry["garment_data"][idx]["segment_file_path"] = os.path.abspath(segment_path)
                        updated_entry["garment_data"][idx]["shirt_detected"] = True
                        continue
                    
                    # Detect shirt in garment image
                    detections = grounded(garment_img_path, "shirt", grounding_dino_model)
                    
                    if len(detections.xyxy) > 0:
                        bbox = detections.xyxy[0]
                        mask = robust_sam(garment_img_path, bbox, sam_model, sam_transform)
                        
                        # Save the mask
                        mask.save(mask_path)
                        
                        # Create segmented image
                        instance_img = Image.open(garment_img_path).convert("RGBA")
                        mask = mask.convert("L")
                        white_background = Image.new("RGB", instance_img.size, (255, 255, 255))
                        composite_img = Image.composite(instance_img, white_background, mask)
                        composite_img.save(segment_path)
                        
                        # Update metadata
                        updated_entry["garment_data"][idx]["mask_file_path"] = os.path.abspath(mask_path)
                        updated_entry["garment_data"][idx]["segment_file_path"] = os.path.abspath(segment_path)
                        updated_entry["garment_data"][idx]["shirt_detected"] = True
                        
                        print(f"[Entry {entry_number}, Garment {idx + 1}] Processed: Mask and segmented images saved.")
                    else:
                        print(f"[Entry {entry_number}, Garment {idx + 1}] No 'shirt' detected.")
                        updated_entry["garment_data"][idx]["mask_file_path"] = None
                        updated_entry["garment_data"][idx]["segment_file_path"] = None
                        updated_entry["garment_data"][idx]["shirt_detected"] = False
                else:
                    print(f"[Entry {entry_number}, Garment {idx + 1}] Garment image path does not exist: {garment_img_path}. Skipping.")
                    updated_entry["garment_data"][idx]["mask_file_path"] = None
                    updated_entry["garment_data"][idx]["segment_file_path"] = None
                    updated_entry["garment_data"][idx]["shirt_detected"] = False
        
        return updated_entry
    
    except Exception as e:
        print(f"[Entry {entry_number}] An error occurred while processing: {e}.")
        return None

def main():
    # Paths
    metadata_file_path = "./chuan_metadata.json"  # Updated to JSON instead of JSONL
    mask_folder_path = "./mask/"
    output_folder_path = "./segment/"
    new_metadata_file_path = "./updated_metadata.json"  # Changed to JSON
    
    # Create necessary directories
    os.makedirs("downloaded_images", exist_ok=True)
    os.makedirs(mask_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Load metadata
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)  # Load as a list of dictionaries
    
    # List to store the results
    results = []
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor with 3 workers
    with ProcessPoolExecutor(max_workers=3, initializer=initializer) as executor:
        # Submit all tasks to the executor
        future_to_entry = {
            executor.submit(process_entry, entry, idx + 1, mask_folder_path, output_folder_path): idx + 1
            for idx, entry in enumerate(metadata)
        }
        
        # Iterate over the completed futures as they finish
        for future in as_completed(future_to_entry):
            entry_number = future_to_entry[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"[Entry {entry_number}] Generated an exception: {exc}")
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    
    # Save the updated metadata to a new JSON file
    with open(new_metadata_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Updated metadata saved to {new_metadata_file_path}")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # If the start method has already been set, this will raise a RuntimeError
        pass
    
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
