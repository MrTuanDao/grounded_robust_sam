# from grounded import grounded
# from robust_sam import robust_sam

# image_path = "instance_folder/instance_18.jpg"
# detections = grounded(image_path, "shirt")
# bbox = detections.xyxy[0]

# mask = robust_sam(image_path, bbox)

# Required Imports
from grounded import grounded, create_grounded_model
from robust_sam import robust_sam, create_sam_model
from PIL import Image
import os
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

def process_garment(garment, garment_index, grounding_dino_model, sam_model, sam_transform, mask_folder_path, segment_folder_path):
    """
    Processes a single garment: downloads the image, detects 'shirt', creates mask and segmented images.
    Updates the garment dictionary with new fields.
    """
    try:
        garment_image_url = garment.get("image")
        if not garment_image_url:
            print(f"[Garment {garment_index}] No image URL found. Skipping.")
            garment["mask_file_path"] = None
            garment["segment_file_path"] = None
            garment["shirt_detected"] = False
            return garment
        
        # Define filenames based on garment index
        garment_filename = f"garment_{garment_index}.jpg"
        garment_path = os.path.join("downloaded_garments", garment_filename)
        
        # Download the garment image
        downloaded_garment_img = download_image(garment_image_url, garment_path)
        
        if downloaded_garment_img and os.path.exists(downloaded_garment_img):
            # Detect 'shirt' in the garment image
            detections = grounded(downloaded_garment_img, "shirt", grounding_dino_model)
            
            if len(detections.xyxy) > 0:
                bbox = detections.xyxy[0]
                
                # Generate mask using robust_sam
                mask = robust_sam(downloaded_garment_img, bbox, sam_model, sam_transform)
                
                # Save the mask image
                mask_filename = f"garment_{garment_index}_mask.png"
                mask_path = os.path.join(mask_folder_path, mask_filename)
                mask.save(mask_path)
                
                # Create segmented image
                instance_img = Image.open(downloaded_garment_img).convert("RGBA")
                mask = mask.convert("L")  # Convert to grayscale
                
                # Create a white background
                white_background = Image.new("RGB", instance_img.size, (255, 255, 255))
                
                # Composite the instance image and the mask onto the white background
                composite_img = Image.composite(instance_img, white_background, mask)
                
                # Save the segmented image
                segment_filename = f"garment_{garment_index}_segmented.png"
                segment_path = os.path.join(segment_folder_path, segment_filename)
                composite_img.save(segment_path)
                
                # Update garment dictionary
                garment["mask_file_path"] = os.path.abspath(mask_path)
                garment["segment_file_path"] = os.path.abspath(segment_path)
                garment["shirt_detected"] = True
                
                print(f"[Garment {garment_index}] Processed: Mask and segmented images saved.")
            else:
                print(f"[Garment {garment_index}] No 'shirt' detected.")
                garment["mask_file_path"] = None
                garment["segment_file_path"] = None
                garment["shirt_detected"] = False
        else:
            print(f"[Garment {garment_index}] Failed to download image. Skipping processing.")
            garment["mask_file_path"] = None
            garment["segment_file_path"] = None
            garment["shirt_detected"] = False
        
        return garment
    
    except Exception as e:
        print(f"[Garment {garment_index}] An error occurred: {e}")
        garment["mask_file_path"] = None
        garment["segment_file_path"] = None
        garment["shirt_detected"] = False
        return garment

def process_entry(entry, entry_number, grounding_dino_model, sam_model, sam_transform, mask_folder_path, segment_folder_path):
    """
    Processes all garments in a single metadata entry.
    Returns the updated entry dictionary.
    """
    try:
        if not entry.get("process_garment_image", False):
            print(f"[Entry {entry_number}] 'process_garment_image' flag is False. Skipping.")
            return entry
        
        garments = entry.get("garment_data", [])
        if not garments:
            print(f"[Entry {entry_number}] No garment data found. Skipping.")
            return entry
        
        for idx, garment in enumerate(garments, start=1):
            updated_garment = process_garment(
                garment,
                garment_index=f"{entry_number}_{idx}",
                grounding_dino_model=grounding_dino_model,
                sam_model=sam_model,
                sam_transform=sam_transform,
                mask_folder_path=mask_folder_path,
                segment_folder_path=segment_folder_path
            )
            garments[idx - 1] = updated_garment  # Update the garment in the list
        
        entry["garment_data"] = garments
        return entry
    
    except Exception as e:
        print(f"[Entry {entry_number}] An error occurred while processing garments: {e}")
        return entry

def main():
    # Paths
    metadata_file_path = "./metadata.json"  # Input metadata file
    mask_folder_path = "./test_mask/"      # Folder to save mask images
    segment_folder_path = "./test_segment/"  # Folder to save segmented images
    new_metadata_file_path = "./test_updated.json"  # Output updated metadata
    
    # Create necessary directories
    os.makedirs("downloaded_garments", exist_ok=True)
    os.makedirs(mask_folder_path, exist_ok=True)
    os.makedirs(segment_folder_path, exist_ok=True)
    
    # Load metadata
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)  # Load as a list of dictionaries
    
    # List to store the updated entries
    updated_metadata = []
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor with 3 workers
    with ProcessPoolExecutor(max_workers=3, initializer=initializer) as executor:
        # Submit all tasks to the executor
        future_to_entry = {
            executor.submit(
                process_entry,
                entry,
                idx + 1,
                grounding_dino_model,
                sam_model,
                sam_transform,
                mask_folder_path,
                segment_folder_path
            ): idx + 1
            for idx, entry in enumerate(metadata)
        }
        
        # Iterate over the completed futures as they finish
        for future in as_completed(future_to_entry):
            entry_number = future_to_entry[future]
            try:
                result = future.result()
                if result is not None:
                    updated_metadata.append(result)
            except Exception as exc:
                print(f"[Entry {entry_number}] Generated an exception: {exc}")
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    
    # Save the updated metadata to a new JSON file
    with open(new_metadata_file_path, "w") as json_file:
        json.dump(updated_metadata, json_file, indent=4)
    
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
