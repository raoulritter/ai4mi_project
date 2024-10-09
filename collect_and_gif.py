import os
import shutil
import argparse
from glob import glob
from PIL import Image

def collect_images_and_create_gif(base_folder, patient_id, slice_number, output_folder):
    """
    Collects images for a specific patient and slice number across all iteration folders,
    saves them to a new folder, and creates a GIF from these images.

    Args:
    base_folder (str): Path to the base folder containing iter folders
    patient_id (str): ID of the patient (e.g., '01')
    slice_number (str): Slice number (e.g., '0152')
    output_folder (str): Path to the folder where results will be saved
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Pattern for the filename we're looking for
    filename_pattern = f'Patient_{patient_id}_{slice_number}.png'

    # List to store paths of found images
    image_paths = []

    # Iterate through all iter folders
    for iter_folder in sorted(glob(os.path.join(base_folder, 'iter*'))):
        # Construct the path to the 'val' folder
        val_folder = os.path.join(iter_folder, 'val')
        
        # Check if the 'val' folder exists
        if not os.path.exists(val_folder):
            print(f"Warning: 'val' folder not found in {iter_folder}")
            continue
        
        # Look for the specific file in the 'val' folder
        for file in os.listdir(val_folder):
            if file == filename_pattern:
                source_path = os.path.join(val_folder, file)
                dest_path = os.path.join(output_folder, f'{os.path.basename(iter_folder)}_{file}')
                
                # Copy the file to the output folder
                shutil.copy2(source_path, dest_path)
                
                # Add the new path to our list
                image_paths.append(dest_path)
                break  # File found, move to next iter folder

    if not image_paths:
        print(f"No images found for Patient {patient_id}, Slice {slice_number}")
        return

    # Sort the image paths to ensure correct order
    image_paths.sort()

    # Create the GIF
    gif_path = os.path.join(output_folder, f'Patient_{patient_id}_Slice_{slice_number}.gif')
    
    # Open all images
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Save the GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=250,  # Duration is in milliseconds
        loop=0
    )

    print(f"GIF created: {gif_path}")

def main():
    parser = argparse.ArgumentParser(description="Collect images and create GIF for a specific patient and slice.")
    parser.add_argument("base_folder", help="Path to the base folder containing iter folders")
    parser.add_argument("patient_id", help="ID of the patient (e.g., '01')")
    parser.add_argument("slice_number", help="Slice number (e.g., '0152')")
    parser.add_argument("output_folder", help="Path to the folder where results will be saved")

    args = parser.parse_args()

    collect_images_and_create_gif(args.base_folder, args.patient_id, args.slice_number, args.output_folder)

if __name__ == "__main__":
    main()