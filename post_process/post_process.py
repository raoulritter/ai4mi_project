# post_process/post_process.py

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure
from tqdm import tqdm

def apply_post_processing(segmentation_data, num_classes):
    """
    Apply post-processing to the segmentation data.

    Args:
        segmentation_data (numpy.ndarray): The segmentation data array.
        num_classes (int): The number of classes including background.

    Returns:
        numpy.ndarray: The post-processed segmentation data.
    """
    processed_segmentation = np.zeros_like(segmentation_data)

    # Define a connectivity structure for 3D morphological operations
    structure = generate_binary_structure(3, 2)  # 18-connectivity

    for class_id in tqdm(range(1, num_classes), desc="Processing classes"):
        # Create binary mask for the current class
        class_mask = (segmentation_data == class_id)
        
        # Apply morphological closing to fill small holes
        class_mask = binary_closing(class_mask, structure=structure)
        # Apply morphological opening to remove small objects
        class_mask = binary_opening(class_mask, structure=structure)
        
        # Label connected components
        labeled_array, num_features = label(class_mask, structure=structure)
        
        if num_features > 0:
            # Find the largest connected component
            component_sizes = np.bincount(labeled_array.ravel())
            # Exclude background component at index 0
            component_sizes[0] = 0
            largest_component_label = component_sizes.argmax()
            # Keep only the largest connected component
            largest_component_mask = (labeled_array == largest_component_label)
            processed_segmentation[largest_component_mask] = class_id
        else:
            print(f"No connected components found for class {class_id}")

    return processed_segmentation

def post_process_nifti_files(input_folder, output_folder, num_classes):
    """
    Post-process all NIfTI files in the input folder and save them to the output folder.

    Args:
        input_folder (str): Path to the input NIfTI files.
        output_folder (str): Path to save the post-processed NIfTI files.
        num_classes (int): The number of classes including background.
    """
    os.makedirs(output_folder, exist_ok=True)
    nifti_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    for nifti_file in tqdm(nifti_files, desc="Post-processing NIfTI files"):
        nifti_input_path = os.path.join(input_folder, nifti_file)
        nifti_output_path = os.path.join(output_folder, nifti_file)
        print(f"Processing '{nifti_input_path}'...")

        # Load the NIfTI file
        nifti_img = nib.load(nifti_input_path)
        segmentation_data = nifti_img.get_fdata()
        segmentation_data = segmentation_data.astype(np.uint8)

        # Apply post-processing
        processed_segmentation = apply_post_processing(segmentation_data, num_classes)

        # Save the post-processed NIfTI file
        new_nifti = nib.Nifti1Image(processed_segmentation, affine=nifti_img.affine, header=nifti_img.header)
        nib.save(new_nifti, nifti_output_path)
        print(f"Saved post-processed file at '{nifti_output_path}'")

if __name__ == '__main__':
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Post-process NIfTI segmentation files.")
        parser.add_argument('--input_folder', type=str, required=True, help="Path to input NIfTI files.")
        parser.add_argument('--output_folder', type=str, required=True, help="Path to save post-processed NIfTI files.")
        parser.add_argument('--num_classes', type=int, required=True, help="Number of classes including background.")
        args = parser.parse_args()

        post_process_nifti_files(args.input_folder, args.output_folder, args.num_classes)

    main()
