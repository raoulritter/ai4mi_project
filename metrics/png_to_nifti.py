import os
import numpy as np
import nibabel as nib
from PIL import Image
import glob

def reconstruct_nifti(png_folder, gt_folder, output_folder):
    """
    Reconstruct NIfTI files from PNG slices.

    Args:
        png_folder (str): Path to the folder containing PNG slice predictions.
        gt_folder (str): Path to the folder containing ground truth NIfTI files.
        output_folder (str): Path to save the reconstructed NIfTI files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all PNG files in the directory
    png_files = glob.glob(os.path.join(png_folder, '*.png'))

    # Extract unique patient IDs from the filenames
    patient_ids = set('_'.join(os.path.basename(f).split('_')[:2]) for f in png_files)

    for patient_id in patient_ids:
        # Collect all PNG files for this patient
        patient_png_files = [f for f in png_files if os.path.basename(f).startswith(patient_id)]

        # Function to extract slice index from filename
        def get_slice_index(filename):
            return int(os.path.splitext(filename)[0].split('_')[-1])

        # Sort the PNG files based on slice index
        patient_png_files.sort(key=lambda x: get_slice_index(os.path.basename(x)))

        # Load the ground truth NIfTI file to get target dimensions and metadata
        gt_nifti_path = os.path.join(gt_folder, patient_id, 'GT.nii.gz')
        if not os.path.exists(gt_nifti_path):
            print(f'Ground truth NIfTI file not found for {patient_id} at {gt_nifti_path}')
            continue

        gt_nifti = nib.load(gt_nifti_path)
        gt_data = gt_nifti.get_fdata()
        gt_header = gt_nifti.header
        gt_affine = gt_nifti.affine
        target_shape = gt_data.shape  # (width, height, num_slices)
        width, height, num_slices = int(target_shape[0]), int(target_shape[1]), int(target_shape[2])

        # Read and stack slices to form a 3D volume
        slices = []
        for png_file in patient_png_files:
            img = Image.open(png_file)
            # Resize the image to match ground truth dimensions
            img_resized = img.resize((width, height), Image.NEAREST)
            # Convert image to array and map pixel values back to class labels
            img_array = np.array(img_resized)
            img_array = np.round(img_array / 63).astype(np.uint8)
            slices.append(img_array)

        # Stack slices along the correct axis
        volume = np.stack(slices, axis=0)  # Shape: (num_slices, height, width)

        # Transpose volume to match NIfTI dimension ordering (X, Y, Z)
        volume = np.transpose(volume, (1, 2, 0))  # Shape: (width, height, num_slices)

        # Ensure the volume matches the ground truth dimensions
        if volume.shape != gt_data.shape:
            print(f'Warning: Volume shape {volume.shape} does not match ground truth shape {gt_data.shape} for {patient_id}')
            continue  # Skip to the next patient if dimensions do not match

        # Create a new NIfTI image with the same header and affine as the ground truth
        new_nifti = nib.Nifti1Image(volume, affine=gt_affine, header=gt_header)

        # Save the new NIfTI file
        output_path = os.path.join(output_folder, f'{patient_id}.nii.gz')
        nib.save(new_nifti, output_path)
        print(f'Saved NIfTI file for {patient_id} at {output_path}')
