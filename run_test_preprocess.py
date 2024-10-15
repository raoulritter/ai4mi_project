#!/usr/bin/env python3

"""
process_test_set.py

This script processes the test set for the SegTHOR medical imaging segmentation challenge.
It performs the following steps:

1. Voxel Clipping:
   - Clipping CT scan intensity values between -1000 and 1000 HU.

2. Resampling:
   - Resampling the images to a common voxel spacing (e.g., 0.977 x 0.977 x 2.5 mm).

3. Intensity Normalization:
   - Normalizing the intensities to have zero mean and unit variance.

4. Slicing:
   - Slicing the preprocessed NIfTI files into 2D PNG images for each axial slice.

The preprocessed NIfTI files are saved in 'data/SEGTHOR_TEST_preprocessed/nifti' folder.
The PNG slices are saved in 'data/SEGTHOR_TEST_preprocessed/img' folder.
"""

import os
import sys
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from tqdm import tqdm
import warnings

def setup_logging():
    """
    Set up logging configuration to write logs to a file.
    """
    logging.basicConfig(
        filename="process_test_set_log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def load_nifti(file_path):
    """
    Load a NIfTI file and return its data, affine matrix, and header.

    Args:
        file_path (Path): Path to the NIfTI file.

    Returns:
        tuple: (data, affine, header) of the NIfTI file.
    """
    nifti_img = nib.load(str(file_path))
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header
    return data, affine, header

def save_nifti(data, affine, header, file_path):
    """
    Save data as a NIfTI file.

    Args:
        data (numpy.ndarray): Image data to save.
        affine (numpy.ndarray): Affine transformation matrix.
        header (nibabel.nifti1.Nifti1Header): NIfTI header.
        file_path (Path): Path to save the NIfTI file.
    """
    nifti_img = nib.Nifti1Image(data, affine, header)
    nib.save(nifti_img, str(file_path))

def voxel_clipping(data, lower_limit=-1000, upper_limit=1000):
    """
    Clip voxel values to a specified range.

    Args:
        data (numpy.ndarray): Input image data.
        lower_limit (int): Lower bound for clipping.
        upper_limit (int): Upper bound for clipping.

    Returns:
        numpy.ndarray: Clipped image data.
    """
    data_clipped = np.clip(data, lower_limit, upper_limit)
    return data_clipped

def resample_image(data, original_spacing, target_spacing, is_label=False):
    """
    Resample image to a target voxel spacing.

    Args:
        data (numpy.ndarray): Input image data.
        original_spacing (tuple): Original voxel spacing.
        target_spacing (tuple): Target voxel spacing.
        is_label (bool): Whether the input is a label map.

    Returns:
        numpy.ndarray: Resampled image data.
    """
    # Calculate the resampling factors
    resize_factor = [
        original_spacing[0] / target_spacing[0],
        original_spacing[1] / target_spacing[1],
        original_spacing[2] / target_spacing[2],
    ]
    new_shape = np.round(np.array(data.shape) * resize_factor).astype(int)

    # Resample
    if is_label:
        # Use nearest-neighbor interpolation for labels
        data_resampled = resize(
            data,
            new_shape,
            order=0,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        )
    else:
        # Use trilinear interpolation for CT scans
        data_resampled = resize(
            data,
            new_shape,
            order=1,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        )
    return data_resampled

def intensity_normalization(data):
    """
    Normalize image intensity to zero mean and unit variance.

    Args:
        data (numpy.ndarray): Input image data.

    Returns:
        numpy.ndarray: Normalized image data.
    """
    mean = np.mean(data)
    std = np.std(data)
    data_normalized = (data - mean) / std
    return data_normalized

def update_affine(affine, original_spacing, target_spacing):
    """
    Update the affine matrix to reflect new voxel sizes.

    Args:
        affine (numpy.ndarray): Original affine matrix.
        original_spacing (tuple): Original voxel spacing.
        target_spacing (tuple): Target voxel spacing.

    Returns:
        numpy.ndarray: Updated affine matrix.
    """
    # Update the affine matrix to reflect the new voxel sizes
    scaling = np.array(target_spacing) / np.array(original_spacing)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(scaling)
    return new_affine

def preprocess_patient(patient_id, input_dir, output_dir, target_spacing):
    """
    Preprocess a single patient's CT scan.

    Args:
        patient_id (str): Patient identifier.
        input_dir (Path): Directory containing the original test set NIfTI files.
        output_dir (Path): Directory to save the preprocessed NIfTI files.
        target_spacing (tuple): Target voxel spacing for resampling.
    """
    logging.info(f"Processing {patient_id}")

    # File paths
    ct_path = input_dir / f"{patient_id}.nii.gz"
    output_ct_path = output_dir / f"{patient_id}.nii.gz"  # Save with same name

    # Load data
    ct_data, ct_affine, ct_header = load_nifti(ct_path)

    # Voxel Clipping
    ct_data = voxel_clipping(ct_data, lower_limit=-1000, upper_limit=1000)
    logging.info(f"Applied voxel clipping to {patient_id}")

    # Get original voxel sizes
    original_spacing = ct_header.get_zooms()
    logging.info(f"{patient_id} original spacing: {original_spacing}")

    # Check if resampling is needed
    if np.allclose(original_spacing, target_spacing, atol=1e-3):
        logging.info(f"No resampling needed for {patient_id}")
        resampled_ct_data = ct_data
    else:
        logging.info(f"Resampling {patient_id} to target spacing {target_spacing}")
        # Resample CT scan
        resampled_ct_data = resample_image(
            ct_data, original_spacing, target_spacing, is_label=False
        )
        # Update affine matrix and header
        ct_affine = update_affine(ct_affine, original_spacing, target_spacing)
        ct_header.set_zooms(target_spacing)

    # Intensity Normalization
    resampled_ct_data = intensity_normalization(resampled_ct_data)
    logging.info(f"Applied intensity normalization to {patient_id}")

    # Save preprocessed data
    save_nifti(resampled_ct_data, ct_affine, ct_header, output_ct_path)
    logging.info(f"Saved preprocessed NIfTI for {patient_id}")

def norm_arr(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min  # In case of constant image
    img = img * 255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def slice_patient(patient_id, nifti_dir, output_dir, shape=(256, 256)):
    """
    Slice a preprocessed NIfTI file into PNG images.

    Args:
        patient_id (str): Patient identifier.
        nifti_dir (Path): Directory containing the preprocessed NIfTI files.
        output_dir (Path): Directory to save the PNG images.
        shape (tuple): Desired shape of the output images.
    """
    ct_path = nifti_dir / f"{patient_id}.nii.gz"

    # Load CT scan
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    # Normalize CT scan
    norm_ct: np.ndarray = norm_arr(ct)

    # Slice and save images
    for idz in range(z):
        img_slice = resize(norm_ct[:, :, idz], shape, order=1, mode='constant', preserve_range=True, anti_aliasing=False).astype(np.uint8)
        filename = f"{patient_id}_{idz:04d}.png"

        save_path: Path = output_dir
        save_path.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            imsave(str(save_path / filename), img_slice)

def main():
    """
    Main function to preprocess and slice all patients in the test set.
    """
    setup_logging()

    # Directories
    input_dir = Path('data/segthor_test/test')
    output_nifti_dir = Path('data/SEGTHOR_TEST_preprocessed/nifti')
    output_img_dir = Path('data/SEGTHOR_TEST_preprocessed/img')

    # Create output directories if they don't exist
    output_nifti_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)

    target_spacing = (0.977, 0.977, 2.5)  # Same as used in training set

    print("Starting preprocessing of test set. Check progress in 'process_test_set_log.txt'.")

    for i in tqdm(range(41, 61), desc="Processing Test Patients"):
        patient_id = f"Patient_{i:02d}"
        try:
            preprocess_patient(patient_id, input_dir, output_nifti_dir, target_spacing)
            slice_patient(patient_id, output_nifti_dir, output_img_dir, shape=(256, 256))
        except Exception as e:
            logging.error(f"Error processing {patient_id}: {str(e)}")

if __name__ == "__main__":
    main()
