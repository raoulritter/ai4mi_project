import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import logging
from pathlib import Path

from tqdm import tqdm

def get_project_root() -> Path:
    """
    Find the project root directory.
    
    Returns:
        Path: The path to the project root directory.
    """
    current_path = Path(__file__).resolve().parent
    while not (current_path / '.git').exists():
        parent = current_path.parent
        if parent == current_path:
            raise RuntimeError("Project root not found. Ensure you're running this from within the cloned repository.")
        current_path = parent
    return current_path

def setup_logging():
    """
    Set up logging configuration to write logs to a file in the insights folder.
    """
    # Get the current script's directory (preprocessing folder)
    current_dir = Path(__file__).resolve().parent
    
    # Create the insights folder if it doesn't exist
    insights_dir = current_dir / 'insights'
    insights_dir.mkdir(exist_ok=True)
    
    # Set up the log file path
    log_file_path = insights_dir / 'preprocessing_log.txt'
    
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_nifti(file_path):
    """
    Load a NIfTI file and return its data, affine matrix, and header.

    Args:
        file_path (Path): Path to the NIfTI file.

    Returns:
        tuple: (data, affine, header) of the NIfTI file.
    """
    nifti_img = nib.load(file_path)
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
    nib.save(nifti_img, file_path)

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
        original_spacing[2] / target_spacing[2]
    ]
    new_shape = np.round(np.array(data.shape) * resize_factor).astype(int)

    # Resample
    if is_label:
        # Use nearest-neighbor interpolation for labels
        data_resampled = resize(
            data,
            new_shape,
            order=0,
            mode='edge',
            preserve_range=True,
            anti_aliasing=False
        )
    else:
        # Use trilinear interpolation for CT scans
        data_resampled = resize(
            data,
            new_shape,
            order=1,
            mode='edge',
            preserve_range=True,
            anti_aliasing=False
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

def preprocess_patient(patient_dir, patient_id, target_spacing):
    """
    Preprocess a single patient's data.

    Args:
        patient_dir (Path): Directory containing patient data.
        patient_id (str): Patient identifier.
        target_spacing (tuple): Target voxel spacing for resampling.
    """
    logging.info(f"Processing {patient_id}")

    # File paths
    ct_path = patient_dir / f"{patient_id}.nii.gz"
    gt_path = patient_dir / "GT_corrected.nii.gz"
    output_ct_path = patient_dir / f"{patient_id}_enhanced.nii.gz"
    output_gt_path = patient_dir / "GT_enhanced.nii.gz"

    # Load data
    ct_data, ct_affine, ct_header = load_nifti(ct_path)
    gt_data, gt_affine, gt_header = load_nifti(gt_path)

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
        resampled_gt_data = gt_data
    else:
        logging.info(f"Resampling {patient_id} to target spacing {target_spacing}")
        # Resample CT scan
        resampled_ct_data = resample_image(ct_data, original_spacing, target_spacing, is_label=False)
        # Resample Ground Truth
        resampled_gt_data = resample_image(gt_data, original_spacing, target_spacing, is_label=True)
        # Update affine matrix and header
        ct_affine = update_affine(ct_affine, original_spacing, target_spacing)
        ct_header.set_zooms(target_spacing)
        gt_affine = ct_affine  # Assuming GT shares the same affine

    # Intensity Normalization
    resampled_ct_data = intensity_normalization(resampled_ct_data)
    logging.info(f"Applied intensity normalization to {patient_id}")

    # Save preprocessed data
    save_nifti(resampled_ct_data, ct_affine, ct_header, output_ct_path)
    save_nifti(resampled_gt_data, gt_affine, gt_header, output_gt_path)
    logging.info(f"Saved enhanced data for {patient_id}")

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

def main():
    """
    Main function to preprocess all patients in the dataset.
    """
    setup_logging()
    project_root = get_project_root()
    base_dir = project_root / 'data' / 'segthor_train' / 'train'
    target_spacing = (0.977, 0.977, 2.5)  # Most common voxel size

    print("Voxel clipping, rescaling and intensity normalizing all patients. Check progress in preprocessing/insights/preprocessing_log.txt")

    for i in tqdm(range(1, 41), desc="Pre-processing Patients"):
        patient_id = f"Patient_{i:02d}"
        patient_dir = base_dir / patient_id
        try:
            preprocess_patient(patient_dir, patient_id, target_spacing)
        except Exception as e:
            logging.error(f"Error processing {patient_id}: {str(e)}")

if __name__ == "__main__":
    main()