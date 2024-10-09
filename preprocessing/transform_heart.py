import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import os
from pathlib import Path

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

def load_nifti(file_path):
    """
    Load a NIfTI file and return the data and affine matrix.

    Parameters:
        file_path (Path): Path to the NIfTI file.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The image data.
            - affine (numpy.ndarray): The affine transformation matrix.
    """
    # Load the NIfTI image using nibabel
    nifti_img = nib.load(file_path)
    # Get the image data as a numpy array
    data = nifti_img.get_fdata()
    # Get the affine transformation matrix
    affine = nifti_img.affine
    return data, affine

def compute_centroid(data, label):
    """
    Compute the centroid of the voxels with the given label.

    Parameters:
        data (numpy.ndarray): The image data.
        label (int): The label to compute the centroid for.

    Returns:
        numpy.ndarray: The centroid coordinates (x, y, z).
    """
    # Find the indices of all voxels with the specified label
    coords = np.argwhere(data == label)
    # Handle the case where the label is not present in the data
    if coords.size == 0:
        return np.array([0, 0, 0])
    # Compute the mean position along each axis (centroid)
    centroid = coords.mean(axis=0)
    return centroid

def correct_heart_segmentation(patient_number, translation_vector, base_path):
    """
    Correct the heart segmentation for the given patient by applying the translation vector.

    Parameters:
        patient_number (int): The patient number to process.
        translation_vector (numpy.ndarray): The translation vector to apply.
        base_path (Path): Base path to the patient data.

    Returns:
        None
    """
    # Construct the path to the patient's GT.nii.gz file
    patient_folder = base_path / f'Patient_{patient_number:02d}'
    gt_path = patient_folder / 'GT.nii.gz'

    # Check if the GT.nii.gz file exists
    if not gt_path.exists():
        print(f"GT.nii.gz not found for patient {patient_number}, skipping.")
        return

    # Load the patient's GT data
    gt_data, affine = load_nifti(gt_path)

    # Create a binary mask for the heart segmentation (label == 2)
    heart_mask = (gt_data == 2).astype(np.float32)  # Use float32 for interpolation

    # Define the identity matrix for the affine transformation
    transformation_matrix = np.eye(3)

    # The offset is the negative of the translation vector
    offset = -translation_vector

    # Apply the affine transformation to shift the heart mask back to its correct position
    corrected_heart_mask = affine_transform(
        heart_mask,
        matrix=transformation_matrix,
        offset=offset,
        order=0,        # Nearest-neighbor interpolation to preserve labels
        mode='constant',# Outside values are set to a constant (0)
        cval=0.0,       # Constant value for points outside boundaries
        output=np.float32
    )

    # Threshold the corrected heart mask to get a binary mask
    corrected_heart_mask = (corrected_heart_mask > 0.5).astype(np.uint8)

    # Create a copy of the GT data to modify
    corrected_gt_data = gt_data.copy()
    # Remove the old (shifted) heart segmentation
    corrected_gt_data[gt_data == 2] = 0
    # Add the corrected heart segmentation
    corrected_gt_data[corrected_heart_mask == 1] = 2

    # Save the corrected GT data to a new NIfTI file
    corrected_gt_img = nib.Nifti1Image(corrected_gt_data, affine)
    corrected_gt_path = patient_folder / 'GT_corrected.nii.gz'
    nib.save(corrected_gt_img, corrected_gt_path)
    print(f"Saved corrected GT for patient {patient_number} at {corrected_gt_path}")

def main():
    """
    Main function to compute the translation vector and correct the heart segmentation for all patients.
    """
    project_root = get_project_root()
    base_path = project_root / 'data' / 'segthor_train' / 'train'

    # Paths to the corrupted and uncorrupted GT files for patient_27
    gt_corrupted_path = base_path / 'Patient_27' / 'GT.nii.gz'
    gt_uncorrupted_path = base_path / 'Patient_27' / 'GT2.nii.gz'

    # Load the corrupted and uncorrupted GT data
    gt_corrupted_data, affine = load_nifti(gt_corrupted_path)
    gt_uncorrupted_data, _ = load_nifti(gt_uncorrupted_path)

    # Compute the centroids of the heart segmentation in both datasets
    label = 2  # The label for the heart
    centroid_corrupted = compute_centroid(gt_corrupted_data, label)
    centroid_uncorrupted = compute_centroid(gt_uncorrupted_data, label)

    # Calculate the translation vector that was applied to the heart segmentation
    translation_vector = centroid_uncorrupted - centroid_corrupted
    print("Translation vector:", translation_vector)

    # Loop over all patients
    for patient_number in range(1, 41):
        # if patient_number == 27:
        #     continue

        print(f"Processing patient {patient_number}...")
        correct_heart_segmentation(patient_number, translation_vector, base_path)


if __name__ == "__main__":
    main()