import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import os
import matplotlib.pyplot as plt

def load_nifti(file_path):
    """
    Load a NIfTI file and return the data and affine matrix.

    Parameters:
        file_path (str): Path to the NIfTI file.

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
    # Compute the mean position along each axis (centroid)
    centroid = coords.mean(axis=0)
    return centroid

def correct_heart_segmentation(patient_number, translation_vector, cor_folder):
    """
    Correct the heart segmentation for the given patient by applying the translation vector.

    Parameters:
        patient_number (int): The patient number to process.
        translation_vector (numpy.ndarray): The translation vector to apply.

    Returns:
        None
    """

    # Construct the path to the patient's GT.nii.gz file
    patient_folder = f'data/segthor_train/train/Patient_{patient_number:02d}'
    gt_path = os.path.join(patient_folder, 'GT.nii.gz')

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

    # Create a folder to store the corrected segmentations
    if not os.path.exists(f"data/SEGTHOR/cor/{cor_folder}"):
        os.makedirs(f"data/SEGTHOR/cor/{cor_folder}") 

    # Determine the path to store the corrected file
    corrected_gt_img = nib.Nifti1Image(corrected_gt_data, affine)
    corrected_folder = f"data/SEGTHOR/cor/{cor_folder}"
    corrected_gt_path = os.path.join(corrected_folder, 'GT_corrected.nii.gz')

    # Save the corrected GT data to a new NIfTI file
    corrected_gt_arr = np.asanyarray(corrected_gt_img.dataobj)
    # plt.imsave("test_image_corrected")
    for i in range(corrected_gt_data.shape[2]):
        # print(corrected_gt_arr[:,:,i])
        plt.imsave(f"data/SEGTHOR/cor/{cor_folder}/Patient_{patient_number:02d}_{i:04d}_C.png", corrected_gt_arr[:,:,i], cmap='gray')
    # nib.save(corrected_gt_img, corrected_gt_path)
    print(f"Saved corrected GT for patient {patient_number} at {corrected_gt_path}")

def main():
    """
    Main function to compute the translation vector and correct the heart segmentation for a patient.
    """
    # Paths to the corrupted and uncorrupted GT files for patient_27
    gt_corrupted_path = 'data/segthor_train/train/Patient_27/GT.nii.gz'
    gt_uncorrupted_path = 'data/segthor_train/train/Patient_27/GT2.nii.gz'

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

    # Prompt the user to input the patient number they want to correct
    for i in range(40):
        patient_number = i+1
        print("Correcting segmentation for Patient ", patient_number)
        if (patient_number in (1,2,13,16,21,22,28,30,35,39)):
            cor_folder = "val"
            print(cor_folder)
        else:
            cor_folder = "train"
            print(cor_folder)
        correct_heart_segmentation(patient_number, translation_vector, cor_folder)
    

if __name__ == "__main__":
    main()
