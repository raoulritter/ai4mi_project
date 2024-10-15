#!/usr/bin/env python3

"""
test_set_inference.py

This module contains functions to perform inference on the test set using a trained model.
It includes functionalities to:

- Load the test slices.
- Perform inference using the best model.
- Save the predicted slices.
- Reconstruct NIfTI files from the predicted slices.
- Post-process the reconstructed NIfTI files.

Functions:
- run_test_inference(args, net, device, K)
- reconstruct_nifti_test_set(png_folder, nifti_folder, output_folder, K)
- TestSliceDataset
"""

import os
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import nibabel as nib
from PIL import Image

from utils import probs2class, save_images
from post_process.post_process import post_process_nifti_files


def run_test_inference(args, net, device, K):
    """
    Perform inference on the test set using the provided model.

    Args:
        args: Command-line arguments.
        net: Trained model.
        device: Device to run inference on (CPU or GPU).
        K: Number of classes (including background).
    """
    # Set up root directory and image directory for test set
    if args.preprocess:
        test_root_dir = Path("data") / "SEGTHOR_TEST_preprocessed"
        img_dir = test_root_dir / 'img'
    else:
        test_root_dir = Path("data") / "segthor_test" / "test"
        img_dir = test_root_dir  # Assuming images are directly under this path

    print("test_root_dir: ", test_root_dir)
    print("img_dir: ", img_dir)

    # Define image transformations
    img_transform = transforms.Compose([
        lambda img: img.convert('L'),  # Convert to grayscale
        lambda img: np.array(img)[np.newaxis, ...],  # Add channel dimension
        lambda nd: nd / 255,  # Normalize to [0, 1]
        lambda nd: torch.tensor(nd, dtype=torch.float32)  # Convert to torch tensor
    ])

    # Create test dataset
    test_set = TestSliceDataset(img_dir=img_dir,
                                img_transform=img_transform)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             num_workers=args.num_workers,
                             shuffle=False)

    # Directory to save predicted slices
    best_folder = args.dest / "best_epoch"
    png_output_dir = best_folder / 'test' / 'png'
    png_output_dir.mkdir(parents=True, exist_ok=True)

    # Perform inference
    print(">>> Starting inference on test set.")
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Inference on test set'):
            img = data['images'].to(device)
            stems = data['stems']

            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)

            predicted_class = probs2class(pred_probs)
            mult: int = 63 if K == 5 else (255 / (K - 1))
            save_images(predicted_class * mult,
                        stems,
                        png_output_dir)

    # Reconstruct NIfTI files from predicted slices
    print(">>> Reconstructing NIfTI files from predicted slices.")

    # Path to original NIfTI files for header and affine
    if args.preprocess:
        nifti_folder = test_root_dir / 'nifti'
    else:
        nifti_folder = Path("data") / "segthor_test" / "test"

    print("NIFTI folder: ", nifti_folder)

    nifti_output_folder = best_folder / 'test' / 'nifti'
    nifti_output_folder.mkdir(parents=True, exist_ok=True)

    reconstruct_nifti_test_set(str(png_output_dir), str(nifti_folder), str(nifti_output_folder), K)

    # Post-processing of the test NIfTI files
    print(">>> Starting post-processing of test NIfTI files.")

    post_process_input_folder = str(nifti_output_folder)
    post_process_output_folder = str(best_folder / 'test' / 'nifti_post_processed')
    num_classes = K  # Number of classes including background

    post_process_nifti_files(post_process_input_folder, post_process_output_folder, num_classes)
    print(">>> Post-processing of test NIfTI files completed successfully.")


def reconstruct_nifti_test_set(png_folder, nifti_folder, output_folder, K):
    """
    Reconstruct NIfTI files from predicted PNG slices for the test set.

    Args:
        png_folder (str): Path to the folder containing predicted PNG slices.
        nifti_folder (str): Path to the folder containing original NIfTI files (for header and affine).
        output_folder (str): Path to save the reconstructed NIfTI files.
        K (int): Number of classes (including background).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all PNG files in the directory
    png_files = glob.glob(os.path.join(png_folder, "*.png"))

    # Extract unique patient IDs from the filenames
    patient_ids = set("_".join(os.path.basename(f).split("_")[:2]) for f in png_files)

    for patient_id in patient_ids:
        # Collect all PNG files for this patient
        patient_png_files = [
            f for f in png_files if os.path.basename(f).startswith(patient_id)
        ]

        # Function to extract slice index from filename
        def get_slice_index(filename):
            return int(os.path.splitext(filename)[0].split("_")[-1])

        # Sort the PNG files based on slice index
        patient_png_files.sort(key=lambda x: get_slice_index(os.path.basename(x)))

        # Load the original NIfTI file to get target dimensions and metadata
        nifti_path = os.path.join(nifti_folder, f"{patient_id}.nii.gz")
        if not os.path.exists(nifti_path):
            print(
                f"Original NIfTI file not found for {patient_id} at {nifti_path}"
            )
            continue

        nifti = nib.load(nifti_path)
        data = nifti.get_fdata()
        header = nifti.header
        affine = nifti.affine
        target_shape = data.shape  # (width, height, num_slices)
        width, height, num_slices = (
            int(target_shape[0]),
            int(target_shape[1]),
            int(target_shape[2]),
        )

        # Read and stack slices to form a 3D volume
        slices = []
        for png_file in patient_png_files:
            img = Image.open(png_file)
            # Resize the image to match original NIfTI dimensions
            img_resized = img.resize((width, height), Image.NEAREST)
            # Convert image to array and map pixel values back to class labels
            img_array = np.array(img_resized)
            # Map pixel values back to class labels
            if K == 5:
                img_array = np.round(img_array / 63).astype(np.uint8)
            else:
                img_array = np.round(img_array / (255 / (K - 1))).astype(np.uint8)
            slices.append(img_array)

        # Stack slices along the correct axis
        volume = np.stack(slices, axis=0)  # Shape: (num_slices, height, width)

        # Transpose volume to match NIfTI dimension ordering (X, Y, Z)
        volume = np.transpose(volume, (1, 2, 0))  # Shape: (width, height, num_slices)

        # Ensure the volume matches the original NIfTI dimensions
        if volume.shape != data.shape:
            print(
                f"Warning: Volume shape {volume.shape} does not match original NIfTI shape {data.shape} for {patient_id}"
            )
            continue  # Skip to the next patient if dimensions do not match

        # Create a new NIfTI image with the same header and affine as the original NIfTI
        new_nifti = nib.Nifti1Image(volume, affine=affine, header=header)

        # Save the new NIfTI file
        output_path = os.path.join(output_folder, f"{patient_id}.nii.gz")
        nib.save(new_nifti, output_path)
        print(f"Saved NIfTI file for {patient_id} at {output_path}")


class TestSliceDataset(Dataset):
    """
    Dataset class for loading test slices without ground truths.

    Args:
        img_dir (Path): Directory containing test images.
        img_transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, img_dir, img_transform=None):
        self.img_dir = Path(img_dir)
        self.img_transform = img_transform

        self.images = sorted(self.img_dir.glob("*.png"))

        # Sort images by patient ID and slice index
        self.images = sorted(
            self.images, key=lambda x: (x.stem.split("_")[1], int(x.stem.split("_")[2]))
        )

        print(f">> Created test dataset with {len(self)} images...")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]

        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)

        # Extract stem
        stem = img_path.stem  # e.g., 'Patient_41_0000'

        sample = {
            "images": img,
            "stems": stem,
        }

        return sample
