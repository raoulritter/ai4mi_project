#!/usr/bin/env python3

"""
run_preprocess.py

This script runs the full preprocessing pipeline for the SegTHOR medical imaging segmentation challenge.
It sequentially executes the following scripts located in the 'preprocessing' directory:

1. transform_heart.py:
   - Corrects the positioning of the heart in the ground truth NIfTI files.
   - Applies a calculated translation vector to align the heart segmentation properly.
   - Saves the corrected ground truth as 'GT_corrected.nii.gz' for each patient.

2. clip_scale_norm.py:
   - Performs voxel clipping to limit CT scan intensity values between -1000 and 1000 HU.
   - Resamples the images to a common voxel spacing (e.g., 0.977 x 0.977 x 2.5 mm).
   - Applies intensity normalization to have zero mean and unit variance.
   - Saves the enhanced CT scans and ground truths as '{Patient_ID}_enhanced.nii.gz' and 'GT_enhanced.nii.gz'.

3. slice_preprocess.py:
   - Slices the enhanced NIfTI files into 2D PNG images for each axial slice.
   - Splits the dataset into training and validation sets.
   - Saves the images and labels in the specified directories.

Each step is crucial for preparing the data before training the segmentation model.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Define the path to the preprocessing directory
    preprocessing_dir = Path(__file__).parent / "preprocessing"

    # Step 1: Run transform_heart.py
    print("Step 1/3: Running transform_heart.py...")
    transform_script = preprocessing_dir / "transform_heart.py"
    if not transform_script.exists():
        print(f"Error: {transform_script} not found.")
        sys.exit(1)
    try:
        subprocess.check_call([sys.executable, str(transform_script)])
        print("Completed transform_heart.py.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running transform_heart.py: {e}")
        sys.exit(1)

    # Step 2: Run clip_scale_norm.py
    print("Step 2/3: Running clip_scale_norm.py...")
    clip_scale_norm_script = preprocessing_dir / "clip_scale_norm.py"
    if not clip_scale_norm_script.exists():
        print(f"Error: {clip_scale_norm_script} not found.")
        sys.exit(1)
    try:
        subprocess.check_call([sys.executable, str(clip_scale_norm_script)])
        print("Completed clip_scale_norm.py.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running clip_scale_norm.py: {e}")
        sys.exit(1)

    # Step 3: Run slice_preprocess.py
    print("Step 3/3: Running slice_preprocess.py...")
    slice_preprocess_script = preprocessing_dir / "slice_preprocess.py"
    if not slice_preprocess_script.exists():
        print(f"Error: {slice_preprocess_script} not found.")
        sys.exit(1)
    try:
        # Provide the necessary arguments for slice_preprocess.py
        source_dir = "data/segthor_train"
        dest_dir = "data/SEGTHOR_preprocessed"
        shape = ["256", "256"]  # Adjust as needed
        retains = "10"  # Number of patients to retain for validation
        seed = "0"
        fold = "0"
        process = "1"  # Number of processes (-1 for all available cores)

        subprocess.check_call(
            [
                sys.executable,
                str(slice_preprocess_script),
                "--source_dir",
                source_dir,
                "--dest_dir",
                dest_dir,
                "--shape",
                *shape,
                "--retains",
                retains,
                "--seed",
                seed,
                "--fold",
                fold,
                "--process",
                process,
            ]
        )
        print("Completed slice_preprocess.py.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running slice_preprocess.py: {e}")
        sys.exit(1)

    print("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
