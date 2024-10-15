import argparse
import random
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import AffineTransform, warp
from tqdm import tqdm


def augment_image_pair(args):
    img_path, gt_path, dest_img_dir, dest_gt_dir = args

    # Read images
    img = imread(str(img_path))
    gt = imread(str(gt_path))

    # Generate random parameters
    # Random rotation angle between -10 and +10 degrees
    angle = random.uniform(-10, 10)
    # Random horizontal flip
    h_flip = random.choice([True, False])
    # Random vertical flip
    v_flip = random.choice([True, False])
    # Random scale between 0.9 and 1.1
    scale = random.uniform(0.9, 1.1)
    # Random translation
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)

    # Create an affine transformation
    tform = AffineTransform(
        scale=(scale, scale), rotation=np.deg2rad(angle), translation=(tx, ty)
    )

    # Apply to image
    augmented_img = warp(
        img, tform.inverse, order=1, mode="edge", preserve_range=True
    ).astype(np.uint8)
    # Apply flips
    # if h_flip:
    #     augmented_img = np.fliplr(augmented_img)
    # if v_flip:
    #     augmented_img = np.flipud(augmented_img)

    # Apply to ground truth
    augmented_gt = warp(
        gt, tform.inverse, order=0, mode="edge", preserve_range=True
    ).astype(np.uint8)
    if h_flip:
        augmented_gt = np.fliplr(augmented_gt)
    if v_flip:
        augmented_gt = np.flipud(augmented_gt)

    # Save augmented images
    # Build output file paths
    filename = (
        img_path.name
    )  # Assuming the filenames are the same in img and gt folders

    # Suppress low contrast warning for ground truth images
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        imsave(str(dest_img_dir / filename), augmented_img)
        imsave(str(dest_gt_dir / filename), augmented_gt)


def main():
    parser = argparse.ArgumentParser(
        description="Data Augmentation for SegTHOR Dataset"
    )
    parser.add_argument(
        "--slice_dir",
        type=str,
        default="data/SEGTHOR_preprocessed",
        help="Path to the directory containing existing slices.",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="data/SEGTHOR_preprocessed",
        help="Path to save the augmented images.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of parallel workers."
    )
    args = parser.parse_args()

    slice_path = Path(args.slice_dir)
    dest_path = Path(args.dest_dir)

    # Paths to train images and ground truths
    train_img_dir = slice_path / "train" / "img"
    train_gt_dir = slice_path / "train" / "gt"

    # Output directories
    dest_img_dir = dest_path / "train" / "img_aug"
    dest_gt_dir = dest_path / "train" / "gt_aug"

    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_gt_dir.mkdir(parents=True, exist_ok=True)

    # Get list of images
    img_files = sorted(train_img_dir.glob("*.png"))
    gt_files = sorted(train_gt_dir.glob("*.png"))

    # Ensure that image and ground truth files match
    assert len(img_files) == len(
        gt_files
    ), "Number of images and ground truths do not match"

    img_filenames = [img.name for img in img_files]
    gt_filenames = [gt.name for gt in gt_files]

    assert (
        img_filenames == gt_filenames
    ), "Image and ground truth filenames do not match"

    # Prepare arguments for multiprocessing
    tasks = []
    for img_path, gt_path in zip(img_files, gt_files):
        tasks.append((img_path, gt_path, dest_img_dir, dest_gt_dir))

    # Process tasks
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            list(tqdm(pool.imap(augment_image_pair, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks, desc="Augmenting Input Slices"):
            augment_image_pair(task)

    print(
        f"Data augmentation completed. Saved augmented images in the '{dest_gt_dir}' and '{dest_img_dir}' folders "
    )


if __name__ == "__main__":
    main()
