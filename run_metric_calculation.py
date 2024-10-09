#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import scipy.ndimage as nd
from tqdm import tqdm
import logging
from scipy.spatial import cKDTree
from collections import defaultdict
from PIL import Image
import glob

def load_nifti(file_path):
    """
    Load a NIfTI file and return its data as a numpy array.
    """
    print(f"Loading NIfTI file: {file_path}")
    return nib.load(file_path).get_fdata()

def load_png_slice(file_path):
    """
    Load a PNG image as a numpy array.
    """
    return np.array(Image.open(file_path).convert('L'), dtype=np.uint8)

def compute_surface_distances(y_true, y_pred, class_label):
    """
    Compute surface distances for both Hausdorff Distance and ASSD using k-d trees.
    """
    y_true_class = (y_true == class_label)
    y_pred_class = (y_pred == class_label)

    if not np.any(y_true_class) and not np.any(y_pred_class):
        return None  # Both masks are empty
    elif not np.any(y_true_class) or not np.any(y_pred_class):
        return np.inf  # One mask is empty; infinite distance

    # Extract surface voxels
    surface_true = y_true_class ^ nd.binary_erosion(y_true_class)
    surface_pred = y_pred_class ^ nd.binary_erosion(y_pred_class)

    # Get coordinates of surface voxels
    surface_true_coords = np.argwhere(surface_true)
    surface_pred_coords = np.argwhere(surface_pred)

    if surface_true_coords.size == 0 or surface_pred_coords.size == 0:
        return np.inf  # No surface points found in one of the masks

    # Build k-d trees
    tree_pred = cKDTree(surface_pred_coords)
    tree_true = cKDTree(surface_true_coords)

    # Compute distances
    distances_true_to_pred, _ = tree_pred.query(surface_true_coords)
    distances_pred_to_true, _ = tree_true.query(surface_pred_coords)

    return distances_true_to_pred, distances_pred_to_true

def hausdorff_and_assd(y_true, y_pred, class_label):
    """
    Calculate both 95th percentile Hausdorff distance and ASSD for a specific class.
    """
    # Calculating the surface distances
    distances = compute_surface_distances(y_true, y_pred, class_label)

    if distances is None:
        return 0.0, 0.0  # Both masks are empty; perfect agreement
    elif isinstance(distances, float) and np.isinf(distances):
        return np.inf, np.inf  # One mask is empty; infinite distance

    distances_true_to_pred, distances_pred_to_true = distances

    if distances_true_to_pred.size == 0 or distances_pred_to_true.size == 0:
        return np.inf, np.inf  # No surface points found in one of the masks

    all_distances = np.concatenate([distances_true_to_pred, distances_pred_to_true])

    # Calculate HD95
    hd95 = np.percentile(all_distances, 95)

    # Calculate ASSD
    assd_value = (np.mean(distances_true_to_pred) + np.mean(distances_pred_to_true)) / 2.0

    return hd95, assd_value

def calculate_hd_assd_metrics(y_true, y_pred, class_labels):
    """
    Calculate HD95 and ASSD metrics for all classes except background.
    """
    metrics = {}
    for class_label in class_labels[1:]:
        hd95, assd_val = hausdorff_and_assd(y_true, y_pred, class_label)
        metrics[class_label] = {'HD95': hd95, 'ASSD': assd_val}
    return metrics

def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice Coefficient between two boolean masks.
    
    Args:
        y_true (np.ndarray): Ground truth boolean mask.
        y_pred (np.ndarray): Predicted boolean mask.
    
    Returns:
        float: Dice coefficient.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    sum_gt = y_true.sum()
    sum_pred = y_pred.sum()

    if sum_gt + sum_pred == 0:
        return 1.0  # Both masks are empty; perfect agreement
    else:
        return (2. * intersection + 1e-8) / (sum_gt + sum_pred + 1e-8)


def calculate_dice_from_pngs(patient_id, gt_png_paths, pred_png_paths, class_labels):
    """
    Calculate 2D Dice metrics for all classes except background from PNG slices.
    """
    metrics = {}
    per_slice_dices = defaultdict(list)  # To store per-slice Dices per class
    for class_label in class_labels[1:]:
        dice_scores = []
        for gt_path, pred_path in zip(gt_png_paths, pred_png_paths):
            # Load the PNG slices
            gt_slice = load_png_slice(gt_path)
            pred_slice = load_png_slice(pred_path)

            # Normalize labels (assuming labels are multiples of 63)
            gt_slice = gt_slice // 63
            pred_slice = pred_slice // 63

            y_true_class = (gt_slice == class_label)
            y_pred_class = (pred_slice == class_label)

            # Dice calculation
            dice_score = dice_coefficient(y_true_class, y_pred_class)

            dice_scores.append(dice_score)
            per_slice_dices[class_label].append(dice_score)

        # Compute mean Dice for the class
        mean_dice = np.mean(dice_scores)
        metrics[class_label] = {'2D_Dice': mean_dice}
    return metrics, per_slice_dices

def three_d_dice_nifti(y_true, y_pred, class_labels):
    """
    Calculate 3D Dice directly on NIfTI volumes.
    """
    metrics = {}
    for class_label in class_labels[1:]:
        y_true_class = (y_true == class_label)
        y_pred_class = (y_pred == class_label)

        intersection = np.logical_and(y_true_class, y_pred_class).sum()
        sum_gt = y_true_class.sum()
        sum_pred = y_pred_class.sum()

        if sum_gt + sum_pred == 0:
            dice = 1.0  # Both masks are empty; perfect agreement
        else:
            dice = (2. * intersection + 1e-8) / (sum_gt + sum_pred + 1e-8)

        metrics[class_label] = dice
    return metrics

def create_viz_folder():
    """
    Create a folder to store visualizations.
    """
    viz_folder = 'viz'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    return viz_folder

def setup_logging():
    """
    Set up logging to a file.
    """
    logging.basicConfig(filename='metrics_log2.txt', level=logging.INFO,
                        format='%(message)s', filemode='w')

def log_metrics(patient_id, dice_metrics, dice_3d_metrics, hd_assd_metrics, class_names):
    """
    Log the metrics for a patient.
    """
    logging.info(f"Patient: {patient_id}")
    logging.info("-" * 80)
    header = "{:<15} {:>12} {:>12} {:>12} {:>12}".format("Class", "2D_Dice", "3D_Dice", "HD95", "ASSD")
    logging.info(header)
    for class_label in dice_metrics.keys():
        class_name = class_names.get(class_label, f"Class {class_label}")
        dice_2d = dice_metrics[class_label]['2D_Dice']
        dice_3d = dice_3d_metrics[class_label]
        hd95 = hd_assd_metrics[class_label]['HD95']
        assd_val = hd_assd_metrics[class_label]['ASSD']
        logging.info("{:<15} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            class_name,
            dice_2d,
            dice_3d,
            hd95 if not np.isinf(hd95) else float('nan'),
            assd_val if not np.isinf(assd_val) else float('nan')
        ))
    # Compute mean metrics for the patient
    mean_dice_2d = np.mean([m['2D_Dice'] for m in dice_metrics.values()])
    mean_dice_3d = np.mean(list(dice_3d_metrics.values()))
    mean_hd95 = np.mean([m['HD95'] for m in hd_assd_metrics.values()])
    mean_assd = np.mean([m['ASSD'] for m in hd_assd_metrics.values()])

    logging.info("{:<15} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        "Mean",
        mean_dice_2d,
        mean_dice_3d,
        mean_hd95,
        mean_assd
    ))
    logging.info("\n")

def calculate_metrics(pred_nifti_dir, pred_png_dir, gt_nifti_dir, gt_png_dir, class_labels, class_names, description):
    """
    Main function to orchestrate the segmentation analysis process.
    """
    # Remove setup_logging() call
    # Use logging.info directly, since logging is configured in main.py

    logging.info(f"Segmentation Metrics ({description})")
    logging.info("=" * 80)

    # For storing overall metrics
    all_dice_metrics = defaultdict(list)      # For 2D Dice (per-slice)
    all_dice_3d_metrics = defaultdict(list)   # For 3D Dice (per-patient)
    all_hd_assd_metrics = defaultdict(list)   # For HD95 and ASSD (per-patient)

    # Get list of patient IDs from NIfTI prediction files
    pred_nifti_files = [f for f in os.listdir(pred_nifti_dir) if f.endswith('.nii.gz')]
    pred_nifti_files.sort()
    patient_ids = [f.replace('.nii.gz', '') for f in pred_nifti_files]

    for patient_id in tqdm(patient_ids, desc=f"Processing Patients ({description})"):
        logging.info(f"\nProcessing Patient: {patient_id}")
        pred_nifti_path = os.path.join(pred_nifti_dir, patient_id + '.nii.gz')
        gt_nifti_path = os.path.join(gt_nifti_dir, patient_id, 'GT.nii.gz')

        if not os.path.exists(gt_nifti_path):
            logging.warning(f"Ground truth NIfTI not found for {patient_id}, skipping.")
            continue

        # Load NIfTI data for HD95 and ASSD
        y_pred_nifti = load_nifti(pred_nifti_path).astype(np.int32)
        y_true_nifti = load_nifti(gt_nifti_path).astype(np.int32)

        # Calculate HD95 and ASSD metrics
        hd_assd_metrics = calculate_hd_assd_metrics(y_true_nifti, y_pred_nifti, class_labels)

        # Load PNG slices for 2D Dice
        gt_png_pattern = os.path.join(gt_png_dir, f'{patient_id}_*.png')
        pred_png_pattern = os.path.join(pred_png_dir, f'{patient_id}_*.png')

        gt_png_paths = sorted(glob.glob(gt_png_pattern))
        pred_png_paths = sorted(glob.glob(pred_png_pattern))

        if not gt_png_paths or not pred_png_paths:
            logging.warning(f"PNG slices not found for {patient_id}, skipping 2D Dice and 3D Dice calculations.")
            dice_metrics = {class_label: {'2D_Dice': np.nan} for class_label in class_labels[1:]}
            dice_3d_metrics = {class_label: np.nan for class_label in class_labels[1:]}
        else:
            if len(gt_png_paths) != len(pred_png_paths):
                logging.warning(f"Number of GT and prediction PNG slices do not match for {patient_id}, skipping 2D Dice and 3D Dice calculations.")
                dice_metrics = {class_label: {'2D_Dice': np.nan} for class_label in class_labels[1:]}
                dice_3d_metrics = {class_label: np.nan for class_label in class_labels[1:]}
            else:
                # Calculate 2D Dice metrics from PNG slices
                dice_metrics, per_slice_dices = calculate_dice_from_pngs(patient_id, gt_png_paths, pred_png_paths, class_labels)
                # Accumulate per-slice 2D Dice metrics
                for class_label in class_labels[1:]:
                    all_dice_metrics[class_label].extend(per_slice_dices[class_label])

                # Calculate 3D Dice directly from NIfTI
                dice_3d_metrics = three_d_dice_nifti(y_true_nifti, y_pred_nifti, class_labels)
                # Accumulate 3D Dice metrics
                for class_label in class_labels[1:]:
                    all_dice_3d_metrics[class_label].append(dice_3d_metrics[class_label])

        # Log metrics
        log_metrics(patient_id, dice_metrics, dice_3d_metrics, hd_assd_metrics, class_names)
        # Accumulate HD and ASSD metrics for overall statistics
        for class_label in class_labels[1:]:
            # HD95 and ASSD
            all_hd_assd_metrics[class_label].append(hd_assd_metrics.get(class_label, {'HD95': np.nan, 'ASSD': np.nan}))

    # After all patients processed, compute overall metrics
    logging.info("=" * 80)
    logging.info(f"Overall Metrics ({description})")
    logging.info("=" * 80)
    logging.info("{:<15} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Class",
        "Mean 2D_Dice",
        "Std 2D_Dice",
        "Mean 3D_Dice",
        "Std 3D_Dice",
        "Mean HD95",
        "Std HD95",
        "Mean ASSD",
        "Std ASSD"
    ))
    for class_label in class_labels[1:]:
        class_name = class_names.get(class_label, f"Class {class_label}")
        # ... [rest of the code as before] ...

    logging.info("=" * 80)
    logging.info(f"Completed metrics calculation for {description}.")
