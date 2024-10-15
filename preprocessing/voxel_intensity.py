import os
import nibabel as nib
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
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

def ensure_directory_exists(path: Path):
    """
    Ensure that the directory exists, creating it if necessary.

    Args:
        path (Path): The directory path to check/create.
    """
    path.mkdir(parents=True, exist_ok=True)

def get_intensity_statistics(base_path: Path) -> list:
    """
    Calculate intensity statistics for CT scans in the SegTHOR dataset.
    
    Args:
        base_path (Path): Path to the directory containing patient subdirectories.
    
    Returns:
        list: A list of statistics for each patient.
    """
    statistics = []
    total_patients = 40
    script_dir = Path(__file__).resolve().parent
    log_file_path = script_dir / 'insights' / 'intensity_log.txt'
    
    # Ensure the directory exists
    ensure_directory_exists(log_file_path.parent)
    
    with open(log_file_path, 'w') as log_file:
        for i in tqdm(range(1, total_patients + 1), total=total_patients, desc="Processing patients"):
            patient_id = f"Patient_{i:02d}"
            file_path = base_path / patient_id / f"{patient_id}.nii.gz"
            
            try:
                nifti_img = nib.load(file_path)
                data = nifti_img.get_fdata()
                voxels = data.flatten()
                
                min_intensity = np.min(voxels)
                max_intensity = np.max(voxels)
                mean_intensity = np.mean(voxels)
                median_intensity = np.median(voxels)
                std_intensity = np.std(voxels)
                
                total_voxels = voxels.size
                lower_threshold = -1000  # HU value for air
                upper_threshold = 1000   # HU value for dense bone
                
                voxels_below_threshold = np.sum(voxels < lower_threshold)
                voxels_above_threshold = np.sum(voxels > upper_threshold)
                
                percent_below = (voxels_below_threshold / total_voxels) * 100
                percent_above = (voxels_above_threshold / total_voxels) * 100
                
                log_patient_statistics(log_file, patient_id, min_intensity, max_intensity, mean_intensity, 
                                       median_intensity, std_intensity, total_voxels, lower_threshold, 
                                       upper_threshold, voxels_below_threshold, voxels_above_threshold, 
                                       percent_below, percent_above)
                
                statistics.append([patient_id, min_intensity, max_intensity, mean_intensity, median_intensity, std_intensity])
                
            except FileNotFoundError:
                log_file.write(f"Warning: File not found for {patient_id}\n\n")
            except Exception as e:
                log_file.write(f"Error processing {patient_id}: {str(e)}\n\n")
    
    return statistics

def log_patient_statistics(log_file, patient_id, min_intensity, max_intensity, mean_intensity, 
                           median_intensity, std_intensity, total_voxels, lower_threshold, 
                           upper_threshold, voxels_below_threshold, voxels_above_threshold, 
                           percent_below, percent_above):
    """
    Write patient statistics to the log file.
    
    Args:
        log_file (file): The open file object to write to.
        patient_id (str): The ID of the patient.
        ... (other parameters are self-explanatory)
    """
    log_file.write(f"Patient {patient_id}:\n")
    log_file.write(f"  Min Intensity: {min_intensity:.2f} HU\n")
    log_file.write(f"  Max Intensity: {max_intensity:.2f} HU\n")
    log_file.write(f"  Mean Intensity: {mean_intensity:.2f} HU\n")
    log_file.write(f"  Median Intensity: {median_intensity:.2f} HU\n")
    log_file.write(f"  Standard Deviation: {std_intensity:.2f} HU\n")
    log_file.write(f"  Total Voxels: {total_voxels}\n")
    log_file.write(f"  Voxels below {lower_threshold} HU: {voxels_below_threshold} ({percent_below:.4f}%)\n")
    log_file.write(f"  Voxels above {upper_threshold} HU: {voxels_above_threshold} ({percent_above:.4f}%)\n")
    log_file.write("-" * 60 + "\n\n")

def main():
    """
    Main function to process the SegTHOR dataset and generate intensity statistics.
    """
    project_root = get_project_root()
    base_path = project_root / 'data' / 'segthor_train' / 'train'
    statistics = get_intensity_statistics(base_path)
    
    headers = ["Patient ID", "Min (HU)", "Max (HU)", "Mean (HU)", "Median (HU)", "Std Dev (HU)"]
    table = tabulate(statistics, headers=headers, tablefmt="grid", floatfmt=".2f")
    
    script_dir = Path(__file__).resolve().parent
    log_file_path = script_dir / 'insights' / 'intensity_log.txt'
    
    # Ensure the directory exists
    ensure_directory_exists(log_file_path.parent)
    
    with open(log_file_path, 'a') as log_file:
        log_file.write("SegTHOR Dataset Intensity Statistics Summary\n")
        log_file.write(table)
        log_file.write("\n")
    
    print(f"Processing complete. Results written to {log_file_path}")

if __name__ == "__main__":
    main()