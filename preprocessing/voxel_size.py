import os
import nibabel as nib
from tabulate import tabulate
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

def ensure_directory_exists(path: Path):
    """
    Ensure that the directory exists, creating it if necessary.

    Args:
        path (Path): The directory path to check/create.
    """
    path.mkdir(parents=True, exist_ok=True)

def get_voxel_sizes(base_path: Path) -> list:
    """
    Calculate voxel sizes for CT scans in the SegTHOR dataset.
    
    Args:
        base_path (Path): Path to the directory containing patient subdirectories.
    
    Returns:
        list: A list of voxel sizes for each patient.
    """
    voxel_sizes = []
    total_patients = 40
    script_dir = Path(__file__).resolve().parent
    log_file_path = script_dir / 'insights' / 'voxel_size_log.txt'
    
    ensure_directory_exists(log_file_path.parent)
    
    with open(log_file_path, 'w') as log_file:
        for i in tqdm(range(1, total_patients + 1), total=total_patients, desc="Processing patients"):
            patient_id = f"Patient_{i:02d}"
            file_path = base_path / patient_id / f"{patient_id}.nii.gz"
            
            try:
                nifti_img = nib.load(file_path)
                header = nifti_img.header
                voxel_size = header.get_zooms()
                voxel_sizes.append([patient_id, *voxel_size])
                
                log_file.write(f"Patient {patient_id}:\n")
                log_file.write(f"  Voxel Size (X, Y, Z): {voxel_size[0]:.3f} mm, {voxel_size[1]:.3f} mm, {voxel_size[2]:.3f} mm\n")
                log_file.write("-" * 60 + "\n\n")
                
            except FileNotFoundError:
                log_file.write(f"Warning: File not found for {patient_id}\n\n")
            except Exception as e:
                log_file.write(f"Error processing {patient_id}: {str(e)}\n\n")
    
    return voxel_sizes

def main():
    """
    Main function to process the SegTHOR dataset and generate voxel size statistics.
    """
    project_root = get_project_root()
    base_path = project_root / 'data' / 'segthor_train' / 'train'
    voxel_sizes = get_voxel_sizes(base_path)
    
    # Prepare the table
    headers = ["Patient ID", "X (mm)", "Y (mm)", "Z (mm)"]
    table = tabulate(voxel_sizes, headers=headers, tablefmt="grid", floatfmt=".3f")
    
    script_dir = Path(__file__).resolve().parent
    log_file_path = script_dir / 'insights' / 'voxel_size_log.txt'
    
    ensure_directory_exists(log_file_path.parent)
    
    with open(log_file_path, 'a') as log_file:
        log_file.write("SegTHOR Dataset Voxel Sizes Summary\n")
        log_file.write(table)
        log_file.write("\n")
    
    print(f"Processing complete. Results written to {log_file_path}")

if __name__ == "__main__":
    main()