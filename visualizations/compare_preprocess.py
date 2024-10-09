import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def get_project_root() -> Path:
    """Find the project root directory."""
    current_path = Path(__file__).resolve().parent
    while not (current_path / ".git").exists():
        parent = current_path.parent
        if parent == current_path:
            raise RuntimeError(
                "Project root not found. Ensure you're running this from within the cloned repository."
            )
        current_path = parent
    return current_path


def load_nifti(file_path):
    """Load a NIfTI file and return its data."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()


def plot_slice(ax, slice_data, title):
    """Plot a single slice with a colorbar."""
    im = ax.imshow(slice_data, cmap="gray")
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")
    return im


def compare_scans(patient_id):
    """Compare original and preprocessed CT scans for a given patient."""
    project_root = get_project_root()
    data_dir = (
        project_root / "data" / "segthor_train" / "train" / f"Patient_{patient_id:02d}"
    )

    original_path = data_dir / f"Patient_{patient_id:02d}.nii.gz"
    preprocessed_path = data_dir / f"Patient_{patient_id:02d}_enhanced.nii.gz"

    original_data = load_nifti(original_path)
    preprocessed_data = load_nifti(preprocessed_path)

    # Select middle slices for axial, sagittal, and coronal views
    axial_slice = original_data.shape[2] // 2
    sagittal_slice = original_data.shape[0] // 2
    coronal_slice = original_data.shape[1] // 2

    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    fig.suptitle(
        f"Patient {patient_id:02d} - Original vs Preprocessed CT Scans",
        fontsize=20,
        y=0.95,
    )

    # Axial view
    im1 = plot_slice(axes[0, 0], original_data[:, :, axial_slice], "Original - Axial")
    im2 = plot_slice(
        axes[0, 1], preprocessed_data[:, :, axial_slice], "Preprocessed - Axial"
    )

    # Sagittal view (flipped vertically)
    im3 = plot_slice(
        axes[1, 0],
        np.flipud(original_data[sagittal_slice, :, :].T),
        "Original - Sagittal",
    )
    im4 = plot_slice(
        axes[1, 1],
        np.flipud(preprocessed_data[sagittal_slice, :, :].T),
        "Preprocessed - Sagittal",
    )

    # Coronal view (flipped vertically)
    im5 = plot_slice(
        axes[2, 0],
        np.flipud(original_data[:, coronal_slice, :].T),
        "Original - Coronal",
    )
    im6 = plot_slice(
        axes[2, 1],
        np.flipud(preprocessed_data[:, coronal_slice, :].T),
        "Preprocessed - Coronal",
    )

    # Add colorbars
    for im, ax in zip([im1, im2, im3, im4, im5, im6], axes.flat):
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1)

    # Create 'images' directory if it doesn't exist
    save_dir = project_root / "visualisations" / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(
        save_dir / f"patient_{patient_id:02d}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_preprocess.py <patient_number>")
        sys.exit(1)

    try:
        patient_id = int(sys.argv[1])
        if patient_id < 1 or patient_id > 40:
            raise ValueError("Patient number must be between 1 and 40")
    except ValueError:
        print("Invalid patient number. Please provide a number between 1 and 40.")
        sys.exit(1)

    compare_scans(patient_id)
    print(
        f"Comparison image for Patient {patient_id:02d} has been saved in the 'visualisations/images' directory."
    )


if __name__ == "__main__":
    main()
