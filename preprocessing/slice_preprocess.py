#!/usr/bin/env python3

import argparse
import random
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Iterable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from tqdm import tqdm

def norm_arr(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min  # In case of constant image
    img = img * 255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

resize_ = partial(resize, mode='constant', preserve_range=True, anti_aliasing=False)

def map_(fn: Callable, iterable: Iterable) -> List:
    return list(map(fn, iterable))

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    if not test_mode:
        ct_path: Path = id_path / f"{id_}_enhanced.nii.gz"
        gt_path: Path = id_path / "GT_enhanced.nii.gz"
    else:
        ct_path: Path = source_path / "test" / f"{id_}_enhanced.nii.gz"
        gt_path: Path = None  # Test mode doesn't have GT

    # Load CT scan
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    # Load GT if available
    if not test_mode and gt_path.exists():
        gt_nib = nib.load(str(gt_path))
        gt = np.asarray(gt_nib.dataobj)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    # Normalize CT scan
    norm_ct: np.ndarray = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        gt_slice *= 63  # Map labels to [0, 252] with steps of 63

        arrays = [img_slice, gt_slice]
        subfolders = ['img', 'gt']

        for save_subfolder, data in zip(subfolders, arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = dest_path / save_subfolder
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz

def get_splits(src_path: Path, retains: int, fold: int) -> tuple[List[str], List[str], List[str]]:
    ids: List[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Found {len(ids)} patient IDs in the training set.")
    assert len(ids) > retains, "Not enough patients to retain for validation."

    random.shuffle(ids)
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: List[str] = ids[validation_slice]
    training_ids: List[str] = [e for e in ids if e not in validation_ids]

    test_ids: List[str] = sorted(map_(lambda p: p.name, (src_path / 'test').glob('*')))
    print(f"Found {len(test_ids)} patient IDs in the test set.")

    return training_ids, validation_ids, test_ids

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    assert src_path.exists(), f"Source directory {src_path} does not exist."
    if dest_path.exists():
        print(f"Destination path {dest_path} already exists. Please remove it before running the script.")
        return

    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} patients to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=(mode == 'test'))
        iterator = tqdm_(split_ids)
        if args.process == 1:
            resolutions = list(map(pfun, iterator))
        elif args.process == -1:
            with Pool() as pool:
                resolutions = pool.map(pfun, iterator)
        else:
            with Pool(args.process) as pool:
                resolutions = pool.map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    # Save spacing information
    with open(dest_path / "spacing.pkl", 'wb') as f:
        import pickle
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionary to {f}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True, help="Path to the source directory (e.g., ../data/segthor_train)")
    parser.add_argument('--dest_dir', type=str, required=True, help="Path to the destination directory for sliced images")
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256], help="Output image shape (height width)")
    parser.add_argument('--retains', type=int, default=10, help="Number of patients to retain for validation")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for splitting")
    parser.add_argument('--fold', type=int, default=0, help="Fold number for cross-validation")
    parser.add_argument('--process', '-p', type=int, default=1, help="Number of processes to use (-1 for all available)")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args

if __name__ == "__main__":
    main(get_args())
