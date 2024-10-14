#!/usr/bin/env python3.7

# MIT License
# (License text omitted for brevity)

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert res.min() == 0, res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)

def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min() <= 31743, ct.min()
    assert 0.896 <= dx <= 1.37, dx
    assert dx == dy
    assert 2 <= dz <= 3.7, dz
    assert x == y == 512, (x, y)
    assert 135 <= z <= 284, z
    return True

resize_ = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int]) -> tuple[float, float, float]:
    ct_path = source_path / 'test' / f"{id_}.nii.gz"
    nib_obj = nib.load(str(ct_path))
    ct = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, x, y, z, dx, dy, dz)

    norm_ct = norm_arr(ct)

    for idz in range(z):
        img_slice = resize_(norm_ct[:, :, idz], shape).astype(np.uint8)
        filename = f"{id_}_{idz:04d}.png"

        save_path = dest_path / "img"
        save_path.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(save_path / filename), img_slice)

    return dx, dy, dz

def get_test_ids(src_path: Path) -> list[str]:
    test_dir = src_path / 'test'
    test_ids = sorted(map_(lambda p: p.name.replace('.nii.gz', ''), test_dir.glob('*.nii.gz')))
    print(f"Found {len(test_ids)} test ids")
    print(test_ids[:10])
    return test_ids

def main(args: argparse.Namespace):
    src_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)

    assert src_path.exists(), f"Source path {src_path} does not exist"
    assert not dest_path.exists(), f"Destination path {dest_path} already exists"

    test_ids = get_test_ids(src_path)
    dest_mode = dest_path / "test"
    print(f"Slicing {len(test_ids)} patients to {dest_mode}")

    pfun = partial(slice_patient,
                   dest_path=dest_mode,
                   source_path=src_path,
                   shape=tuple(args.shape))
    iterator = tqdm_(test_ids)
    if args.process == 1:
        resolutions = list(map(pfun, iterator))
    else:
        with Pool(args.process) as pool:
            resolutions = pool.map(pfun, iterator)

    resolution_dict = dict(zip(test_ids, resolutions))

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionary to {f}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters for test data')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="Number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args

if __name__ == "__main__":
    main(get_args())
