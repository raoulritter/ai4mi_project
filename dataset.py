#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, subset) -> list[tuple[Path, Path]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    # our new implementation for sorting images ours above was first
    # Sort images and labels by patient ID and slice index
    images = sorted(images, key=lambda x: (x.stem.split('_')[1], int(x.stem.split('_')[2])))
    full_labels = sorted(full_labels, key=lambda x: (x.stem.split('_')[1], int(x.stem.split('_')[2])))

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    # Original getitem

    # def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
    #     img_path, gt_path = self.files[index]

    #     img: Tensor = self.img_transform(Image.open(img_path))
    #     gt: Tensor = self.gt_transform(Image.open(gt_path))

    #     _, W, H = img.shape
    #     K, _, _ = gt.shape
    #     assert gt.shape == (K, W, H)

    #     # Ours added patient ID (also in dict below)
    #     stem = img_path.stem  # 'Patient_03_0000'
    #     patient_id = stem.split('_')[1]  # '03

    #     return {"images": img,
    #             "gts": gt,
    #             "stems": img_path.stem,
    #             "patient_id": patient_id}

    # Ours getitem

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        # Extract patient ID and slice index
        stem = img_path.stem  # 'Patient_03_0000'
        parts = stem.split('_')
        patient_id = f"{parts[0]}_{parts[1]}"  # 'Patient_03'
        slice_idx = int(parts[2])  # '0000'

        return {
            "images": img,
            "gts": gt,
            "stems": stem,
            "patient_id": patient_id,
            "slice_idx": slice_idx
        }
