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

import argparse
import warnings
import os
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

# Import our own models (placeholders now)
import torch.nn as nn

class SAM2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SAM2, self).__init__()
        # Placeholder implementation

    def forward(self, x):
        return x  # Placeholder

    def init_weights(self):
        pass  # Placeholder

class VMUNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VMUNet, self).__init__()
        # Placeholder implementation

    def forward(self, x):
        return x  # Placeholder

    def init_weights(self):
        pass  # Placeholder

# Imports we have added
import logging
import datetime
import os
from collections import defaultdict
import nibabel as nib
from typing import Dict, List, Tuple

# For tuning
import torch.optim.lr_scheduler as lr_scheduler

# Post process the NIfTI files
from post_process.post_process import post_process_nifti_files

# Calculating Metrics
from run_metric_calculation import calculate_metrics
from metrics.png_to_nifti import reconstruct_nifti

# Test Set inference
from inference.test_set_inference import run_test_inference

# Creating Zip-file of all results
import zipfile

# Load environment variables
from dotenv import load_dotenv


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}

# Create a dictionary mapping model names to classes
model_dict = {
    'ENet': ENet,
    'SAM2': SAM2,
    'VMUNet': VMUNet
}


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, Any]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.backends.mps.is_available() or torch.cuda.is_available()
    if gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']

    # Get the model class based on args.model_name
    if args.model_name in model_dict:
        model_class = model_dict[args.model_name]
    else:
        raise ValueError(f"Unknown model name {args.model_name}")

    # for VMamba, check if weights are in the right path
    if args.model == "VMUNet":
        # Check if the pre-trained weights file exists
        weight_path = './pre_trained_weights/vmamba_small_e238_ema.pth'
        if not os.path.exists(weight_path):
            raise KeyError(f"Pre-trained weights file not found: {weight_path}.\nPlease download the weights, see: https://github.com/JCruan519/VM-UNet/blob/main/pre_trained_weights/README.md")

    # Set the number of kernels and learning rate based on the tuning flag
    if args.tuning:
        kernels = 32
        lr = 0.001    
        weight_decay = 1e-4 
    else:
        kernels = 16
        lr = 0.0005
        weight_decay = 0.0

    # Initialize the network
    # net = datasets_params[args.dataset]['net'](1, K, kernels=kernels)
    net = model_class(1, K, kernels=kernels)
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    # Set up the scheduler if tuning
    if args.tuning:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None


    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    
    # Change the root_dir based on the --preprocess flag
    if args.preprocess:
        root_dir = Path("data") / "SEGTHOR_preprocessed"
    else:
        root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug,
                             augment=args.augmentation)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", args.wandb_project),
        entity=os.getenv("WANDB_ENTITY", args.wandb_entity),
        name=f"{args.dataset}_{args.mode}_lr{lr}_epochs{args.epochs}",
        config={
            "learning_rate": lr,
            "epochs": args.epochs,
            "batch_size": B,
            "dataset": args.dataset,
            "mode": args.mode,
            "optimizer": "Adam",
            "model": net.__class__.__name__,
        },
    )

    return (net, optimizer, device, train_loader, val_loader, K, root_dir, scheduler)


def runTraining(args, current_time, log_file):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K, root_dir, scheduler = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    # Initialize lists to store metrics over epochs
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    # For per-class Dice scores
    per_class_train_dices = {k: [] for k in range(1, K)}  # Exclude background class 0
    per_class_val_dices = {k: [] for k in range(1, K)}    # Exclude background class 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:

                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

                    # Log to wandb
                    if m == "train":
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/dice": log_dice[e, :j, 1:].mean().item(),
                                "epoch": e,
                            }
                        )
                    else:  # validation
                        wandb.log(
                            {
                                "val/loss": loss.item(),
                                "val/dice": log_dice[e, :j, 1:].mean().item(),
                                "epoch": e,
                            }
                        )

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

            # Log best model to wandb
            wandb.log({"best_dice": best_dice, "best_epoch": e})

        # After both training and validation phases are complete for the epoch

        # Calculate mean metrics
        mean_train_loss = log_loss_tra[e, :].mean().item()
        mean_val_loss = log_loss_val[e, :].mean().item()
        mean_train_dice = log_dice_tra[e, :, 1:].mean().item()  # Exclude background class
        mean_val_dice = log_dice_val[e, :, 1:].mean().item()

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step(mean_val_loss)

        # Append mean metrics to lists
        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        train_dices.append(mean_train_dice)
        val_dices.append(mean_val_dice)

        # Append per-class Dice scores
        for k in range(1, K):  # Exclude background class
            class_train_dice = log_dice_tra[e, :, k].mean().item()
            class_val_dice = log_dice_val[e, :, k].mean().item()

            per_class_train_dices[k].append(class_train_dice)
            per_class_val_dices[k].append(class_val_dice)

        # Log metrics
        logging.info(f'Epoch {e}')
        logging.info(f'Training Loss: {mean_train_loss:.4f}')
        logging.info(f'Validation Loss: {mean_val_loss:.4f}')
        logging.info(f'Training Dice Coefficient (mean over classes): {mean_train_dice:.4f}')
        logging.info(f'Validation Dice Coefficient (mean over classes): {mean_val_dice:.4f}')

        # Log per-class Dice scores
        for k in range(1, K):  # Exclude background class
            class_train_dice = log_dice_tra[e, :, k].mean().item()
            class_val_dice = log_dice_val[e, :, k].mean().item()

            logging.info(f'Class {k} Training Dice: {class_train_dice:.4f}')
            logging.info(f'Class {k} Validation Dice: {class_val_dice:.4f}')

        # Log epoch metrics to wandb
        wandb.log(
            {
                "train/epoch_loss": log_loss_tra[e].mean().item(),
                "train/epoch_dice": log_dice_tra[e, :, 1:].mean().item(),
                "val/epoch_loss": log_loss_val[e].mean().item(),
                "val/epoch_dice": log_dice_val[e, :, 1:].mean().item(),
                "epoch": e,
            }
        )

    # Post-training reconstruction of the raw output from PNG slices to NIfTI files
    print(">>> Training complete. Reconstructing NIfTI files from the best epoch predictions.")

    best_folder = args.dest / "best_epoch"
    png_folder = best_folder / 'val'
    gt_folder = 'data/segthor_train/train'  # Adjust the path if necessary
    nifti_output_folder = best_folder / 'nifti'

    reconstruct_nifti(str(png_folder), gt_folder, str(nifti_output_folder))

    # Post-processing of the raw output to "better" (read: smoother) predictions
    print(">>> Starting post-processing of NIfTI files.")

    post_process_input_folder = str(nifti_output_folder)
    post_process_output_folder = str(best_folder / 'nifti_post_processed')
    num_classes = K  # Number of classes including background

    post_process_nifti_files(post_process_input_folder, post_process_output_folder, num_classes)
    print(">>> Post-processing completed successfully.")

    # Metric Calculation
    # Prepare common parameters for metric calculation
    class_labels = list(range(K))  # [0, 1, 2, 3, 4] for SEGTHOR
    class_names = {0: 'Background', 1: 'Esophagus', 2: 'Heart', 3: 'Trachea', 4: 'Aorta'}

    # Determine ground truth filename based on whether --preprocess is set
    if args.preprocess:
        gt_filename = 'GT_enhanced.nii.gz'
    else:
        gt_filename = 'GT.nii.gz'

    # Metric Calculation for Raw Predictions
    print(">>> Starting metrics calculation for raw predictions.")
    calculate_metrics(
        pred_nifti_dir=str(nifti_output_folder),
        pred_png_dir=str(best_folder / 'val'),
        gt_nifti_dir=str(gt_folder),
        gt_png_dir=str(root_dir / 'val' / 'gt'),
        class_labels=class_labels,
        class_names=class_names,
        description='Raw Predictions',
        gt_filename=gt_filename
    )

    # Metric Calculation for Post-Processed Predictions
    print(">>> Starting metrics calculation for post-processed predictions.")
    calculate_metrics(
        pred_nifti_dir=post_process_output_folder,
        pred_png_dir=str(best_folder / 'val'),  # PNGs are the same
        gt_nifti_dir=str(gt_folder),
        gt_png_dir=str(root_dir / 'val' / 'gt'),
        class_labels=class_labels,
        class_names=class_names,
        description='Post-Processed Predictions',
        gt_filename=gt_filename
    )

    # Log the collected metrics over epochs
    logging.info('Training and Validation Losses over Epochs:')
    logging.info(f'train_loss = {train_losses}')
    logging.info(f'val_loss = {val_losses}')

    logging.info('Training and Validation Dice Coefficients over Epochs:')
    logging.info(f'train_dice = {train_dices}')
    logging.info(f'val_dice = {val_dices}')

    # Log per-class Dice scores
    for k in range(1, K):
        logging.info(f'Class {k} Training Dice over Epochs: {per_class_train_dices[k]}')
        logging.info(f'Class {k} Validation Dice over Epochs: {per_class_val_dices[k]}')

    # Test Set Inference
    print(">>> Starting inference on test set.")

    # Load the best model (if not already loaded)
    net = torch.load(args.dest / "bestmodel.pkl", map_location=device)
    net.to(device)
    net.eval()

    # Perform inference on the test set
    run_test_inference(args, net, device, K)

    # Zip summary file creation
    print(">>> Creating summary zip file.")

    # Determine statuses for preprocess, augmentation, and tuning
    pre_status = 'on' if args.preprocess else 'off'
    aug_status = 'on' if args.augmentation else 'off'
    tuning_status = 'on' if args.tuning else 'off'

    # Construct the zip file name
    zip_filename = f"experiment_{args.model_name}_pre-{pre_status}_aug-{aug_status}_tuning-{tuning_status}_{current_time}.zip"

    # Place the zip file in the root directory
    zip_path = Path('.') / zip_filename

    # Define the base directory (project root) to calculate relative paths
    base_dir = Path('.').resolve()

    # Ensure paths are absolute
    best_epoch_folder = (args.dest / 'best_epoch').resolve()
    best_epoch_txt = (args.dest / 'best_epoch.txt').resolve()
    bestmodel_pkl = (args.dest / 'bestmodel.pkl').resolve()
    bestweights_pt = (args.dest / 'bestweights.pt').resolve()
    log_file = log_file.resolve()

    # Initialize the zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the best_epoch folder and its contents
        for root, dirs, files in os.walk(best_epoch_folder):
            for file in files:
                file_path = (Path(root) / file).resolve()
                # Compute the relative path within the zip file
                arcname = file_path.relative_to(base_dir)
                # Write the file to the zip archive
                zipf.write(file_path, arcname)

        # Add the best_epoch.txt file
        if best_epoch_txt.exists():
            arcname = best_epoch_txt.relative_to(base_dir)
            zipf.write(best_epoch_txt, arcname)

        # Add the bestmodel.pkl file
        if bestmodel_pkl.exists():
            arcname = bestmodel_pkl.relative_to(base_dir)
            zipf.write(bestmodel_pkl, arcname)

        # Add the bestweights.pt file
        if bestweights_pt.exists():
            arcname = bestweights_pt.relative_to(base_dir)
            zipf.write(bestweights_pt, arcname)

        # Add the logging file
        if log_file.exists():
            arcname = log_file.relative_to(base_dir)
            zipf.write(log_file, arcname)

    print(f">>> Summary zip file created at {zip_path}")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='SEGTHOR', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    
    # Added by to configure model settings
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['ENet', 'SAM2', 'VMUNet'],
                        help="Name of the model to use. Choices are 'ENet', 'SAM2', or 'VMUNet'.")
    parser.add_argument('--preprocess', action='store_true',
                        help="If set, the program will use the preprocessed data directory.")
    parser.add_argument('--augmentation', action='store_true',
                        help="If set, the program will include augmented data in training.")
    parser.add_argument('--tuning', action='store_true',
                        help="If set, the program will perform tuning. Can only be set when --model_name='ENet'.")

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ai4mi",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="mae-testing",
        help="Weights & Biases entity (username or team name)",
    )

    args = parser.parse_args()

    # Enforce that --augmentation requires --preprocess
    if args.augmentation and not args.preprocess:
        parser.error("--augmentation requires --preprocess.")

    # Enforce that --tuning can only be set when --model_name='ENet'
    if args.tuning and args.model_name != 'ENet':
        parser.error("--tuning can only be set when --model_name='ENet'.")

    # Convert args.dest to an absolute path
    args.dest = args.dest.resolve()

    # Setup logging
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logging')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'training_{current_time}.txt'

    # Configure logging
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    # Log initial arguments
    logging.info('Starting training with arguments:')
    logging.info(args)

    pprint(args)

    runTraining(args, current_time, log_file)


if __name__ == '__main__':
    main()
