import wandb
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from operator import itemgetter
from dataset import SliceDataset
from ENet import ENet
from losses import CrossEntropy
from ShallowNet import shallowCNN
from utils import Dcm, class2one_hot, dice_coef, probs2class, probs2one_hot, save_images, tqdm_
from warmup_cosine_annealing_lr import WarmupCosineAnnealingLR
import os
from shutil import rmtree, copytree

datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["TOY2"] = {"K": 2, "net": shallowCNN, "B": 2}
datasets_params["SEGTHOR"] = {"K": 5, "net": ENet, "B": 8}

def setup(args, lr) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    load_dotenv()
    gpu: bool = (
        args.gpu and torch.backends.mps.is_available() or torch.cuda.is_available()
    )
    if gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]["K"]
    net = datasets_params[args.dataset]["net"](1, K)
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.epochs)

    B: int = datasets_params[args.dataset]["B"]
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,
            lambda nd: torch.tensor(nd, dtype=torch.float32),
        ]
    )

    gt_transform = transforms.Compose(
        [
            lambda img: np.array(img)[...],
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],
            lambda t: class2one_hot(t, K=K),
            itemgetter(0),
        ]
    )

    train_set = SliceDataset(
        "train",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    train_loader = DataLoader(
        train_set, batch_size=B, num_workers=args.num_workers, shuffle=True
    )

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set, batch_size=B, num_workers=args.num_workers, shuffle=False
    )

    args.dest.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=os.getenv('WANDB_PROJECT', args.wandb_project),
        entity=os.getenv('WANDB_ENTITY', args.wandb_entity),
        config={
            "learning_rate": lr,
            "epochs": args.epochs,
            "batch_size": B,
            "dataset": args.dataset,
            "mode": args.mode,
            "optimizer": "Adam",
            "model": net.__class__.__name__,
            "warmup_epochs": args.warmup_epochs,
        },
    )

    return (net, optimizer, scheduler, device, train_loader, val_loader, K)

def runTraining(args):
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args, args.lr)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))
    elif args.mode in ["partial"] and args.dataset in ["SEGTHOR", "SEGTHOR_STUDENTS"]:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])
    else:
        raise ValueError(args.mode, args.dataset)

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt:
                        opt.zero_grad()

                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)

                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j : j + B, :] = dice_coef(pred_seg, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()

                    if opt:
                        loss.backward()
                        opt.step()

                    if m == "val":
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(
                                predicted_class * mult,
                                data["stems"],
                                args.dest / f"iter{e:03d}" / m,
                            )

                    j += B
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                    }
                    if K > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, K)
                        }
                    tq_iter.set_postfix(postfix_dict)

                    if m == "train":
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/dice": log_dice[e, :j, 1:].mean().item(),
                                "epoch": e,
                            }
                        )
                    else:
                        wandb.log(
                            {
                                "val/loss": loss.item(),
                                "val/dice": log_dice[e, :j, 1:].mean().item(),
                                "epoch": e,
                            }
                        )

        scheduler.step()

        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        wandb.log(
            {
                "train/epoch_loss": log_loss_tra[e].mean().item(),
                "train/epoch_dice": log_dice_tra[e, :, 1:].mean().item(),
                "val/epoch_loss": log_loss_val[e].mean().item(),
                "val/epoch_dice": log_dice_val[e, :, 1:].mean().item(),
                "epoch": e,
            }
        )

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(
                f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            )
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", "w") as f:
                f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

            wandb.log({"best_dice": best_dice, "best_epoch": e})

    wandb.finish()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--dataset", default="TOY2", choices=datasets_params.keys())
    parser.add_argument("--mode", default="full", choices=["partial", "full"])
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory to save the results (predictions and weights).",
    )

    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )

    args = parser.parse_args()

    args.wandb_project = os.getenv('WANDB_PROJECT', 'ai4mi')
    args.wandb_entity = os.getenv('WANDB_ENTITY', )

    pprint(args)

    runTraining(args)

if __name__ == "__main__":
    main()