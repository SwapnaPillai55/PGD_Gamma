# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from glob import glob
import warnings
import time
import random
import numpy as np
import shutil
import torchvision.utils as vutils

from utils_image_VIRNet import batch_PSNR, batch_SSIM
from tensorboardX import SummaryWriter
from math import ceil
from loss import loss_fn
from networks import weight_init_kaiming
from datasets import DenoisingDatasets_Gamma
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
from pathlib import Path
from options import set_opts

import evidential_deep_learning as edl
from Networks_Sid.models.convnext_isotropic_x2_f8_3c import convnext_isotropic_base


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(SEED)
warnings.simplefilter("ignore", Warning, lineno=0)

criterion2 = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")


class SimplifiedEvidentialGammaLoss(nn.Module):
    def __init__(self, regularization_coefficient):
        super(SimplifiedEvidentialGammaLoss, self).__init__()
        self.lambda_reg = regularization_coefficient

    def forward(self, alpha_pred, beta_pred, y_true):
        alpha_pred = F.softplus(alpha_pred) + 1e-6
        beta_pred = F.softplus(beta_pred) + 1e-6
        y_true = y_true + 1e-6

        nll_loss = -(
            alpha_pred * torch.log(beta_pred)
            - torch.lgamma(alpha_pred)
            + (alpha_pred - 1) * torch.log(y_true)
            - beta_pred * y_true
        )

        total_loss = nll_loss
        return torch.mean(total_loss)


def EvidentialRegressionLoss(true, pred):
    return edl.losses.continuous.EvidentialRegression(true, pred, coeff=1e-2)


args = set_opts()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if isinstance(args.gpu_id, int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in list(args.gpu_id))

_C = 3
_lr_min = 1e-6
_modes = ["train", "test_bsd50"]


def train_model(net, datasets, optimizer, lr_scheduler, criterion):
    clip_grad_next = args.clip_grad_D

    batch_size = {
        "train": args.batch_size,
        "test_bsd50": 1,
    }

    # deterministic DataLoader generator
    g = torch.Generator()
    g.manual_seed(SEED)

    data_loader = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=batch_size[phase],
            shuffle=True if phase == "train" else False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        for phase in datasets.keys()
    }

    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {
        phase: ceil(num_data[phase] / batch_size[phase])
        for phase in datasets.keys()
    }

    writer = SummaryWriter(args.log_dir)

    if args.resume:
        step = args.step
        step_img = args.step_img
    else:
        step = 0
        step_img = {x: 0 for x in _modes}

    param_next = [x for name, x in net.named_parameters()]
    best_psnr = 0.0
    best_ssim = 0.0

    # local RNGs for reproducible Gamma noise
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    for epoch in range(args.epoch_start, args.epochs):
        loss_per_epoch = {x: 0 for x in ["Loss", "lh", "KLG", "KLIG"]}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_next = 0
        tic = time.time()

        net.train()
        lr = optimizer.param_groups[0]["lr"]

        if lr < _lr_min:
            sys.exit("Reached the minimal learning rate")

        phase = "train"

        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda(non_blocking=True) for x in data]

            optimizer.zero_grad()

            # deterministic multiplicative Gamma noise
            L = py_rng.randint(5, 20)

            noise = np_rng.gamma(
                shape=L,
                scale=1.0 / L,
                size=tuple(im_gt.shape),
            ).astype(np.float32)

            im_noisy = (im_gt.detach().cpu().numpy() * noise).astype(np.float32)
            im_noisy = torch.from_numpy(im_noisy).to(im_gt.device)

            alpha_pred, beta_pred = net(im_noisy)

            alpha_pred = F.softplus(alpha_pred) + 1e-6
            beta_pred = F.softplus(beta_pred) + 1e-6
            im_gt = im_gt + 1e-6

            im_denoise = alpha_pred / beta_pred

            criterion = SimplifiedEvidentialGammaLoss(
                regularization_coefficient=0.01
            )

            loss = criterion(alpha_pred, beta_pred, im_gt)
            loss.backward()

            total_norm_next = nn.utils.clip_grad_norm_(
                param_next,
                clip_grad_next,
            )

            grad_norm_next = (
                grad_norm_next * (ii / (ii + 1))
                + total_norm_next / (ii + 1)
            )

            optimizer.step()

            loss_per_epoch["Loss"] += loss.item() / num_iter_epoch[phase]

            mse = F.mse_loss(im_denoise, im_gt)
            mse_per_epoch[phase] += mse.item()

            if (ii + 1) % args.print_freq == 0:
                log_str = (
                    "[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, "
                    "mse={:.2e}, lr={:.1e}"
                )
                print(
                    log_str.format(
                        epoch + 1,
                        args.epochs,
                        phase,
                        ii + 1,
                        num_iter_epoch[phase],
                        mse,
                        lr,
                    )
                )

                writer.add_scalar("Train Loss Iter", loss.item(), step)
                writer.add_scalar("Train MSE Iter", mse.item(), step)
                step += 1

        mse_per_epoch[phase] /= (ii + 1)

        log_str = "{:s}: Loss={:+.2e}, mse={:.3e}"
        print(log_str.format(phase, loss_per_epoch["Loss"], mse_per_epoch[phase]))

        writer.add_scalar("Loss_epoch", loss_per_epoch["Loss"], epoch)
        print("-" * 150)

        # =========================================================
        # Test
        # =========================================================
        net.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}

        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                im_noisy, im_gt = [x.cuda(non_blocking=True) for x in data]

                with torch.no_grad():
                    L = 10

                    noise = np_rng.gamma(
                        shape=L,
                        scale=1.0 / L,
                        size=tuple(im_gt.shape),
                    ).astype(np.float32)

                    im_noisy = (
                        im_gt.detach().cpu().numpy() * noise
                    ).astype(np.float32)

                    im_noisy = torch.from_numpy(im_noisy).to(im_gt.device)

                    alpha_pred, beta_pred = net(im_noisy)

                    alpha_pred = F.softplus(alpha_pred) + 1e-6
                    beta_pred = F.softplus(beta_pred) + 1e-6
                    im_gt = im_gt + 1e-6

                    im_denoise = alpha_pred / beta_pred

                im_denoise = torch.clamp(im_denoise, 0.0, 1.0)
                im_gt = torch.clamp(im_gt, 0.0, 1.0)

                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)

                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter

                if (ii + 1) % 20 == 0:
                    log_str = (
                        "[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, "
                        "psnr={:4.2f}, ssim={:5.4f}"
                    )
                    print(
                        log_str.format(
                            epoch + 1,
                            args.epochs,
                            phase,
                            ii + 1,
                            num_iter_epoch[phase],
                            psnr_iter,
                            ssim_iter,
                        )
                    )

            psnr_per_epoch[phase] /= (ii + 1)
            ssim_per_epoch[phase] /= (ii + 1)

            log_str = "{:s}: PSNR={:4.2f}, SSIM={:5.4f}"
            print(log_str.format(phase, psnr_per_epoch[phase], ssim_per_epoch[phase]))

        if (
            psnr_per_epoch["test_bsd50"] > best_psnr
            and ssim_per_epoch["test_bsd50"] > best_ssim
        ):
            print("\n ===== found better metrics =====")

            best_psnr = psnr_per_epoch["test_bsd50"]
            best_ssim = ssim_per_epoch["test_bsd50"]

            save_path_model = os.path.join(args.model_dir, f"model_{epoch + 1}")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "seed": SEED,
                    "step": step + 1,
                    "step_img": {x: step_img[x] for x in _modes},
                    "grad_norm_next": clip_grad_next,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "rng_python_state": random.getstate(),
                    "rng_numpy_state": np.random.get_state(),
                    "rng_torch_state": torch.get_rng_state(),
                    "rng_cuda_state": torch.cuda.get_rng_state_all(),
                },
                save_path_model,
            )

            save_path_model_state = os.path.join(
                args.model_dir,
                f"model_state_{epoch + 1}",
            )

            torch.save(net.state_dict(), save_path_model_state)

            print(
                f"Model saved at epoch {epoch + 1} with "
                f"PSNR: {psnr_per_epoch['test_bsd50']}, "
                f"SSIM: {ssim_per_epoch['test_bsd50']}"
            )

        writer.add_scalars("PSNR_epoch_test", psnr_per_epoch, epoch)
        writer.add_scalars("SSIM_epoch_test", ssim_per_epoch, epoch)

        toc = time.time()
        print(f"This epoch took time {toc - tic:.2f}")

        lr_scheduler.step()

    writer.close()
    print("Reached the maximum epochs! Training complete.")


def main():
    net = convnext_isotropic_base()
    net = nn.DataParallel(net).cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    args.milestones = [10, 25, 35, 45]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        args.milestones,
        args.gamma,
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)

            args.epoch_start = checkpoint["epoch"]
            args.step = checkpoint["step"]
            args.step_img = checkpoint["step_img"]

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            net.load_state_dict(checkpoint["model_state_dict"])

            if "rng_python_state" in checkpoint:
                random.setstate(checkpoint["rng_python_state"])
            if "rng_numpy_state" in checkpoint:
                np.random.set_state(checkpoint["rng_numpy_state"])
            if "rng_torch_state" in checkpoint:
                torch.set_rng_state(checkpoint["rng_torch_state"])
            if "rng_cuda_state" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["rng_cuda_state"])

            print(f'=> Loaded checkpoint {args.resume} epoch {checkpoint["epoch"]}')
        else:
            sys.exit("Please provide the correct model path!")
    else:
        net = weight_init_kaiming(net)
        args.epoch_start = 0

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

    for arg in vars(args):
        print(f"{arg:<15s}: {str(getattr(args, arg))}")

    simulate_dir = Path(args.simulate_dir)

    train_im_list = (
        list(simulate_dir.glob("*.jpg"))
        + list(simulate_dir.glob("*.png"))
        + list(simulate_dir.glob("*.bmp"))
    )
    train_im_list = sorted([str(x) for x in train_im_list])

    test_case1_bsd50_h5 = Path("test_data").joinpath(
        "noise_niid",
        "BSD50_niid_case1.hdf5",
    )

    test_im_list = (Path("test_data") / "BSD50").glob("*.jpg")
    test_im_list = sorted([str(x) for x in test_im_list])

    datasets = {
        "train": DenoisingDatasets_Gamma.SimulateTrain(
            train_im_list,
            5000 * args.batch_size,
            args.patch_size,
            radius=args.radius,
            noise_estimate=True,
        ),
        "test_bsd50": DenoisingDatasets_Gamma.SimulateTest(
            test_im_list,
            test_case1_bsd50_h5,
        ),
    }

    print(f"\nBegin training with GPU: {str(args.gpu_id)}")
    train_model(net, datasets, optimizer, scheduler, EvidentialRegressionLoss)


if __name__ == "__main__":
    main()
