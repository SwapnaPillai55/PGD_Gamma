#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_float
from pathlib import Path
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from matplotlib import pyplot as plt
import time

from Networks_Sid.models.convnext_isotropic_x2_f8_3c import convnext_isotropic_base

import random
from PIL import Image
import os


# =========================================================
# SEED
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
seed = np.random.RandomState(112311)

use_gpu = True
case = 2
C = 3
dep_U = 4


# =========================================================
# CHECKPOINTS
# =========================================================
ckpt_root = './models/model_L5to20_DEBRIS'
ckpt_files = sorted([f for f in os.listdir(ckpt_root) if f.startswith('model_state_')])

if len(ckpt_files) == 0:
    print(f"No checkpoints found in: {ckpt_root}")
    sys.exit(1)


# =========================================================
# IMAGES
# =========================================================
image_dir = Path('test_data') / 'noisy_URPC_testset'
image_files = sorted([f for ext in ('*.png', '*.jpg', '*.jpeg') for f in image_dir.glob(ext)])

if len(image_files) == 0:
    print(f"No images found in: {image_dir}")
    sys.exit(1)


# =========================================================
# FLEXIBLE CHECKPOINT LOADER
# =========================================================
def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def add_module_prefix(state_dict):
    return {
        (k if k.startswith("module.") else "module." + k): v
        for k, v in state_dict.items()
    }


def remove_module_prefix(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_checkpoint_flexible(net, checkpoint, use_gpu_local):
    state = get_state_dict(checkpoint)

    if use_gpu_local:
        # net is DataParallel, so it normally expects "module." keys
        candidates = [
            state,
            add_module_prefix(state),
            remove_module_prefix(state),
        ]

        last_err = None
        for cand in candidates:
            try:
                net.load_state_dict(cand, strict=True)
                return True
            except Exception as e:
                last_err = e

            try:
                net.module.load_state_dict(remove_module_prefix(cand), strict=True)
                return True
            except Exception as e:
                last_err = e

        print("Failed to load checkpoint into DataParallel model:", last_err)
        return False

    else:
        candidates = [
            state,
            remove_module_prefix(state),
        ]

        last_err = None
        for cand in candidates:
            try:
                net.load_state_dict(cand, strict=True)
                return True
            except Exception as e:
                last_err = e

            try:
                load_state_dict_cpu(net, cand)
                return True
            except Exception as e:
                last_err = e

        print("Failed to load checkpoint into CPU model:", last_err)
        return False


# =========================================================
# TEST EACH CHECKPOINT
# =========================================================
for ckpt_name in ckpt_files:

    print("\n" + "=" * 70)
    print(f"Loading the Model: {ckpt_name}")
    print("=" * 70)

    i = ckpt_name.split('_')[-1]

    # Load checkpoint
    try:
        checkpoint = torch.load(os.path.join(ckpt_root, ckpt_name), map_location="cpu")
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print("Error loading model checkpoint:", e)
        continue

    # Build model
    net = convnext_isotropic_base()

    if use_gpu:
        if not torch.cuda.is_available():
            print("CUDA not available but use_gpu=True. Falling back to CPU.")
            use_gpu_local = False
        else:
            use_gpu_local = True
    else:
        use_gpu_local = False

    if use_gpu_local:
        net = torch.nn.DataParallel(net).cuda()

    ok = load_checkpoint_flexible(net, checkpoint, use_gpu_local)
    if not ok:
        continue

    net.eval()

    all_psnr_vals = []
    all_ssim_vals = []

    output_dir = f'./R2_results_fig_gamma_L5to20_URPC_testset_seed123_check/Proposed_{i}'
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # LOOP IMAGES
    # =========================================================
    for im_path in image_files:
        im_name = im_path.name

        im_read = cv2.imread(str(im_path))
        if im_read is None:
            print(f"Warning: failed to read {im_path}, skipping.")
            continue

        im_gt = img_as_float(im_read[:, :, ::-1])
        H, W, _ = im_gt.shape

        if case == 1:
            sigma = peaks(256)
        elif case == 2:
            sigma = sincos_kernel()
        elif case == 3:
            sigma = generate_gauss_kernel_mix(256, 256)
        else:
            sys.exit('Please input the corrected test case: 1, 2 or 3')

        sigma = 10 / 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((75 - 10) / 255.0)
        sigma = cv2.resize(sigma, (W, H))

        im_noisy_np = ((im_gt ** 1) * 1).astype(np.float32)
        im_noisy = torch.from_numpy(im_noisy_np.transpose((2, 0, 1))[np.newaxis, ...])

        if use_gpu_local:
            im_noisy = im_noisy.cuda(non_blocking=True)
        else:
            print(f'Begin Testing on CPU for {im_name}')

        with torch.no_grad():
            if use_gpu_local:
                torch.cuda.synchronize()
            tic = time.perf_counter()

            alpha, beta = net(im_noisy)

            alpha_pred = F.softplus(alpha) + 1e-6
            beta_pred = F.softplus(beta) + 1e-6
            im_denoise_t = torch.clamp(alpha_pred / beta_pred, 0.0, 1.0)

            if use_gpu_local:
                torch.cuda.synchronize()
            toc = time.perf_counter()

        im_denoise_np = np.transpose(
            im_denoise_t.squeeze(0).detach().cpu().numpy(),
            (1, 2, 0)
        ).astype(np.float32)

        im_denoise_np = np.clip(im_denoise_np, 0.0, 1.0)
        im_denoise_u8 = (im_denoise_np * 255.0 + 0.5).astype(np.uint8)

        im_gt_np = np.clip(im_gt.astype(np.float32), 0.0, 1.0)
        im_gt_u8 = (im_gt_np * 255.0 + 0.5).astype(np.uint8)

        psnr_val = compare_psnr(im_gt_u8, im_denoise_u8, data_range=255)
        ssim_val = compare_ssim(im_gt_u8, im_denoise_u8, data_range=255, channel_axis=2, win_size=7)

        all_psnr_vals.append(psnr_val)
        all_ssim_vals.append(ssim_val)

        im_gt_avg = np.mean(im_gt_u8, axis=2)
        im_noisy_u8 = (np.clip(im_noisy_np, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        im_noisy_avg = np.mean(im_noisy_u8, axis=2)

        im_denoise_avg = np.mean(im_denoise_np, axis=2)
        im_denoise_avg_u8 = (im_denoise_avg * 255.0 + 0.5).astype(np.uint8)

        # plt.subplot(131)
        # plt.imshow(im_gt_avg, cmap='gray')
        # plt.title('Groundtruth')
        # plt.subplot(132)
        # plt.imshow(im_noisy_avg, cmap='gray')
        # plt.title('Noisy Image')
        # plt.subplot(133)
        # plt.imshow(im_denoise_avg_u8, cmap='gray')
        # plt.title('Denoised Image')
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()

        base_name, ext = os.path.splitext(im_name)
        output_path = os.path.join(output_dir, f'{base_name}_DN{ext}')

        img_pil = Image.fromarray(im_denoise_avg_u8)
        img_pil.save(output_path)

    # =========================================================
    # Summary per checkpoint
    # =========================================================
    if all_psnr_vals and all_ssim_vals:
        avg_psnr = float(np.mean(all_psnr_vals))
        avg_ssim = float(np.mean(all_ssim_vals))
        print("\n" + "=" * 50)
        print('Ignore PSNR and SSIM below in case of real images')
        # print(f"[{ckpt_name}] Average PSNR over the test set: {avg_psnr:.4f}")
        # print(f"[{ckpt_name}] Average SSIM over the test set: {avg_ssim:.4f}")
        print("=" * 50)
    else:
        print(f"[{ckpt_name}] No images were processed, so no average PSNR or SSIM could be computed.")