import sys
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
from skimage import img_as_float, img_as_ubyte
from pathlib import Path
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
import time

from Networks_Sid.models.convnext_isotropic_x2_f2_32_3c import convnext_isotropic_base

import random
import os


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

use_gpu = True
case = 2

ckpt_root = "./models/model_sassed"

ckpt_files = sorted(
    [f for f in os.listdir(ckpt_root) if f.startswith("model_state_")]
)

if len(ckpt_files) == 0:
    print(f"No checkpoints found in: {ckpt_root}")
    sys.exit(1)


image_dir = Path("test_data") / "noisy_sas_partitioned/test"

image_files = sorted(
    [f for ext in ("*.png", "*.jpg", "*.jpeg") for f in image_dir.glob(ext)]
)

if len(image_files) == 0:
    print(f"No images found in: {image_dir}")
    sys.exit(1)


# ==========================================================
# Test each checkpoint
# ==========================================================
for ckpt_name in ckpt_files:

    i = ckpt_name.split("_")[-1]

    print("\n" + "=" * 70)
    print(f"Processing Proposed_{i}  |  checkpoint: {ckpt_name}")
    print("=" * 70)

    ckpt_path = os.path.join(ckpt_root, ckpt_name)

    try:
        checkpoint = torch.load(ckpt_path)
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_name}: {e}")
        continue

    net = convnext_isotropic_base()

    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(net, checkpoint)

    net.eval()

    all_psnr_vals = []
    all_ssim_vals = []

    output_dir = f"./Result_figs/sassed/Proposed_{i}"
    os.makedirs(output_dir, exist_ok=True)

    for im_path in image_files:
        im_name = im_path.name

        im_bgr = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        if im_bgr is None:
            continue

        im_gt = img_as_float(im_bgr[:, :, ::-1])
        H, W, _ = im_gt.shape

        if case == 1:
            sigma = peaks(256)
        elif case == 2:
            sigma = sincos_kernel()
        elif case == 3:
            sigma = generate_gauss_kernel_mix(256, 256)
        else:
            sys.exit("Please input the correct test case: 1, 2 or 3")

        sigma = 10 / 255.0 + (
            (sigma - sigma.min()) / (sigma.max() - sigma.min())
        ) * ((75 - 10) / 255.0)

        sigma = cv2.resize(sigma, (W, H))

        # ------------------------------------------------------------
        # Input image
        # For synthetic noisy input, currently you are using clean image directly.
        # ------------------------------------------------------------
        im_noisy = im_gt.astype(np.float32)

        im_noisy_tensor = torch.from_numpy(
            im_noisy.transpose((2, 0, 1))[np.newaxis, :]
        ).float()

        if use_gpu:
            im_noisy_tensor = im_noisy_tensor.cuda(non_blocking=True)

        # ------------------------------------------------------------
        # Denoising
        # ------------------------------------------------------------
        with torch.inference_mode():
            alpha, beta = net(im_noisy_tensor)

            alpha_pred = F.softplus(alpha) + 1e-6
            beta_pred = F.softplus(beta) + 1e-6

            im_denoise = alpha_pred / beta_pred
            im_denoise = torch.clamp(im_denoise, 0.0, 1.0)

        im_denoise = np.transpose(
            im_denoise.squeeze().cpu().numpy(), (1, 2, 0)
        )

        im_denoise_uint8 = img_as_ubyte(im_denoise.clip(0, 1))
        im_gt_uint8 = img_as_ubyte(im_gt.clip(0, 1))

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        psnr_val = compare_psnr(im_gt_uint8, im_denoise_uint8, data_range=255)
        ssim_val = compare_ssim(
            im_gt_uint8,
            im_denoise_uint8,
            data_range=255,
            channel_axis=2,
            win_size=7,
        )

        all_psnr_vals.append(psnr_val)
        all_ssim_vals.append(ssim_val)

        # ------------------------------------------------------------
        # Save denoised grayscale image
        # ------------------------------------------------------------
        im_denoise_avg = np.mean(im_denoise_uint8, axis=2)

        base_name, ext = os.path.splitext(im_name)
        output_path = os.path.join(output_dir, f"{base_name}_DN{ext}")

        cv2.imwrite(output_path, np.uint8(im_denoise_avg))

    if all_psnr_vals and all_ssim_vals:
        avg_psnr = np.mean(all_psnr_vals)
        avg_ssim = np.mean(all_ssim_vals)

        print(f"Finished Proposed_{i}")
        print("Ignore PSNR and SSIM below in case of real images")
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print(f"Proposed_{i}: no images were processed.")