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
from matplotlib import pyplot as plt
import time

from Networks_Sid.models.convnext_isotropic_x2_f8_3c import convnext_isotropic_base


import random
from PIL import Image
import os

# Fixing the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
seed = np.random.RandomState(112311)

use_gpu = True
case = 2
C = 3
dep_U = 4

# ------------------------------------------------------------
# Loop over all model_state_{i} in ./model_28Jan_gamma_for_KLSG_R2
# ------------------------------------------------------------

ckpt_root = './models/sassed_seed123_L5to20_f8_500' # ==================================================================================

ckpt_files = sorted([f for f in os.listdir(ckpt_root) if f.startswith('model_state_')])

if len(ckpt_files) == 0:
    print(f"No checkpoints found in: {ckpt_root}")
    sys.exit(1)

# Get list of all image files in the demo directory (unchanged)

image_dir = Path('test_data') / 'noisy_sas_partitioned/test'

image_files = sorted([f for ext in ('*.png', '*.jpg', '*.jpeg') for f in image_dir.glob(ext)])


if len(image_files) == 0:
    print(f"No PNG images found in: {image_dir}")
    sys.exit(1)

# ==========================================================
# Test each checkpoint
# ==========================================================
for ckpt_name in ckpt_files:

    print("\n" + "="*70)
    print(f"Loading the Model: {ckpt_name}")
    print("="*70)

    # Extract i from model_state_{i}
    i = ckpt_name.split('_')[-1]

    # Load the pretrained model (ONLY path changed)
    try:
        checkpoint = torch.load(os.path.join(ckpt_root, ckpt_name))
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print("Error loading model checkpoint:", e)
        continue

    net = convnext_isotropic_base()  # unchanged
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(net, checkpoint)
    net.eval()

    # Initialize lists to store PSNR and SSIM values (per checkpoint)
    all_psnr_vals = []
    all_ssim_vals = []

    # Output directory per model_state_{i} (ONLY changed output root + Proposed_{i})

    output_dir = f'./Result_figs/sassed_f2_32depth1_L10to30_valL20_steps50_testset_f8_500/Proposed_{i}' #==================================================================
    


    os.makedirs(output_dir, exist_ok=True)

    # Loop through each image file (UNCHANGED BODY)
    for im_path in image_files:
        im_name = im_path.name
       # print(f"Processing {im_name}")

        # Load and process the image
        im_gt = img_as_float(cv2.imread(str(im_path))[:, :, ::-1])  # Read and convert image to float32
        H, W, _ = im_gt.shape

        # Generate the sigma map (based on case)
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

        #=============================================================================
        # Generate noise and add it to the ground truth image
        L = 1
        noise = np.random.gamma(shape=L, scale=1.0/L, size=im_gt.shape)  # for num looks = 1
        im_noisy = ((im_gt ** 1)* 1 ).astype(np.float32) # for synthetic
        #im_noisy = ((im_gt ** 1) ).astype(np.float32) # for real
        noise = noise.astype(np.float32)
        #=============================================================================

        # Prepare the noisy image tensor
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))[np.newaxis, ])  # Add batch dimension
        if use_gpu:
            im_noisy = im_noisy.cuda()
            print(f'Begin Testing on GPU for {im_name}')
        else:
            print(f'Begin Testing on CPU for {im_name}')

        # Perform denoising
        with torch.autograd.set_grad_enabled(False):
            torch.cuda.synchronize()
            tic = time.perf_counter()

            # Forward pass through the network
            alpha, beta = net(im_noisy)

            # Apply softplus to ensure positivity
            alpha_pred = F.softplus(alpha) + 1e-6
            beta_pred = F.softplus(beta) + 1e-6

            im_denoise = alpha_pred / beta_pred

            im_denoise = torch.clamp(im_denoise, 0.0, 1.0)
            im_gt_tensor = torch.from_numpy(im_gt).float()
            im_gt_tensor = torch.clamp(im_gt_tensor, 0.0, 1.0)

            torch.cuda.synchronize()
            toc = time.perf_counter()

        # Convert tensors to numpy arrays
        im_denoise = np.transpose(im_denoise.squeeze().cpu().numpy(), (1, 2, 0))
        im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

        im_gt_numpy = img_as_ubyte(im_gt)

        # Calculate PSNR and SSIM
        psnr_val = compare_psnr(im_gt_numpy, im_denoise, data_range=255)
        ssim_val = compare_ssim(im_gt_numpy, im_denoise, data_range=255, channel_axis=2, win_size=7)

        # Append values to the lists
        all_psnr_vals.append(psnr_val)
        all_ssim_vals.append(ssim_val)

       # print(f'Image name: {im_name}, PSNR={psnr_val:5.2f}, SSIM={ssim_val:7.4f}, time={toc - tic:.4f}')

        # --- Plotting and saving code (unchanged) ---
        im_gt_avg = np.mean(im_gt_numpy, axis=2)
        im_denoise_avg = np.mean(im_denoise, axis=2)

        plt.subplot(131)
        plt.imshow(im_gt_avg, cmap='gray')
        plt.title('Groundtruth')
        plt.subplot(132)
        im_noisy_numpy = np.transpose(im_noisy.squeeze().cpu().numpy(), (1, 2, 0))
        im_noisy_numpy = img_as_ubyte(im_noisy_numpy.clip(0, 1))
        im_noisy_avg = np.mean(im_noisy_numpy, axis=2)
        plt.imshow(im_noisy_avg, cmap='gray')
        plt.title('Noisy Image')
        plt.subplot(133)
        plt.imshow(im_denoise_avg, cmap='gray')
        plt.title('Denoised Image')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        base_name, ext = os.path.splitext(im_name)
        output_path = os.path.join(output_dir, f'{base_name}_DN{ext}')

        # Save the denoised image as a grayscale image
        img = Image.fromarray(np.uint8(im_denoise_avg))
        img.save(output_path)

        #print(f"Saved denoised image as {output_path}")

    # After the loop, compute and print the averages (per checkpoint)
    if all_psnr_vals and all_ssim_vals:
        avg_psnr = np.mean(all_psnr_vals)
        avg_ssim = np.mean(all_ssim_vals)
        print("\n" + "="*50)
        print('Ignore PSNR and SSIM below in case of real images')
        #print(f"[{ckpt_name}] Average PSNR over the test set: {avg_psnr:.4f}")
        #print(f"[{ckpt_name}] Average SSIM over the test set: {avg_ssim:.4f}")
        print("="*50)
    else:
        print(f"[{ckpt_name}] No images were processed, so no average PSNR or SSIM could be computed.")