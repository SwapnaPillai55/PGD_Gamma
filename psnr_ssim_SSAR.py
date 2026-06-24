import os
import re
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Paths ---
# Denoised folder: contains S1_L1_DN.png, S10_L1_DN.png, etc.
denoised_dir = "/home/dinesh/swapna/MLE_Denoiser_June2026/SSAR_f8_steps10k/Proposed_51"

# GT folder: contains S1.png, S10.png, etc.
# Note: Ensure these filenames match exactly (e.g., if GT is S1.png, use f"{prefix}.png")
#gt_dir = "/home/dinesh/swapna/AE_Den/AE_Den/data/SSAR/gray"

gt_dir = "/home/dinesh/swapna/MLE_Denoiser_June2026/test_data/SSAR/gray"


denoised_files = sorted(
    [f for f in os.listdir(denoised_dir) if f.lower().endswith(".png")]
)

psnr_list = []
ssim_list = []

print(f"Found {len(denoised_files)} files. Starting evaluation...\n")

for den_name in denoised_files:
    # 1. Extract prefix using Regex (grabs everything before the first underscore)
    # Example: 'S10_L1_DN.png' -> 'S10'
    match = re.match(r"^([^_]+)", den_name)
    if not match:
        continue
    
    prefix = match.group(1)
    
    # 2. Define GT filename
    # Based on your prompt, GT is 'S1.png', 'S2.png', etc.
    gt_name = f"{prefix}.png"

    den_path = os.path.join(denoised_dir, den_name)
    gt_path = os.path.join(gt_dir, gt_name)

    # 3. Check for existence
    if not os.path.exists(gt_path):
        print(f"Missing GT: {den_name} -> Expected {gt_name}")
        continue

    try:
        # 4. Read and convert to float
        den = imread(den_path, as_gray=True).astype(np.float32)
        gt = imread(gt_path, as_gray=True).astype(np.float32)

        # Scale to [0, 1]
        if den.max() > 1.0:
            den /= 255.0
        if gt.max() > 1.0:
            gt /= 255.0

        # 5. Check Shape
        if den.shape != gt.shape:
            print(f"Shape mismatch: {den_name} {den.shape} vs {gt_name} {gt.shape}")
            continue

        # 6. Metrics
        p = psnr(gt, den, data_range=1.0)
        s = ssim(gt, den, data_range=1.0)

        psnr_list.append(p)
        ssim_list.append(s)

        print(f"{den_name} | PSNR: {p:.4f} | SSIM: {s:.4f}")

    except Exception as e:
        print(f"Error processing {den_name}: {e}")

# --- Summary ---
if psnr_list:
    print("\n" + "="*40)
    print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print("="*40)
else:
    print("No valid pairs evaluated.")