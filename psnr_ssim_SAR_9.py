import cv2
import numpy as np
import re
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

clean_dir = Path('/home/dinesh/swapna/MLE_Denoiser_June2026/test_data/SAR_9/gray')
denoised_dir = Path('/home/dinesh/swapna/MLE_Denoiser_June2026/SAR_9_f8_steps10k/Proposed_51')


def get_clean_name(denoised_name):
    name = re.sub(r'_DN\.png$', '.png', denoised_name)
    name = re.sub(r'_L\d+\.png$', '.png', name)
    return name.replace('.png', '_gray.png')


psnrs, ssims = [], []
denoised_files = sorted(denoised_dir.glob('*_DN.png'))

print(f"Using denoised folder: {denoised_dir}")
print(f"Found {len(denoised_files)} denoised images")

print("\nAll filename mappings:")
for d_path in denoised_files:
    clean_name = get_clean_name(d_path.name)
    clean_path = clean_dir / clean_name
    print(f"{d_path.name} --> {clean_name} | Exists: {clean_path.exists()}")

print("\nPer-image results:")
for d_path in denoised_files:
    clean_name = get_clean_name(d_path.name)
    clean_path = clean_dir / clean_name

    if not clean_path.exists():
        print(f"Missing GT: {clean_name}")
        continue

    img_c = cv2.imread(str(clean_path), 0)
    img_d = cv2.imread(str(d_path), 0)

    if img_c is None:
        print(f"Could not read GT: {clean_path}")
        continue

    if img_d is None:
        print(f"Could not read denoised: {d_path}")
        continue

    if img_c.shape != img_d.shape:
        img_d = cv2.resize(img_d, (img_c.shape[1], img_c.shape[0]))

    psnr_val = compare_psnr(img_c, img_d, data_range=255)
    ssim_val = compare_ssim(img_c, img_d, data_range=255)

    psnrs.append(psnr_val)
    ssims.append(ssim_val)

    print(f"{d_path.name} vs {clean_name} | PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}")

print("\n" + "=" * 50)

if psnrs:
    print(f"Average PSNR: {np.mean(psnrs):.4f}")
    print(f"Average SSIM: {np.mean(ssims):.4f}")
    print(f"Matched images: {len(psnrs)}")
else:
    print("No valid image matches found.")