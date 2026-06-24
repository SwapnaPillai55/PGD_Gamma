import os
import cv2
import numpy as np

#NOISY_DIR = "/home/dinesh/swapna/MLE_Denoiser_June2026/test_data/noisy_KLSG_testset"
#DENOISED_DIR = '/home/dinesh/swapna/MLE_Denoiser_June2026/R2_results_fig_gamma_L5to20_KLSG_testset/Proposed_11'

NOISY_DIR = "/home/dinesh/swapna/MLE_Denoiser_June2026/test_data/noisy_URPC_testset"
DENOISED_DIR = '/home/dinesh/swapna/MLE_Denoiser_June2026/R2_results_fig_gamma_L5to20_URPC_testset_seed123_check/Proposed_26'

#NOISY_DIR = "/home/dinesh/swapna/MLE_Denoiser_June2026/test_data/noisy_DEBRIS_testset"
#DENOISED_DIR = '/home/dinesh/swapna/MLE_Denoiser_June2026/R2_results_fig_gamma_L5to20_DEBRIS_testset_seed123/Proposed_7'


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def epd_roa_safe(noisy, denoised, min_den=1.0, eps=1e-8):
    noisy = noisy.astype(np.float64)
    denoised = denoised.astype(np.float64)

    # Horizontal: (i,j) / (i,j+1)
    noisy_num_h = noisy[:, :-1]
    noisy_den_h = noisy[:, 1:]
    den_num_h = denoised[:, :-1]
    den_den_h = denoised[:, 1:]

    mask_h = (noisy_den_h > min_den) & (den_den_h > min_den)

    noisy_h = np.abs(noisy_num_h[mask_h] / (noisy_den_h[mask_h] + eps))
    den_h = np.abs(den_num_h[mask_h] / (den_den_h[mask_h] + eps))

    epd_h = np.sum(den_h) / (np.sum(noisy_h) + eps)

    # Vertical: (i,j) / (i+1,j)
    noisy_num_v = noisy[:-1, :]
    noisy_den_v = noisy[1:, :]
    den_num_v = denoised[:-1, :]
    den_den_v = denoised[1:, :]

    mask_v = (noisy_den_v > min_den) & (den_den_v > min_den)

    noisy_v = np.abs(noisy_num_v[mask_v] / (noisy_den_v[mask_v] + eps))
    den_v = np.abs(den_num_v[mask_v] / (den_den_v[mask_v] + eps))

    epd_v = np.sum(den_v) / (np.sum(noisy_v) + eps)

    return epd_h, epd_v, np.mean(mask_h), np.mean(mask_v)


epd_hd_all = []
epd_vd_all = []

files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.lower().endswith(IMAGE_EXTS)
])

for file in files:
    noisy_path = os.path.join(NOISY_DIR, file)

    name, ext = os.path.splitext(file)
    denoised_file = f"{name}_DN{ext}"
    #denoised_file = f"{name}{ext}"
    denoised_path = os.path.join(DENOISED_DIR, denoised_file)

    if not os.path.exists(denoised_path):
        print(f"Missing denoised image: {denoised_file}")
        continue

    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)

    if noisy is None or denoised is None:
        print(f"Error reading pair: {file}")
        continue

    if noisy.shape != denoised.shape:
        print(f"Size mismatch: {file}")
        continue

    epd_hd, epd_vd, valid_h, valid_v = epd_roa_safe(
        noisy,
        denoised,
        min_den=1.0
    )

    epd_hd_all.append(epd_hd)
    epd_vd_all.append(epd_vd)

    print(
        f"{file} --> "
        f"EPD-ROA-HD: {epd_hd:.4f}, "
        f"EPD-ROA-VD: {epd_vd:.4f}, "
        f"Valid-HD: {valid_h*100:.2f}%, "
        f"Valid-VD: {valid_v*100:.2f}%"
    )


print("\n================ FINAL RESULTS ================\n")

if len(epd_hd_all) > 0:
    mean_hd = np.mean(epd_hd_all)
    mean_vd = np.mean(epd_vd_all)
    overall_mean = (mean_hd + mean_vd) / 2

    print(f"Mean EPD-ROA-HD      : {mean_hd:.4f}")
    print(f"Mean EPD-ROA-VD      : {mean_vd:.4f}")
    print(f"Overall Mean EPD-ROA : {overall_mean:.4f}")
else:
    print("No valid image pairs found.")
