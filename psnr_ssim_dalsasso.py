from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Could not read: {path}")
    return img.astype(np.float32) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--restored_dir", required=True)
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    restored_dir = Path(args.restored_dir)
    restored_files = list(restored_dir.iterdir())

    # Explicit sequence as requested
    sequence = [
        ("marais1", "Marais 1"),
        ("limagne", "Limagne"),
        ("saclay", "Saclay"),
        ("lely", "Lely"),
        ("ramb", "Rambouillet"),
        ("risoul", "Risoul"),
        ("marais2", "Marais 2")
    ]

    print(f"\n{'Image Case':<15} | {'Ver':<3} | {'PSNR (Avg ± Std)':<18} | {'SSIM (Avg ± Std)':<18}")
    print("-" * 65)

    all_case_psnr = []
    all_case_ssim = []

    for prefix, display_name in sequence:
        # Match GT image (handles both .png and .jpg)
        gt_path = next((p for p in clean_dir.iterdir() if p.stem.lower() == prefix.lower()), None)
        
        if gt_path is None:
            print(f"{display_name:<15} | --- | Not Found in Clean Dir")
            continue

        # Match all 20 denoised versions (e.g., marais1_noisy_1_DN.png)
        matches = [r for r in restored_files if r.name.lower().startswith(prefix.lower()) and "_DN" in r.name]
        
        if not matches:
            print(f"{display_name:<15} |  0  | No Denoised Files Found")
            continue

        clean_img = read_gray(gt_path)
        psnrs, ssims = [], []

        for m_path in matches:
            pred_img = read_gray(m_path)
            
            # Ensure dimensions match
            if clean_img.shape != pred_img.shape:
                pred_img = cv2.resize(pred_img, (clean_img.shape[1], clean_img.shape[0]))

            p = peak_signal_noise_ratio(clean_img, pred_img, data_range=1.0)
            s = structural_similarity(clean_img, pred_img, data_range=1.0)
            
            psnrs.append(p)
            ssims.append(s)

        # Calculate metrics
        m_psnr, s_psnr = np.mean(psnrs), np.std(psnrs)
        m_ssim, s_ssim = np.mean(ssims), np.std(ssims)

        print(f"{display_name:<15} | {len(matches):<3} | {m_psnr:>5.2f} ± {s_psnr:<5.4f} | {m_ssim:>5.4f} ± {s_ssim:<5.4f}")

        all_case_psnr.append(m_psnr)
        all_case_ssim.append(m_ssim)

    # Final Average across all cases
    if all_case_psnr:
        print("-" * 65)
        print(f"{'OVERALL AVG':<15} | --- | {np.mean(all_case_psnr):>5.2f} ± {np.std(all_case_psnr):<5.4f} | {np.mean(all_case_ssim):>5.4f} ± {np.std(all_case_ssim):<5.4f}")
        print("=" * 65)

if __name__ == "__main__":
    main()
