"""
Pipeline xử lý tăng sáng và khử nhiễu ảnh thiếu sáng.
Người thực hiện: Dương Tiến Anh Khoa (ghép bước) & Vũ Minh Anh (tích hợp)

Hai cấu hình thực nghiệm:
- Cấu hình 1: Gamma correction (gamma=0.6) + Gaussian Blur (k=5, sigma=1.0)
- Cấu hình 2: CLAHE (clip=2.0, grid=8x8) + Bilateral Filter (d=9, sigma_color=75, sigma_space=75)
"""

import os
import glob
from pathlib import Path
import cv2
import numpy as np

# Import từ các module của nhóm
from src.enhance import gamma_correction, clahe, histogram_stretch

# Xác định thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CONFIG1_DIR = RESULTS_DIR / "config1"
CONFIG2_DIR = RESULTS_DIR / "config2"


def run_config1(img: np.ndarray) -> np.ndarray:
    """Cấu hình 1: Gamma Correction (gamma=0.6) + Gaussian Blur (5x5)."""
    # 1. Tăng sáng bằng Gamma
    enhanced = gamma_correction(img, gamma=0.6)
    # 2. Khử nhiễu bằng Gaussian Blur
    denoised = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=1.0)
    return denoised


def run_config2(img: np.ndarray) -> np.ndarray:
    """Cấu hình 2: CLAHE (clip=2.0, grid=8x8) + Bilateral Filter (d=9)."""
    # 1. Tăng sáng thích nghi bằng CLAHE
    enhanced = clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    # 2. Khử nhiễu bảo toàn biên cạnh bằng Bilateral
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


def process_all_images():
    """Chạy toàn bộ ảnh trong thư mục data qua cả 2 cấu hình và lưu kết quả."""
    CONFIG1_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG2_DIR.mkdir(parents=True, exist_ok=True)

    # Quét tất cả ảnh trong thư mục data
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(str(DATA_DIR / ext)))

    image_paths = sorted(set(image_paths))
    total = len(image_paths)
    print(f"🔍 Tìm thấy {total} ảnh trong {DATA_DIR}")

    if total == 0:
        print("❌ Không tìm thấy ảnh trong thư mục data!")
        return

    print("🚀 Bắt đầu chạy pipeline 2 cấu hình...")

    for idx, path_str in enumerate(image_paths, 1):
        fpath = Path(path_str)
        fname = fpath.name

        # Đọc ảnh an toàn
        img_array = np.fromfile(str(fpath), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  ⚠ Lỗi đọc ảnh: {fname}")
            continue

        # Chạy Cấu hình 1
        res1 = run_config1(img)
        save1_path = CONFIG1_DIR / fname
        cv2.imwrite(str(save1_path), res1)

        # Chạy Cấu hình 2
        res2 = run_config2(img)
        save2_path = CONFIG2_DIR / fname
        cv2.imwrite(str(save2_path), res2)

        if idx % 10 == 0 or idx == total:
            print(f"  ✓ Đã xử lý: {idx}/{total} ảnh")

    print("\n🎉 Hoàn thành xử lý toàn bộ ảnh!")
    print(f"  📁 Kết quả Cấu hình 1: {CONFIG1_DIR}")
    print(f"  📁 Kết quả Cấu hình 2: {CONFIG2_DIR}")


if __name__ == "__main__":
    process_all_images()
