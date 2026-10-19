"""
Mô-đun đánh giá chất lượng ảnh.
Người thực hiện: Vũ Minh Anh

Các chỉ số:
- Entropy (Shannon) – đo lượng thông tin
- RMS Contrast – độ tương phản
- PSNR – tỷ lệ tín hiệu trên nhiễu đỉnh (cần ảnh tham chiếu)
- SSIM – chỉ số tương đồng cấu trúc (cần ảnh tham chiếu)
"""

import cv2
import numpy as np
import pandas as pd
import os
import time
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================
# HÀM PHỤ TRỢ
# ============================================================

def _to_gray(img):
    """Chuyển ảnh sang mức xám nếu đang là ảnh màu."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================================================
# CÁC CHỈ SỐ ĐÁNH GIÁ
# ============================================================

def compute_entropy(img):
    """
    Entropy Shannon – đo lượng thông tin trong ảnh.
    Giá trị càng cao → ảnh càng giàu chi tiết.
    """
    return float(shannon_entropy(_to_gray(img)))


def compute_contrast(img):
    """
    RMS Contrast – độ lệch chuẩn mức xám.
    Giá trị càng cao → ảnh càng có độ tương phản rõ.
    """
    return float(np.std(_to_gray(img)))


def compute_psnr(img, ref):
    """
    PSNR – Peak Signal-to-Noise Ratio.
    So sánh ảnh kết quả với ảnh tham chiếu.
    Giá trị càng cao → ảnh càng gần tham chiếu (tốt).
    Yêu cầu: img và ref phải cùng kích thước.
    """
    return float(peak_signal_noise_ratio(ref, img))


def compute_ssim(img, ref):
    """
    SSIM – Structural Similarity Index.
    Đo độ tương đồng cấu trúc với ảnh tham chiếu.
    Giá trị từ 0 đến 1, càng gần 1 → càng giống.
    Yêu cầu: img và ref phải cùng kích thước.
    """
    return float(structural_similarity(_to_gray(ref), _to_gray(img)))


def compute_all_metrics(img, ref=None):
    """
    Tính tất cả metrics, trả về dict.
    Nếu không có ảnh tham chiếu (ref=None) → bỏ qua PSNR, SSIM.
    """
    result = {
        'entropy': compute_entropy(img),
        'contrast': compute_contrast(img),
    }
    if ref is not None:
        result['psnr'] = compute_psnr(img, ref)
        result['ssim'] = compute_ssim(img, ref)
    return result


def measure_time(func, *args, **kwargs):
    """
    Đo thời gian chạy 1 hàm.
    Trả về: (kết_quả_hàm, thời_gian_giây)
    """
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


# ============================================================
# XUẤT KẾT QUẢ RA CSV
# ============================================================

def evaluate_directory(processed_dir, reference_dir=None, output_csv='results/metrics.csv'):
    """
    Đánh giá tất cả ảnh trong thư mục processed_dir.
    Nếu có reference_dir → tính thêm PSNR, SSIM.
    Kết quả ghi vào file CSV.
    """
    rows = []
    for fname in sorted(os.listdir(processed_dir)):
        fpath = os.path.join(processed_dir, fname)
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img = cv2.imread(fpath)
        if img is None:
            print(f"  ⚠ Không đọc được: {fname}")
            continue

        # Tìm ảnh tham chiếu tương ứng
        ref = None
        if reference_dir:
            ref_path = os.path.join(reference_dir, fname)
            if os.path.exists(ref_path):
                ref = cv2.imread(ref_path)

        metrics = compute_all_metrics(img, ref)
        metrics['filename'] = fname
        rows.append(metrics)
        print(f"  ✓ {fname}: entropy={metrics['entropy']:.2f}, contrast={metrics['contrast']:.2f}")

    df = pd.DataFrame(rows)
    # Sắp xếp cột cho dễ đọc
    cols = ['filename', 'entropy', 'contrast']
    if 'psnr' in df.columns:
        cols += ['psnr', 'ssim']
    df = df[cols]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n→ Đã lưu kết quả vào {output_csv}")
    return df


# ============================================================
# VẼ BIỂU ĐỒ SO SÁNH
# ============================================================

def plot_comparison_charts(csv_path='results/metrics.csv', output_dir='results/charts'):
    """
    Đọc file CSV metrics và vẽ biểu đồ cột cho từng chỉ số.
    Lưu biểu đồ vào output_dir.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for metric in ['entropy', 'contrast', 'psnr', 'ssim']:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(df)), df[metric], color='steelblue')
        plt.xticks(range(len(df)), df['filename'], rotation=45, ha='right', fontsize=8)
        plt.ylabel(metric.upper())
        plt.title(f'So sánh {metric.upper()} giữa các ảnh')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ✓ Đã lưu biểu đồ: {save_path}")


# ============================================================
# CHẠY TRỰC TIẾP
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("ĐÁNH GIÁ CHẤT LƯỢNG ẢNH")
    print("=" * 50)

    # 1. Đánh giá bộ ảnh gốc thiếu sáng (Baseline)
    print("\n📊 Đánh giá ảnh gốc (Baseline data/raw):")
    if os.path.exists('data/raw'):
        evaluate_directory('data/raw', 'data/reference', 'results/metrics.csv')
        plot_comparison_charts('results/metrics.csv', 'results/charts')

    # 2. Đánh giá cấu hình 1
    print("\n📊 Cấu hình 1:")
    if os.path.exists('results/config1'):
        evaluate_directory('results/config1', 'data/reference', 'results/metrics_config1.csv')
        plot_comparison_charts('results/metrics_config1.csv', 'results/charts/config1')
    else:
        print("  ⚠ Chưa có thư mục results/config1 (chờ Khoa chạy pipeline)")

    # 3. Đánh giá cấu hình 2
    print("\n📊 Cấu hình 2:")
    if os.path.exists('results/config2'):
        evaluate_directory('results/config2', 'data/reference', 'results/metrics_config2.csv')
        plot_comparison_charts('results/metrics_config2.csv', 'results/charts/config2')
    else:
        print("  ⚠ Chưa có thư mục results/config2 (chờ Khoa chạy pipeline)")

