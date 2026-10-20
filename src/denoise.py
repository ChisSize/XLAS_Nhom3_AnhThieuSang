import random
import shutil
from pathlib import Path

import cv2


# =========================================================
# 1. CẤU HÌNH
# =========================================================

BASE_DIR = Path(
    r"D:\XLAS\XLAS_Nhom3_AnhThieuSang-main"
    r"\XLAS_Nhom3_AnhThieuSang-main"
)

SRC_DIR = BASE_DIR / "results" / "enhance" / "pipeline"   # ảnh đầu vào
OUTPUT_DIR = BASE_DIR / "results" / "denoise"             # kết quả khử nhiễu

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Tham số các bộ lọc
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0
MEDIAN_KERNEL = 5
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Số ảnh lấy ra để xử lý (None = lấy tất cả ảnh trong pipeline)
NUM_IMAGES = None

# True: xóa thư mục denoise cũ trước khi chạy (tránh ảnh cũ bị lẫn/lặp)
RESET_OUTPUT = True

RANDOM_SEED = 42

METHODS = ["Gaussian", "Median", "Bilateral"]

# --- Giảm dung lượng ảnh đầu ra ---
SAVE_FORMAT = ".jpg"     # ".jpg" nhẹ nhất | ".png" nặng hơn nhiều
JPEG_QUALITY = 80        # 60-95, càng thấp càng nhẹ
PNG_COMPRESSION = 9      # 0-9, chỉ dùng khi SAVE_FORMAT = ".png"
MAX_SIDE = 1024          # thu nhỏ nếu cạnh dài hơn số này (None = giữ nguyên)

# Cảnh báo nếu tổng dung lượng vượt ngưỡng này (MB)
WARN_TOTAL_MB = 25


# =========================================================
# 2. LẤY DANH SÁCH ẢNH
# =========================================================

def get_image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# =========================================================
# 3. CHIA ẢNH CHO 3 PHƯƠNG PHÁP (mỗi ảnh chỉ thuộc 1 phương pháp)
# =========================================================

def split_among_methods(image_paths):
    paths = list(image_paths)
    random.Random(RANDOM_SEED).shuffle(paths)

    if NUM_IMAGES is not None:
        paths = paths[:NUM_IMAGES]

    base = len(paths) // len(METHODS)
    remainder = len(paths) % len(METHODS)

    groups = {}
    start = 0
    for i, method in enumerate(METHODS):
        size = base + (1 if i < remainder else 0)
        groups[method] = paths[start:start + size]
        start += size
    return groups


# =========================================================
# 4. KHỬ NHIỄU + LƯU ẢNH NHẸ
# =========================================================

def denoise(image, method):
    if method == "Gaussian":
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, sigmaX=GAUSSIAN_SIGMA)
    if method == "Median":
        return cv2.medianBlur(image, MEDIAN_KERNEL)
    if method == "Bilateral":
        return cv2.bilateralFilter(
            image,
            BILATERAL_DIAMETER,
            BILATERAL_SIGMA_COLOR,
            BILATERAL_SIGMA_SPACE,
        )
    raise ValueError(f"Phương pháp không hợp lệ: {method}")


def resize_if_needed(image):
    if MAX_SIDE is None:
        return image
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def process_image(image_path, method):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return False

    # Khử nhiễu ở kích thước gốc, rồi mới thu nhỏ khi lưu
    result = resize_if_needed(denoise(image, method))

    relative_path = image_path.relative_to(SRC_DIR)
    save_path = (OUTPUT_DIR / method / relative_path).with_suffix(SAVE_FORMAT)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_FORMAT.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]

    if not cv2.imwrite(str(save_path), result, params):
        print(f"Lỗi lưu ảnh: {save_path}")
        return False
    return True


# =========================================================
# 5. TÍNH DUNG LƯỢNG
# =========================================================

def folder_size_mb(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0.0
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    return total / (1024 * 1024)


# =========================================================
# 6. CHẠY
# =========================================================

def main():
    print("=" * 60)
    print(f"Ảnh đầu vào: {SRC_DIR}")
    print(f"Ảnh đầu ra:  {OUTPUT_DIR}")
    print(f"Định dạng lưu: {SAVE_FORMAT} | JPEG quality: {JPEG_QUALITY} | Cạnh dài tối đa: {MAX_SIDE}")
    print("=" * 60)

    if not SRC_DIR.is_dir():
        print("LỖI: Không tìm thấy thư mục results/enhance/pipeline.")
        return

    image_paths = get_image_paths(SRC_DIR)
    if not image_paths:
        print("Không tìm thấy ảnh trong results/enhance/pipeline.")
        return

    print(f"Tổng số ảnh tìm thấy: {len(image_paths)}")
    print(f"Dung lượng ảnh đầu vào: {folder_size_mb(SRC_DIR):.1f} MB")

    if RESET_OUTPUT and OUTPUT_DIR.exists():
        print(f"Đang xóa kết quả cũ: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = split_among_methods(image_paths)

    print("\nSố ảnh cho từng phương pháp:")
    for method, paths in groups.items():
        print(f"  {method}: {len(paths)} ảnh")

    success = {m: 0 for m in METHODS}
    failed = 0

    for method, paths in groups.items():
        print(f"\n--- {method} ---")
        for i, path in enumerate(paths, start=1):
            if process_image(path, method):
                success[method] += 1
                print(f"[{i}/{len(paths)}] {path.name}")
            else:
                failed += 1

    total_mb = folder_size_mb(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)
    for method in METHODS:
        print(f"{method}: {success[method]} ảnh | {folder_size_mb(OUTPUT_DIR / method):.1f} MB")
    print(f"Thất bại: {failed}")
    print(f"Tổng đã xử lý: {sum(success.values())} (mỗi ảnh chỉ 1 bộ lọc, không lặp)")
    print(f"Tổng dung lượng kết quả: {total_mb:.1f} MB")

    if total_mb > WARN_TOTAL_MB:
        print(f"\nCẢNH BÁO: Tổng dung lượng vượt {WARN_TOTAL_MB} MB.")
        print("Hãy giảm JPEG_QUALITY (ví dụ 70) hoặc MAX_SIDE (ví dụ 800) rồi chạy lại.")


if __name__ == "__main__":
    main()
