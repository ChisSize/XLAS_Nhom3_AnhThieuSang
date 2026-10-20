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
# 4. KHỬ NHIỄU
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


def process_image(image_path, method):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return False

    relative_path = image_path.relative_to(SRC_DIR)
    save_path = OUTPUT_DIR / method / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    result = denoise(image, method)

    if not cv2.imwrite(str(save_path), result):
        print(f"Lỗi lưu ảnh: {save_path}")
        return False
    return True


# =========================================================
# 5. CHẠY
# =========================================================

def main():
    print("=" * 60)
    print(f"Ảnh đầu vào: {SRC_DIR}")
    print(f"Ảnh đầu ra:  {OUTPUT_DIR}")
    print("=" * 60)

    if not SRC_DIR.is_dir():
        print("LỖI: Không tìm thấy thư mục results/enhance/pipeline.")
        return

    image_paths = get_image_paths(SRC_DIR)
    if not image_paths:
        print("Không tìm thấy ảnh trong results/enhance/pipeline.")
        return

    print(f"Tổng số ảnh tìm thấy: {len(image_paths)}")

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

    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)
    for method in METHODS:
        print(f"{method}: {success[method]} ảnh")
    print(f"Thất bại: {failed}")
    print(f"Tổng đã xử lý: {sum(success.values())} (mỗi ảnh chỉ 1 bộ lọc, không lặp)")


if __name__ == "__main__":
    main()
