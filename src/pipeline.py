
from pathlib import Path
import sys


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# Đường dẫn thư mục gốc của repository
BASE_DIR = Path(
    r"D:\XLAS\XLAS_Nhom3_AnhThieuSang-main"
    r"\XLAS_Nhom3_AnhThieuSang-main"
)

SRC_DIR = BASE_DIR / "src"

# Thư mục ảnh đầu vào và kết quả
INPUT_DIR = BASE_DIR / "data" / "Data"
ENHANCE_DIR = BASE_DIR / "results" / "enhance"
PIPELINE_DIR = ENHANCE_DIR / "pipeline"
DENOISE_DIR = BASE_DIR / "results" / "denoise"

IMG_EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


# =========================================================
# 2. KIỂM TRA CẤU HÌNH
# =========================================================

def check_configuration():
    print("=" * 60)
    print("KIỂM TRA CẤU HÌNH PIPELINE")
    print("=" * 60)

    print(f"Thư mục gốc:       {BASE_DIR}")
    print(f"Thư mục code:      {SRC_DIR}")
    print(f"Ảnh đầu vào:       {INPUT_DIR}")
    print(f"Kết quả tăng sáng: {PIPELINE_DIR}")
    print(f"Kết quả khử nhiễu: {DENOISE_DIR}")

    if not BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục gốc repository: {BASE_DIR}"
        )

    if not SRC_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục src: {SRC_DIR}"
        )

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh đầu vào: {INPUT_DIR}"
        )

    enhance_file = SRC_DIR / "enhance.py"
    denoise_file = SRC_DIR / "Khu_Nhieu.py"

    if not enhance_file.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file tăng sáng: {enhance_file}"
        )

    if not denoise_file.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file khử nhiễu: {denoise_file}"
        )


# =========================================================
# 3. ĐẾM ẢNH
# =========================================================

def get_image_paths(folder):
    folder = Path(folder)

    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )



# =========================================================
# 4. BƯỚC 1: TĂNG SÁNG
# =========================================================

def run_enhance():
    print("\n" + "=" * 60)
    print("BƯỚC 1: TĂNG SÁNG ẢNH")
    print("=" * 60)

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import enhance

    # Gọi trực tiếp hàm xử lý có sẵn trong enhance.py
    # Không cần sửa file enhance.py
    enhance.run_all_images(
        INPUT_DIR,
        ENHANCE_DIR
    )

    images = get_image_paths(PIPELINE_DIR)

    if not images:
        raise RuntimeError(
            f"Không tìm thấy ảnh sau tăng sáng tại: {PIPELINE_DIR}"
        )

    print(f"\nTăng sáng hoàn tất: {len(images)} ảnh.")
    print(f"Kết quả lưu tại: {PIPELINE_DIR}")

    return len(images)


# =========================================================
# 5. BƯỚC 2: KHỬ NHIỄU
# =========================================================

def run_denoise():
    print("\n" + "=" * 60)
    print("BƯỚC 2: KHỬ NHIỄU ẢNH")
    print("=" * 60)

    images = get_image_paths(PIPELINE_DIR)

    if not images:
        raise RuntimeError(
            f"Không có ảnh để khử nhiễu tại: {PIPELINE_DIR}"
        )

    print(f"Số ảnh đưa vào khử nhiễu: {len(images)}")

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import Khu_Nhieu

    # Đặt đúng đường dẫn cho module khử nhiễu
    Khu_Nhieu.BASE_DIR = BASE_DIR
    Khu_Nhieu.SRC_DIR = PIPELINE_DIR
    Khu_Nhieu.OUTPUT_DIR = DENOISE_DIR

    Khu_Nhieu.main()

    print("\nKhử nhiễu hoàn tất.")
    print(f"Kết quả lưu tại: {DENOISE_DIR}")



# =========================================================
# 6. CHẠY TOÀN BỘ PIPELINE
# =========================================================

def main():
    try:
        check_configuration()

        # Bước 1: Tăng sáng
        run_enhance()

        # Bước 2: Khử nhiễu
        run_denoise()

        print("\n" + "=" * 60)
        print("PIPELINE HOÀN TẤT")
        print("=" * 60)
        print(f"Ảnh sau tăng sáng: {PIPELINE_DIR}")
        print(f"Ảnh sau khử nhiễu: {DENOISE_DIR}")

    except Exception as error:
        print("\n" + "=" * 60)
        print("PIPELINE GẶP LỖI")
        print("=" * 60)
        print(error)
        raise


if __name__ == "__main__":
    main()
