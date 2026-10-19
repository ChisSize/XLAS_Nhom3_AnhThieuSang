
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Khai báo đường dẫn
# =========================
# Sửa đường dẫn này theo thư mục bạn đã giải nén
SRC_DIR = r"D:\XLAS\XLAS_Nhom3_AnhThieuSang-main\XLAS_Nhom3_AnhThieuSang-main\data"

# Thư mục lưu kết quả
OUTPUT_DIR = r"D:\XLAS\KetQua_KhuNhieu"

# =========================
# 2. Kiểm tra thư mục ảnh
# =========================
if not os.path.isdir(SRC_DIR):
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh: {SRC_DIR}\n"
        "Hãy kiểm tra lại đường dẫn sau khi giải nén ZIP."
    )

image_paths = []

# Hỗ trợ JPG, JPEG, PNG và chữ hoa/chữ thường
for ext in ("*.jpg", "*.jpeg", "*.png",
            "*.JPG", "*.JPEG", "*.PNG"):
    image_paths.extend(
        glob.glob(os.path.join(SRC_DIR, ext))
    )

image_paths = sorted(set(image_paths))

print("Số ảnh tìm thấy:", len(image_paths))

if not image_paths:
    raise FileNotFoundError(
        "Không tìm thấy ảnh trực tiếp trong thư mục src."
    )

# Tạo thư mục kết quả riêng cho từng phương pháp
method_dirs = {
    "brightness": os.path.join(OUTPUT_DIR, "brightness"),
    "gaussian": os.path.join(OUTPUT_DIR, "gaussian"),
    "median": os.path.join(OUTPUT_DIR, "median"),
    "bilateral": os.path.join(OUTPUT_DIR, "bilateral")
}

for folder in method_dirs.values():
    os.makedirs(folder, exist_ok=True)

# =========================
# 3. Gamma Correction
# =========================
gamma = 0.5

lookup_table = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in range(256)
]).clip(0, 255).astype(np.uint8)

# =========================
# 4. Xử lý toàn bộ ảnh
# =========================
success_count = 0
failed_files = []
first_result = None

for index, image_path in enumerate(image_paths, start=1):

    # Đọc ảnh
    image = cv2.imread(image_path)

    if image is None:
        failed_files.append(image_path)
        print("Không đọc được:", image_path)
        continue

    # 4.1. Tăng sáng bằng Gamma Correction
    bright_image = cv2.LUT(image, lookup_table)

    # 4.2. Gaussian Blur
    gaussian_image = cv2.GaussianBlur(
        bright_image,
        (5, 5),
        sigmaX=1.0
    )

    # 4.3. Median Blur
    median_image = cv2.medianBlur(
        bright_image,
        5
    )

    # 4.4. Bilateral Filter
    bilateral_image = cv2.bilateralFilter(
        bright_image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # Tên file gốc, giữ nguyên phần mở rộng
    filename = os.path.basename(image_path)

    # Lưu kết quả
    results = {
        "brightness": bright_image,
        "gaussian": gaussian_image,
        "median": median_image,
        "bilateral": bilateral_image
    }

    for method, result in results.items():
        save_path = os.path.join(
            method_dirs[method],
            filename
        )

        if not cv2.imwrite(save_path, result):
            print("Lỗi lưu ảnh:", save_path)

    # Lưu kết quả ảnh đầu tiên để hiển thị so sánh
    if first_result is None:
        first_result = [
            ("Original", image),
            ("Gamma Correction", bright_image),
            ("Gaussian Blur", gaussian_image),
            ("Median Blur", median_image),
            ("Bilateral Filter", bilateral_image)
        ]

    success_count += 1
    print(f"Đã xử lý: {index}/{len(image_paths)}")

# =========================
# 5. Tổng kết
# =========================
print("\n========== KẾT QUẢ ==========")
print("Tổng số ảnh:", len(image_paths))
print("Xử lý thành công:", success_count)
print("Không đọc được:", len(failed_files))
print("Thư mục kết quả:", OUTPUT_DIR)

# =========================
# 6. Hiển thị kết quả ảnh đầu tiên
# =========================
if first_result is not None:
    plt.figure(figsize=(18, 9))

    for i, (title, img) in enumerate(first_result, start=1):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 3, i)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
