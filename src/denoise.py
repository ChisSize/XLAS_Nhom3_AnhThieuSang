import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# =========================
# 1. Khai báo đường dẫn
# =========================
SRC_DIR = r"D:\XLAS\XLAS_Nhom3_AnhThieuSang-main\XLAS_Nhom3_AnhThieuSang-main\data"
OUTPUT_DIR = r"D:\XLAS\KetQua_KhuNhieu"

SHUFFLE = True      # True: xáo trộn ngẫu nhiên trước khi chia; False: chia theo thứ tự tên file
SEED = 42

# =========================
# 2. Đọc danh sách ảnh
# =========================
if not os.path.isdir(SRC_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {SRC_DIR}")

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    image_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))

image_paths = sorted(set(image_paths))
print("Số ảnh tìm thấy:", len(image_paths))

if not image_paths:
    raise FileNotFoundError("Không tìm thấy ảnh trong thư mục data.")

if SHUFFLE:
    random.Random(SEED).shuffle(image_paths)

# =========================
# 3. Chia đều thành 4 nhóm
# =========================
methods = ["brightness", "gaussian", "median", "bilateral"]
groups = np.array_split(np.array(image_paths), 4)   # 76 ảnh -> 4 nhóm x 19 ảnh

method_dirs = {}
for m in methods:
    method_dirs[m] = os.path.join(OUTPUT_DIR, m)
    os.makedirs(method_dirs[m], exist_ok=True)

for m, g in zip(methods, groups):
    print(f"{m}: {len(g)} ảnh")

# =========================
# 4. Gamma Correction
# =========================
gamma = 0.5
lookup_table = np.array([
    ((i / 255.0) ** gamma) * 255 for i in range(256)
]).clip(0, 255).astype(np.uint8)

# =========================
# 5. Hàm xử lý theo phương pháp
# =========================
def process(image, method):
    bright = cv2.LUT(image, lookup_table)      # tăng sáng trước
    if method == "brightness":
        return bright
    if method == "gaussian":
        return cv2.GaussianBlur(bright, (5, 5), sigmaX=1.0)
    if method == "median":
        return cv2.medianBlur(bright, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(bright, d=9, sigmaColor=75, sigmaSpace=75)

# =========================
# 6. Xử lý + lưu
# =========================
failed_files = []
samples = {}      # lưu 1 ảnh mẫu mỗi nhóm để hiển thị
success = {m: 0 for m in methods}

for method, group in zip(methods, groups):
    for path in group:
        image = cv2.imread(str(path))
        if image is None:
            failed_files.append(str(path))
            print("Không đọc được:", path)
            continue

        result = process(image, method)
        save_path = os.path.join(method_dirs[method], os.path.basename(str(path)))

        if not cv2.imwrite(save_path, result):
            print("Lỗi lưu ảnh:", save_path)
            continue

        success[method] += 1
        if method not in samples:
            samples[method] = (image, result)

    print(f"Xong nhóm {method}: {success[method]}/{len(group)}")

# =========================
# 7. Tổng kết
# =========================
print("\n========== KẾT QUẢ ==========")
print("Tổng số ảnh:", len(image_paths))
print("Xử lý thành công:", sum(success.values()))
print("Không đọc được:", len(failed_files))
print("Thư mục kết quả:", OUTPUT_DIR)

# =========================
# 8. Hiển thị mẫu: hàng trên = gốc, hàng dưới = kết quả
# =========================
titles = {
    "brightness": "Gamma Correction",
    "gaussian": "Gaussian Blur",
    "median": "Median Blur",
    "bilateral": "Bilateral Filter",
}

plt.figure(figsize=(18, 9))
for i, m in enumerate(methods, start=1):
    if m not in samples:
        continue
    orig, res = samples[m]

    plt.subplot(2, 4, i)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Original ({m})")
    plt.axis("off")

    plt.subplot(2, 4, i + 4)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title(titles[m])
    plt.axis("off")

plt.tight_layout()
plt.show()
