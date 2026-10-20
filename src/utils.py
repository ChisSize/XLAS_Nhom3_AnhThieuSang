from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Đường dẫn project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Thư mục con cho biểu đồ histogram
HIST_LOW_DIR = RESULTS_DIR / "histograms" / "low_light"
HIST_NORMAL_DIR = RESULTS_DIR / "histograms" / "normal_bright"

# Tạo sẵn các thư mục
for folder in (HIST_LOW_DIR, HIST_NORMAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
LOW_LIGHT_THRESHOLD = 50


def read_image(file_path):
    try:
        img_array = np.fromfile(str(file_path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def plot_histogram(img_bgr, gray, mean_brightness, file_path, save_path):
    plt.figure(figsize=(12, 5))

    # 1. Grayscale Histogram
    plt.subplot(1, 2, 1)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.75)
    plt.axvline(x=mean_brightness, color='r', linestyle='--',
                label=f'Mean = {mean_brightness:.2f}')
    plt.title(f'Grayscale Histogram - {file_path.name}')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.legend()

    # 2. RGB Histogram
    plt.subplot(1, 2, 2)
    for i, (col, label) in enumerate(zip(('b', 'g', 'r'), ('Blue', 'Green', 'Red'))):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=label)
    plt.title(f'RGB Histogram - {file_path.name}')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_all_images(data_folder):
    report_data = []

    image_files = sorted(
        f for f in data_folder.glob('*') if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Bắt đầu xử lý {len(image_files)} ảnh trong thư mục data...")

    for file_path in image_files:
        img_bgr = read_image(file_path)
        if img_bgr is None:
            print(f"✘ Không đọc được: {file_path.name}")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))

        if mean_b < LOW_LIGHT_THRESHOLD:
            status, hist_dir = 'Low-Light', HIST_LOW_DIR
        else:
            status, hist_dir = 'Normal/Bright', HIST_NORMAL_DIR

        plot_histogram(
            img_bgr, gray, mean_b, file_path,
            save_path=hist_dir / f"hist_{file_path.stem}.png",
        )

        report_data.append({
            'filename': file_path.name,
            'mean_brightness': round(mean_b, 2),
            'status': status,
        })
        print(f"✔ [Xử lý xong] {file_path.name} | Độ sáng TB: {mean_b:.2f} | {status}")

    csv_file = RESULTS_DIR / "brightness_report.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'mean_brightness', 'status'])
        writer.writeheader()
        writer.writerows(report_data)

    print(f"\nHoàn thành! Báo cáo: {csv_file}")
    print(f"Biểu đồ ít sáng: {HIST_LOW_DIR}")
    print(f"Biểu đồ đủ sáng: {HIST_NORMAL_DIR}")


if __name__ == "__main__":
    process_all_images(DATA_DIR)