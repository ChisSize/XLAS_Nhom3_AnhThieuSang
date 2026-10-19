from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Xác định đường dẫn project và thư mục data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"  # Thư mục lưu kết quả (đã có sẵn trong cấu trúc dự án)

# Tạo thư mục results nếu chưa tồn tại
RESULTS_DIR.mkdir(exist_ok=True)

# Định dạng các đuôi ảnh cần lấy
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

def analyze_and_plot_image(file_path, save_plot=True):
    try:
        # Đọc file thành mảng byte để xử lý đường dẫn tiếng Việt có dấu
        img_array = np.fromfile(str(file_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        img_bgr = None

    if img_bgr is None:
        return None
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Tính độ sáng trung bình (Grayscale Mean Brightness)
    mean_brightness = np.mean(gray)
    
    if save_plot:
        # Vẽ biểu đồ Histogram
        plt.figure(figsize=(12, 5))
        
        # 1. Grayscale Histogram
        plt.subplot(1, 2, 1)
        plt.hist(gray.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.75)
        plt.axvline(x=mean_brightness, color='r', linestyle='--', label=f'Mean = {mean_brightness:.2f}')
        plt.title(f'Grayscale Histogram - {file_path.name}')
        plt.xlabel('Pixel Intensity (0-255)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 2. RGB Histogram
        plt.subplot(1, 2, 2)
        colors = ('b', 'g', 'r')
        labels = ('Blue', 'Green', 'Red')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=labels[i])
        plt.title(f'RGB Histogram - {file_path.name}')
        plt.xlabel('Pixel Intensity (0-255)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        
        # Lưu biểu đồ Histogram vào thư mục results/
        plot_path = RESULTS_DIR / f"hist_{file_path.stem}.png"
        plt.savefig(plot_path)
        plt.close()  # Đóng plot để giải phóng RAM
        
    return mean_brightness

def process_all_images(data_folder):
    """
    Duyệt qua tất cả ảnh trong thư mục data, tính toán và xuất kết quả ra file CSV.
    """
    report_data = []
    
    # Tìm tất cả tệp ảnh trong thư mục data
    image_files = [f for f in data_folder.glob('*') if f.suffix.lower() in IMAGE_EXTENSIONS]
    
    print(f"Bắt đầu xử lý {len(image_files)} ảnh trong thư mục data...")
    
    for file_path in image_files:
        mean_b = analyze_and_plot_image(file_path, save_plot=True)
        
        if mean_b is not None:
            report_data.append({
                'filename': file_path.name,
                'mean_brightness': round(mean_b, 2),
                'status': 'Low-Light' if mean_b < 50 else 'Normal/Bright'
            })
            print(f"✔ [Xử lý xong] {file_path.name} | Độ sáng TB: {mean_b:.2f}")

    # Xuất kết quả thống kê ra tệp CSV để làm báo cáo
    csv_file = RESULTS_DIR / "brightness_report.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'mean_brightness', 'status'])
        writer.writeheader()
        writer.writerows(report_data)
        
    print(f"\n Hoàn thành! Đã lưu báo cáo tại: {csv_file}")
    print(f"Biểu đồ Histogram của từng ảnh được lưu trong: {RESULTS_DIR}")

# Chạy chương trình
if __name__ == "__main__":
    process_all_images(DATA_DIR)