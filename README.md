# Đồ án Xử lý ảnh số – Nhóm 3
## Đề tài: Tăng Cường & Khử Nhiễu Ảnh Chụp Thiếu Sáng

Dự án nghiên cứu và hiện thực hóa các giải thuật tăng cường chất lượng ảnh số chụp trong môi trường thiếu sáng (Low-Light Image Enhancement), kết hợp các kỹ thuật cân bằng histogram, biến đổi gamma và khử nhiễu bảo toàn biên cạnh.

---

## 👥 Phân công thành viên

| Thành viên | Vai trò & Nhiệm vụ chính | Sản phẩm đầu ra |
|---|---|---|
| **Nguyễn Hoàng Khiêm** | Dữ liệu & Tiền xử lý | Bộ 76 ảnh, khảo sát độ sáng, vẽ histogram |
| **Trần Chí Vỹ** | Phương pháp tăng sáng | `src/enhance.py` (Gamma, Giãn histogram, CLAHE) |
| **Dương Tiến Anh Khoa** | Khử nhiễu & Pipeline | `src/denoise.py`, `src/pipeline.py` (2 Cấu hình) |
| **Vũ Minh Anh** | Đánh giá, Demo & Báo cáo | `src/metrics.py`, `src/gui.py`, `results/metrics.csv`, charts |

---

## 📂 Cấu trúc thư mục

```text
XLAS_Nhom3_AnhThieuSang/
├── data/                       # 76 ảnh thiếu sáng thực tế
├── src/
│   ├── __init__.py
│   ├── utils.py                # Xử lý và đọc ảnh, vẽ histogram (Khiêm)
│   ├── enhance.py              # Gamma, Giãn phân vị, CLAHE (Vỹ)
│   ├── denoise.py              # Gaussian, Median, Bilateral (Khoa)
│   ├── pipeline.py             # Ghép 2 cấu hình thực nghiệm (Khoa & Anh)
│   ├── metrics.py              # Tính Entropy, RMS Contrast, đo thời gian (Anh)
│   └── gui.py                  # Giao diện demo tương tác Streamlit (Anh)
├── results/
│   ├── config1/                # Kết quả Cấu hình 1 (76 ảnh)
│   ├── config2/                # Kết quả Cấu hình 2 (76 ảnh)
│   ├── charts/                 # Biểu đồ so sánh định lượng
│   ├── brightness_report.csv   # Thống kê độ sáng ảnh gốc (Khiêm)
│   └── metrics.csv             # Bảng so sánh chỉ số định lượng (Anh)
└── report/
    ├── slides.pptx             # Slide báo cáo
    └── sections/               # Các phần báo cáo chi tiết
        └── 04_danh_gia_demo.md # Phần đánh giá & demo (Anh)
