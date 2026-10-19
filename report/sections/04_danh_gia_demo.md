# Phần 4: Đánh giá Thực nghiệm và Phát triển Ứng dụng Demo

**Người thực hiện:** Vũ Minh Anh  
**Đề tài:** Tăng cường sáng và khử nhiễu ảnh chụp trong điều kiện thiếu sáng

---

## 4.1. Các chỉ số đánh giá chất lượng ảnh (Evaluation Metrics)

Để đánh giá toàn diện hiệu năng của các thuật toán, nhóm sử dụng kết hợp các độ đo không tham chiếu (no-reference) và có tham chiếu (full-reference):

- **Entropy (Độ phong phú thông tin):** Đo lường mức độ chi tiết và lượng thông tin chứa đựng trong ảnh:
  $$\text{Entropy} = -\sum_{i=0}^{255} p_i \log_2(p_i)$$
  *(Trong đó $p_i$ là xác suất xuất hiện của mức xám $i$; Entropy càng cao biểu thị ảnh càng chứa nhiều chi tiết rõ ràng).*

- **Contrast (Độ tương phản RMS):** Đo độ lệch chuẩn mức sáng điểm ảnh so với giá trị sáng trung bình:
  $$C_{\text{RMS}} = \sqrt{\frac{1}{M \times N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} \left(I(x, y) - \bar{I}\right)^2}$$
  *(Trong đó $\bar{I}$ là mức xám trung bình của toàn bộ bức ảnh kích thước $M \times N$).*

- **PSNR (Peak Signal-to-Noise Ratio):** Tỷ lệ tín hiệu đỉnh trên nhiễu so với ảnh tham chiếu (Ground Truth):
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right), \quad \text{MSE} = \frac{1}{M \times N}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} [I(x, y) - K(x, y)]^2$$
  *(PSNR càng lớn chứng tỏ ảnh kết quả càng ít biến dạng so với ảnh tham chiếu chuẩn).*

- **SSIM (Structural Similarity Index Measure):** Đánh giá độ tương đồng cấu trúc theo cảm nhận thị giác người:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  *(SSIM có giá trị trong đoạn $[0, 1]$, giá trị càng gần 1 thì cấu trúc hình học và độ tương phản càng tương đồng).*

## 4.2. Bảng so sánh kết quả thực nghiệm giữa 2 cấu hình

Kết quả trung bình đánh giá trên tập kiểm thử gồm 60 ảnh thiếu sáng và ảnh tham chiếu:

| Bộ dữ liệu / Cấu hình | Entropy (bit) | Contrast (RMS) | PSNR (dB) | SSIM | Thời gian xử lý (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ảnh gốc thiếu sáng (Input)** | 5.12 | 24.30 | 14.20 | 0.512 | — |
| **Cấu hình 1 (Gamma + Gaussian)** | 6.84 | 46.15 | 19.85 | 0.748 | **18.5 ms** |
| **Cấu hình 2 (CLAHE + Bilateral)** | **7.42** | **58.70** | **23.40** | **0.841** | 62.1 ms |
| **Ảnh tham chiếu (Reference)** | 7.65 | 62.40 | $\infty$ | 1.000 | — |

<!-- [Chèn biểu đồ so sánh: Biểu đồ cột thể hiện mức độ cải thiện của Entropy, PSNR và SSIM giữa 2 cấu hình] -->

*Nhận xét:* Cấu hình 2 vượt trội hơn về mọi mặt chất lượng hình ảnh (SSIM đạt 0.841 so với 0.748 của Cấu hình 1). Tuy nhiên, Cấu hình 1 đạt ưu thế lớn về thời gian tính toán, nhanh gấp hơn 3 lần so với Cấu hình 2.

## 4.3. Mô tả ứng dụng giao diện trực quan (GUI Demo Streamlit)

Ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit** nhằm trực quan hóa các chức năng:

- **Tải ảnh linh hoạt:** Cho phép người dùng tải lên ảnh bất kỳ hoặc chọn ảnh từ tập dữ liệu mẫu (`data/raw/`).
- **Lựa chọn chế độ linh hoạt:** Hỗ trợ chạy nhanh theo cấu hình định sẵn (Config 1 / Config 2) hoặc chế độ tùy chỉnh thủ công các tham số ($\gamma$, clipLimit, $\sigma$, ...).
- **So sánh trực quan đa góc nhìn (Side-by-side):** Hiển thị song song ảnh gốc, ảnh đã xử lý và ảnh tham chiếu kèm theo biểu đồ histogram động.
- **Bảng điểm chỉ số tức thì:** Tự động tính và hiển thị bảng đối chiếu Entropy, Contrast, PSNR, SSIM ngay sau khi xử lý xong.

<!-- [Chèn ảnh chụp màn hình giao diện ứng dụng Streamlit: Chế độ so sánh side-by-side và bảng metrics trực tiếp] -->

## 4.4. Kết luận và Hướng phát triển

- **Kết luận:**
  - Nhóm đã hoàn thành nghiên cứu lý thuyết và hiện thực hóa thành công các giải pháp tăng sáng kết hợp khử nhiễu.
  - Cấu hình 2 (CLAHE kết hợp Bilateral Filter) cho chất lượng thị giác vượt trội, các chi tiết được bộc lộ rõ ràng mà không làm mờ biên hay nhiễu màu.
- **Hướng phát triển tiếp theo:**
  - Mở rộng ứng dụng các mô hình học sâu hiện đại như RetinexNet hoặc Zero-DCE để tăng cường khả năng thích ứng trong môi trường ánh sáng phức tạp.
  - Tối ưu hóa thuật toán bằng CUDA / C++ để xử lý luồng video thời gian thực từ camera an ninh ban đêm.
