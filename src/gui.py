"""
Giao diện Demo Tăng sáng & Khử nhiễu ảnh thiếu sáng.
Người thực hiện: Vũ Minh Anh
Thư viện: Streamlit
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.metrics import compute_all_metrics

# ============================================================
# CÁC HÀM XỬ LÝ ẢNH (TĂNG SÁNG & KHỬ NHIỄU)
# ============================================================

def apply_gamma(img, gamma=0.5):
    """Tăng sáng bằng Gamma Correction: s = c * r^gamma."""
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.0, grid_size=8):
    """Tăng sáng và cân bằng tương phản bằng CLAHE trên kênh L (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_hist_stretch(img, low_pct=2, high_pct=98):
    """Giãn histogram theo percentile."""
    res = np.zeros_like(img)
    for c in range(3):
        low_val = np.percentile(img[:, :, c], low_pct)
        high_val = np.percentile(img[:, :, c], high_pct)
        if high_val > low_val:
            stretched = (img[:, :, c] - low_val) * (255.0 / (high_val - low_val))
            res[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            res[:, :, c] = img[:, :, c]
    return res


def apply_denoise(img, method, **params):
    """Khử nhiễu ảnh."""
    if method == "Gaussian Blur":
        k = params.get("k", 5)
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), params.get("sigma", 1.0))
    elif method == "Median Filter":
        k = params.get("k", 5)
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(img, k)
    elif method == "Bilateral Filter":
        d = params.get("d", 9)
        sc = params.get("sigma_color", 75)
        ss = params.get("sigma_space", 75)
        return cv2.bilateralFilter(img, d, sc, ss)
    return img


# ============================================================
# GIAO DIỆN CHÍNH (STREAMLIT)
# ============================================================

st.set_page_config(
    page_title="Xử Lý Ảnh Số – Tăng Sáng & Khử Nhiễu",
    page_icon="🔦",
    layout="wide"
)

st.title("🔦 Hệ Thống Tăng Cường & Đánh Giá Ảnh Thiếu Sáng")
st.caption("Đồ án **Xử lý ảnh số** – Nhóm XX | Phụ trách đánh giá & demo: **Vũ Minh Anh**")

# Sidebar: Chọn nguồn ảnh
st.sidebar.header("📁 1. Chọn nguồn dữ liệu")
source_option = st.sidebar.radio("Nguồn ảnh:", ["Chọn từ thư mục data/raw", "Tải ảnh từ máy tính"])

raw_img = None
ref_img = None
img_name = ""

if source_option == "Chọn từ thư mục data/raw":
    if os.path.exists("data/raw"):
        raw_files = [f for f in sorted(os.listdir("data/raw")) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if raw_files:
            img_name = st.sidebar.selectbox("Chọn ảnh mẫu:", raw_files)
            raw_path = os.path.join("data/raw", img_name)
            raw_img = cv2.imread(raw_path)

            ref_path = os.path.join("data/reference", img_name)
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
        else:
            st.sidebar.warning("Thư mục data/raw đang trống.")
    else:
        st.sidebar.error("Chưa có thư mục data/raw.")
else:
    uploaded_file = st.sidebar.file_uploader("Tải ảnh thiếu sáng:", type=["jpg", "png", "jpeg"])
    uploaded_ref = st.sidebar.file_uploader("Tải ảnh tham chiếu (tùy chọn, để đo PSNR/SSIM):", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_name = uploaded_file.name
    if uploaded_ref is not None:
        ref_bytes = np.frombuffer(uploaded_ref.read(), np.uint8)
        ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

# Sidebar: Tùy chọn xử lý
st.sidebar.header("⚙️ 2. Cấu hình xử lý")
pipeline_mode = st.sidebar.radio(
    "Chế độ:",
    ["Tùy chỉnh thủ công", "⚡ Cấu hình 1 (Gamma + Gaussian)", "⚡ Cấu hình 2 (CLAHE + Bilateral)"]
)

enhance_method = "Gamma"
denoise_method = "Không"

if pipeline_mode == "⚡ Cấu hình 1 (Gamma + Gaussian)":
    enhance_method = "Gamma"
    gamma_val = 0.5
    denoise_method = "Gaussian Blur"
    k_gauss = 5
    sigma_gauss = 1.0
elif pipeline_mode == "⚡ Cấu hình 2 (CLAHE + Bilateral)":
    enhance_method = "CLAHE"
    clip_val = 2.0
    grid_val = 8
    denoise_method = "Bilateral Filter"
    d_bilateral = 9
    sc_bilateral = 75
    ss_bilateral = 75
else:
    # Tùy chỉnh thủ công
    st.sidebar.subheader("Tăng sáng:")
    enhance_method = st.sidebar.selectbox("Thuật toán tăng sáng:", ["Gamma", "CLAHE", "Histogram Stretching", "Không"])
    if enhance_method == "Gamma":
        gamma_val = st.sidebar.slider("Hệ số Gamma (γ < 1 làm sáng):", 0.1, 2.0, 0.5, 0.05)
    elif enhance_method == "CLAHE":
        clip_val = st.sidebar.slider("Clip Limit:", 1.0, 10.0, 2.0, 0.5)
        grid_val = st.sidebar.slider("Tile Grid Size:", 4, 16, 8, 2)
    elif enhance_method == "Histogram Stretching":
        low_p = st.sidebar.slider("Percentile cận dưới (%):", 0, 10, 2, 1)
        high_p = st.sidebar.slider("Percentile cận trên (%):", 90, 100, 98, 1)

    st.sidebar.subheader("Khử nhiễu:")
    denoise_method = st.sidebar.selectbox("Thuật toán khử nhiễu:", ["Không", "Gaussian Blur", "Median Filter", "Bilateral Filter"])
    if denoise_method == "Gaussian Blur":
        k_gauss = st.sidebar.slider("Kernel Size:", 3, 15, 5, 2)
        sigma_gauss = st.sidebar.slider("Sigma:", 0.1, 5.0, 1.0, 0.1)
    elif denoise_method == "Median Filter":
        k_med = st.sidebar.slider("Kernel Size:", 3, 15, 5, 2)
    elif denoise_method == "Bilateral Filter":
        d_bilateral = st.sidebar.slider("Đường kính d:", 3, 15, 9, 2)
        sc_bilateral = st.sidebar.slider("Sigma Color:", 10, 150, 75, 5)
        ss_bilateral = st.sidebar.slider("Sigma Space:", 10, 150, 75, 5)


# ============================================================
# THỰC HIỆN XỬ LÝ & HIỂN THỊ
# ============================================================

if raw_img is not None:
    # 1. Pipeline xử lý
    start_time = time.time()

    # Bước 1: Tăng sáng
    if enhance_method == "Gamma":
        enhanced = apply_gamma(raw_img, gamma_val)
    elif enhance_method == "CLAHE":
        enhanced = apply_clahe(raw_img, clip_val, grid_val)
    elif enhance_method == "Histogram Stretching":
        enhanced = apply_hist_stretch(raw_img, low_p, high_p)
    else:
        enhanced = raw_img.copy()

    # Bước 2: Khử nhiễu
    if denoise_method == "Gaussian Blur":
        processed = apply_denoise(enhanced, "Gaussian Blur", k=k_gauss, sigma=sigma_gauss)
    elif denoise_method == "Median Filter":
        processed = apply_denoise(enhanced, "Median Filter", k=k_med)
    elif denoise_method == "Bilateral Filter":
        processed = apply_denoise(enhanced, "Bilateral Filter", d=d_bilateral, sigma_color=sc_bilateral, sigma_space=ss_bilateral)
    else:
        processed = enhanced.copy()

    elapsed_ms = (time.time() - start_time) * 1000

    # 2. Hiển thị ảnh so sánh
    st.subheader(f"🖼️ So sánh trực quan: `{img_name}` (Xử lý trong: {elapsed_ms:.1f} ms)")

    col1, col2, col3 = st.columns(3 if ref_img is not None else 2)

    with col1:
        st.markdown("**Ảnh gốc thiếu sáng**")
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.markdown(f"**Ảnh kết quả ({pipeline_mode})**")
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    if ref_img is not None:
        with col3:
            st.markdown("**Ảnh tham chiếu đủ sáng**")
            st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 3. Tính toán và hiển thị bảng chỉ số định lượng
    st.markdown("---")
    st.subheader("📊 Bảng chỉ số định lượng (Metrics)")

    raw_metrics = compute_all_metrics(raw_img, ref_img)
    proc_metrics = compute_all_metrics(processed, ref_img)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    # Entropy
    delta_entropy = proc_metrics['entropy'] - raw_metrics['entropy']
    mcol1.metric("Entropy (Lượng thông tin)", f"{proc_metrics['entropy']:.3f}", f"{delta_entropy:+.3f}")

    # Contrast
    delta_contrast = proc_metrics['contrast'] - raw_metrics['contrast']
    mcol2.metric("Contrast (Tương phản RMS)", f"{proc_metrics['contrast']:.2f}", f"{delta_contrast:+.2f}")

    # PSNR & SSIM
    if ref_img is not None:
        delta_psnr = proc_metrics.get('psnr', 0) - raw_metrics.get('psnr', 0)
        mcol3.metric("PSNR (Độ trung thực)", f"{proc_metrics.get('psnr', 0):.2f} dB", f"{delta_psnr:+.2f} dB")

        delta_ssim = proc_metrics.get('ssim', 0) - raw_metrics.get('ssim', 0)
        mcol4.metric("SSIM (Tương đồng cấu trúc)", f"{proc_metrics.get('ssim', 0):.4f}", f"{delta_ssim:+.4f}")
    else:
        mcol3.info("Cần ảnh tham chiếu để tính PSNR")
        mcol4.info("Cần ảnh tham chiếu để tính SSIM")

    # 4. Biểu đồ Histogram
    with st.expander("📈 Xem biểu đồ Histogram mức xám (Trước vs Sau)"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        ax[0].hist(gray_raw.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.8)
        ax[0].set_title("Histogram ảnh gốc (Dồn về bên trái - tối)")
        ax[0].set_xlim([0, 256])

        ax[1].hist(gray_proc.ravel(), bins=256, range=[0, 256], color='steelblue', alpha=0.8)
        ax[1].set_title("Histogram sau xử lý (Phân bố đều hơn)")
        ax[1].set_xlim([0, 256])

        st.pyplot(fig)

else:
    st.info("👈 Hãy chọn hoặc tải ảnh ở thanh bên trái để bắt đầu demo!")
