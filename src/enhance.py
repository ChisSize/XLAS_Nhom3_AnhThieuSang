import cv2
import numpy as np


def gamma_correction(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Tăng sáng bằng gamma correction. Gamma < 1 làm sáng vùng tối."""
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def histogram_stretch(img: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Giãn histogram về [0, 255] theo percentile, bỏ qua điểm nhiễu ở hai đầu."""
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return img.copy()
    out = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Cân bằng histogram thích ứng theo ô. Ảnh màu được xử lý trên kênh L của LAB."""
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe_op.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return clahe_op.apply(img)

# Ghi chú: CLAHE chỉ tăng sáng kênh L (độ sáng) trong không gian LAB, nên màu không bị lệch. Nếu áp trực tiếp lên từng kênh BGR sẽ làm đổi màu ảnh.

if __name__ == "__main__":
    import glob
    import os

    paths = glob.glob("data/raw/*.jpg") + glob.glob("data/raw/*.png")
    if not paths:
        print("Chưa có ảnh trong data/raw/")
    for p in paths[:3]:
        img = cv2.imread(p)
        if img is None:
            print("Không đọc được:", p)
            continue
        print(os.path.basename(p), img.shape)
        print("  gamma:", gamma_correction(img, 0.6).shape)
        print("  stretch:", histogram_stretch(img).shape)
        print("  clahe:", clahe(img).shape)