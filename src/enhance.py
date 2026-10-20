"""
Kỹ thuật cải thiện (phần của Trần Chí Vỹ):
  1. Gamma correction
  2. Giãn histogram theo percentile (mặc định 2–98%)
  3. CLAHE (áp dụng trên kênh L của LAB để không làm lệch màu)

Cách dùng:
    from enhance_all import run_all_images
    run_all_images("data/raw_images", "output", params={"gamma": 0.5})

Kết quả:
    output/gamma/      ảnh sau gamma
    output/stretch/    ảnh sau giãn histogram
    output/clahe/      ảnh sau CLAHE
    output/pipeline/   ảnh sau cả 3 bước (gamma -> giãn -> CLAHE)
    output/summary.csv bảng tham số và độ sáng trung bình trước/sau
"""

from pathlib import Path
import csv
import sys
import cv2
import numpy as np

# Tham số mặc định. Có thể truyền dict khác vào run_all_images để ghi đè.
DEFAULT_PARAMS = {
    "gamma": 0.6,          # < 1 làm sáng vùng tối, > 1 làm tối
    "low_pct": 2,          # percentile dưới cho giãn histogram
    "high_pct": 98,        # percentile trên cho giãn histogram
    "clahe_clip": 2.0,     # clipLimit của CLAHE
    "clahe_grid": (8, 8),  # tileGridSize của CLAHE
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------- đọc / ghi an toàn với đường dẫn tiếng Việt trên Windows ----------
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Không ghi được ảnh: {path}")
    buf.tofile(str(path))


# ---------- 3 kỹ thuật ----------
def gamma_correction(img, gamma):
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def percentile_stretch(img, low_pct, high_pct):
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c].astype(np.float32)
        lo = np.percentile(channel, low_pct)
        hi = np.percentile(channel, high_pct)
        if hi <= lo:  # ảnh phẳng, không giãn được
            out[:, :, c] = img[:, :, c]
            continue
        stretched = (channel - lo) / (hi - lo) * 255.0
        out[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
    return out


def clahe_lab(img, clip, grid):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------- chạy hàng loạt ----------
def run_all_images(input_dir, output_dir, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"Không có ảnh nào trong {input_dir}")

    rows = []
    for i, f in enumerate(files, start=1):
        img = read_image(f)
        if img is None:
            print(f"[{i}/{len(files)}] Bỏ qua (không đọc được): {f.name}")
            continue

        g = gamma_correction(img, p["gamma"])
        s = percentile_stretch(img, p["low_pct"], p["high_pct"])
        c = clahe_lab(img, p["clahe_clip"], p["clahe_grid"])
        full = clahe_lab(
            percentile_stretch(gamma_correction(img, p["gamma"]),
                               p["low_pct"], p["high_pct"]),
            p["clahe_clip"], p["clahe_grid"])

        name = f.stem + ".png"
        write_image(output_dir / "gamma" / name, g)
        write_image(output_dir / "stretch" / name, s)
        write_image(output_dir / "clahe" / name, c)
        write_image(output_dir / "pipeline" / name, full)

        rows.append({
            "file": f.name,
            "brightness_before": round(float(img.mean()), 2),
            "brightness_gamma": round(float(g.mean()), 2),
            "brightness_stretch": round(float(s.mean()), 2),
            "brightness_clahe": round(float(c.mean()), 2),
            "brightness_pipeline": round(float(full.mean()), 2),
        })
        print(f"[{i}/{len(files)}] Xong: {f.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nHoàn thành {len(rows)} ảnh. Kết quả lưu tại: {output_dir.resolve()}")
    print("Tham số đã dùng:", p)
    return rows

    
if __name__ == "__main__":
    

    ROOT = Path(__file__).resolve().parent.parent   # thư mục gốc project
    INPUT = ROOT / "data" / "Data"            # thư mục chứa ảnh thiếu sáng
    OUTPUT = ROOT / "results" / "enhance"           # nơi lưu kết quả

    if not INPUT.exists():
        print(f"Không thấy thư mục ảnh: {INPUT}")
        print("Hãy tạo thư mục đó và copy ảnh vào, hoặc sửa dòng INPUT.")
        sys.exit(1)

    run_all_images(INPUT, OUTPUT)