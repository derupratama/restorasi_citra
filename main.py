import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# ==========================================
# 0. Folder Output
# ==========================================
os.makedirs("output/landscape", exist_ok=True)
os.makedirs("output/portrait", exist_ok=True)

# ==========================================
# 1. RGB → Grayscale
# ==========================================
def rgb_to_grayscale(img_arr):
    r = img_arr[..., 0].astype(np.float32)
    g = img_arr[..., 1].astype(np.float32)
    b = img_arr[..., 2].astype(np.float32)
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

# ==========================================
# 2. Noise
# ==========================================
def add_salt_and_pepper(img, prob):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob / 2] = 255
    noisy[(rnd >= prob / 2) & (rnd < prob)] = 0
    return noisy

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

# ==========================================
# 3. Padding & Convolution
# ==========================================
def pad_image(img, pad):
    return np.pad(img, ((pad, pad), (pad, pad)), mode="edge")

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    pad = kh // 2
    padded = pad_image(img, pad)
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    for y in range(H):
        for x in range(W):
            region = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(region * kernel)

    return np.clip(out, 0, 255).astype(np.uint8)

# ==========================================
# 4. Filter
# ==========================================
def min_filter(img, ksize=3):
    pad = ksize // 2
    padded = pad_image(img, pad)
    H, W = img.shape
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            out[y, x] = np.min(padded[y:y+ksize, x:x+ksize])
    return out

def max_filter(img, ksize=3):
    pad = ksize // 2
    padded = pad_image(img, pad)
    H, W = img.shape
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            out[y, x] = np.max(padded[y:y+ksize, x:x+ksize])
    return out

def median_filter(img, ksize=3):
    pad = ksize // 2
    padded = pad_image(img, pad)
    H, W = img.shape
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            out[y, x] = np.median(padded[y:y+ksize, x:x+ksize])
    return out

def mean_filter(img, ksize=3):
    kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    return convolve2d(img, kernel)

# ==========================================
# 5. MSE
# ==========================================
def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# ==========================================
# 6. Histogram MSE + Angka
# ==========================================
def plot_mse_histogram(mse_data, title):
    noise_types = list(mse_data.keys())
    filter_names = list(next(iter(mse_data.values())).keys())

    x = np.arange(len(noise_types))
    width = 0.2

    plt.figure(figsize=(11, 5))

    for i, filt in enumerate(filter_names):
        values = [mse_data[n][filt] for n in noise_types]
        bars = plt.bar(x + i * width, values, width, label=filt)

        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    plt.xlabel("Jenis Noise")
    plt.ylabel("MSE")
    plt.title(title)
    plt.xticks(x + width * 1.5, noise_types, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. Proses Citra
# ==========================================
def process_image(path, out_folder):
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = rgb_to_grayscale(rgb)

    sp_levels = [0.01, 0.05]
    gauss_levels = [10, 30]

    noise_gray = {}
    for lv in sp_levels:
        noise_gray[f"SP {lv}"] = add_salt_and_pepper(gray, lv)
    for lv in gauss_levels:
        noise_gray[f"Gauss {lv}"] = add_gaussian_noise(gray, lv)

    filters = {
        "Min": min_filter,
        "Max": max_filter,
        "Median": median_filter,
        "Mean": mean_filter
    }

    mse_data = {}

    print(f"\n=== MSE untuk {out_folder} ===")
    for noise_name, noisy_img in noise_gray.items():
        mse_data[noise_name] = {}
        for filt_name, filt_func in filters.items():
            filtered = filt_func(noisy_img, 3)
            error = mse(gray, filtered)
            mse_data[noise_name][filt_name] = error
            print(f"{noise_name} + {filt_name} → MSE = {error:.2f}")

    plot_mse_histogram(mse_data, f"Histogram MSE ({out_folder})")

# ==========================================
# 8. Jalankan
# ==========================================
process_image("Gambar Landscape.jpg", "output/landscape")
process_image("Gambar Potrait.jpg", "output/portrait")
