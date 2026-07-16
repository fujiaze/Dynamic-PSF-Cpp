# -*- coding: utf-8 -*-
"""快速诊断: 图像值域 + 边缘 vs 中心背景值 + PSF 拟合失败根因"""
import sys
import os
import numpy as np
from astropy.io import fits

FITS_PATH = r"f:\Astro dev\Astro CS Normalization Database\testdata\results\Galaxy_Center_T4\panel1\Red\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red\01_calibrated.fits"

with fits.open(FITS_PATH, mode='readonly') as hdul:
    data = hdul[0].data.astype(np.float32)

h, w = data.shape
print("图像尺寸: %dx%d" % (w, h))
print("dtype: %s" % data.dtype)
print("值域: [%.4f, %.4f]" % (data.min(), data.max()))
print("中位数: %.4f" % np.median(data))
print("均值: %.4f" % data.mean())
print("标准差: %.4f" % data.std())

# 检查负值比例
n_neg = np.sum(data < 0)
n_zero = np.sum(data == 0)
n_small = np.sum((data >= 0) & (data < 1))
print("\n负值: %d (%.2f%%)" % (n_neg, n_neg / data.size * 100))
print("零值: %d (%.2f%%)" % (n_zero, n_zero / data.size * 100))
print("[0,1) 值: %d (%.2f%%)" % (n_small, n_small / data.size * 100))

# 转 uint16 后的效果
img_u16 = np.clip(data, 0, 65535).astype(np.uint16)
print("\n转 uint16 后:")
print("  值域: [%d, %d]" % (img_u16.min(), img_u16.max()))
print("  零值: %d (%.2f%%)" % (np.sum(img_u16 == 0), np.sum(img_u16 == 0) / img_u16.size * 100))
print("  饱和(65535): %d (%.2f%%)" % (np.sum(img_u16 == 65535), np.sum(img_u16 == 65535) / img_u16.size * 100))

# 径向背景分布
cx_c, cy_c = w / 2, h / 2
r_max = (w**2 + h**2)**0.5 / 2

print("\n--- 径向背景分布 (中位数) ---")
yy, xx = np.mgrid[0:h, 0:w]
r = np.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
r_norm = r / r_max

for r_lo, r_hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    mask = (r_norm >= r_lo) & (r_norm < r_hi)
    vals = data[mask]
    print("  r=[%.1f,%.1f): median=%.4f, mean=%.4f, std=%.4f, min=%.4f, max=%.4f, n_zero=%.1f%%" %
          (r_lo, r_hi, np.median(vals), vals.mean(), vals.std(), vals.min(), vals.max(),
           np.sum(vals == 0) / len(vals) * 100))

# 检查几个边缘星点的像素值
print("\n--- 边缘星点像素值检查 ---")
import json
FSYN_PATH = os.path.join(os.path.dirname(FITS_PATH), "03_fsyn.json")
with open(FSYN_PATH, "r", encoding="utf-8") as f:
    fsyn = json.load(f)

stars = fsyn["stars"]
# 成功的星 (中心区域)
center_stars = [s for s in stars if np.sqrt((s["cx"]-cx_c)**2 + (s["cy"]-cy_c)**2) / r_max < 0.3]
print("中心区域成功星 (r<0.3): %d 颗" % len(center_stars))
if center_stars:
    s = center_stars[0]
    x, y = int(s["cx"]), int(s["cy"])
    patch = data[max(0,y-10):y+11, max(0,x-10):x+11]
    print("  星点 (%d,%d): B=%.4f, A=%.4f, 像素patch median=%.4f, max=%.4f" %
          (x, y, s["B"], s["A"], np.median(patch), patch.max()))

# 检查边缘 Gaia 星位置 (r~0.7) 的像素值
print("\n--- 边缘 Gaia 星位置像素值 (r~0.7, PSF 失败区域) ---")
# 读取几个边缘位置的 Gaia 星
sys.path.insert(0, r"f:\Astro dev\Astro CS Normalization Database\lib\photometric_calib\gradient_estimator\python")
sys.path.insert(0, r"f:\Astro dev\Astro CS Normalization Database\lib\integration_test\python")

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + ";" + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_MINGW_BIN)

for d in [r"lib\astro_image_io", r"lib\gaia_xpsd_client"]:
    full = os.path.join(r"f:\Astro dev\Astro CS Normalization Database", d)
    os.environ["PATH"] = full + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(full)

from step3_integrate import read_wcs_from_fits
from gaia_spectrum_client import GaiaSpectrumClient

wcs_t = read_wcs_from_fits(FITS_PATH)
ra_c, dec_c = wcs_t.pixel_to_sky_batch(np.array([w/2.0]), np.array([h/2.0]))

# 锥形搜索
with GaiaSpectrumClient(r"f:\Astro dev\Astro CS Normalization Database\GaiaDR3SP", db_type=2) as client:
    stars_g = client.cone_search_with_spectrum(float(ra_c[0]), float(dec_c[0]), 6.0, 8.0, 10.7)
    ra_arr = np.array([s.ra for s in stars_g])
    dec_arr = np.array([s.dec for s in stars_g])
    mag_arr = np.array([s.mag_g for s in stars_g])
    px, py = wcs_t.sky_to_pixel_batch(ra_arr, dec_arr)

    in_img = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px_in, py_in, mag_in = px[in_img], py[in_img], mag_arr[in_img]
    r_in = np.sqrt((px_in - cx_c)**2 + (py_in - cy_c)**2) / r_max

    # 边缘星 (r~0.7)
    edge_mask = (r_in >= 0.65) & (r_in < 0.75)
    edge_idx = np.where(edge_mask)[0]

    print("边缘 Gaia 星 (r~0.7): %d 颗" % len(edge_idx))
    for i in edge_idx[:5]:
        x, y = int(px_in[i]), int(py_in[i])
        if 10 <= x < w-10 and 10 <= y < h-10:
            patch = data[y-10:y+11, x-10:x+11]
            print("  星 (%d,%d) mag=%.2f: patch median=%.4f, max=%.4f, 中心值=%.4f, 信噪比=%.1f" %
                  (x, y, mag_in[i], np.median(patch), patch.max(), data[y,x],
                   (patch.max() - np.median(patch)) / (np.std(patch) + 1e-10)))

    # 中心星 (r~0.2)
    center_mask = (r_in >= 0.15) & (r_in < 0.25)
    center_idx_g = np.where(center_mask)[0]
    print("\n中心 Gaia 星 (r~0.2): %d 颗" % len(center_idx_g))
    for i in center_idx_g[:5]:
        x, y = int(px_in[i]), int(py_in[i])
        if 10 <= x < w-10 and 10 <= y < h-10:
            patch = data[y-10:y+11, x-10:x+11]
            print("  星 (%d,%d) mag=%.2f: patch median=%.4f, max=%.4f, 中心值=%.4f, 信噪比=%.1f" %
                  (x, y, mag_in[i], np.median(patch), patch.max(), data[y,x],
                   (patch.max() - np.median(patch)) / (np.std(patch) + 1e-10)))

print("\n诊断完成")
