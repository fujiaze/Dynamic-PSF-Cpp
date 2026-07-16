# -*- coding: utf-8 -*-
"""直接对边缘星和中心星运行 PSF 拟合, 对比结果和状态码"""
import sys
import os
import numpy as np
from astropy.io import fits

PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
FITS_PATH = os.path.join(PROJECT_ROOT, "testdata", "results", "Galaxy_Center_T4", "panel1", "Red",
                         "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red", "01_calibrated.fits")

# 加载图像
with fits.open(FITS_PATH, mode='readonly') as hdul:
    data = hdul[0].data.astype(np.float32)

h, w = data.shape
cx_c, cy_c = w / 2, h / 2
r_max = (w**2 + h**2)**0.5 / 2

# 环境设置
_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + ";" + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_MINGW_BIN)

for d in [r"lib\astro_image_io", r"lib\gaia_xpsd_client", r"lib\dynamic_psf"]:
    full = os.path.join(PROJECT_ROOT, d)
    os.environ["PATH"] = full + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(full)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "photometric_calib", "gradient_estimator", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "integration_test", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "dynamic_psf", "python"))

from step3_integrate import read_wcs_from_fits
from gaia_spectrum_client import GaiaSpectrumClient
from dynamic_psf import DynamicPSF, DPSFFitParamsPy, DPSF_FIT_STATUS_NAMES

wcs_t = read_wcs_from_fits(FITS_PATH)
ra_c, dec_c = wcs_t.pixel_to_sky_batch(np.array([w/2.0]), np.array([h/2.0]))

# Gaia 锥形搜索
with GaiaSpectrumClient(os.path.join(PROJECT_ROOT, "GaiaDR3SP"), db_type=2) as client:
    stars_g = client.cone_search_with_spectrum(float(ra_c[0]), float(dec_c[0]), 6.0, 8.0, 10.7)
    ra_arr = np.array([s.ra for s in stars_g])
    dec_arr = np.array([s.dec for s in stars_g])
    mag_arr = np.array([s.mag_g for s in stars_g])
    px, py = wcs_t.sky_to_pixel_batch(ra_arr, dec_arr)

in_img = (px >= 0) & (px < w) & (py >= 0) & (py < h)
px_in, py_in, mag_in = px[in_img], py[in_img], mag_arr[in_img]
r_in = np.sqrt((px_in - cx_c)**2 + (py_in - cy_c)**2) / r_max

print("=" * 70)
print("PSF 拟合对比: 中心星 vs 边缘星")
print("=" * 70)

# 选 10 颗中心星和 10 颗边缘星
center_idx = np.where((r_in >= 0.15) & (r_in < 0.25))[0][:10]
edge_idx = np.where((r_in >= 0.65) & (r_in < 0.75))[0][:10]

# 转 uint16 (与 step3 一致)
img_u16 = np.clip(data, 0, 65535).astype(np.uint16)

# 用原始 float32 图像也试一下
params = DPSFFitParamsPy(fitRadius=8)

print("\n--- 中心星 (r~0.2, uint16 转换) ---")
for i in center_idx:
    x, y = float(px_in[i]), float(py_in[i])
    # 检查 patch 原始值
    xi, yi = int(x), int(y)
    patch_orig = data[max(0,yi-8):yi+9, max(0,xi-8):xi+9]
    patch_u16 = img_u16[max(0,yi-8):yi+9, max(0,xi-8):xi+9]

    result = DynamicPSF.fit(img_u16, x, y, params)
    status_name = DPSF_FIT_STATUS_NAMES.get(result.status, "UNKNOWN(%d)" % result.status)

    print("  星 (%.0f,%.0f) mag=%.2f: status=%s, B=%.1f, A=%.1f, cx=%.2f, cy=%.2f, fwhm=%.2f, "
          "orig_max=%.0f, u16_max=%d, 饱和像素=%d" %
          (x, y, mag_in[i], status_name, result.B, result.A, result.cx, result.cy,
           result.fwhm_x, patch_orig.max(), int(patch_u16.max()),
           np.sum(patch_u16 >= 65535)))

print("\n--- 边缘星 (r~0.7, uint16 转换) ---")
for i in edge_idx:
    x, y = float(px_in[i]), float(py_in[i])
    xi, yi = int(x), int(y)
    patch_orig = data[max(0,yi-8):yi+9, max(0,xi-8):xi+9]
    patch_u16 = img_u16[max(0,yi-8):yi+9, max(0,xi-8):xi+9]

    result = DynamicPSF.fit(img_u16, x, y, params)
    status_name = DPSF_FIT_STATUS_NAMES.get(result.status, "UNKNOWN(%d)" % result.status)

    print("  星 (%.0f,%.0f) mag=%.2f: status=%s, B=%.1f, A=%.1f, cx=%.2f, cy=%.2f, fwhm=%.2f, "
          "orig_max=%.0f, u16_max=%d, 饱和像素=%d" %
          (x, y, mag_in[i], status_name, result.B, result.A, result.cx, result.cy,
           result.fwhm_x, patch_orig.max(), int(patch_u16.max()),
           np.sum(patch_u16 >= 65535)))

# 也用 float32 图像直接拟合 (修改 DynamicPSF 接受 float32)
print("\n--- 边缘星 (r~0.7, float32 原始, 无 clip) ---")
# DynamicPSF.fit 会把 float32 转 uint16, 我们需要绕过这个转换
# 直接调用 C++ DLL, 传入 float 数据
# 但 C++ 接口只接受 uint16, 所以我们手动缩放
# 方案: 把 float32 数据缩放到 [0, 65535] 范围
data_scaled = np.clip(data * (65535.0 / data.max()), 0, 65535).astype(np.uint16)
print("缩放后图像值域: [%d, %d]" % (data_scaled.min(), data_scaled.max()))

for i in edge_idx:
    x, y = float(px_in[i]), float(py_in[i])
    xi, yi = int(x), int(y)
    patch_scaled = data_scaled[max(0,yi-8):yi+9, max(0,xi-8):xi+9]

    result = DynamicPSF.fit(data_scaled, x, y, params)
    status_name = DPSF_FIT_STATUS_NAMES.get(result.status, "UNKNOWN(%d)" % result.status)

    print("  星 (%.0f,%.0f) mag=%.2f: status=%s, B=%.1f, A=%.1f, cx=%.2f, cy=%.2f, fwhm=%.2f, "
          "scaled_max=%d, 饱和像素=%d" %
          (x, y, mag_in[i], status_name, result.B, result.A, result.cx, result.cy,
           result.fwhm_x, int(patch_scaled.max()), np.sum(patch_scaled >= 65535)))

print("\n诊断完成")
