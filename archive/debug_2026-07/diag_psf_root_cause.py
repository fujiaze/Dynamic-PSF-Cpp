# -*- coding: utf-8 -*-
"""
PSF 拟合根因诊断脚本
功能: 诊断为什么 PSF 拟合在图像边缘 100% 失败
用途: 提取特定星的 fit window, 分析像素值/背景估计/初始参数/LM发散原因
"""
import os
import sys
import numpy as np

# 项目根目录
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__))))
_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except OSError:
        pass

_ASTRO_IO_DIR = os.path.join(_PROJECT_ROOT, "lib", "astro_image_io")
os.environ["PATH"] = _ASTRO_IO_DIR + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_ASTRO_IO_DIR)
    except OSError:
        pass

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "photometric_calib",
                                "spectrum_integrator", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "photometric_calib",
                                "gradient_estimator", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "dynamic_psf", "python"))

from dynamic_psf import DynamicPSF, DPSFFitParamsPy
from wcs_transform import WCSTransform
from astro_image_io import ImageReader
from gaia_spectrum_client import GaiaSpectrumClient


def analyze_patch(image, cx, cy, fitRadius=8):
    """手动提取 fit window 并分析"""
    h, w = image.shape[:2]
    x0 = max(0, int(cx) - fitRadius)
    y0 = max(0, int(cy) - fitRadius)
    x1 = min(w, int(cx) + fitRadius + 1)
    y1 = min(h, int(cy) + fitRadius + 1)
    rw = x1 - x0
    rh = y1 - y0

    patch = image[y0:y1, x0:x1].astype(np.float64)

    # 背景估计 (复制 C++ 逻辑)
    vals = np.sort(patch.flatten())
    m = len(vals)
    median_val = (vals[m//2-1] + vals[m//2]) / 2.0 if m % 2 == 0 else vals[m//2]

    lower_half = vals[vals < median_val]
    if len(lower_half) == 0:
        lower_half = np.array([median_val])
    nh = len(lower_half)
    med_lh = (lower_half[nh//2-1] + lower_half[nh//2]) / 2.0 if nh % 2 == 0 else lower_half[nh//2]
    mad_lh = np.median(np.abs(lower_half - med_lh))
    threshold = 2.0 * 1.4826 * mad_lh
    filtered = lower_half[np.abs(lower_half - med_lh) <= threshold]
    if len(filtered) == 0:
        filtered = np.array([med_lh])
    nf = len(filtered)
    bkg0 = (filtered[nf//2-1] + filtered[nf//2]) / 2.0 if nf % 2 == 0 else filtered[nf//2]

    max_val = patch.max()
    A0 = max_val - bkg0
    sx0 = 0.15 * rw

    return {
        'rect': (x0, y0, x1, y1),
        'rw': rw, 'rh': rh,
        'patch': patch,
        'median_val': median_val,
        'bkg0': bkg0,
        'max_val': max_val,
        'A0': A0,
        'sx0': sx0,
        'local_cx': cx - x0,
        'local_cy': cy - y0,
    }


def main():
    # 使用 Galaxy Center panel1 Red 帧 (477 stars, scale=0.0076)
    fits_path = os.path.join(
        _PROJECT_ROOT, "testdata", "results", "Galaxy_Center_T4", "panel1", "Red",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red",
        "01_calibrated.fits")

    if not os.path.isfile(fits_path):
        print("校准图像不存在: %s" % fits_path)
        return

    print("=" * 70)
    print("PSF 拟合根因诊断")
    print("图像: %s" % os.path.basename(fits_path))
    print("=" * 70)

    # 1. 读取 WCS
    from astropy.io import fits
    with fits.open(fits_path, mode='readonly') as hdul:
        header = hdul[0].header
        ctype1 = str(header.get('CTYPE1', 'RA---TAN'))
        ctype2 = str(header.get('CTYPE2', 'DEC--TAN'))
        crval1 = float(header.get('CRVAL1', 0.0))
        crval2 = float(header.get('CRVAL2', 0.0))
        crpix1 = float(header.get('CRPIX1', 0.0))
        crpix2 = float(header.get('CRPIX2', 0.0))
        cd11 = float(header.get('CD1_1', 0.0))
        cd12 = float(header.get('CD1_2', 0.0))
        cd21 = float(header.get('CD2_1', 0.0))
        cd22 = float(header.get('CD2_2', 0.0))
        sip_order = int(header.get('A_ORDER', 0))
        sip_a = None
        sip_b = None
        if sip_order > 0:
            sip_a = [0.0] * 36
            sip_b = [0.0] * 36
            for i in range(sip_order + 1):
                for j in range(sip_order + 1 - i):
                    key_a = 'A_%d_%d' % (i, j)
                    key_b = 'B_%d_%d' % (i, j)
                    if key_a in header:
                        sip_a[i * 6 + j] = float(header[key_a])
                    if key_b in header:
                        sip_b[i * 6 + j] = float(header[key_b])

    wcs_transform = WCSTransform(
        crpix1=crpix1, crpix2=crpix2,
        crval1=crval1, crval2=crval2,
        cd11=cd11, cd12=cd12, cd21=cd21, cd22=cd22,
        sip_order=sip_order,
        sip_a=sip_a if sip_order > 0 else None,
        sip_b=sip_b if sip_order > 0 else None,
        ctype1=ctype1, ctype2=ctype2,
    )

    # 2. 加载图像
    reader = ImageReader()
    image_data = reader.read(fits_path)
    image = image_data.data
    img_w, img_h = image_data.width, image_data.height
    print("图像: %dx%d, dtype=%s, range=[%.1f, %.1f]" % (
        img_w, img_h, image.dtype, float(image.min()), float(image.max())))

    # 3. Gaia 锥形搜索
    ra_center_arr, dec_center_arr = wcs_transform.pixel_to_sky_batch(
        np.array([img_w / 2.0]), np.array([img_h / 2.0]))
    ra_center = float(ra_center_arr[0])
    dec_center = float(dec_center_arr[0])

    corner_xs = np.array([0.0, float(img_w), 0.0, float(img_w)])
    corner_ys = np.array([0.0, 0.0, float(img_h), float(img_h)])
    corner_ra, corner_dec = wcs_transform.pixel_to_sky_batch(corner_xs, corner_ys)
    from astropy.coordinates import angular_separation
    max_sep_rad = 0.0
    for i in range(4):
        sep = angular_separation(
            ra_center * np.pi / 180.0, dec_center * np.pi / 180.0,
            float(corner_ra[i]) * np.pi / 180.0, float(corner_dec[i]) * np.pi / 180.0)
        if sep > max_sep_rad:
            max_sep_rad = sep
    fov_radius_deg = float(max_sep_rad * 180.0 / np.pi)
    cone_radius_deg = fov_radius_deg * 1.05

    gaia_data_dir = os.path.join(_PROJECT_ROOT, "GaiaDR3SP")
    with GaiaSpectrumClient(gaia_data_dir, db_type=2) as client:
        gaia_stars = client.cone_search_with_spectrum(
            ra_center, dec_center, cone_radius_deg, 8.0, 16.0)

    gaia_ra = np.array([s.ra for s in gaia_stars], dtype=np.float64)
    gaia_dec = np.array([s.dec for s in gaia_stars], dtype=np.float64)
    gaia_px, gaia_py = wcs_transform.sky_to_pixel_batch(gaia_ra, gaia_dec)
    in_img = (gaia_px >= 0) & (gaia_px < img_w) & (gaia_py >= 0) & (gaia_py < img_h)
    gaia_idx_in = np.where(in_img)[0]
    gaia_px_in = gaia_px[in_img]
    gaia_py_in = gaia_py[in_img]
    gaia_mag_in = np.array([gaia_stars[i].mag_g for i in gaia_idx_in])
    print("Gaia 星: 总 %d, 图像内 %d" % (len(gaia_stars), len(gaia_idx_in)))

    # 4. 全量 PSF 拟合
    cx_list = gaia_px_in.tolist()
    cy_list = gaia_py_in.tolist()
    params = DPSFFitParamsPy(fitRadius=8)
    psf_results = DynamicPSF.fit_batch(image, cx_list, cy_list, params=params)

    n_ok = sum(1 for r in psf_results if int(r.status) == 0)
    print("PSF 拟合: 成功 %d / %d (%.1f%%)" % (n_ok, len(psf_results), 100.0 * n_ok / len(psf_results)))

    # 5. 按径向距离分析成功率
    cx_c, cy_c = img_w / 2.0, img_h / 2.0
    r_max = float(np.sqrt(img_w**2 + img_h**2)) / 2.0
    r_arr = np.sqrt((gaia_px_in - cx_c)**2 + (gaia_py_in - cy_c)**2) / r_max

    print("\n径向成功率分布:")
    for r_lo, r_hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (r_arr >= r_lo) & (r_arr < r_hi)
        n = mask.sum()
        if n == 0:
            continue
        n_s = sum(1 for i, r in enumerate(psf_results) if mask[i] and int(r.status) == 0)
        print("  r=[%.1f,%.1f): n=%d, 成功=%d (%.1f%%)" % (
            r_lo, r_hi, n, n_s, 100.0 * n_s / n if n > 0 else 0))

    # 6. 深入分析: 挑选成功和失败的边缘星, 查看 patch
    print("\n" + "=" * 70)
    print("深入分析: 中心成功星 vs 边缘失败星")
    print("=" * 70)

    # 中心成功星 (r < 0.3, status=0)
    center_ok = [(i, psf_results[i]) for i in range(len(psf_results))
                 if r_arr[i] < 0.3 and int(psf_results[i].status) == 0]
    # 边缘失败星 (r >= 0.5, status != 0)
    edge_fail = [(i, psf_results[i]) for i in range(len(psf_results))
                 if r_arr[i] >= 0.5 and int(psf_results[i].status) != 0]

    print("\n中心成功星 (r<0.3, status=0): %d 颗" % len(center_ok))
    print("边缘失败星 (r>=0.5, status!=0): %d 颗" % len(edge_fail))

    # 分析 3 颗中心成功星
    print("\n--- 中心成功星分析 ---")
    for idx, (i, psf) in enumerate(center_ok[:3]):
        cx, cy = gaia_px_in[i], gaia_py_in[i]
        mag = gaia_mag_in[i]
        info = analyze_patch(image, cx, cy)
        print("\n  星 #%d: (%.1f, %.1f) r=%.2f mag=%.2f" % (
            i, cx, cy, r_arr[i], mag))
        print("  patch: %dx%d, range=[%.1f, %.1f]" % (
            info['rw'], info['rh'], info['patch'].min(), info['patch'].max()))
        print("  bkg0=%.1f, max=%.1f, A0=%.1f, sx0=%.2f" % (
            info['bkg0'], info['max_val'], info['A0'], info['sx0']))
        print("  PSF: status=%d, B=%.1f, A=%.1f, cx=%.2f, cy=%.2f, fwhm=%.2f/%.2f" % (
            psf.status, psf.B, psf.A, psf.cx, psf.cy, psf.fwhm_x, psf.fwhm_y))
        # 显示 patch 中心 5x5
        p = info['patch']
        pcx, pcy = int(info['local_cx']), int(info['local_cy'])
        y0 = max(0, pcy - 2)
        y1 = min(p.shape[0], pcy + 3)
        x0 = max(0, pcx - 2)
        x1 = min(p.shape[1], pcx + 3)
        print("  中心 5x5 像素值:")
        for row in p[y0:y1]:
            print("    " + "  ".join("%7.1f" % v for v in row[x0:x1]))

    # 分析 5 颗边缘失败星
    print("\n--- 边缘失败星分析 ---")
    for idx, (i, psf) in enumerate(edge_fail[:5]):
        cx, cy = gaia_px_in[i], gaia_py_in[i]
        mag = gaia_mag_in[i]
        info = analyze_patch(image, cx, cy)
        print("\n  星 #%d: (%.1f, %.1f) r=%.2f mag=%.2f" % (
            i, cx, cy, r_arr[i], mag))
        print("  patch: %dx%d, range=[%.1f, %.1f]" % (
            info['rw'], info['rh'], info['patch'].min(), info['patch'].max()))
        print("  bkg0=%.1f, max=%.1f, A0=%.1f, sx0=%.2f" % (
            info['bkg0'], info['max_val'], info['A0'], info['sx0']))
        print("  PSF: status=%d, B=%.1f, A=%.1f, cx=%.2f, cy=%.2f, fwhm=%.2f/%.2f" % (
            psf.status, psf.B, psf.A, psf.cx, psf.cy, psf.fwhm_x, psf.fwhm_y))
        # 显示 patch 中心 5x5
        p = info['patch']
        pcx, pcy = int(info['local_cx']), int(info['local_cy'])
        y0 = max(0, pcy - 2)
        y1 = min(p.shape[0], pcy + 3)
        x0 = max(0, pcx - 2)
        x1 = min(p.shape[1], pcx + 3)
        print("  中心 5x5 像素值:")
        for row in p[y0:y1]:
            print("    " + "  ".join("%7.1f" % v for v in row[x0:x1]))

        # 分析: 是否有亮星在 patch 中?
        patch_max = info['patch'].max()
        patch_median = np.median(info['patch'])
        bright_pixels = np.sum(info['patch'] > patch_median + 3 * (patch_max - patch_median))
        print("  分析: patch_median=%.1f, bright_pixels=%d, A0/max_ratio=%.3f" % (
            patch_median, bright_pixels, info['A0'] / info['max_val'] if info['max_val'] > 0 else 0))

    # 7. 关键对比: 中心 vs 边缘的 A0 和 bkg0 分布
    print("\n" + "=" * 70)
    print("中心 vs 边缘: A0 和 bkg0 统计")
    print("=" * 70)

    center_A0 = []
    center_bkg0 = []
    edge_A0 = []
    edge_bkg0 = []

    for i in range(len(psf_results)):
        cx, cy = gaia_px_in[i], gaia_py_in[i]
        info = analyze_patch(image, cx, cy)
        if r_arr[i] < 0.3:
            center_A0.append(info['A0'])
            center_bkg0.append(info['bkg0'])
        elif r_arr[i] >= 0.5:
            edge_A0.append(info['A0'])
            edge_bkg0.append(info['bkg0'])

    if center_A0:
        print("中心 (r<0.3): n=%d, A0: median=%.1f, mean=%.1f, [%.1f, %.1f]" % (
            len(center_A0), np.median(center_A0), np.mean(center_A0),
            min(center_A0), max(center_A0)))
        print("             bkg0: median=%.1f, mean=%.1f, [%.1f, %.1f]" % (
            np.median(center_bkg0), np.mean(center_bkg0),
            min(center_bkg0), max(center_bkg0)))
    if edge_A0:
        print("边缘 (r>=0.5): n=%d, A0: median=%.1f, mean=%.1f, [%.1f, %.1f]" % (
            len(edge_A0), np.median(edge_A0), np.mean(edge_A0),
            min(edge_A0), max(edge_A0)))
        print("             bkg0: median=%.1f, mean=%.1f, [%.1f, %.1f]" % (
            np.median(edge_bkg0), np.mean(edge_bkg0),
            min(edge_bkg0), max(edge_bkg0)))

    # 8. uint16 截断分析
    print("\n" + "=" * 70)
    print("uint16 截断分析")
    print("=" * 70)
    n_clipped_center = 0
    n_clipped_edge = 0
    for i in range(len(psf_results)):
        cx, cy = gaia_px_in[i], gaia_py_in[i]
        info = analyze_patch(image, cx, cy)
        n_clipped = np.sum(info['patch'] > 65535)
        if n_clipped > 0:
            if r_arr[i] < 0.3:
                n_clipped_center += 1
            elif r_arr[i] >= 0.5:
                n_clipped_edge += 1
    print("中心 (r<0.3): patch 有像素>65535 的星数=%d" % n_clipped_center)
    print("边缘 (r>=0.5): patch 有像素>65535 的星数=%d" % n_clipped_edge)

    # 9. 星等分布对比
    print("\n" + "=" * 70)
    print("星等分布对比")
    print("=" * 70)
    center_mag = [gaia_mag_in[i] for i in range(len(psf_results)) if r_arr[i] < 0.3]
    edge_mag = [gaia_mag_in[i] for i in range(len(psf_results)) if r_arr[i] >= 0.5]
    if center_mag:
        print("中心 (r<0.3): mag median=%.2f, [%.2f, %.2f]" % (
            np.median(center_mag), min(center_mag), max(center_mag)))
    if edge_mag:
        print("边缘 (r>=0.5): mag median=%.2f, [%.2f, %.2f]" % (
            np.median(edge_mag), min(edge_mag), max(edge_mag)))

    # 10. 关键: 检查边缘星 patch 中是否有多个亮峰 (星点拥挤)
    print("\n" + "=" * 70)
    print("星点拥挤分析: patch 中亮峰数")
    print("=" * 70)
    from scipy.ndimage import maximum_filter

    center_n_peaks = []
    edge_n_peaks = []
    for i in range(len(psf_results)):
        cx, cy = gaia_px_in[i], gaia_py_in[i]
        info = analyze_patch(image, cx, cy)
        p = info['patch']
        # 局部极大值 (3x3 窗口)
        max_filt = maximum_filter(p, size=3)
        peaks = (p == max_filt) & (p > info['bkg0'] + 3 * (info['bkg0'] * 0.1 + 10))
        n_peaks = np.sum(peaks)
        if r_arr[i] < 0.3:
            center_n_peaks.append(n_peaks)
        elif r_arr[i] >= 0.5:
            edge_n_peaks.append(n_peaks)

    if center_n_peaks:
        print("中心 (r<0.3): 亮峰数 median=%.1f, mean=%.1f, [%d, %d]" % (
            np.median(center_n_peaks), np.mean(center_n_peaks),
            min(center_n_peaks), max(center_n_peaks)))
    if edge_n_peaks:
        print("边缘 (r>=0.5): 亮峰数 median=%.1f, mean=%.1f, [%d, %d]" % (
            np.median(edge_n_peaks), np.mean(edge_n_peaks),
            min(edge_n_peaks), max(edge_n_peaks)))


if __name__ == "__main__":
    main()
