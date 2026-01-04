"""Toy JPEG 2000-style compressor using 2D Haar wavelet.

This is not a full JPEG 2000 implementation, but a didactic
approximation that highlights the main ideas:
- wavelet transform instead of block DCT
- multi-resolution decomposition (LL, LH, HL, HH subbands)
- scalar quantization of subbands
"""

import numpy as np


def haar_1d(signal):
    """Single-level 1D Haar transform on the last axis.

    Input length must be even. Returns (approx, detail).
    """
    signal = signal.astype(np.float64)
    even = signal[..., ::2]
    odd = signal[..., 1::2]
    approx = (even + odd) / 2.0
    detail = (even - odd) / 2.0
    return approx, detail


def inverse_haar_1d(approx, detail):
    """Inverse 1D Haar transform (single level)."""
    even = approx + detail
    odd = approx - detail
    out = np.empty((approx.shape[0], approx.shape[1] * 2), dtype=np.float64)
    out[..., ::2] = even
    out[..., 1::2] = odd
    return out


def haar_2d(image):
    """Single-level 2D Haar transform.

    Returns four subbands: LL, LH, HL, HH.
    """
    image = image.astype(np.float64)

    # Transform rows
    approx_rows, detail_rows = haar_1d(image)

    # Transform columns on both approx and detail parts
    approx_cols, detail_cols = haar_1d(approx_rows.T)
    approx_cols = approx_cols.T
    detail_cols = detail_cols.T

    approx_cols_d, detail_cols_d = haar_1d(detail_rows.T)
    approx_cols_d = approx_cols_d.T
    detail_cols_d = detail_cols_d.T

    LL = approx_cols
    LH = detail_cols  # horizontal detail
    HL = approx_cols_d  # vertical detail
    HH = detail_cols_d  # diagonal detail

    return LL, LH, HL, HH


def inverse_haar_2d(LL, LH, HL, HH):
    """Inverse single-level 2D Haar transform."""
    # Reconstruct detail_rows (from HL, HH)
    approx_rows_d = HL
    detail_rows_d = HH
    detail_rows = inverse_haar_1d(approx_rows_d.T, detail_rows_d.T).T

    # Reconstruct approx_rows (from LL, LH)
    approx_rows = inverse_haar_1d(LL.T, LH.T).T

    # Now inverse along rows to get full image
    image = inverse_haar_1d(approx_rows, detail_rows)
    return image


def multilevel_haar_2d(image, levels=1):
    """Recursive multi-level 2D Haar transform on the LL band.

    Parameters
    ----------
    image : 2D array
        Input grayscale image.
    levels : int
        Number of decomposition levels. For JPEG2000-style pyramids
        a typical value is 3.

    Returns
    -------
    LL : 2D array
        Coarsest approximation after `levels` decompositions.
    details : list of (LH, HL, HH)
        Detail subbands for each level, from first (finest) to last
        (coarsest) decomposition.
    """

    img = image.astype(np.float64)
    LL = img
    details = []

    for _ in range(levels):
        LL, LH, HL, HH = haar_2d(LL)
        details.append((LH, HL, HH))

    return LL, details


def inverse_multilevel_haar_2d(LL, details):
    """Inverse of multilevel_haar_2d.

    Parameters
    ----------
    LL : 2D array
        Coarsest LL band.
    details : list of (LH, HL, HH)
        Detail subbands for each level, in the same order as returned
        by multilevel_haar_2d (from finest to coarsest).
    """

    current = LL
    # Reconstruct from coarsest to finest
    for LH, HL, HH in reversed(details):
        current = inverse_haar_2d(current, LH, HL, HH)
    return current


def quantize_subbands(LL, LH, HL, HH, step_LL=4.0, step_H=8.0):
    """Uniform scalar quantization for each subband.

    step_LL: quantization step for low-frequency LL band
    step_H:  quantization step for high-frequency bands (LH, HL, HH)
    """
    q_LL = np.round(LL / step_LL)
    q_LH = np.round(LH / step_H)
    q_HL = np.round(HL / step_H)
    q_HH = np.round(HH / step_H)
    return q_LL, q_LH, q_HL, q_HH


def dequantize_subbands(q_LL, q_LH, q_HL, q_HH, step_LL=4.0, step_H=8.0):
    """Inverse of quantize_subbands."""
    LL = q_LL * step_LL
    LH = q_LH * step_H
    HL = q_HL * step_H
    HH = q_HH * step_H
    return LL, LH, HL, HH


def jpeg2000_like_compress_grayscale(image, step_LL=4.0, step_H=8.0):
    """Toy JPEG 2000-like compression for a grayscale image.

    - Pads image to even dimensions
    - Applies 2D Haar wavelet (1 level)
    - Quantizes subbands
    - Reconstructs image via inverse transform
    - Returns reconstructed image, MSE, and a crude "bit" estimate
      based on number of non-zero quantized coefficients.
    """
    img = image.astype(np.float64)
    h, w = img.shape

    # Ensure even dimensions (required by 1-level Haar)
    h_pad = h % 2
    w_pad = w % 2
    if h_pad or w_pad:
        img = np.pad(img, ((0, h_pad), (0, w_pad)), mode="edge")

    # Forward transform
    LL, LH, HL, HH = haar_2d(img)

    # Quantize
    q_LL, q_LH, q_HL, q_HH = quantize_subbands(LL, LH, HL, HH, step_LL, step_H)

    # Bit-cost proxy: count non-zero coefficients
    total_coeffs = (
        q_LL.size + q_LH.size + q_HL.size + q_HH.size
    )
    nonzero_coeffs = (
        np.count_nonzero(q_LL)
        + np.count_nonzero(q_LH)
        + np.count_nonzero(q_HL)
        + np.count_nonzero(q_HH)
    )
    compression_ratio = total_coeffs / max(nonzero_coeffs, 1)

    # Dequantize and inverse transform
    LL_d, LH_d, HL_d, HH_d = dequantize_subbands(
        q_LL, q_LH, q_HL, q_HH, step_LL, step_H
    )
    recon = inverse_haar_2d(LL_d, LH_d, HL_d, HH_d)

    # Remove padding
    recon = recon[:h, :w]

    mse = np.mean((image.astype(np.float64) - recon) ** 2)

    return recon, mse, compression_ratio, {
        "nonzero_coeffs": nonzero_coeffs,
        "total_coeffs": total_coeffs,
        "step_LL": step_LL,
        "step_H": step_H,
    }


def jpeg2000_like_compress_grayscale_multilevel(image, step_LL=4.0, step_H=8.0, levels=3):
    """JPEG 2000-like compression with a multi-level LL pyramid.

    This uses multilevel_haar_2d to recursively decompose the LL band
    `levels` times, mimicking the resolution pyramid of real JPEG 2000.
    """

    img = image.astype(np.float64)
    h, w = img.shape

    # Ensure dimensions are multiples of 2**levels
    scale = 2 ** levels
    h_pad = (scale - h % scale) % scale
    w_pad = (scale - w % scale) % scale
    if h_pad or w_pad:
        img = np.pad(img, ((0, h_pad), (0, w_pad)), mode="edge")

    # Forward multi-level transform
    LL, details = multilevel_haar_2d(img, levels=levels)

    # Quantize: LL with step_LL, all detail subbands with step_H
    q_LL = np.round(LL / step_LL)
    q_details = []
    for LH, HL, HH in details:
        q_LH = np.round(LH / step_H)
        q_HL = np.round(HL / step_H)
        q_HH = np.round(HH / step_H)
        q_details.append((q_LH, q_HL, q_HH))

    # Bit-cost proxy
    nonzero_coeffs = np.count_nonzero(q_LL)
    total_coeffs = q_LL.size
    for q_LH, q_HL, q_HH in q_details:
        nonzero_coeffs += (
            np.count_nonzero(q_LH)
            + np.count_nonzero(q_HL)
            + np.count_nonzero(q_HH)
        )
        total_coeffs += q_LH.size + q_HL.size + q_HH.size

    compression_ratio = total_coeffs / max(nonzero_coeffs, 1)

    # Dequantize and inverse transform
    LL_d = q_LL * step_LL
    details_d = []
    for q_LH, q_HL, q_HH in q_details:
        LH_d = q_LH * step_H
        HL_d = q_HL * step_H
        HH_d = q_HH * step_H
        details_d.append((LH_d, HL_d, HH_d))

    recon = inverse_multilevel_haar_2d(LL_d, details_d)

    # Remove padding
    recon = recon[:h, :w]

    mse = np.mean((image.astype(np.float64) - recon) ** 2)

    return recon, mse, compression_ratio, {
        "nonzero_coeffs": nonzero_coeffs,
        "total_coeffs": total_coeffs,
        "step_LL": step_LL,
        "step_H": step_H,
        "levels": levels,
    }
