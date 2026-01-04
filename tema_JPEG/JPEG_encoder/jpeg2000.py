"""Toy JPEG 2000-style compressor using 2D Haar wavelet.

This is NOT a full JPEG 2000 implementation, but a didactic
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
