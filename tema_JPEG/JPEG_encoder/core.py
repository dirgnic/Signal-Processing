"""
JPEG Core Functions
DCT encoding/decoding and quantization matrices
"""

import numpy as np
from scipy.fft import dctn, idctn

# Standard JPEG quantization matrix for luminance
Q_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

# Standard JPEG quantization matrix for chrominance
Q_CHROMINANCE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float64)


def jpeg_encode_block(block, Q):
    """Encode a single 8x8 block using DCT and quantization"""
    dct_block = dctn(block.astype(np.float64), norm='ortho')
    quantized = np.round(dct_block / Q)
    return quantized


def jpeg_decode_block(quantized, Q):
    """Decode a single 8x8 block using de-quantization and IDCT"""
    dequantized = quantized * Q
    return idctn(dequantized, norm='ortho')


def scale_quantization_matrix(Q, quality_factor):
    """Scale quantization matrix based on quality factor (1-100)"""
    qf = np.clip(quality_factor, 1, 100)
    if qf < 50:
        scale = 5000 / qf
    else:
        scale = 200 - 2 * qf
    
    Q_scaled = np.floor((Q * scale + 50) / 100)
    Q_scaled = np.clip(Q_scaled, 1, 255)
    return Q_scaled
