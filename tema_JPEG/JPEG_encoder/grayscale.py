"""
Grayscale JPEG Compression
Task 1: Complete JPEG on all 8x8 blocks
"""

import numpy as np
from .core import jpeg_encode_block, jpeg_decode_block, Q_LUMINANCE


def jpeg_compress_grayscale(img, Q=None):
    """Compress a grayscale image using JPEG algorithm"""
    if Q is None:
        Q = Q_LUMINANCE
    
    h, w = img.shape
    
    # Pad to multiple of 8
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad)), mode='edge')
    
    h_new, w_new = img_padded.shape
    compressed = np.zeros_like(img_padded, dtype=np.float64)
    
    total_coeffs = 0
    nonzero_coeffs = 0
    
    # Process each 8x8 block
    for i in range(0, h_new, 8):
        for j in range(0, w_new, 8):
            block = img_padded[i:i+8, j:j+8]
            
            quantized = jpeg_encode_block(block, Q)
            total_coeffs += 64
            nonzero_coeffs += np.count_nonzero(quantized)
            
            reconstructed = jpeg_decode_block(quantized, Q)
            compressed[i:i+8, j:j+8] = reconstructed
    
    # Remove padding
    compressed = compressed[:h, :w]
    
    compression_ratio = total_coeffs / max(nonzero_coeffs, 1)
    return compressed, compression_ratio


def compute_metrics(original, compressed):
    """Compute MSE and PSNR between original and compressed images"""
    mse = np.mean((original.astype(np.float64) - compressed) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr
