"""
Adaptive JPEG Compression
Task 3: Compress to target MSE threshold using binary search
"""

import numpy as np
from .core import Q_LUMINANCE, scale_quantization_matrix
from .grayscale import jpeg_compress_grayscale


def jpeg_compress_with_quality(img, quality_factor):
    """Compress image with specified quality factor (1-100)"""
    Q_scaled = scale_quantization_matrix(Q_LUMINANCE, quality_factor)
    compressed, ratio = jpeg_compress_grayscale(img, Q_scaled)
    mse = np.mean((img.astype(np.float64) - compressed) ** 2)
    return compressed, mse, ratio


def find_quality_for_target_mse(img, target_mse, tolerance=0.5, max_iterations=20):
    """Find optimal quality factor for target MSE using binary search"""
    low, high = 1, 100
    best_qf = 50
    best_mse = float('inf')
    history = []
    
    for iteration in range(max_iterations):
        mid = (low + high) // 2
        _, mse, _ = jpeg_compress_with_quality(img, mid)
        history.append((mid, mse))
        
        if abs(mse - target_mse) < tolerance:
            return mid, mse, history
        
        if abs(mse - target_mse) < abs(best_mse - target_mse):
            best_mse = mse
            best_qf = mid
        
        # Higher QF = lower MSE
        if mse > target_mse:
            low = mid + 1
        else:
            high = mid - 1
        
        if low > high:
            break
    
    return best_qf, best_mse, history
