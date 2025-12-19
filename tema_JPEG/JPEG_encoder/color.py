"""
Color Space Conversion
RGB to Y'CbCr and back (ITU-R BT.601 standard)
"""

import numpy as np


def rgb_to_ycbcr(rgb):
    """Convert RGB image to Y'CbCr color space"""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycbcr):
    """Convert Y'CbCr image to RGB color space"""
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)
