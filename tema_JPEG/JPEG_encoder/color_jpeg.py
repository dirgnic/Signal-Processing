"""
Color JPEG Compression
Task 2: JPEG with RGB to Y'CbCr conversion and 4:2:0 subsampling
"""

import numpy as np
from .core import jpeg_encode_block, jpeg_decode_block, Q_LUMINANCE, Q_CHROMINANCE
from .color import rgb_to_ycbcr, ycbcr_to_rgb


def jpeg_compress_color(img_rgb, Q_luma=None, Q_chroma=None, subsample=True):
    """Compress a color image using JPEG algorithm with Y'CbCr"""
    if Q_luma is None:
        Q_luma = Q_LUMINANCE
    if Q_chroma is None:
        Q_chroma = Q_CHROMINANCE
    
    # Convert to Y'CbCr
    ycbcr = rgb_to_ycbcr(img_rgb.astype(np.float64))
    
    h, w = ycbcr.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    
    compressed_channels = []
    
    for c in range(3):
        channel = ycbcr[:, :, c]
        channel_padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='edge')
        
        # 4:2:0 subsampling for chrominance
        if subsample and c > 0:
            channel_padded = channel_padded[::2, ::2]
        
        h_c, w_c = channel_padded.shape
        compressed = np.zeros_like(channel_padded)
        
        Q = Q_luma if c == 0 else Q_chroma
        
        for i in range(0, h_c, 8):
            for j in range(0, w_c, 8):
                block = channel_padded[i:i+8, j:j+8]
                if block.shape == (8, 8):
                    quantized = jpeg_encode_block(block, Q)
                    reconstructed = jpeg_decode_block(quantized, Q)
                    compressed[i:i+8, j:j+8] = reconstructed
        
        # Upsample chrominance
        if subsample and c > 0:
            compressed = np.repeat(np.repeat(compressed, 2, axis=0), 2, axis=1)
            compressed = compressed[:h+h_pad, :w+w_pad]
        
        compressed_channels.append(compressed[:h, :w])
    
    # Reconstruct and convert back to RGB
    ycbcr_compressed = np.stack(compressed_channels, axis=-1)
    rgb_compressed = ycbcr_to_rgb(ycbcr_compressed)
    
    return rgb_compressed
