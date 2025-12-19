"""
Video JPEG Compression (MJPEG)
Task 4: Frame-by-frame JPEG compression
"""

import numpy as np
from .core import Q_LUMINANCE, Q_CHROMINANCE, scale_quantization_matrix
from .color_jpeg import jpeg_compress_color


def generate_synthetic_video(n_frames=30, size=(128, 128)):
    """Generate a synthetic video with a moving object"""
    frames = []
    
    for t in range(n_frames):
        frame = np.zeros((*size, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(size[0]):
            frame[i, :, 0] = int(50 + 50 * np.sin(2 * np.pi * i / size[0] + t * 0.1))
            frame[i, :, 2] = int(100 + 50 * np.cos(2 * np.pi * i / size[0] + t * 0.1))
        
        # Moving circle
        cx = int(size[1] * (0.3 + 0.4 * np.sin(2 * np.pi * t / n_frames)))
        cy = int(size[0] * (0.3 + 0.4 * np.cos(2 * np.pi * t / n_frames)))
        radius = 15
        
        for i in range(max(0, cy - radius), min(size[0], cy + radius)):
            for j in range(max(0, cx - radius), min(size[1], cx + radius)):
                if (i - cy)**2 + (j - cx)**2 < radius**2:
                    frame[i, j] = [255, 200, 0]
        
        frames.append(frame)
    
    return frames


def compress_video_mjpeg(frames, quality_factor=75):
    """Compress video using MJPEG (frame-by-frame JPEG)"""
    Q_luma = scale_quantization_matrix(Q_LUMINANCE, quality_factor)
    Q_chroma = scale_quantization_matrix(Q_CHROMINANCE, quality_factor)
    
    compressed_frames = []
    mse_per_frame = []
    
    for frame in frames:
        compressed = jpeg_compress_color(frame, Q_luma, Q_chroma, subsample=True)
        compressed_frames.append(compressed)
        
        mse = np.mean((frame.astype(np.float64) - compressed.astype(np.float64)) ** 2)
        mse_per_frame.append(mse)
    
    return compressed_frames, mse_per_frame
