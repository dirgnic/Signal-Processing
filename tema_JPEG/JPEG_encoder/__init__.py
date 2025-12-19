"""
JPEG Encoder Package
A complete implementation of JPEG compression algorithm
"""

from .core import (
    Q_LUMINANCE,
    Q_CHROMINANCE,
    jpeg_encode_block,
    jpeg_decode_block,
    scale_quantization_matrix
)

from .color import rgb_to_ycbcr, ycbcr_to_rgb

from .grayscale import jpeg_compress_grayscale, compute_metrics

from .color_jpeg import jpeg_compress_color

from .adaptive import (
    jpeg_compress_with_quality,
    find_quality_for_target_mse
)

from .video import (
    generate_synthetic_video,
    compress_video_mjpeg
)

from .huffman import (
    HuffmanNode,
    build_huffman_tree,
    get_huffman_codes,
    compute_entropy,
    compute_average_code_length
)

__all__ = [
    'Q_LUMINANCE',
    'Q_CHROMINANCE',
    'jpeg_encode_block',
    'jpeg_decode_block',
    'scale_quantization_matrix',
    'rgb_to_ycbcr',
    'ycbcr_to_rgb',
    'jpeg_compress_grayscale',
    'compute_metrics',
    'jpeg_compress_color',
    'jpeg_compress_with_quality',
    'find_quality_for_target_mse',
    'generate_synthetic_video',
    'compress_video_mjpeg',
    'HuffmanNode',
    'build_huffman_tree',
    'get_huffman_codes',
    'compute_entropy',
    'compute_average_code_length'
]
