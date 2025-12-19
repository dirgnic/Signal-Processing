"""
Test MJPEG compression on a real video file (kitty.mp4)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from JPEG_encoder import (
    jpeg_compress_color,
    compress_video_mjpeg
)

# Try to import cv2 for video reading
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
    import cv2
    HAS_CV2 = True


def load_video_frames(video_path, max_frames=50, resize=None):
    """Load frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video info:")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS: {fps:.2f}")
    print(f"    Total frames: {total_frames}")
    print(f"    Duration: {total_frames/fps:.2f}s")
    
    # Sample frames evenly
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = range(total_frames)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed (must be multiple of 8 for JPEG)
            if resize:
                frame = cv2.resize(frame, resize)
            else:
                # Make dimensions multiple of 8
                h, w = frame.shape[:2]
                new_h = (h // 8) * 8
                new_w = (w // 8) * 8
                if new_h != h or new_w != w:
                    frame = frame[:new_h, :new_w]
            
            frames.append(frame)
    
    cap.release()
    return frames, fps


def test_video_compression(video_path):
    """Test MJPEG compression on a video"""
    print("\n" + "="*60)
    print("TESTING MJPEG ON YOUR VIDEO: kitty.mp4")
    print("="*60)
    
    # Load frames
    print("\nLoading video frames...")
    frames, fps = load_video_frames(video_path, max_frames=30, resize=None)
    print(f"  Loaded {len(frames)} frames")
    print(f"  Frame shape: {frames[0].shape}")
    
    # Test with different quality factors
    quality_factors = [30, 60, 90]
    results = {}
    
    for qf in quality_factors:
        print(f"\n  Compressing with QF={qf}...")
        compressed, mse_list = compress_video_mjpeg(frames, qf)
        avg_mse = np.mean(mse_list)
        avg_psnr = 10 * np.log10(255**2 / avg_mse)
        results[qf] = {
            'frames': compressed,
            'mse': mse_list,
            'avg_mse': avg_mse,
            'avg_psnr': avg_psnr
        }
        print(f"    Avg MSE: {avg_mse:.2f}")
        print(f"    Avg PSNR: {avg_psnr:.2f} dB")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Visualize sample frames
    n_samples = min(6, len(frames))
    sample_indices = np.linspace(0, len(frames)-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(len(quality_factors) + 1, n_samples, figsize=(3*n_samples, 3*(len(quality_factors)+1)))
    
    # Original frames
    for idx, frame_idx in enumerate(sample_indices):
        axes[0, idx].imshow(frames[frame_idx])
        axes[0, idx].set_title(f'Original\nFrame {frame_idx}', fontsize=9)
        axes[0, idx].axis('off')
    
    # Compressed frames
    for row, qf in enumerate(quality_factors, 1):
        for idx, frame_idx in enumerate(sample_indices):
            axes[row, idx].imshow(results[qf]['frames'][frame_idx])
            mse = results[qf]['mse'][frame_idx]
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
            axes[row, idx].set_title(f'QF={qf}\nPSNR={psnr:.1f}dB', fontsize=9)
            axes[row, idx].axis('off')
    
    plt.suptitle('MJPEG Compression on kitty.mp4', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/kitty_mjpeg_frames.png', dpi=150)
    plt.close()
    print(f"\n  Saved: output/kitty_mjpeg_frames.png")
    
    # MSE per frame plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for qf in quality_factors:
        ax.plot(results[qf]['mse'], label=f'QF={qf} (Avg PSNR={results[qf]["avg_psnr"]:.1f}dB)', marker='o', markersize=3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('MSE')
    ax.set_title('kitty.mp4 - MSE per Frame')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('output/kitty_mjpeg_mse.png', dpi=150)
    plt.close()
    print(f"  Saved: output/kitty_mjpeg_mse.png")
    
    # Side-by-side comparison (first frame)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(frames[0])
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    for idx, qf in enumerate(quality_factors):
        axes[idx+1].imshow(results[qf]['frames'][0])
        axes[idx+1].set_title(f'QF={qf}\nPSNR={10*np.log10(255**2/results[qf]["mse"][0]):.1f}dB', fontsize=12)
        axes[idx+1].axis('off')
    
    plt.suptitle('First Frame Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/kitty_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: output/kitty_comparison.png")
    
    print("\n" + "="*60)
    print("VIDEO COMPRESSION TEST COMPLETE!")
    print("="*60)
    
    return True


if __name__ == '__main__':
    video_path = os.path.join(os.path.dirname(__file__), 'my_cat_video', 'kitty.mp4')
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        sys.exit(1)
    
    test_video_compression(video_path)
