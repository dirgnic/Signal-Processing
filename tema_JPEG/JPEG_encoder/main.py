"""
JPEG Encoder - Main Demo Script
Demonstrates all 4 tasks from the homework
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
    Q_LUMINANCE, Q_CHROMINANCE,
    scale_quantization_matrix,
    jpeg_compress_grayscale, compute_metrics,
    jpeg_compress_color,
    jpeg_compress_with_quality, find_quality_for_target_mse,
    generate_synthetic_video, compress_video_mjpeg,
    jpeg_encode_block, build_huffman_tree, get_huffman_codes,
    compute_entropy, compute_average_code_length,
    jpeg2000_like_compress_grayscale
)

# Import test images
try:
    from scipy.datasets import ascent, face
except ImportError:
    from scipy.misc import ascent, face

# Create output directory
os.makedirs('output', exist_ok=True)


def task1_grayscale_jpeg():
    """Task 1: Complete JPEG on all 8x8 blocks"""
    print("\n" + "="*60)
    print("TASK 1: Grayscale JPEG Compression")
    print("="*60)
    
    X = ascent()
    X_compressed, ratio = jpeg_compress_grayscale(X)
    mse, psnr = compute_metrics(X, X_compressed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(X, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(X_compressed, cmap='gray')
    axes[1].set_title(f'JPEG Compressed\nMSE={mse:.2f}, PSNR={psnr:.2f}dB')
    axes[1].axis('off')
    
    diff = np.abs(X.astype(np.float64) - X_compressed)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference\nCompression ratio: {ratio:.2f}x')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/task1_grayscale.png', dpi=150)
    plt.close()
    
    print(f"  Image size: {X.shape}")
    print(f"  MSE: {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Saved: output/task1_grayscale.png")
    
    return True


def task2_color_jpeg():
    """Task 2: Color JPEG with Y'CbCr and 4:2:0 subsampling"""
    print("\n" + "="*60)
    print("TASK 2: Color JPEG Compression (Y'CbCr)")
    print("="*60)
    
    img = face()
    
    # Show multiple quality levels to demonstrate compression
    quality_factors = [10, 50, 90]  # Low, medium, high quality
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')  # Empty cell
    
    mse_default = 0
    psnr_default = 0
    
    for idx, qf in enumerate(quality_factors):
        Q_luma = scale_quantization_matrix(Q_LUMINANCE, qf)
        Q_chroma = scale_quantization_matrix(Q_CHROMINANCE, qf)
        img_compressed = jpeg_compress_color(img, Q_luma, Q_chroma, subsample=True)
        
        mse = np.mean((img.astype(np.float64) - img_compressed.astype(np.float64))**2)
        psnr = 10 * np.log10(255**2 / mse)
        
        axes[0, idx+1].imshow(img_compressed)
        axes[0, idx+1].set_title(f'QF={qf}\nPSNR={psnr:.1f}dB')
        axes[0, idx+1].axis('off')
        
        # Difference (amplified)
        diff = np.abs(img.astype(np.float64) - img_compressed.astype(np.float64))
        amplify = 10 if qf >= 50 else 3
        axes[1, idx+1].imshow(np.clip(diff * amplify, 0, 255).astype(np.uint8))
        axes[1, idx+1].set_title(f'Difference ({amplify}x)')
        axes[1, idx+1].axis('off')
        
        if qf == 50:
            mse_default = mse
            psnr_default = psnr
    
    plt.tight_layout()
    plt.savefig('output/task2_color.png', dpi=150)
    plt.close()
    
    print(f"  Image size: {img.shape}")
    print(f"  Quality factors tested: {quality_factors}")
    print(f"  QF=10 (low): visible artifacts, high compression")
    print(f"  QF=50 (standard): MSE={mse_default:.2f}, PSNR={psnr_default:.1f}dB")
    print(f"  QF=90 (high): nearly identical to original")
    print(f"  Saved: output/task2_color.png")
    
    return True


def task3_adaptive_mse():
    """Task 3: Adaptive compression with target MSE"""
    print("\n" + "="*60)
    print("TASK 3: Adaptive Compression (Target MSE)")
    print("="*60)
    
    X = ascent()
    target_mse_values = [10, 50, 100, 200]
    
    fig, axes = plt.subplots(2, len(target_mse_values), figsize=(16, 8))
    
    for idx, target_mse in enumerate(target_mse_values):
        qf, achieved_mse, history = find_quality_for_target_mse(X, target_mse)
        compressed, _, ratio = jpeg_compress_with_quality(X, qf)
        psnr = 10 * np.log10(255**2 / achieved_mse)
        
        axes[0, idx].imshow(compressed, cmap='gray')
        axes[0, idx].set_title(f'Target={target_mse}\nQF={qf}, MSE={achieved_mse:.1f}')
        axes[0, idx].axis('off')
        
        mses = [h[1] for h in history]
        axes[1, idx].plot(range(len(history)), mses, 'b-o')
        axes[1, idx].axhline(target_mse, color='r', linestyle='--')
        axes[1, idx].set_xlabel('Iteration')
        axes[1, idx].set_ylabel('MSE')
        axes[1, idx].set_title(f'Binary Search\nPSNR={psnr:.1f}dB')
        axes[1, idx].grid(True)
        
        print(f"  Target MSE={target_mse}: QF={qf}, Achieved MSE={achieved_mse:.2f}")
    
    plt.tight_layout()
    plt.savefig('output/task3_adaptive.png', dpi=150)
    plt.close()
    
    print(f"  Saved: output/task3_adaptive.png")
    
    # Optional interactive mode: user-specified target MSE
    try:
        user_input = input("  Enter a target MSE for extra demo (or press Enter to skip): ")
    except EOFError:
        user_input = ""

    if user_input.strip() != "":
        try:
            target_mse_user = float(user_input)
            qf_user, achieved_mse_user, history_user = find_quality_for_target_mse(X, target_mse_user)
            compressed_user, _, ratio_user = jpeg_compress_with_quality(X, qf_user)
            psnr_user = 10 * np.log10(255**2 / achieved_mse_user) if achieved_mse_user > 0 else float('inf')

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(X, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(compressed_user, cmap='gray')
            axes[1].set_title(f'JPEG (QF={qf_user})\nMSE={achieved_mse_user:.2f}, PSNR={psnr_user:.2f}dB')
            axes[1].axis('off')

            diff_user = np.abs(X.astype(np.float64) - compressed_user)
            axes[2].imshow(diff_user, cmap='hot')
            axes[2].set_title(f'Difference (target MSE={target_mse_user})\nRatio: {ratio_user:.2f}x')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig('output/task3_adaptive_user.png', dpi=150)
            plt.close()

            print(f"  User target MSE: {target_mse_user}")
            print(f"  Found QF: {qf_user}")
            print(f"  Achieved MSE: {achieved_mse_user:.4f}")
            print(f"  PSNR: {psnr_user:.2f} dB")
            print(f"  Compression ratio: {ratio_user:.2f}x")
            print(f"  Saved: output/task3_adaptive_user.png")
        except ValueError:
            print("  Invalid input: please enter a numeric MSE value next time.")
    
    return True


def load_video_frames(video_path, max_frames=30):
    """Load frames from a video file using OpenCV"""
    try:
        import cv2
    except ImportError:
        print("  Installing OpenCV...")
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', '-q'])
        import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    
    frames = []
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make dimensions multiple of 8
            h, w = frame.shape[:2]
            frame = frame[:(h//8)*8, :(w//8)*8]
            frames.append(frame)
    
    cap.release()
    return frames, fps


def task4_video_mjpeg():
    """Task 4: Video compression with MJPEG on real video (kitty.mp4)"""
    print("\n" + "="*60)
    print("TASK 4: Video Compression (MJPEG) - kitty.mp4")
    print("="*60)
    
    # Try to load real video first
    video_path = os.path.join(os.path.dirname(__file__), 'my_cat_video', 'kitty.mp4')
    
    if os.path.exists(video_path):
        print(f"  Loading video: kitty.mp4")
        frames, fps = load_video_frames(video_path, max_frames=30)
    else:
        print("  Video not found, using synthetic video...")
        frames = generate_synthetic_video(n_frames=24, size=(128, 128))
    
    quality_factors = [30, 60, 90]
    results = {}
    
    for qf in quality_factors:
        print(f"  Compressing with QF={qf}...")
        compressed, mse_list = compress_video_mjpeg(frames, qf)
        results[qf] = {'frames': compressed, 'mse': mse_list}
    
    # Visualize sample frames - spread across entire video
    fig, axes = plt.subplots(len(quality_factors) + 1, 6, figsize=(18, 12))
    
    # Use evenly spaced indices across all loaded frames
    n_samples = 6
    sample_indices = [int(i * (len(frames) - 1) / (n_samples - 1)) for i in range(n_samples)]
    
    for idx, frame_idx in enumerate(sample_indices):
        if frame_idx < len(frames):
            axes[0, idx].imshow(frames[frame_idx])
            axes[0, idx].set_title(f'Original\nFrame {frame_idx}')
            axes[0, idx].axis('off')
    
    for row, qf in enumerate(quality_factors, 1):
        for idx, frame_idx in enumerate(sample_indices):
            if frame_idx < len(results[qf]['frames']):
                axes[row, idx].imshow(results[qf]['frames'][frame_idx])
                mse = results[qf]['mse'][frame_idx]
                axes[row, idx].set_title(f'QF={qf}\nMSE={mse:.1f}')
                axes[row, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/task4_video_frames.png', dpi=150)
    plt.close()
    
    # MSE per frame plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for qf in quality_factors:
        avg_psnr = 10 * np.log10(255**2 / np.mean(results[qf]['mse']))
        ax.plot(results[qf]['mse'], label=f'QF={qf} (PSNR={avg_psnr:.1f}dB)', marker='o', markersize=3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('MSE')
    ax.set_title('kitty.mp4 - MSE per Frame')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('output/task4_video_mse.png', dpi=150)
    plt.close()
    
    print(f"  Number of frames: {len(frames)}")
    print(f"  Frame size: {frames[0].shape}")
    for qf in quality_factors:
        avg_mse = np.mean(results[qf]['mse'])
        avg_psnr = 10 * np.log10(255**2 / avg_mse)
        print(f"  QF={qf}: Avg MSE={avg_mse:.2f}, Avg PSNR={avg_psnr:.2f}dB")
    print(f"  Saved: output/task4_video_frames.png")
    print(f"  Saved: output/task4_video_mse.png")
    
    return True


def bonus_huffman():
    """Bonus: Huffman coding demonstration"""
    print("\n" + "="*60)
    print("BONUS: Huffman Coding")
    print("="*60)
    
    X = ascent()
    block = X[:8, :8]
    quantized = jpeg_encode_block(block, Q_LUMINANCE)
    
    tree = build_huffman_tree(quantized)
    codes = get_huffman_codes(tree)
    
    entropy = compute_entropy(quantized)
    avg_len = compute_average_code_length(codes, quantized)
    efficiency = entropy / avg_len * 100
    
    print(f"  Block size: 8x8")
    print(f"  Unique symbols: {len(codes)}")
    print(f"  Entropy: {entropy:.2f} bits/symbol")
    print(f"  Avg Huffman code length: {avg_len:.2f} bits/symbol")
    print(f"  Coding efficiency: {efficiency:.1f}%")
    
    print("\n  Sample Huffman codes (shortest 5):")
    sorted_codes = sorted(codes.items(), key=lambda x: len(x[1]))
    for symbol, code in sorted_codes[:5]:
        print(f"    Symbol {symbol:4d}: {code}")
    
    return True


def bonus_jpeg2000_like():
    """Bonus: JPEG 2000-like wavelet compression demo (grayscale)"""
    print("\n" + "="*60)
    print("BONUS: JPEG 2000-like (Haar Wavelet) Compression")
    print("="*60)

    X = ascent()

    # Baseline: block-DCT JPEG from Task 1 (recompute here)
    X_jpeg, ratio_jpeg = jpeg_compress_grayscale(X)
    mse_jpeg, psnr_jpeg = compute_metrics(X, X_jpeg)

    # JPEG 2000-like: 1-level Haar wavelet + quantization
    recon_wt, mse_wt, ratio_wt, stats = jpeg2000_like_compress_grayscale(
        X, step_LL=4.0, step_H=8.0
    )
    psnr_wt = 10 * np.log10(255**2 / mse_wt) if mse_wt > 0 else float("inf")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(X, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(X_jpeg, cmap="gray")
    axes[1].set_title(
        f"JPEG (DCT)\nMSE={mse_jpeg:.2f}, PSNR={psnr_jpeg:.2f}dB\nRatio={ratio_jpeg:.2f}x"
    )
    axes[1].axis("off")

    axes[2].imshow(recon_wt, cmap="gray")
    axes[2].set_title(
        f"JPEG 2000-like (Haar)\nMSE={mse_wt:.2f}, PSNR={psnr_wt:.2f}dB\nRatio={ratio_wt:.2f}x"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("output/bonus_jpeg2000_like.png", dpi=150)
    plt.close()

    print("  Grayscale test image size:", X.shape)
    print("  --- JPEG (block DCT) ---")
    print(f"    MSE: {mse_jpeg:.4f}")
    print(f"    PSNR: {psnr_jpeg:.2f} dB")
    print(f"    Compression ratio (non-zero coeffs): {ratio_jpeg:.2f}x")
    print("  --- JPEG 2000-like (Haar) ---")
    print(f"    MSE: {mse_wt:.4f}")
    print(f"    PSNR: {psnr_wt:.2f} dB")
    print(f"    Compression ratio (non-zero coeffs): {ratio_wt:.2f}x")
    print(
        "    Non-zero wavelet coeffs:",
        stats.get("nonzero_coeffs"),
        "/",
        stats.get("total_coeffs"),
    )
    print("  Saved: output/bonus_jpeg2000_like.png")

    return True


def main():
    """Run all tasks"""
    print("\n" + "#"*60)
    print("#" + " "*20 + "JPEG ENCODER DEMO" + " "*21 + "#")
    print("#"*60)
    
    success = True
    
    success &= task1_grayscale_jpeg()
    success &= task2_color_jpeg()
    success &= task3_adaptive_mse()
    success &= task4_video_mjpeg()
    success &= bonus_huffman()
    success &= bonus_jpeg2000_like()
    
    print("\n" + "="*60)
    if success:
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
    else:
        print("SOME TASKS FAILED!")
    print("="*60)
    print(f"Output files saved in: {os.path.abspath('output')}")
    
    return success


if __name__ == '__main__':
    main()
