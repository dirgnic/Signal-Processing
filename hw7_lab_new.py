from scipy import datasets, ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

N = 256
FREQ_CUTOFF = 120
PIXEL_NOISE = 200

os.makedirs('plots_lab7', exist_ok=True)

def load_image():
    X = datasets.face(gray=True)
    plt.imshow(X, cmap=plt.cm.gray)
    plt.savefig('plots_lab7/original.png')
    plt.close()
    return X

def compute_fft(X):
    Y = np.fft.fft2(X)
    freq_db = 20 * np.log10(abs(Y))
    
    plt.imshow(freq_db)
    plt.colorbar()
    plt.savefig('plots_lab7/fft_spectrum.png')
    plt.close()
    
    return Y, freq_db

def rotate_and_fft(X, angle=45):
    X_rot = ndimage.rotate(X, angle)
    plt.imshow(X_rot, cmap=plt.cm.gray)
    plt.savefig('plots_lab7/rotated.png')
    plt.close()
    
    Y_rot = np.fft.fft2(X_rot)
    plt.imshow(20 * np.log10(abs(Y_rot)))
    plt.colorbar()
    plt.savefig('plots_lab7/fft_rotated.png')
    plt.close()
    
    return X_rot, Y_rot

def filter_high_freq(X, Y, freq_db, cutoff=FREQ_CUTOFF):
    Y_filt = Y.copy()
    Y_filt[freq_db > cutoff] = 0
    X_filt = np.real(np.fft.ifft2(Y_filt))
    
    plt.imshow(X_filt, cmap=plt.cm.gray)
    plt.title(f'Filtered (cutoff={cutoff})')
    plt.savefig('plots_lab7/filtered.png')
    plt.close()
    
    return X_filt

def add_noise(X, noise_level=PIXEL_NOISE):
    noise = np.random.randint(-noise_level, high=noise_level+1, size=X.shape)
    X_noisy = X + noise
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(X, cmap=plt.cm.gray)
    axes[0].set_title('Original')
    axes[1].imshow(X_noisy, cmap=plt.cm.gray)
    axes[1].set_title('Noisy')
    plt.savefig('plots_lab7/noisy.png')
    plt.close()
    
    return X_noisy

def exercise1():
    print("Ex 1 - Imagini si spectru")
    n1, n2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    
    x1 = np.sin(2*np.pi*n1 + 3*np.pi*n2)
    x2 = np.sin(4*np.pi*n1) + np.cos(6*np.pi*n2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(x1, cmap='gray')
    axes[0,0].set_title('sin(2πn1 + 3πn2)')
    
    Y1 = np.fft.fft2(x1)
    axes[0,1].imshow(20*np.log10(abs(Y1)+1e-10))
    axes[0,1].set_title('Spectru')
    
    axes[1,0].imshow(x2, cmap='gray')
    axes[1,0].set_title('sin(4πn1) + cos(6πn2)')
    
    Y2 = np.fft.fft2(x2)
    axes[1,1].imshow(20*np.log10(abs(Y2)+1e-10))
    axes[1,1].set_title('Spectru')
    
    plt.tight_layout()
    plt.savefig('plots_lab7/ex1_signals.png')
    plt.close()
    
    Y3 = np.zeros((N, N), dtype=complex)
    Y3[0, 5] = Y3[0, N-5] = 1
    x3 = np.real(np.fft.ifft2(Y3))
    
    Y4 = np.zeros((N, N), dtype=complex)
    Y4[5, 0] = Y4[N-5, 0] = 1
    x4 = np.real(np.fft.ifft2(Y4))
    
    Y5 = np.zeros((N, N), dtype=complex)
    Y5[5, 5] = Y5[N-5, N-5] = 1
    x5 = np.real(np.fft.ifft2(Y5))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x3, cmap='gray')
    axes[0].set_title('Y[0,5]=Y[0,N-5]=1')
    axes[1].imshow(x4, cmap='gray')
    axes[1].set_title('Y[5,0]=Y[N-5,0]=1')
    axes[2].imshow(x5, cmap='gray')
    axes[2].set_title('Y[5,5]=Y[N-5,N-5]=1')
    plt.tight_layout()
    plt.savefig('plots_lab7/ex1_inverse.png')
    plt.close()
    print("Salvat\n")

def exercise2(X):
    print("Ex 2 - Compresie prin atenuare frecvente inalte")
    Y = np.fft.fft2(X)
    freq_db = 20 * np.log10(abs(Y))
    
    snr_target = 30
    cutoffs = [100, 120, 140, 160]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, cutoff in enumerate(cutoffs):
        Y_comp = Y.copy()
        Y_comp[freq_db > cutoff] = 0
        X_comp = np.real(np.fft.ifft2(Y_comp))
        
        signal_power = np.mean(X**2)
        noise_power = np.mean((X - X_comp)**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        axes[i//2, i%2].imshow(X_comp, cmap='gray')
        axes[i//2, i%2].set_title(f'Cutoff={cutoff}, SNR={snr:.1f}dB')
    
    plt.tight_layout()
    plt.savefig('plots_lab7/ex2_compression.png')
    plt.close()
    print("Salvat\n")

def exercise3(X):
    print("Ex 3 - Eliminare zgomot")
    noise = np.random.randint(-PIXEL_NOISE, high=PIXEL_NOISE+1, size=X.shape)
    X_noisy = X + noise
    
    signal_power = np.mean(X**2)
    noise_power = np.mean(noise**2)
    snr_before = 10 * np.log10(signal_power / noise_power)
    
    Y_noisy = np.fft.fft2(X_noisy)
    freq_db = 20 * np.log10(abs(Y_noisy))
    
    Y_clean = Y_noisy.copy()
    Y_clean[freq_db > 140] = 0
    X_clean = np.real(np.fft.ifft2(Y_clean))
    
    noise_after = X - X_clean
    noise_power_after = np.mean(noise_after**2)
    snr_after = 10 * np.log10(signal_power / noise_power_after)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(X, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(X_noisy, cmap='gray')
    axes[1].set_title(f'Noisy (SNR={snr_before:.1f}dB)')
    axes[2].imshow(X_clean, cmap='gray')
    axes[2].set_title(f'Denoised (SNR={snr_after:.1f}dB)')
    plt.tight_layout()
    plt.savefig('plots_lab7/ex3_denoise.png')
    plt.close()
    
    print(f"SNR inainte: {snr_before:.2f}dB")
    print(f"SNR dupa: {snr_after:.2f}dB\n")

def main():
    print("Laborator 7 - FFT 2D (Imagini)\n")
    
    X = load_image()
    Y, freq_db = compute_fft(X)
    X_rot, Y_rot = rotate_and_fft(X)
    X_filt = filter_high_freq(X, Y, freq_db)
    X_noisy = add_noise(X)
    
    exercise1()
    exercise2(X)
    exercise3(X)
    
    print("Laborator 7 finalizat")

if __name__ == '__main__':
    main()
