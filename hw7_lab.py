from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt


def imports():
    X = datasets.face(gray=True)
    plt.imshow(X, cmap=plt.cm.gray)
    plt.show()
    return X


def fourier_transform_image():
    Y = np.fft.fft2(X)
    freq_db = 20*np.log10(abs(Y))
    plt.imshow(freq_db)
    plt.colorbar()
    plt.show()

def rotate_image():
    rotate_angle = 45
    X45 = ndimage.rotate(X, rotate_angle)
    plt.imshow(X45, cmap=plt.cm.gray)
    plt.show()

def fourier_transform_rotated_image():
    Y45 = np.fft.fft2(X45)
    plt.imshow(20*np.log10(abs(Y45)))
    plt.colorbar()
    plt.show()

def filter_frequencies():
    freq_x = np.fft.fftfreq(X.shape[1])
    freq_y = np.fft.fftfreq(X.shape[0])
    plt.stem(freq_x, freq_db[:][0])
    plt.show()


def filter_high_frequencies():
    freq_cutoff = 120

    Y_cutoff = Y.copy()
    Y_cutoff[freq_db > freq_cutoff] = 0
    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                    # in practice use irfft2
    plt.imshow(X_cutoff, cmap=plt.cm.gray)
    plt.show()

def add_noise():
    pixel_noise = 200

    noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
    X_noisy = X + noise
    plt.imshow(X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.show()
    plt.imshow(X_noisy, cmap=plt.cm.gray)
    plt.title('Noisy')
    plt.show()

    