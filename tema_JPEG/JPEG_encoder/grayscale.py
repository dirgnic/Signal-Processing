import numpy as np
from .core import enc8, dec8, Q_Y

def compress_gray(img, Q=None):
    if Q is None: Q = Q_Y
    h, w = img.shape
    hp, wp = (8-h%8)%8, (8-w%8)%8
    pad = np.pad(img, ((0,hp),(0,wp)), mode='edge')
    out = np.zeros_like(pad, dtype=float)
    nz = 0
    for i in range(0, pad.shape[0], 8):
        for j in range(0, pad.shape[1], 8):
            q = enc8(pad[i:i+8,j:j+8], Q)
            nz += np.count_nonzero(q)
            out[i:i+8,j:j+8] = dec8(q, Q)
    return out[:h,:w], (pad.size>>6<<6)/max(nz,1)

def mse_psnr(o, c):
    m = np.mean((o.astype(float)-c)**2)
    p = 10*np.log10(255*255/m) if m > 0 else float('inf')
    return m, p
