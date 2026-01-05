import numpy as np
from .core import Q_Y, scaleQ
from .grayscale import compress_gray

def compress_qf(img, qf):
    c, r = compress_gray(img, scaleQ(Q_Y, qf))
    return c, np.mean((img.astype(float)-c)**2), r

def find_qf(img, t, tol=0.5, mx=20):
    lo, hi, bqf, bm = 1, 100, 50, float('inf')
    h = []
    for _ in range(mx):
        mid = (lo+hi)>>1
        _, m, _ = compress_qf(img, mid)
        h.append((mid, m))
        if abs(m-t) < tol: return mid, m, h
        if abs(m-t) < abs(bm-t): bm, bqf = m, mid
        if m > t: lo = mid+1
        else: hi = mid-1
        if lo > hi: break
    return bqf, bm, h
