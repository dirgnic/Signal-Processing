import numpy as np

def haar2d(img):
    img = img.astype(float)
    ar, dr = (img[:,::2]+img[:,1::2])/2, (img[:,::2]-img[:,1::2])/2
    LL = (ar[::2,:]+ar[1::2,:])/2
    LH = (ar[::2,:]-ar[1::2,:])/2
    HL = (dr[::2,:]+dr[1::2,:])/2
    HH = (dr[::2,:]-dr[1::2,:])/2
    return LL, LH, HL, HH

def ihaar2d(LL, LH, HL, HH):
    ar = np.zeros((LL.shape[0]<<1, LL.shape[1]), dtype=float)
    ar[::2,:], ar[1::2,:] = LL+LH, LL-LH
    dr = np.zeros((HL.shape[0]<<1, HL.shape[1]), dtype=float)
    dr[::2,:], dr[1::2,:] = HL+HH, HL-HH
    out = np.zeros((ar.shape[0], ar.shape[1]<<1), dtype=float)
    out[:,::2], out[:,1::2] = ar+dr, ar-dr
    return out

def compress_jp2k(img, sL=4.0, sH=8.0):
    h, w = img.shape
    pad = np.pad(img.astype(float), ((0,h%2),(0,w%2)), mode='edge')
    LL, LH, HL, HH = haar2d(pad)
    qLL, qLH, qHL, qHH = np.round(LL/sL), np.round(LH/sH), np.round(HL/sH), np.round(HH/sH)
    nz = np.count_nonzero(qLL)+np.count_nonzero(qLH)+np.count_nonzero(qHL)+np.count_nonzero(qHH)
    r = ihaar2d(qLL*sL, qLH*sH, qHL*sH, qHH*sH)[:h,:w]
    return r, np.mean((img.astype(float)-r)**2), (qLL.size<<2)/max(nz,1)

def compress_jp2k_ml(img, sL=4.0, sH=8.0, levels=3):
    h, w = img.shape
    p = 1<<levels
    pad = np.pad(img.astype(float), ((0,(p-h%p)%p),(0,(p-w%p)%p)), mode='edge')
    LL, det = pad, []
    for _ in range(levels):
        LL, LH, HL, HH = haar2d(LL)
        det.append((np.round(LH/sH), np.round(HL/sH), np.round(HH/sH)))
    qLL = np.round(LL/sL)
    nz = np.count_nonzero(qLL) + sum(np.count_nonzero(d[0])+np.count_nonzero(d[1])+np.count_nonzero(d[2]) for d in det)
    tot = qLL.size + sum(d[0].size*3 for d in det)
    cur = qLL*sL
    for qLH, qHL, qHH in reversed(det): cur = ihaar2d(cur, qLH*sH, qHL*sH, qHH*sH)
    return cur[:h,:w], np.mean((img.astype(float)-cur[:h,:w])**2), tot/max(nz,1)
