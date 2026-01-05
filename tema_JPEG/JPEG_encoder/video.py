import numpy as np
from .core import Q_Y, Q_C, scaleQ
from .color_jpeg import compress_color

def compress_video(frames, qf=75):
    Qy, Qc = scaleQ(Q_Y, qf), scaleQ(Q_C, qf)
    out, ms = [], []
    for f in frames:
        c = compress_color(f, Qy, Qc, sub=True)
        out.append(c)
        ms.append(np.mean((f.astype(float)-c.astype(float))**2))
    return out, ms
