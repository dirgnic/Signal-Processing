import numpy as np
from .core import enc8, dec8, Q_Y, Q_C
from .color import rgb2ycbcr, ycbcr2rgb

def compress_color(rgb, Qy=None, Qc=None, sub=True):
    if Qy is None: Qy = Q_Y
    if Qc is None: Qc = Q_C
    ycc = rgb2ycbcr(rgb.astype(np.float64))
    h, w = ycc.shape[:2]
    hp, wp = (8-h%8)%8, (8-w%8)%8
    
    channels = []
    for c in range(3):
        ch = np.pad(ycc[:,:,c], ((0,hp),(0,wp)), mode='edge')
        if sub and c > 0: ch = ch[::2,::2]
        Q = Qy if c == 0 else Qc
        out = np.zeros_like(ch)
        for i in range(0, ch.shape[0], 8):
            for j in range(0, ch.shape[1], 8):
                blk = ch[i:i+8,j:j+8]
                if blk.shape == (8,8):
                    out[i:i+8,j:j+8] = dec8(enc8(blk, Q), Q)
        if sub and c > 0:
            out = np.repeat(np.repeat(out,2,0),2,1)[:h+hp,:w+wp]
        channels.append(out[:h,:w])
    return ycbcr2rgb(np.stack(channels, axis=-1))

