import numpy as np
from scipy.fft import dctn
from collections import Counter
import heapq
from .core import Q_Y, Q_C, scaleQ
from .color import rgb2ycbcr
from .huffman import zigzag

DC_BITS = [0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
DC_VALS = [0,1,2,3,4,5,6,7,8,9,10,11]
AC_BITS = [0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d]
AC_VALS = [0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa]

DC_C_BITS = [0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
DC_C_VALS = [0,1,2,3,4,5,6,7,8,9,10,11]
AC_C_BITS = [0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77]
AC_C_VALS = [0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa]

def _huff(bits, vals):
    c, code, k = {}, 0, 0
    for ln in range(1, 17):
        for _ in range(bits[ln]):
            c[vals[k]] = (code, ln); code += 1; k += 1
        code <<= 1
    return c

DC_T = _huff(DC_BITS, DC_VALS); AC_T = _huff(AC_BITS, AC_VALS)
DC_C_T = _huff(DC_C_BITS, DC_C_VALS); AC_C_T = _huff(AC_C_BITS, AC_C_VALS)

class BitW:
    def __init__(s): s.buf, s.acc, s.n = bytearray(), 0, 0
    def write(s, v, nb):
        for i in range(nb-1, -1, -1):
            s.acc = (s.acc<<1)|((v>>i)&1); s.n += 1
            if s.n == 8:
                s.buf.append(s.acc&0xFF)
                if s.acc == 0xFF: s.buf.append(0)
                s.acc, s.n = 0, 0
    def flush(s):
        if s.n: 
            s.acc = (s.acc<<(8-s.n)) | ((1<<(8-s.n))-1)
            s.buf.append(s.acc&0xFF)
            if s.acc == 0xFF: s.buf.append(0)
            s.acc, s.n = 0, 0

def _cat(v):
    if v == 0: return 0, 0
    c = abs(v).bit_length()
    b = (~abs(v))&((1<<c)-1) if v < 0 else abs(v)
    return c, b

def _app0(): return b'\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
def _dqt(Q, tid=0):
    d = bytearray(b'\xFF\xDB')
    p = bytearray([tid&0x0F]) + bytearray(int(x)&0xFF for x in Q.flatten())
    d += bytes([len(p)+2>>8, (len(p)+2)&0xFF]) + p
    return bytes(d)

def _sof0_g(h, w): return b'\xFF\xC0\x00\x0B\x08' + bytes([h>>8,h&0xFF,w>>8,w&0xFF,1,1,0x11,0])
def _sof0_c(h, w): return b'\xFF\xC0\x00\x11\x08' + bytes([h>>8,h&0xFF,w>>8,w&0xFF,3,1,0x22,0,2,0x11,1,3,0x11,1])

def _dht(bits, vals, tc, th):
    d = bytearray(b'\xFF\xC4')
    p = bytearray([((tc&1)<<4)|(th&0x0F)]) + bytearray(bits[1:17]) + bytearray(vals)
    d += bytes([len(p)+2>>8, (len(p)+2)&0xFF]) + p
    return bytes(d)

def _sos_g(): return b'\xFF\xDA\x00\x08\x01\x01\x00\x00\x3F\x00'
def _sos_c(): return b'\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00'

def _enc_blk(bw, blk, Q, dc_t, ac_t, pdc):
    dct = dctn((blk.astype(np.int16)-128).astype(np.float64), norm='ortho')
    qb = np.round(dct/Q).astype(int)
    zz = zigzag(qb)
    dc, diff = int(zz[0]), int(zz[0])-pdc
    c, b = _cat(diff)
    cd, l = dc_t.get(c, dc_t.get(0, (0,1)))
    bw.write(cd, l)
    if c > 0: bw.write(b, c)
    run = 0
    for k in range(1, 64):
        ac = int(zz[k])
        if ac == 0: run += 1; continue
        while run > 15:
            cd, l = ac_t.get(0xF0, ac_t.get(0, (0,1))); bw.write(cd, l); run -= 16
        c, b = _cat(ac)
        cd, l = ac_t.get((run<<4)|c, ac_t.get(0, (0,1))); bw.write(cd, l)
        if c > 0: bw.write(b, c)
        run = 0
    if run > 0: cd, l = ac_t.get(0, (0,1)); bw.write(cd, l)
    return dc

def enc_gray(img, qf=75):
    img = np.asarray(img).astype(np.uint8)
    h, w = img.shape
    hp, wp = (8-h%8)%8, (8-w%8)%8
    pad = np.pad(img, ((0,hp),(0,wp)), mode='edge')
    Q = scaleQ(Q_Y, qf)
    bw, pdc = BitW(), 0
    for i in range(0, pad.shape[0], 8):
        for j in range(0, pad.shape[1], 8):
            pdc = _enc_blk(bw, pad[i:i+8,j:j+8], Q, DC_T, AC_T, pdc)
    bw.flush()
    return b'\xFF\xD8' + _app0() + _dqt(Q) + _sof0_g(h,w) + _dht(DC_BITS,DC_VALS,0,0) + _dht(AC_BITS,AC_VALS,1,0) + _sos_g() + bytes(bw.buf) + b'\xFF\xD9'

def save_gray(img, path, qf=75):
    with open(path, 'wb') as f: f.write(enc_gray(img, qf))

def enc_color(rgb, qf=75):
    rgb = np.asarray(rgb).astype(np.uint8)
    h, w = rgb.shape[:2]
    ycc = rgb2ycbcr(rgb.astype(np.float64))
    hp, wp = (16-h%16)%16, (16-w%16)%16
    Y = np.pad(ycc[:,:,0], ((0,hp),(0,wp)), mode='edge')
    Cb = np.pad(ycc[:,:,1], ((0,hp),(0,wp)), mode='edge')[::2,::2]
    Cr = np.pad(ycc[:,:,2], ((0,hp),(0,wp)), mode='edge')[::2,::2]
    Qy, Qc = scaleQ(Q_Y, qf), scaleQ(Q_C, qf)
    bw, pdc = BitW(), {'Y':0,'Cb':0,'Cr':0}
    for i in range(0, Y.shape[0], 16):
        for j in range(0, Y.shape[1], 16):
            for by in (0,8):
                for bx in (0,8):
                    pdc['Y'] = _enc_blk(bw, Y[i+by:i+by+8,j+bx:j+bx+8], Qy, DC_T, AC_T, pdc['Y'])
            ci, cj = i>>1, j>>1
            pdc['Cb'] = _enc_blk(bw, Cb[ci:ci+8,cj:cj+8], Qc, DC_C_T, AC_C_T, pdc['Cb'])
            pdc['Cr'] = _enc_blk(bw, Cr[ci:ci+8,cj:cj+8], Qc, DC_C_T, AC_C_T, pdc['Cr'])
    bw.flush()
    return b'\xFF\xD8' + _app0() + _dqt(Qy,0) + _dqt(Qc,1) + _sof0_c(h,w) + _dht(DC_BITS,DC_VALS,0,0) + _dht(AC_BITS,AC_VALS,1,0) + _dht(DC_C_BITS,DC_C_VALS,0,1) + _dht(AC_C_BITS,AC_C_VALS,1,1) + _sos_c() + bytes(bw.buf) + b'\xFF\xD9'

def save_color(rgb, path, qf=75):
    with open(path, 'wb') as f: f.write(enc_color(rgb, qf))

def _c2(v): return 0 if v==0 else abs(v).bit_length()

def _scan(img, Q):
    h, w = img.shape
    pad = np.pad(img, ((0,(8-h%8)%8),(0,(8-w%8)%8)), mode='edge')
    dc, ac, pdc = Counter(), Counter(), 0
    for i in range(0, pad.shape[0], 8):
        for j in range(0, pad.shape[1], 8):
            zz = zigzag(np.round(dctn((pad[i:i+8,j:j+8].astype(np.int16)-128).astype(float), norm='ortho')/Q).astype(int))
            dc[_c2(int(zz[0])-pdc)] += 1; pdc = int(zz[0]); run = 0
            for k in range(1, 64):
                v = int(zz[k])
                if v == 0: run += 1; continue
                while run > 15: ac[0xF0] += 1; run -= 16
                ac[(run<<4)|_c2(v)] += 1; run = 0
            ac[0] += 1
            
    if 0 not in ac: ac[0] = 1
    if 0xF0 not in ac: ac[0xF0] = 1
    for c in range(12):
        if c not in dc: dc[c] = 1
    return dc, ac

def _canon(freq):
    if not freq: return [0]*17, [], {}
    
    h = [type('N',(object,),{'s':s,'f':f,'l':None,'r':None,'__lt__':lambda a,b:a.f<b.f})() for s,f in freq.items()]
    heapq.heapify(h)
    while len(h)>1:
        a,b = heapq.heappop(h), heapq.heappop(h)
        n = type('N',(object,),{'s':None,'f':a.f+b.f,'l':a,'r':b,'__lt__':lambda a,b:a.f<b.f})()
        heapq.heappush(h, n)
    def gc(n,d=0,c=None):
        if c is None: c={}
        if n.s is not None: c[n.s]=max(1,d)
        if n.l: gc(n.l,d+1,c)
        if n.r: gc(n.r,d+1,c)
        return c
    lens = gc(h[0]) if h else {}
    
    bits = [0]*33
    for l in lens.values(): bits[l] += 1
 
    i = 32
    while i > 16:
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0: j -= 1
            bits[i] -= 2
            bits[i-1] += 1
            bits[j+1] += 2
            bits[j] -= 1
        i -= 1

    while bits[0] > 0:
        j = 1
        while bits[j] == 0: j += 1
        bits[0] -= 1
        bits[j] -= 1
        bits[j-1] += 2 if j > 1 else 1
    
    # Kraft inequality must be strict: sum(bits[i] * 2^(16-i)) < 2^16
    kraft = sum(bits[i] * (1 << (16-i)) for i in range(1, 17))
    if kraft >= (1 << 16):
        for i in range(16, 0, -1):
            if bits[i] > 0 and i < 16:
                bits[i] -= 1
                bits[i+1] += 2
                break
    syms = sorted(lens.keys(), key=lambda s:(lens[s],s))
    new_lens = {}
    idx = 0
    for ln in range(1, 17):
        for _ in range(bits[ln]):
            if idx < len(syms):
                new_lens[syms[idx]] = ln
                idx += 1
    syms = sorted(new_lens.keys(), key=lambda s:(new_lens[s],s))
    bits_out, vals, codes = [0]*17, [], {}
    code, pl = 0, 0
    for s in syms:
        ln = new_lens[s]
        code <<= (ln-pl)
        codes[s] = (code,ln)
        bits_out[ln] += 1
        vals.append(s)
        code += 1
        pl = ln
    return bits_out, vals, codes

def enc_gray_a(img, qf=75):
    img = np.asarray(img).astype(np.uint8)
    h, w = img.shape; Q = scaleQ(Q_Y, qf)
    dc_freq, ac_freq = _scan(img, Q)
    dcb, dcv, dcc = _canon(dc_freq)
    acb, acv, acc = _canon(ac_freq)
    pad = np.pad(img, ((0,(8-h%8)%8),(0,(8-w%8)%8)), mode='edge')
    bw, pdc = BitW(), 0
    for i in range(0, pad.shape[0], 8):
        for j in range(0, pad.shape[1], 8): pdc = _enc_blk(bw, pad[i:i+8,j:j+8], Q, dcc, acc, pdc)
    bw.flush()
    return b'\xFF\xD8' + _app0() + _dqt(Q) + _sof0_g(h,w) + _dht(dcb,dcv,0,0) + _dht(acb,acv,1,0) + _sos_g() + bytes(bw.buf) + b'\xFF\xD9'

def save_gray_a(img, path, qf=75):
    with open(path, 'wb') as f: f.write(enc_gray_a(img, qf))

def cmp_adapt(img, qf=75): return len(enc_gray(img, qf)), len(enc_gray_a(img, qf))
