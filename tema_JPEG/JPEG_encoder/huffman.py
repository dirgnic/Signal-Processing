import heapq
import numpy as np
from collections import Counter

class HNode:
    def __init__(s, sym=None, f=0, l=None, r=None): s.sym, s.f, s.l, s.r = sym, f, l, r
    def __lt__(s, o): return s.f < o.f

ZZ = [(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
      (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
      (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
      (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)]

def zigzag(b): return np.array([b[r,c] for r,c in ZZ], dtype=b.dtype)
def izigzag(a):
    b = np.zeros((8,8), dtype=a.dtype)
    for k,(r,c) in enumerate(ZZ): b[r,c] = a[k]
    return b

def build_tree(d):
    f = list(d.flatten().astype(int)) if isinstance(d, np.ndarray) else list(d)
    h = [HNode(s,c) for s,c in Counter(f).items()]
    heapq.heapify(h)
    while len(h) > 1:
        l, r = heapq.heappop(h), heapq.heappop(h)
        heapq.heappush(h, HNode(f=l.f+r.f, l=l, r=r))
    return h[0] if h else None

def get_codes(n, p='', c=None):
    if c is None: c = {}
    if n is None: return c
    if n.sym is not None: c[n.sym] = p or '0'
    get_codes(n.l, p+'0', c); get_codes(n.r, p+'1', c)
    return c

def entropy(d):
    f = list(d.flatten().astype(int)) if isinstance(d, np.ndarray) else list(d)
    if not f: return 0.0
    cnt = Counter(f); t = sum(cnt.values())
    return -sum((v/t)*np.log2(v/t) for v in cnt.values())

def avg_len(c, d):
    f = list(d.flatten().astype(int)) if isinstance(d, np.ndarray) else list(d)
    if not f: return 0.0
    cnt = Counter(f); t = sum(cnt.values())
    return sum(len(c[s])*cnt[s] for s in cnt)/t

def rle_ac(ac):
    syms, run = [], 0
    for x in [int(v) for v in np.asarray(ac).flatten()[:63]]:
        if x == 0:
            run += 1
            if run == 16: syms.append(('Z',)); run = 0
        else:
            syms.append(('A', run, x)); run = 0
    syms.append(('E',))
    return syms

def enc_block(q, pdc=0):
    zz = zigzag(q)
    dc = int(zz[0])
    return [('D', dc-pdc)] + rle_ac(zz[1:]), dc

def dec_rle(s):
    ac = []
    for x in s:
        if x[0] == 'E': break
        if x[0] == 'Z': ac.extend([0]*(1<<4))
        elif x[0] == 'A': ac.extend([0]*x[1] + [x[2]])
    return (ac + [0]*63)[:63]

def dec_block(s, pdc=0):
    dc = pdc + s[0][1]
    zz = np.zeros(64, dtype=int)
    zz[0] = dc; zz[1:] = dec_rle(s[1:])
    return izigzag(zz), dc
