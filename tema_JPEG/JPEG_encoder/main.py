"""JPEG Encoder - Demo Script"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
sys.path.insert(0, os.path.dirname(BASE_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

from JPEG_encoder import (
    Q_Y, Q_C, scaleQ, compress_gray, mse_psnr, compress_color,
    compress_qf, find_qf, compress_video, enc8, build_tree, get_codes,
    entropy, avg_len, enc_block, dec_block, compress_jp2k, compress_jp2k_ml,
    enc_gray, save_gray, enc_color, save_color, save_gray_a, cmp_adapt
)

try: from scipy.datasets import ascent, face
except: from scipy.misc import ascent, face

def task1():
    print("\n[1] gray")
    X = ascent()
    c, r = compress_gray(X)
    m, p = mse_psnr(X, c)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(X, cmap='gray'); ax[0].set_title('orig'); ax[0].axis('off')
    ax[1].imshow(c, cmap='gray'); ax[1].set_title(f'mse={m:.0f} psnr={p:.0f}'); ax[1].axis('off')
    ax[2].imshow(np.abs(X-c), cmap='hot'); ax[2].set_title(f'r={r:.1f}x'); ax[2].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'task1.png'), dpi=150); plt.close()
    save_gray(X.astype(np.uint8), os.path.join(OUTPUT_DIR, 'task1.jpg'))
    print(f"  mse={m:.1f} psnr={p:.1f} r={r:.1f}")
    return True

def task2():
    print("\n[2] color")
    img = face()
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax[0,0].imshow(img); ax[0,0].set_title('orig'); ax[0,0].axis('off'); ax[1,0].axis('off')
    for i, qf in enumerate([10, 50, 90]):
        Qy, Qc = scaleQ(Q_Y, qf), scaleQ(Q_C, qf)
        c = compress_color(img, Qy, Qc)
        m = np.mean((img.astype(float)-c.astype(float))**2)
        p = 10*np.log10(255*255/m)
        ax[0,i+1].imshow(c); ax[0,i+1].set_title(f'qf={qf} p={p:.0f}'); ax[0,i+1].axis('off')
        ax[1,i+1].imshow(np.clip(np.abs(img-c)*5, 0, 255).astype(np.uint8)); ax[1,i+1].set_title('diff'); ax[1,i+1].axis('off')
        save_color(img, os.path.join(OUTPUT_DIR, f'task2_qf{qf}.jpg'), qf)
        print(f"  qf={qf} psnr={p:.1f}")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'task2.png'), dpi=150); plt.close()
    return True

def task3():
    print("\n[3] adapt")
    X = ascent()
    ts = [10, 50, 100, 200]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for i, t in enumerate(ts):
        qf, m, h = find_qf(X, t)
        c, _, _ = compress_qf(X, qf)
        p = 10*np.log10(255*255/m)
        ax[0,i].imshow(c, cmap='gray'); ax[0,i].set_title(f't={t} qf={qf}'); ax[0,i].axis('off')
        ax[1,i].plot([x[1] for x in h], 'b-o'); ax[1,i].axhline(t, color='r', ls='--')
        ax[1,i].set_xlabel('i'); ax[1,i].set_ylabel('mse'); ax[1,i].grid(True)
        print(f"  t={t} qf={qf} mse={m:.1f}")
    
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'task3.png'), dpi=150); plt.close()
    
    try:
        u = input("  target MSE (enter to skip): ")
        if u.strip():
            t = float(u)
            qf, mse, _ = find_qf(X, t)
            c, _, r = compress_qf(X, qf)
            p = 10*np.log10(255*255/mse)
            print(f"  t={t}: qf={qf} mse={mse:.1f} psnr={p:.1f} r={r:.1f}x")
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(X, cmap='gray'); ax[0].set_title('orig'); ax[0].axis('off')
            ax[1].imshow(c, cmap='gray'); ax[1].set_title(f'qf={qf}'); ax[1].axis('off')
            ax[2].imshow(np.abs(X-c), cmap='hot'); ax[2].set_title('diff'); ax[2].axis('off')
            plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f'task3_mse{int(t)}.png'), dpi=150); plt.close()
            print(f"  saved task3_mse{int(t)}.png")
    except: pass
    return True

def load_vid(p, n=30):
    try: import cv2
    except: import subprocess as sp; sp.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', '-q']); import cv2
    cap = cv2.VideoCapture(p)
    if not cap.isOpened(): return None, None
    tot, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    frames, idxs = [], np.linspace(0, tot-1, min(n, tot), dtype=int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w = f.shape[:2]
            frames.append(f[:(h>>3<<3), :(w>>3<<3)])
    cap.release()
    return frames, fps

def task4():
    print("\n[4] vid")
    vp = os.path.join(BASE_DIR, 'my_cat_video', 'kitty.mp4')
    if os.path.exists(vp):
        frames, _ = load_vid(vp, 30)
        print(f"  {len(frames)} frames")
    else:
        print("  synth frames")
        frames = [np.random.randint(0, 256, (128,128,3), dtype=np.uint8) for _ in range(24)]
    res = {}
    for qf in [5, 30, 90]:
        c, ms = compress_video(frames, qf)
        res[qf] = (c, ms)
        print(f"  qf={qf} psnr={10*np.log10(255*255/np.mean(ms)):.1f}")
    fig, ax = plt.subplots(4, 6, figsize=(15, 12))
    si = [int(i*(len(frames)-1)/5) for i in range(6)]
    for j, idx in enumerate(si):
        ax[0,j].imshow(frames[idx]); ax[0,j].set_title(f'f{idx}'); ax[0,j].axis('off')
    for r, qf in enumerate([5, 30, 90], 1):
        for j, idx in enumerate(si):
            ax[r,j].imshow(res[qf][0][idx]); ax[r,j].set_title(f'qf={qf}'); ax[r,j].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'task4.png'), dpi=150); plt.close()
    return True

def bonus_huff():
    print("\n[5] huff")
    X = ascent()
    q = enc8(X[:8,:8], Q_Y)
    s, _ = enc_block(q, 0)
    t = build_tree(s)
    c = get_codes(t)
    e, a = entropy(s), avg_len(c, s)
    print(f"  n={len(s)} u={len(c)} ent={e:.2f} avg={a:.2f} eff={e/a*100:.0f}%")
    return True

def bonus_jp2k():
    print("\n[6] jp2k")
    X = ascent()
    c1, m1, _ = compress_jp2k(X)
    c3, m3, _ = compress_jp2k_ml(X, levels=3)
    cj, _ = compress_gray(X)
    mj, pj = mse_psnr(X, cj)
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    ax[0].imshow(X, cmap='gray'); ax[0].set_title('orig'); ax[0].axis('off')
    ax[1].imshow(cj, cmap='gray'); ax[1].set_title(f'jpeg p={pj:.0f}'); ax[1].axis('off')
    ax[2].imshow(c1, cmap='gray'); ax[2].set_title(f'haar1 p={10*np.log10(255*255/m1):.0f}'); ax[2].axis('off')
    ax[3].imshow(c3, cmap='gray'); ax[3].set_title(f'haar3 p={10*np.log10(255*255/m3):.0f}'); ax[3].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'bonus_jp2k.png'), dpi=150); plt.close()
    print(f"  jpeg={pj:.0f} h1={10*np.log10(255*255/m1):.0f} h3={10*np.log10(255*255/m3):.0f}")
    return True

def bonus_adapt():
    print("\n[+] adapt huff")
    X = ascent()
    for qf in [50, 75, 90]:
        s, a = cmp_adapt(X, qf)
        print(f"  qf={qf} s={s} a={a} -{(s-a)/s*100:.1f}%")
    save_gray_a(X.astype(np.uint8), os.path.join(OUTPUT_DIR, 'adapt.jpg'))
    return True

def main():
    ok = task1() & task2() & task3() & task4() & bonus_huff() & bonus_jp2k() & bonus_adapt()
    print(f"\ndone")

if __name__ == '__main__':
    main()
