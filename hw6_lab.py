import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, filtfilt

np.random.seed(0)
os.makedirs('plots_lab6', exist_ok=True)

print("Laborator 6 - Convolutie si Filtre\n")

# Ex 1
print("Ex 1 - sinc^2: esantionare si reconstructie")
B = 1
t = np.linspace(-3, 3, 3000)
x_orig = np.sinc(B * t)**2

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

for i, fs in enumerate([1, 1.5, 2, 4]):
    Ts = 1 / fs
    n = np.arange(-int(3*fs), int(3*fs)+1)
    ts = n * Ts
    xs = np.sinc(B * ts)**2
    
    x_rec = sum(xs[k] * np.sinc((t - ts[k]) / Ts) for k in range(len(ts)))
    
    axes[i].plot(t, x_orig, 'b-', alpha=0.4, label='x(t)')
    axes[i].stem(ts, xs, linefmt='r-', markerfmt='ro', basefmt='k-', label='x[n]')
    axes[i].plot(t, x_rec, 'g--', alpha=0.7, label='x_rec(t)')
    axes[i].set_title(f'fs={fs}Hz, Ts={Ts:.2f}s')
    axes[i].legend()
    axes[i].grid(alpha=0.3)
    axes[i].set_xlim(-3, 3)

plt.tight_layout()
plt.savefig('plots_lab6/ex1_sinc_esantionare.pdf')
plt.savefig('plots_lab6/ex1_sinc_esantionare.png')
plt.close()

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fs = 2

for i, B in enumerate([0.5, 1, 2]):
    x_orig = np.sinc(B * t)**2
    Ts = 1 / fs
    n = np.arange(-int(3*fs), int(3*fs)+1)
    ts = n * Ts
    xs = np.sinc(B * ts)**2
    
    x_rec = sum(xs[k] * np.sinc((t - ts[k]) / Ts) for k in range(len(ts)))
    
    axes[i].plot(t, x_orig, 'b-', alpha=0.4)
    axes[i].stem(ts, xs, linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[i].plot(t, x_rec, 'g--', alpha=0.7)
    axes[i].set_title(f'B={B}, fs={fs}Hz, Nyquist={2*B}Hz')
    axes[i].grid(alpha=0.3)
    axes[i].set_xlim(-3, 3)

plt.tight_layout()
plt.savefig('plots_lab6/ex1_variatie_B.pdf')
plt.savefig('plots_lab6/ex1_variatie_B.png')
plt.close()
print("Obs: fs<2B -> aliasing, fs>=2B -> perfect\n")

# Ex 2 - Convolutie repetata
print("Ex 2 - Convolutie repetata")
x = np.random.rand(100)
x1 = np.convolve(x, x)
x2 = np.convolve(x1, x1)
x3 = np.convolve(x2, x2)
x4 = np.convolve(x3, x3)
x5 =  np.convolve(x4, x4)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(x)
axes[0].set_title('Iteratia 0 (Original)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x1)
axes[1].set_title('Iteratia 1: x * x')
axes[1].grid(True, alpha=0.3)

axes[2].plot(x2)
axes[2].set_title('Iteratia 2: (x*x) * (x*x)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(x3)
axes[3].set_title('Iteratia 3: rez_anterior * rez_anterior')
axes[3].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('plots_lab6/ex2_convolutie_repetata.pdf', format='pdf')
plt.savefig('plots_lab6/ex2_convolutie_repetata.png', format='png')
plt.close()

print("Obsev: Forma devine gaussiana, lung creste exponential\n")

# Ex 3 - Inmultirea polinoamelor
print("Ex 3 - Inmultirea polinoamelor")
N = 100
p = np.random.randint(0, 1000, size=N)
q = np.random.randint(0, 1000, size=N)

r_conv = np.convolve(p, q)

p_fft = np.fft.fft(p, 2 * N - 1)
q_fft = np.fft.fft(q, 2 * N - 1)
r_fft = np.real(np.fft.ifft(p_fft * q_fft))

similar = np.allclose(r_conv, r_fft)
print(f"Rezultate identice (convolutie vs FFT): {similar}\n")

# Ex 4
print("Ex 4 - Deplasare circulara")
n = 20
x = np.sin(3 * np.linspace(0, 2*np.pi, n, endpoint=False))
d = 5
y = np.roll(x, d)

X = np.fft.fft(x)
Y = np.fft.fft(y)

r1 = np.fft.ifft(X * np.conj(Y))
d1 = np.argmax(np.abs(r1))

r2 = np.fft.ifft(Y / (X + 1e-10))
d2 = int(np.round(np.angle(r2[1]) * n / (2*np.pi))) % n

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].stem(x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('x[n] original')
axes[0].grid(alpha=0.3)

axes[1].stem(y, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'y[n] deplasat circular cu d={d}')
axes[1].grid(alpha=0.3)

axes[2].plot(np.abs(r1), 'g-o', label=f'IFFT(X*conj(Y)): d={d1}')
axes[2].axvline(d1, color='g', linestyle='--', alpha=0.5)
axes[2].set_title('Recuperare d prin corelatie')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab6/ex4_deplasare_circulara.pdf')
plt.savefig('plots_lab6/ex4_deplasare_circulara.png')
plt.close()

print(f"d original={d}, metoda1(corelatie)={d1}, metoda2(deconv)={d2}")
print("Obs: metoda1=corelatie robusta, metoda2=deconvolutie sensibila zgomot\n")

# Ex 5 - Ferestre
print("Ex 5 - Ferestre")

def rect_window(N):
    return np.ones(N)

def hanning_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))

f = 100
A = 1
phi = 0
Nw = 200
N_total = 1000
t = np.linspace(0, 1, N_total)
signal = A * np.sin(2 * np.pi * f * t + phi)

rect_win = np.pad(rect_window(Nw), (0, N_total - Nw), constant_values=0)
hann_win = np.pad(hanning_window(Nw), (0, N_total - Nw), constant_values=0)

sig_rect = signal * rect_win
sig_hann = signal * hann_win

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(t, signal)
axes[0].set_title('Semnal Original (f=100Hz)')
axes[0].set_xlabel('Timp (s)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, sig_rect)
axes[1].set_title('Fereastra Dreptunghiulara (Nw=200)')
axes[1].set_xlabel('Timp (s)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, sig_hann)
axes[2].set_title('Fereastra Hanning (Nw=200)')
axes[2].set_xlabel('Timp (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab6/ex5_ferestre.pdf', format='pdf')
plt.savefig('plots_lab6/ex5_ferestre.png', format='png')
plt.close()
print("Ferestre aplicate si salvate\n")

# Ex 6 - Analiza trafic (Train.csv)
print("Ex 6 - Analiza Date Trafic")

data = pd.read_csv("data_lab5/Train.csv", parse_dates=["Datetime"], dayfirst=True)
traffic = data["Count"].values
fs = 1 / 3600

# a) 3 zile
print("a) Selectie 3 zile (72 ore)")
x = traffic[:72]
t_hours = np.arange(len(x))

# b) Medie alunecatoare
print("b) Filtrare medie alunecatoare")
windows = [5, 9, 13, 17]

fig, axes = plt.subplots(len(windows) + 1, 1, figsize=(14, 12))
axes[0].plot(t_hours, x, linewidth=1)
axes[0].set_title('Semnal Original (3 zile)')
axes[0].set_ylabel('Vehicule')
axes[0].grid(True, alpha=0.3)

for i, w in enumerate(windows):
    filtered = np.convolve(x, np.ones(w), 'valid') / w
    t_filt = np.arange(len(filtered))
    axes[i+1].plot(t_filt, filtered, linewidth=1, color='red')
    axes[i+1].set_title(f'Medie Alunecatoare w={w}')
    axes[i+1].set_ylabel('Vehicule')
    axes[i+1].grid(True, alpha=0.3)

axes[-1].set_xlabel('Ore')
plt.tight_layout()
plt.savefig('plots_lab6/ex6b_medie_alunecatoare.pdf', format='pdf')
plt.savefig('plots_lab6/ex6b_medie_alunecatoare.png', format='png')
plt.close()
print("Observatie: Ferestre mai mari -> netezire mai puternica, intarziere mai mare\n")

# c) Alegere frecventa taiere
print("c) Frecventa de taiere")
cutoff_pct = 0.05
f_nyquist = fs / 2
f_cutoff_hz = fs * cutoff_pct
f_norm = f_cutoff_hz / f_nyquist

print(f"Frecventa esantionare: fs = {fs:.10f} Hz")
print(f"Frecventa Nyquist: {f_nyquist:.10f} Hz")
print(f"Frecventa taiere (5% din fs): {f_cutoff_hz:.10f} Hz")
print(f"Frecventa taiere normalizata [0,1]: {f_norm:.4f}")
print("Argumentare: Pastram doar variatiile lente (trenduri), eliminam variatii orare\n")

# d) Proiectare filtre
print("d) Proiectare filtre ordin 5")
order = 5
rp = 5

b_butter, a_butter = butter(order, f_norm, btype='low')
b_cheby, a_cheby = cheby1(order, rp, f_norm, btype='low')
print("Filtre Butterworth si Chebyshev create\n")

# e) Aplicare filtre
print("e) Filtrare semnal")
x_butter = filtfilt(b_butter, a_butter, x)
x_cheby = filtfilt(b_cheby, a_cheby, x)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(t_hours, x, linewidth=1, alpha=0.7, label='Original')
axes[0].set_title('Semnal Original')
axes[0].set_ylabel('Vehicule')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_hours, x_butter, linewidth=1.5, color='blue', label='Butterworth')
axes[1].set_title('Filtru Butterworth (ord=5)')
axes[1].set_ylabel('Vehicule')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_hours, x_cheby, linewidth=1.5, color='red', label='Chebyshev')
axes[2].set_title(f'Filtru Chebyshev (ord=5, rp={rp}dB)')
axes[2].set_xlabel('Ore')
axes[2].set_ylabel('Vehicule')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab6/ex6e_comparatie_filtre.pdf', format='pdf')
plt.savefig('plots_lab6/ex6e_comparatie_filtre.png', format='png')
plt.close()

print("Alegere: Butterworth - raspuns plat in banda de trecere, fara distorsiuni\n")

# f) Variatie parametri
print("f) Optimizare parametri")

orders = [3, 5, 7]
rps = [2, 5, 10]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(t_hours, x, linewidth=1, alpha=0.5, label='Original', color='gray')
for ord in orders:
    b, a = butter(ord, f_norm, btype='low')
    x_filt = filtfilt(b, a, x)
    axes[0].plot(t_hours, x_filt, linewidth=1.5, label=f'Butterworth ord={ord}')

axes[0].set_title('Variatie Ordin - Butterworth')
axes[0].set_ylabel('Vehicule')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_hours, x, linewidth=1, alpha=0.5, label='Original', color='gray')
for rp_val in rps:
    b, a = cheby1(5, rp_val, f_norm, btype='low')
    x_filt = filtfilt(b, a, x)
    axes[1].plot(t_hours, x_filt, linewidth=1.5, label=f'Chebyshev rp={rp_val}dB')

axes[1].set_title('Variatie Atenuare Ondulatie - Chebyshev (ord=5)')
axes[1].set_xlabel('Ore')
axes[1].set_ylabel('Vehicule')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab6/ex6f_optimizare.pdf', format='pdf')
plt.savefig('plots_lab6/ex6f_optimizare.png', format='png')
plt.close()

print("Parametri optimi:")
print("- Ordin: 3-5 (compromis intre performanta si intarziere)")
print("- rp: 2-5 dB (mai putin ripple, tranzitie mai buna)")
print("- Alegere finala: Butterworth ordin 5 pentru date trafic\n")

# Comparatie finala
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(t_hours, x, linewidth=1, alpha=0.6, label='Original', color='lightgray')
ax.plot(t_hours, x_butter, linewidth=2, label='Butterworth (ord=5)', color='blue')
ax.plot(t_hours, x_cheby, linewidth=2, label='Chebyshev (ord=5, rp=5dB)', color='red')
ax.set_title('Comparatie Finala: Filtrare Date Trafic (3 zile)')
ax.set_xlabel('Ore')
ax.set_ylabel('Numar Vehicule')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab6/ex6_comparatie_finala.pdf', format='pdf')
plt.savefig('plots_lab6/ex6_comparatie_finala.png', format='png')
plt.close()

print("Laborator 6 finalizat - toate graficele in plots_lab6/")

# 4 - * inultire elem cu elem, / impart elem cu elem (cerc taiat)