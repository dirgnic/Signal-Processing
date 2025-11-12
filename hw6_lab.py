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

# Ex 1 - Convolutie repetata
print("Ex 1 - Convolutie repetata")
x = np.random.rand(100)
x1 = np.convolve(x, x)
x2 = np.convolve(x1, x1)
x3 = np.convolve(x2, x2)

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
plt.savefig('plots_lab6/ex1_convolutie_repetata.pdf', format='pdf')
plt.savefig('plots_lab6/ex1_convolutie_repetata.png', format='png')
plt.close()

print("Obsev: Forma devine gaussiana, lung creste exponential\n")

# Ex 2 - Inmultirea polinoamelor
print("Ex 2 - Inmultirea polinoamelor")
N = 100
p = np.random.randint(0, 1000, size=N)
q = np.random.randint(0, 1000, size=N)

r_conv = np.convolve(p, q)

p_fft = np.fft.fft(p, 2 * N - 1)
q_fft = np.fft.fft(q, 2 * N - 1)
r_fft = np.real(np.fft.ifft(p_fft * q_fft))

similar = np.allclose(r_conv, r_fft)
print(f"Rezultate identice (convolutie vs FFT): {similar}\n")

# Ex 3 - Ferestre
print("Ex 3 - Ferestre")

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
plt.savefig('plots_lab6/ex3_ferestre.pdf', format='pdf')
plt.savefig('plots_lab6/ex3_ferestre.png', format='png')
plt.close()
print("Ferestre aplicate si salvate\n")

# Ex 4 - Analiza trafic
print("Ex 4 - Analiza Date Trafic")

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
plt.savefig('plots_lab6/ex4b_medie_alunecatoare.pdf', format='pdf')
plt.savefig('plots_lab6/ex4b_medie_alunecatoare.png', format='png')
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
plt.savefig('plots_lab6/ex4e_comparatie_filtre.pdf', format='pdf')
plt.savefig('plots_lab6/ex4e_comparatie_filtre.png', format='png')
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
plt.savefig('plots_lab6/ex4f_optimizare.pdf', format='pdf')
plt.savefig('plots_lab6/ex4f_optimizare.png', format='png')
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
plt.savefig('plots_lab6/ex4_comparatie_finala.pdf', format='pdf')
plt.savefig('plots_lab6/ex4_comparatie_finala.png', format='png')
plt.close()

print("Laborator 6 finalizat - toate graficele in plots_lab6/")
