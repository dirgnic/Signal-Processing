import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('plots_lab5', exist_ok=True)

print("Laborator 5 - Analiza Fourier\n")

data = pd.read_csv("data_lab5/Train.csv", parse_dates=["Datetime"], dayfirst=True)
x = data["Count"].values
dates = data["Datetime"].values

N = len(x)
fs = 1 / 3600

print(f"N={N}, Durata={N/24:.1f} zile\n")

# a)
print("a) fs = 1/3600 Hz = {:.10f} Hz\n".format(fs))

# b)
print("b) Interval: {} -> {}".format(dates[0], dates[-1]))
print("Durata: {} zile\n".format(int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))))

# c)
f_nyq = fs / 2
print("c) f_max = fs/2 = {:.10f} Hz\n".format(f_nyq))

# d)
print("d) Calculare FFT")
X = np.fft.fft(x)
X_mag = np.abs(X / N)
freqs = fs * np.linspace(0, N/2, int(N/2)) / N
X_mag = X_mag[:int(N/2)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(x, linewidth=0.5)
axes[0].set_title('Semnal')
axes[0].set_xlabel('Esantion')
axes[0].set_ylabel('Vehicule')
axes[0].grid(True, alpha=0.3)

axes[1].plot(freqs, X_mag, linewidth=0.5)
axes[1].set_title('Modul FFT')
axes[1].set_xlabel('Frecventa (Hz)')
axes[1].set_ylabel('Magnitudine')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab5/ex_d.pdf')
plt.savefig('plots_lab5/ex_d.png')
plt.close()
print("Salvat\n")

# e)
print("e) Componenta DC")
dc = np.abs(X[0]) / N
mean_val = np.mean(x)
print("DC={:.2f}, Media={:.2f}".format(dc, mean_val))
print("Prezenta DC: {}\n".format('DA' if dc > 10 else 'NU'))

x_no_dc = x - mean_val
X_no_dc = np.fft.fft(x_no_dc)
X_no_dc_mag = np.abs(X_no_dc / N)
X_no_dc_mag = X_no_dc_mag[:int(N/2)]

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(x, linewidth=0.5)
axes[0].axhline(y=mean_val, color='r', linestyle='--')
axes[0].set_title('Original')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_no_dc, linewidth=0.5, color='green')
axes[1].set_title('Fara DC')
axes[1].grid(True, alpha=0.3)

axes[2].plot(freqs, X_no_dc_mag, linewidth=0.5, color='purple')
axes[2].set_title('FFT fara DC')
axes[2].set_xlabel('Frecventa (Hz)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab5/ex_e.pdf')
plt.savefig('plots_lab5/ex_e.png')
plt.close()

# f)
print("f) Top 4 frecvente")
top_idx = np.argsort(X_no_dc_mag)[-4:][::-1]
top_freqs = freqs[top_idx]
top_mags = X_no_dc_mag[top_idx]

for i in range(4):
    period_days = 1 / (top_freqs[i] * 3600 * 24) if top_freqs[i] > 0 else 0
    per_year = top_freqs[i] * 3600 * 24 * 365
    print("{:.12f} Hz, Mag={:10.2f}, T={:8.1f} zile, {:.1f}/an".format(
        top_freqs[i], top_mags[i], period_days, per_year))
print()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(freqs, X_no_dc_mag, linewidth=0.5, alpha=0.6)
ax.scatter(top_freqs, top_mags, color='red', s=100, zorder=5)
ax.set_title('Top 4 Frecvente')
ax.set_xlabel('Frecventa (Hz)')
ax.set_ylabel('Magnitudine')
ax.grid(True, alpha=0.3)
plt.savefig('plots_lab5/ex_f.pdf')
plt.savefig('plots_lab5/ex_f.png')
plt.close()

# g)
print("g) O luna (28 zile)")
day_of_week = [pd.Timestamp(d).weekday() for d in dates]
monday_idx = [i for i, dow in enumerate(day_of_week) if dow == 0 and i > 1000]

if monday_idx:
    start = monday_idx[0]
    end = start + 28 * 24
    month_x = x[start:end]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(month_x, linewidth=1)
    ax.set_title('Luna (start: {})'.format(dates[start]))
    ax.set_xlabel('Ore')
    ax.set_ylabel('Vehicule')
    ax.grid(True, alpha=0.3)
    plt.savefig('plots_lab5/ex_g.pdf')
    plt.savefig('plots_lab5/ex_g.png')
    plt.close()
    print("Start: {}\n".format(dates[start]))

# h)
print("h) Metoda: analizam pattern zilnic/saptamanal/anual din FFT\n")

# i)
print("i) Filtrare")
cutoff = 0.05
n_keep = int(len(X_no_dc) * cutoff)

X_filt = X_no_dc.copy()
X_filt[n_keep:-n_keep] = 0
x_filt = np.real(np.fft.ifft(X_filt)) + mean_val

if monday_idx:
    start = monday_idx[0]
    end = start + 28 * 24
    
    orig = x[start:end]
    filt = x_filt[start:end]
    noise = orig - filt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(orig, linewidth=1, color='blue')
    axes[0].set_title('Original')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(filt, linewidth=1, color='red')
    axes[1].set_title('Filtrat ({}%)'.format(cutoff*100))
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(noise, linewidth=0.5, color='green')
    axes[2].set_title('Zgomot eliminat')
    axes[2].set_xlabel('Ore')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots_lab5/ex_i.pdf')
    plt.savefig('plots_lab5/ex_i.png')
    plt.close()
    print("Filtrat: {} frecvente\n".format(n_keep))

print("Finalizat")
