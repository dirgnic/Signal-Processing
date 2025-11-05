import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile
from scipy import signal
import os

os.makedirs('plots_lab4', exist_ok=True)

def fourier_transform(x, number_of_components=None):
    n = len(x) if number_of_components is None else number_of_components
    X = np.zeros(n, dtype=np.complex128)
    
    for m in range(n):
        component_sum = 0
        for k in range(n):
            component_sum += x[k] * np.exp(-2 * np.pi * 1j * k * m / n)
        X[m] = component_sum
    
    return X


# ex
# Comparatie timp de executie DFT custom vs numpy.fft

print("\nex 1")

N_values = [128, 256, 512, 1024, 2048, 4096, 8192]
custom_times = []
numpy_times = []

for N in N_values:
    # semnal test
    x = np.random.rand(N)
    # masuram timp in python cum trebe, eventual... scara log
    start_time = time.time()
    X_custom = fourier_transform(x)
    custom_time = time.time() - start_time
    custom_times.append(custom_time)

    start_time = time.time()
    X_numpy = np.fft.fft(x)
    numpy_time = time.time() - start_time
    numpy_times.append(numpy_time)

    # vectori de timp (ptr DFT si ptr FFT) pentru fiecare valoarea din N values
    
    print(f"  Custom DFT: {custom_time:.4f}s, NumPy FFT: {numpy_time:.6f}s")

# Plotam rezultatele
plt.figure(figsize=(10, 6))
plt.semilogy(N_values, custom_times, label='Custom DFT')
plt.semilogy(N_values, numpy_times, label='NumPy FFT')
plt.xlabel('Dimensiunea vectorului N')
plt.ylabel('Timp de execuție (s) - scara log')
plt.title('Comparație timp de execuție: DFT custom vs NumPy FFT')
plt.savefig('plots_lab4/ex1_comparatie_timpi.pdf', format='pdf')
plt.savefig('plots_lab4/ex1_comparatie_timpi.png', format='png')
plt.show()

print(f"\nSpeedup maxim: {max(custom_times) / min(numpy_times):.2f}x")

# ex 2
# Aliasing - esantionare sub-Nyquist

print("\nex 2")

# Parametri semnalului original
f_signal = 100  # Hz - frecventa semnalului original
A = 1.0
phi = 0

# Frecventa de esantionare sub-Nyquist
fs_sub_nyquist = 150  # Hz - Mai mica decat 2*f_signal = 200 Hz
print(f"Frecventa semnalului: {f_signal} Hz, frecventa Nyquist: {2 * f_signal} Hz")
print(f"Frecventa de esantionare aleasa: {fs_sub_nyquist} Hz (sub-Nyquist)")

# Construim semnalul continuu vizual, esantionam cu frecventa sub-Nyquist
t_continuous = np.linspace(0, 0.1, 10000)
signal_continuous = A * np.sin(2 * np.pi * f_signal * t_continuous + phi)
t_sampled = np.arange(0, 0.1, 1/fs_sub_nyquist)
signal_sampled = A * np.sin(2 * np.pi * f_signal * t_sampled + phi)

# Frecventa aparenta (aliased)
f_alias = abs(f_signal - fs_sub_nyquist)
print(f"Frecventa aparenta (alias): {f_alias} Hz")
# Alte doua semnale care produc acelasi alias
f_signal2 = fs_sub_nyquist + f_alias  # 200 Hz
f_signal3 = 2 * fs_sub_nyquist - f_alias  # 250 Hz

signal2_continuous = A * np.sin(2 * np.pi * f_signal2 * t_continuous + phi)
signal2_sampled = A * np.sin(2 * np.pi * f_signal2 * t_sampled + phi)

signal3_continuous = A * np.sin(2 * np.pi * f_signal3 * t_continuous + phi)
signal3_sampled = A * np.sin(2 * np.pi * f_signal3 * t_sampled + phi)

print(f"Semnalul 2: {f_signal2} Hz")
print(f"Semnalul 3: {f_signal3} Hz")

# Plotare
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Semnalul 1
axs[0].plot(t_continuous, signal_continuous, label=f'Semnal continuu {f_signal} Hz')
axs[0].plot(t_sampled, signal_sampled, label=f'Esantioane (fs={fs_sub_nyquist} Hz)')
axs[0].plot(t_sampled, signal_sampled)
axs[0].set_ylabel('Amplitudine')
axs[0].set_title(f'Semnal 1: f = {f_signal} Hz')

# Semnalul 2
axs[1].plot(t_continuous, signal2_continuous, label=f'Semnal continuu {f_signal2} Hz')
axs[1].plot(t_sampled, signal2_sampled, label=f'Esantioane (fs={fs_sub_nyquist} Hz)')
axs[1].plot(t_sampled, signal2_sampled)
axs[1].set_ylabel('Amplitudine')
axs[1].set_title(f'Semnal 2: f = {f_signal2} Hz')

# Semnalul 3
axs[2].plot(t_continuous, signal3_continuous, label=f'Semnal continuu {f_signal3} Hz')
axs[2].plot(t_sampled, signal3_sampled, label=f'Esantioane (fs={fs_sub_nyquist} Hz)')
axs[2].plot(t_sampled, signal3_sampled)
axs[2].set_xlabel('Timp (s)')
axs[2].set_ylabel('Amplitudine')
axs[2].set_title(f'Semnal 3: f = {f_signal3} Hz')

fig.suptitle('Fenomenul de Aliasing - Esantionare sub-Nyquist')
plt.tight_layout()
plt.savefig('plots_lab4/ex2_aliasing_sub_nyquist.pdf', format='pdf')
plt.savefig('plots_lab4/ex2_aliasing_sub_nyquist.png', format='png')
plt.show()

# ex 3
# Fara aliasing - esantionare peste Nyquist

print("ex 3")

# Frecventa de esantionare peste Nyquist
fs_over_nyquist = 500  # Hz - Mai mare decat 2*f_signal = 200 Hz
print(f"Frecventa de esantionare aleasa: {fs_over_nyquist} Hz (peste Nyquist)")

# Esantionam cu frecventa peste Nyquist
t_sampled_good = np.arange(0, 0.1, 1/fs_over_nyquist)

signal1_sampled_good = A * np.sin(2 * np.pi * f_signal * t_sampled_good + phi)
signal2_sampled_good = A * np.sin(2 * np.pi * f_signal2 * t_sampled_good + phi)
signal3_sampled_good = A * np.sin(2 * np.pi * f_signal3 * t_sampled_good + phi)

# Plotare
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Semnalul 1
axs[0].plot(t_continuous, signal_continuous, label=f'Semnal continuu {f_signal} Hz')
axs[0].plot(t_sampled_good, signal1_sampled_good, label=f'Esantioane (fs={fs_over_nyquist} Hz)')
axs[0].plot(t_sampled_good, signal1_sampled_good)
axs[0].set_ylabel('Amplitudine')
axs[0].set_title(f'Semnal 1: f = {f_signal} Hz', fontsize=12)

# Semnalul 2
axs[1].plot(t_continuous, signal2_continuous, label=f'Semnal continuu {f_signal2} Hz')
axs[1].plot(t_sampled_good, signal2_sampled_good, label=f'Esantioane (fs={fs_over_nyquist} Hz)')
axs[1].plot(t_sampled_good, signal2_sampled_good)
axs[1].set_ylabel('Amplitudine')
axs[1].set_title(f'Semnal 2: f = {f_signal2} Hz')

# Semnalul 3
axs[2].plot(t_continuous, signal3_continuous, label=f'Semnal continuu {f_signal3} Hz')
axs[2].plot(t_sampled_good, signal3_sampled_good, label=f'Esantioane (fs={fs_over_nyquist} Hz)')
axs[2].plot(t_sampled_good, signal3_sampled_good)
axs[2].set_xlabel('Timp (s)')
axs[2].set_ylabel('Amplitudine')
axs[2].set_title(f'Semnal 3: f = {f_signal3} Hz')

fig.suptitle('Fără Aliasing - Esantionare peste Nyquist')
plt.tight_layout()
plt.savefig('plots_lab4/ex3_fara_aliasing_over_nyquist.pdf', format='pdf')
plt.savefig('plots_lab4/ex3_fara_aliasing_over_nyquist.png', format='png')
plt.show()

# ex 4
# Frecventa de esantionare pentru contrabas

print("ex 4")

f_min_contrabas = 40  # Hz
f_max_contrabas = 200  # Hz

# Pentru un semnal trece-banda, frecventa minima de esantionare este:
# fs >= 2 * B, unde B este latimea de banda
B = f_max_contrabas - f_min_contrabas
fs_min_bandpass = 2 * B

print(f"Frecventele contrabasului: {f_min_contrabas} Hz - {f_max_contrabas} Hz")
print(f"Latimea de banda (B): {B} Hz")
print(f"Frecventa minima de esantionare (fs >= 2B): {fs_min_bandpass} Hz")
print(f"Se foloseste fs >= 2*f_max = {2*f_max_contrabas} Hz")

# Salvam rezultatul intr-un fisier text
with open('plots_lab4/ex4_contrabas_frecventa.txt', 'w') as f:
    f.write("ex 4: Frecventa de esantionare pentru contrabas\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Frecventele contrabasului: {f_min_contrabas} Hz - {f_max_contrabas} Hz\n")
    f.write(f"Latimea de banda (B): {B} Hz\n")
    f.write(f"Frecventa minima de esantionare (teoretic, trece-banda): {fs_min_bandpass} Hz\n")
    f.write(f"Frecventa minima de esantionare (practic, sigur): {2*f_max_contrabas} Hz\n")


# ex 5 si 6
# Spectrograma pentru vocale

print("ex 5 si 6")
print("Gen un semnal audio sintetic cu vocale pentru demo")

# Generam un semnal sintetic care simuleaza vocale
# Vocalele au formanti (frecvente caracteristice)
duration = 0.5  # secunde per vocala
fs_audio = 44100  # Hz - frecventa standard audio

# Formanti aproximativi pentru vocale (in Hz)
# Sursa: valori standard din fonetica
formants = {
    'a': [730, 1090, 2440],
    'e': [530, 1840, 2480],
    'i': [270, 2290, 3010],
    'o': [570, 840, 2410],
    'u': [300, 870, 2240]
}

def generate_vowel(formants_list, duration, fs):
    """Genereaza un semnal sintetic pentru o vocala"""
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    
    # Adaugam formantii cu amplitudini descrescatoare
    for i, freq in enumerate(formants_list):
        amplitude = 1.0 / (i + 1)  # Amplitudine descrescatoare
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Normalizam
    signal = signal / np.max(np.abs(signal))
    return signal

# Generam semnalul complet cu toate vocalele
vowels_order = ['a', 'e', 'i', 'o', 'u']
full_signal = []

for vowel in vowels_order:
    vowel_signal = generate_vowel(formants[vowel], duration, fs_audio)
    full_signal.extend(vowel_signal)

full_signal = np.array(full_signal)

# Salvam ca fisier WAV pentru referinta
wavfile.write('plots_lab4/ex5_vocale_sintetice.wav', fs_audio, (full_signal * 32767).astype(np.int16))
print(f"Semnal audio salvat: plots_lab4/ex5_vocale_sintetice.wav")

file_names = ['audio/a.wav', 'audio/e.wav', 'audio/i.wav', 'audio/o.wav', 'audio/u.wav']
full_signal = []

print("Incarcarea si concatenarea vocalelor din fisiere WAV:")

for i, file_name in enumerate(file_names):
    try:
        # Citim fisierul WAV
        fs_read, signal_data = wavfile.read(file_name)

        # Verificam si setam fs_audio la prima citire
        if i == 0:
            fs_audio = fs_read
            print(f"Frecventa de esantionare setata la {fs_audio} Hz.")
        elif fs_read != fs_audio:
            print(f"ATENTIE: Fisierul {file_name} are fs diferita ({fs_read} Hz). Se recomanda uniformizarea lor.")

        # Normalizam semnalul la [-1, 1] daca nu este deja (pentru int16)
        if signal_data.dtype == np.int16:
            signal_normalized = signal_data.astype(np.float64) / 32768.0
        else:
            signal_normalized = signal_data.astype(np.float64)

        full_signal.extend(signal_normalized)
        print(f" - Fisier '{file_name}' incarcat.")

    except FileNotFoundError:
        print(f"EROARE: Fisierul '{file_name}' nu a fost gasit!")
        # Puteti alege sa iesiti sau sa continuati cu fisierul urmator

full_signal = np.array(full_signal)

# --- Pasul 2: Salvati semnalul complet (Optional, doar pentru referinta) ---
# Aceasta linie devine redundanta daca nu generati, dar o puteti pastra
# pentru a salva concatenarea finala.
wavfile.write('plots_lab4/ex5_vocale_reale_concatenate.wav', fs_audio, (full_signal * 32767).astype(np.int16))
print(f"\nSemnal audio concatenat salvat: plots_lab4/ex5_vocale_reale_concatenate.wav")
# sprectrograma

def compute_spectrogram(signal, fs, window_percent=1, overlap_percent=50):
    """
    Calculeaza spectrograma unui semnal
    
    Parameters:
    - signal: semnalul de intrare
    - fs: frecventa de esantionare
    - window_percent: procentul din semnal pentru fiecare fereastra
    - overlap_percent: procentul de suprapunere intre ferestre
    """
    N = len(signal)
    window_size = int(N * window_percent / 100)
    overlap_size = int(window_size * overlap_percent / 100)
    step_size = window_size - overlap_size
    
    # Calculam numarul de ferestre
    num_windows = (N - overlap_size) // step_size
    
    # Matricea pentru spectrograma
    spectrogram = []
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        
        if end > N:
            break
        
        # Extragem fereastra
        window = signal[start:end]
        
        # Aplicam o fereastra Hanning pentru a reduce efectele de margine
        window = window * np.hanning(len(window))
        
        # Calculam FFT
        fft_result = np.fft.fft(window)
        
        # Luam doar jumatate (frecvente pozitive) si valoarea absoluta
        fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        spectrogram.append(fft_magnitude)
    
    # Convertim la matrice (transpunem pentru afisare corecta)
    spectrogram = np.array(spectrogram).T
    
    return spectrogram, window_size

# Calculam spectrograma
spectrogram, window_size = compute_spectrogram(full_signal, fs_audio, window_percent=1, overlap_percent=50)

# Cream axa de frecventa si timp pentru afisare
freq_axis = np.linspace(0, fs_audio/2, spectrogram.shape[0])
time_axis = np.linspace(0, len(full_signal)/fs_audio, spectrogram.shape[1])

# Plotam spectrograma
plt.figure(figsize=(14, 8))
plt.imshow(20 * np.log10(spectrogram + 1e-10), aspect='auto', origin='lower', 
           cmap='viridis', extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
plt.colorbar(label='Magnitudine (dB)')
plt.ylabel('Frecve (Hz)', fontsize=12)
plt.xlabel('Timp (s)', fontsize=12)
plt.title('Spectrograma Vocale: a, e, i, o, u', fontsize=14, fontweight='bold')
plt.ylim([0, 4000])  # Limitam la frecventele relevante pentru voce

# Adaugam linii verticale pentru delimitarea vocalelor
for i in range(1, len(vowels_order)):
    plt.axvline(x=i * duration, color='white', linestyle='--', linewidth=2, alpha=0.7)

# Adaugam etichete pentru vocale
for i, vowel in enumerate(vowels_order):
    plt.text((i + 0.5) * duration, 3700, vowel.upper(), 
             color='white', fontsize=20, fontweight='bold',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig('plots_lab4/ex6_spectrograma_vocale.pdf', format='pdf')
plt.savefig('plots_lab4/ex6_spectrograma_vocale.png', format='png', dpi=300)
plt.show()

print("Spectrograma salvată în plots_lab4/ex6_spectrograma_vocale.png")
print("\nRăspuns Ex. 5: Da, vocalele pot fi distinse pe baza spectrogramei!")
print("Fiecare vocală are formanti (frecvențe caracteristice) diferiți,")
print("vizibili ca benzi orizontale în spectrogramă.")


# ex 7
# Calculul puterii zgomotului

print("ex 7 - Calculul puterii zgomotului")

# Date
P_signal_dB = 90  # dB
SNR_dB = 80  # dB

# Formula: SNR_dB = P_signal_dB - P_noise_dB
# Deci: P_noise_dB = P_signal_dB - SNR_dB
P_noise_dB = P_signal_dB - SNR_dB

print(f"Puterea semnalului: P_signal = {P_signal_dB} dB")
print(f"Raport semnal-zgomot: SNR = {SNR_dB} dB")
print(f"\nFormula: SNR = P_signal - P_noise")
print(f"Rezultat: P_noise = P_signal - SNR")
print(f"P_noise = {P_signal_dB} - {SNR_dB} = {P_noise_dB} dB")

# Cream o vizualizare
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Putere Semnal\n(dB)', 'SNR\n(dB)', 'Putere Zgomot\n(dB)']
values = [P_signal_dB, SNR_dB, P_noise_dB]
colors = ['#2E86AB', '#A23B72', '#F18F01']

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Adaugam valorile pe bare
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{value} dB', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Putere (dB)', fontsize=12)
ax.set_title('Exercițiul 7: Calculul Puterii Zgomotului', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Adaugam formula
formula_text = f'SNR = P_signal - P_noise\n{SNR_dB} dB = {P_signal_dB} dB - P_noise\nP_noise = {P_noise_dB} dB'
ax.text(0.5, 0.95, formula_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('plots_lab4/ex7_putere_zgomot.pdf', format='pdf')
plt.savefig('plots_lab4/ex7_putere_zgomot.png', format='png')
plt.show()

# Salvam rezultatul
with open('plots_lab4/ex7_putere_zgomot.txt', 'w') as f:
    f.write("ex 7: Calculul puterii zgomotului\n")
    f.write(f"Date:\n")
    f.write(f"  Puterea semnalului: P_signal = {P_signal_dB} dB\n")
    f.write(f"  Raport semnal-zgomot: SNR = {SNR_dB} dB\n\n")
    f.write(f"Formula: SNR = P_signal - P_noise\n\n")
    f.write(f"Calcul:\n")
    f.write(f"  P_noise = P_signal - SNR\n")
    f.write(f"  P_noise = {P_signal_dB} dB - {SNR_dB} dB\n")
    f.write(f"  P_noise = {P_noise_dB} dB\n\n")
    f.write(f"Raspuns: Puterea zgomotului este {P_noise_dB} dB\n")

print(f"\nRaspuns: Puterea zgomotului este {P_noise_dB} dB")
