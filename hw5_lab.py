import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('plots_lab5', exist_ok=True)

print("Laborator 5 - Analiza Semnalelor")


data = pd.read_csv("data_lab5/Train.csv", parse_dates=["Datetime"], dayfirst=True)
sample_ids = data["ID"].values
traffic_count = data["Count"].values
sample_rate_hz = 1 / 3600

num_samples = len(traffic_count)
duration_hours = num_samples
duration_days = duration_hours / 24

print(f"\nNumar total de esantioane: {num_samples}")
print(f"Durata totala: {duration_hours} ore ({duration_days:.2f} zile)")

# Exercitiul a) - Frecventa de esantionare
print("Exercitiul a) - Frecventa de esantionare")

answer_a = f"""
Semnalul este esantionat o data pe ora.
Frecventa de esantionare: 1 esantion/ora
In Hz: fs = 1/3600 Hz = {sample_rate_hz:.10f} Hz

Explicatie:
- Un esantion pe ora inseamna T = 3600 secunde
- Frecventa de esantionare fs = 1/T = 1/3600 Hz
"""
print(answer_a)

with open('plots_lab5/raspuns_a.txt', 'w') as f:
    f.write(answer_a)


# Exercitiul b) - Intervalul de timp
print("Exercitiul b) - Intervalul de timp")

first_date = data["Datetime"].iloc[0]
last_date = data["Datetime"].iloc[-1]
samples_per_day = 24

answer_b = f"""
Numar total de esantioane: {num_samples}
Esantioane pe zi: {samples_per_day}
Durata: {num_samples} / {samples_per_day} = {num_samples / samples_per_day:.0f} zile

Data de inceput: {first_date}
Data de sfarsit: {last_date}

Interval total: {(last_date - first_date).days} zile
"""
print(answer_b)

with open('plots_lab5/raspuns_b.txt', 'w') as f:
    f.write(answer_b)


# Exercitiul c) - Frecventa maxima
print("Exercitiul c) - Frecventa maxima")

nyquist_freq = sample_rate_hz / 2

answer_c = f"""
Daca semnalul a fost esantionat corect (fara aliere), frecventa maxima 
prezenta in semnal este frecventa Nyquist:

f_max = fs / 2 = {nyquist_freq:.10f} Hz

In termeni practici:
f_max = 1 / (2 * 3600) Hz = 1 / 7200 Hz

Aceasta inseamna ca cea mai rapida variatie detectabila este o perioada de 2 ore.
"""
print(answer_c)

with open('plots_lab5/raspuns_c.txt', 'w') as f:
    f.write(answer_c)


# Exercitiul d) - Transformata Fourier
print("Exercitiul d) - Transformata Fourier")

fft_result = np.fft.fft(traffic_count)
fft_magnitude = np.abs(fft_result)
fft_freqs = np.fft.fftfreq(num_samples, d=3600)

positive_freq_indices = fft_freqs > 0
positive_freqs = fft_freqs[positive_freq_indices]
positive_magnitudes = fft_magnitude[positive_freq_indices]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(sample_ids, traffic_count, linewidth=0.5)
axes[0].set_title('Semnalul Original - Trafic')
axes[0].set_xlabel('Numar esantion')
axes[0].set_ylabel('Numar de vehicule')
axes[0].grid(True, alpha=0.3)

axes[1].plot(positive_freqs, positive_magnitudes, linewidth=0.5)
axes[1].set_title('Modulul Transformatei Fourier')
axes[1].set_xlabel('Frecventa (Hz)')
axes[1].set_ylabel('Magnitudine')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab5/ex_d_transformata_fourier.pdf', format='pdf')
plt.savefig('plots_lab5/ex_d_transformata_fourier.png', format='png')
plt.close()

print("Transformata Fourier calculata si salvata")


# Exercitiul e) - Componenta continua
print("Exercitiul e) - Componenta continua")

dc_component = fft_result[0]
dc_magnitude = np.abs(dc_component)
mean_value = np.mean(traffic_count)

answer_e = f"""
Componenta continua (DC) din transformata Fourier:
Magnitudine: {dc_magnitude:.2f}
Valoare medie a semnalului: {mean_value:.2f}

Decizie: {'DA' if dc_magnitude > 1000 else 'NU'}, semnalul prezinta o componenta continua semnificativa.

Eliminarea componentei continue:
"""
print(answer_e)

traffic_no_dc = traffic_count - mean_value

fft_no_dc = np.fft.fft(traffic_no_dc)
fft_mag_no_dc = np.abs(fft_no_dc)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(sample_ids, traffic_count, linewidth=0.5, label='Original')
axes[0].axhline(y=mean_value, color='r', linestyle='--', label=f'Media = {mean_value:.2f}')
axes[0].set_title('Semnal Original cu Media')
axes[0].set_xlabel('Numar esantion')
axes[0].set_ylabel('Numar vehicule')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sample_ids, traffic_no_dc, linewidth=0.5, color='green')
axes[1].set_title('Semnal fara Componenta Continua')
axes[1].set_xlabel('Numar esantion')
axes[1].set_ylabel('Numar vehicule (centrat)')
axes[1].grid(True, alpha=0.3)

positive_mag_no_dc = fft_mag_no_dc[positive_freq_indices]
axes[2].plot(positive_freqs, positive_mag_no_dc, linewidth=0.5, color='purple')
axes[2].set_title('Transformata Fourier fara DC')
axes[2].set_xlabel('Frecventa (Hz)')
axes[2].set_ylabel('Magnitudine')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab5/ex_e_fara_dc.pdf', format='pdf')
plt.savefig('plots_lab5/ex_e_fara_dc.png', format='png')
plt.close()

print("Componenta continua eliminata")

with open('plots_lab5/raspuns_e.txt', 'w') as f:
    f.write(answer_e)
    f.write(f"\nComponenta DC eliminata prin scaderea mediei: {mean_value:.2f}")


# Exercitiul f) - Top 4 frecvente
print("Exercitiul f) - Top 4 frecvente principale")

positive_mag_no_dc = fft_mag_no_dc[positive_freq_indices]
top_4_indices = np.argsort(positive_mag_no_dc)[-4:][::-1]
top_4_freqs_hz = positive_freqs[top_4_indices]
top_4_magnitudes = positive_mag_no_dc[top_4_indices]

top_4_periods_hours = 1 / (top_4_freqs_hz + 1e-10)
top_4_periods_days = top_4_periods_hours / 24
top_4_per_year = top_4_freqs_hz * 3600 * 24 * 365

answer_f = f"""
Top 4 frecvente principale din semnal:

Frecventa (Hz)           Magnitudine      Perioada (ore)    Perioada (zile)   Aparitii/an
"""

for i in range(4):
    answer_f += f"\n{top_4_freqs_hz[i]:.12f}    {top_4_magnitudes[i]:12.2f}    {top_4_periods_hours[i]:10.2f}        {top_4_periods_days[i]:10.2f}      {top_4_per_year[i]:8.2f}"

answer_f += """

Interpretarea frecventelor:
"""

for i, freq_per_year in enumerate(top_4_per_year):
    if abs(freq_per_year - 365) < 10:
        interpretation = "Variatie zilnica (trafic diferit in fiecare zi)"
    elif abs(freq_per_year - 1) < 0.1:
        interpretation = "Variatie anuala (pattern anual)"
    elif abs(freq_per_year - 0.5) < 0.1:
        interpretation = "Variatie semestriala (pattern la 6 luni)"
    elif abs(freq_per_year - 2) < 0.2:
        interpretation = "Variatie bianuala (pattern la 18 luni)"
    else:
        interpretation = f"Variatie periodica (aproximativ {365/freq_per_year:.1f} zile)"
    
    answer_f += f"\n{i+1}. {interpretation}"

print(answer_f)

with open('plots_lab5/raspuns_f.txt', 'w') as f:
    f.write(answer_f)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(positive_freqs, positive_mag_no_dc, linewidth=0.5, alpha=0.7, label='Spectru complet')
ax.scatter(top_4_freqs_hz, top_4_magnitudes, color='red', s=100, zorder=5, label='Top 4 frecvente')
for i, (freq, mag) in enumerate(zip(top_4_freqs_hz, top_4_magnitudes)):
    ax.annotate(f'#{i+1}', xy=(freq, mag), xytext=(5, 5), textcoords='offset points')
ax.set_title('Top 4 Frecvente Principale')
ax.set_xlabel('Frecventa (Hz)')
ax.set_ylabel('Magnitudine')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab5/ex_f_top_frecvente.pdf', format='pdf')
plt.savefig('plots_lab5/ex_f_top_frecvente.png', format='png')
plt.close()


# Exercitiul g) - O luna de trafic incepand cu luni
print("Exercitiul g) - O luna de trafic (28 zile, incepand cu luni)")

start_index = 1001
remaining_data = data.iloc[start_index:]
monday_indices = remaining_data[remaining_data["Datetime"].dt.weekday == 0].index

if len(monday_indices) > 0:
    first_monday_idx = monday_indices[0]
    month_duration_days = 28
    month_samples = month_duration_days * 24
    
    end_idx = min(first_monday_idx + month_samples, len(data))
    month_data = data.iloc[first_monday_idx:end_idx]
    month_traffic = month_data["Count"].values
    month_time = np.arange(len(month_traffic))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(month_time, month_traffic, linewidth=1)
    ax.set_title(f'Trafic pentru o Luna (28 zile, incepand cu {month_data["Datetime"].iloc[0]})')
    ax.set_xlabel('Ore de la inceputul perioadei')
    ax.set_ylabel('Numar vehicule')
    ax.grid(True, alpha=0.3)
    
    for day in range(0, month_duration_days + 1, 7):
        ax.axvline(x=day*24, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots_lab5/ex_g_luna_trafic.pdf', format='pdf')
    plt.savefig('plots_lab5/ex_g_luna_trafic.png', format='png')
    plt.close()
    
    print(f"Grafic salvat pentru o luna incepand cu: {month_data['Datetime'].iloc[0]}")
else:
    print("Nu s-a gasit o zi de luni dupa esantionul 1000")


# Exercitiul h) - Determinare data de start
print("Exercitiul h) - Metoda de determinare a datei de inceput")

answer_h = """
Metoda de determinare a datei de inceput prin analiza semnalului:

Pasi:
1. Identificare pattern zilnic
   - Analizam FFT si gasim frecventa dominanta de 365 aparitii/an (1 zi)
   - Aceasta ne arata ca exista un ciclu zilnic clar in trafic

2. Identificare pattern saptamanal
   - Cautam pattern cu perioada de 7 zile
   - Zilele de weekend au trafic diferit fata de zilele lucratoare
   
3. Identificare pattern anual
   - Cautam variatii sezoniere (vara/iarna, sarbatori)
   - Identificam lunile prin pattern-uri caracteristice
   
4. Sincronizare cu calendarul
   - Folosind pattern-urile saptamanale, identificam zilele de luni
   - Folosind pattern-urile lunare/sezoniere, identificam lunile
   - Combinam informatiile pentru a determina data exacta

Neajunsuri si limitari:

1. Dependenta de durata semnalului
   - Pentru semnal de 2+ ani: putem determina luna si ziua cu acuratete mare
   - Pentru semnal de 1 an: putem determina ziua dar luna este ambigua
   - Pentru semnal de < 1 saptamana: imposibil de determinat ziua

2. Factori care afecteaza acuratetea:
   - Evenimente exceptionale (sarbatori, calamitati) distorsioneaza pattern-ul
   - Schimbari in comportament (pandemie, constructii) afecteaza regularitatea
   - Zgomot in date reduce claritatea pattern-urilor
   - Esantioane lipsa sau eronate creeaza discontinuitati

3. Presupuneri necesare:
   - Traficul urmeaza pattern-uri regulate saptamanale
   - Nu exista modificari majore de infrastructura in perioada masurarii
   - Pattern-ul seasonal este consistent
"""

print(answer_h)

with open('plots_lab5/raspuns_h.txt', 'w') as f:
    f.write(answer_h)


# Exercitiul i) - Filtrare (eliminare componente de frecventa inalta)

print("Exercitiul i) - Filtrare (eliminare componente inalta frecventa)")

cutoff_percentage = 0.95
num_freqs_to_keep = int(len(fft_no_dc) * cutoff_percentage)

fft_filtered = fft_no_dc.copy()
fft_filtered[num_freqs_to_keep:-num_freqs_to_keep] = 0

traffic_filtered = np.real(np.fft.ifft(fft_filtered))
traffic_filtered_with_mean = traffic_filtered + mean_value

if len(monday_indices) > 0:
    month_original = month_traffic
    month_filtered = traffic_filtered_with_mean[first_monday_idx:end_idx]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(month_time, month_original, linewidth=1, alpha=0.7, label='Original')
    axes[0].plot(month_time, month_filtered, linewidth=2, label='Filtrat', color='red')
    axes[0].set_title('Comparatie: Semnal Original vs Filtrat')
    axes[0].set_xlabel('Ore')
    axes[0].set_ylabel('Numar vehicule')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(month_time, month_original - month_filtered[:len(month_original)], linewidth=0.5, color='green')
    axes[1].set_title('Diferenta (zgomot eliminat)')
    axes[1].set_xlabel('Ore')
    axes[1].set_ylabel('Diferenta')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots_lab5/ex_i_filtrare.pdf', format='pdf')
    plt.savefig('plots_lab5/ex_i_filtrare.png', format='png')
    plt.close()
    
    print(f"Filtrare aplicata: pastrate {cutoff_percentage*100}% din componentele de frecventa joasa")
else:
    print("Date insuficiente pentru vizualizare luna filtrata")

