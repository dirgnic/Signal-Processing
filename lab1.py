import matplotlib.pyplot as plt
import numpy as np

axa_reala = np.arange(0, 0.03, 0.0005)

x = lambda a: np.cos(520 * np.pi * a + np.pi / 3)
y = lambda a: np.cos(280 * np.pi * a - np.pi / 3)
z = lambda a: np.sin(120 * np.pi * a + np.pi / 3)

fig, axs = plt.subplots(3)

x_values = x(axa_reala)
y_values = y(axa_reala)
z_values = z(axa_reala)

axs[0].plot(axa_reala, x_values)
axs[1].plot(axa_reala, y_values)
axs[2].plot(axa_reala, z_values)
plt.show()

samples = np.linspace(0, 0.03, 6)
x_values_sampled = x(samples)
y_values_sampled = y(samples)
z_values_sampled = z(samples)

fig, axs = plt.subplots(3)

axs[0].stem(samples, x_values_sampled)
axs[1].stem(samples, y_values_sampled)
axs[2].stem(samples, z_values_sampled)

plt.show()
samples = np.linspace(0, 1, 1600)  # 0 to 0.1 seconds
m = lambda a: np.sin(400 * np.pi * a + np.pi / 3)

# Calculate for the entire interval
m_values = m(samples)

# Only show a subsection, e.g., from 0 to 0.03 seconds
subsection_mask = (samples >= 0) & (samples <= 0.03)
samples_sub = samples[subsection_mask]
m_sub = m_values[subsection_mask]

fig, axs = plt.subplots(1)
axs.stem(samples_sub, m_sub)

plt.show()


# Define signal parameters
T = 1  # period (seconds)
fs = 1000 # sampling frequency (Hz)
time = np.linspace(0, T, T * fs, endpoint=False)

# Generate sine wave
frequency_sine = 800  # frequency (Hz)
sine_wave = np.sin(2 * np.pi * frequency_sine * time)


# Generate sawtooth wave
frequency_sawtooth = 240  # frequency (Hz)
sawtooth_wave = 2 * (frequency_sawtooth * time -
                     np.floor(frequency_sawtooth * time + 0.5))


# Plot the waves
fig, axs = plt.subplots(1)
axs.plot(time, sawtooth_wave, label='Sawtooth Wave (240 Hz)',
            color='orange')

plt.show()

# (a)
t = 1
n = 1600
axs.stem("1600 samples", np.linspace(0, t, n), plt.sinusoidal_signal(1, 400, t, 0, n))

# (b)
t = 3
n = 800
axs.stem("800 samples 3 seconds", np.linspace(0, t, n), plt.sinusoidal_signal(1, 800, t, 0, n))

# (c)
t = 1
n = 1_000_000
signal = plt.sinusoidal_signal(1, 240, t, 0, n)
sawtooth_signal = np.arcsin(np.abs(signal))
sawtooth_signal = np.mod(sawtooth_signal, 0.1)
plt.plot("Sawtooth", np.linspace(0, t, n), sawtooth_signal, xlim=[0, 6e-4])

# (d)
t = 1000
n = 10_000
plt.plot(np.linspace(0, t, n), np.sign(plt.sinusoidal_signal(1, 300, t, 0, n)))

# (e)
cmap = "binary"
random_signal = np.random.rand(128, 128)
plt.imshow(random_signal, cmap=cmap)
plt.savefig(fname=f"./plots/random_image.pdf", format="pdf")
plt.show()
plt.close()

# (f)
generated_signal = np.zeros((128, 128))

values = np.linspace(0, 2 * np.pi, 128)
for i in range(len(generated_signal)):
    generated_signal[i] = np.sin(values) * np.tan(values)

min_value = np.min(generated_signal)
max_value = np.max(generated_signal)

generated_signal = (generated_signal - min_value) / (max_value - min_value)

plt.imshow(generated_signal, cmap=cmap)
plt.savefig(fname=f"./plots/generated_image.pdf", format="pdf")
plt.show()
