import numpy as np
import matplotlib.pyplot as plt

# 1 a
t = np.arange(0, 0.03 + 1e-12, 0.0005) # 0 - 0.03 interval, 0.0005 perioada

# b
x = lambda a: np.cos(520 * np.pi * a + np.pi / 3)
y = lambda a: np.cos(280 * np.pi * a - np.pi / 3)
z = lambda a: np.cos(120 * np.pi * a + np.pi / 3)

_,axs = plt.subplots(3)
axs[0].plot(t, x(t)) # t proiectat pe axa reala def
axs[0].set_title("x")
axs[1].plot(t, y(t))
axs[1].set_title("y")
axs[2].plot(t, z(t))
axs[2].set_title("z")
plt.show()

# definim functii pentru x/y/z

# c - esantionare
fs = 200 # frecventa esantionare
T = 1/fs # perioada
ts = np.arange(0, 0.03 + 1e-12, T)   # momentele de eșantionare
n   = np.arange(len(ts))             #  nr indecși esantioane

x_values_sampled = x(ts)
y_values_sampled = y(ts)
z_values_sampled = z(ts)

fig, axs = plt.subplots(3)

axs[0].stem(ts, x_values_sampled)
axs[0].set_title("x[n]")
axs[1].stem(ts, y_values_sampled)
axs[1].set_title("y[n]")
axs[2].stem(ts, z_values_sampled)
axs[2].set_title("z[n]")

plt.show()


# 2 a
f = 400 # frecventa semnal
fs = 1600 # frecventa esantioanre >= 2*f
N = 1600
ts = np.arange(N)/fs  # momentele de eșantionare

x = np.sin(2*np.pi*f*ts)

T = 1/f  # 1.25 ms
plt.figure()
plt.plot(ts, x)
plt.xlim(0, 3*T)  # aratam doar 3 perioade
plt.title("Sinus 400 Hz – primele 3 perioade")
plt.xlabel("t [s]")
plt.ylabel("ampl.")
plt.grid(True)
plt.show()

plt.figure();

# 2 b
f = 800 # frecventa semnal
fs = 3200 # frecventa esantioanre >= 2*f
d = 3
ts = np.arange(0, d, 1/fs)  # momentele de eșantionare

x = np.sin(2*np.pi*f*ts)

T = 1/f  # 1.25 ms
plt.figure()
plt.plot(ts, x)
plt.xlim(0, 3*T)  # aratam doar o perioada
plt.title("Sinus 800 Hz – primele 3 perioade")
plt.xlabel("t [s]")
plt.ylabel("ampl.")
plt.show()


# 2 c
f = 240 # frecventa semnal
fs = 1200 # frecventa esantioanre >= 2*f
dur = 0.05
ts = np.arange(0, dur, 1/fs)
saw = 2*(f*ts - np.floor(0.5 + f*ts))

T = 1/f  # 1.25 ms
plt.figure()
plt.plot(ts, saw)
plt.xlim(0, 3*T)  # aratam doar o perioada
plt.title("SAW")
plt.xlabel("t [s]")
plt.ylabel("ampl.")
plt.show()

# 2 d
f = 300 # frecventa semnal
fs = 1200 # frecventa esantioanre >= 2*f
dur = 0.05
ts = np.arange(0, dur, 1/fs)
sq = np.sign(np.sin(2*np.pi*f*ts))

T = 1/f  # 1.25 ms
plt.figure()
plt.plot(ts, sq)
plt.xlim(0, 3*T)  # aratam doar o perioada
plt.title("SQ")
plt.xlabel("t [s]")
plt.ylabel("ampl.")
plt.show()

# 2 e
# semnal 2D
s = np.random.rand(128, 128)
plt.figure();
plt.imshow(s, cmap='gray', interpolation='nearest')
plt.title("2D random")
plt.colorbar()
plt.axis('off'); plt.show()

# 2 f
# semnal 2D
# base random
I = np.arange(128*128).reshape(128, 128)
I = I / I.max()  # normalize 0..1

# reverse horiz/vertical
rev_h = np.flip(I, axis=1)
rev_v = np.flip(I, axis=0)

# combine
nice = (I + rev_h + rev_v) / 3

plt.imshow(nice, cmap='plasma', interpolation='nearest')
plt.title("Symmetric gradient)")
plt.axis('off')
plt.show()

# 3
# a - T = 1/fs = 1/2000 s = 0.5 ms
# b - 2000×4=8000 biti = 100 bytes / sec = 3.6 MB / hr
'''
rezumat:

Nyquist:
f esantioante >= 2*f semnal

Construiește timp continuu pe [0, D] cu pas dt:
t = np.arange(0, D + 1e-12, dt)

Construiește timp discret pentru fs Hz și durată D:
fs = 1000
t = np.arange(0, D, 1/fs)
n = np.arange(len(t))

Cos / Sin în radiani:
y = np.cos(2*np.pi*f*t + phi) 
# f - frecv semnal, t pcte
y = np.sin(2*np.pi*f*t)

Stem plot (discret):
plt.stem(n, y, use_line_collection=True)

Sawtooth / Square:
saw = 2*(f*t - np.floor(0.5 + f*t))
sq  = np.sign(np.sin(2*np.pi*f*t)); sq[sq==0] = 1

'''
