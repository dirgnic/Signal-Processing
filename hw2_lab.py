import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig
import sounddevice as sd

FS = 8200  # frecv esantionare

def exe_1():
    fs = FS
    T = 0.03
    t = np.arange(0, T, 1/FS)

    A = 1.0
    f = 400 # frecv semnal
    phi = 0.1

    sin = A * np.sin(2*np.pi*f*t + phi)
    # Cos identic - deplas faza cu +π/2
    cos = A * np.cos(2*np.pi*f*t + (phi - np.pi/2))  # echivalent sin

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(t, sin)
    axs[0].set_title("sinus")
    axs[0].set_xlabel("t [s]")

    axs[1].plot(t, cos)
    axs[1].set_title("cosinus (în fază cu sin)")
    axs[1].set_xlabel("t [s]")
    plt.show()


def exe_2():
# 2 a
    fs = FS
    T = 0.03
    t = np.arange(0, T, 1/FS)

    A = 1.0
    f = 400 # frecv semnal
    phi = 0.1

    sin = lambda q: A * np.sin(2*np.pi*f*t + q)

    fig, axs = plt.subplots(1, 4, constrained_layout=True)
    axs[0].plot(t, sin(0.1))
    axs[0].set_title("phi1")
    axs[0].set_xlabel("t [s]")

    axs[1].plot(t, sin(0.2))
    axs[1].set_title("phi2")
    axs[1].set_xlabel("t [s]")

    axs[2].plot(t, sin(0.3))
    axs[2].set_title("phi3")
    axs[2].set_xlabel("t [s]")

    axs[3].plot(t, sin(0.4))
    axs[3].set_title("phi4")
    axs[3].set_xlabel("t [s]")
    plt.show()
# 2 b
    # alegem una dintre sinusoide (phi=0.1)
    x = sin(0.1)
    z = np.random.normal(0, 1, len(x))

    # norme L2
    x_norm = np.linalg.norm(x)
    z_norm = np.linalg.norm(z)

    SNRs = [0.1, 1, 10, 100]
    make_noisy = lambda SNR: x + (x_norm / (z_norm * np.sqrt(SNR))) * z

    fig, axs = plt.subplots(1, 4, constrained_layout=True)

    for i, SNR in enumerate(SNRs):
        y = make_noisy(SNR)
        axs[i].plot(t, y)
        axs[i].set_title(f"SNR = {SNR}")
        axs[i].set_xlabel("t [s]")

    fig.suptitle("sin zgomote")
    plt.show()



def exe_3():
    fs = FS
    dur = 3.0
    t = np.arange(0, dur, 1/fs)

    x = 1.0 * np.sin(2*np.pi*440*t)

    sd.play(x, fs)
    sd.wait()
    # salvam pe device
    wav.write("ex3_ton440.wav", fs, (x.astype(np.float32)))
    # verificam - putem da load de pe disc?
    fs_loaded, x_loaded = wav.read("ex3_ton440.wav")
    sd.play(x_loaded, fs_loaded)
    sd.wait()

    # 2 a
    f = 400  # frecventa semnal
    fs = 1600  # frecventa esantioanre >= 2*f
    N = 1600
    ts = np.arange(N) / fs  # momentele de eșantionare

    x = np.sin(2 * np.pi * f * ts)

    T = 1 / f  # 1.25 ms

    sd.play(x, fs)
    sd.wait()
    # salvam pe device
    wav.write("ex3_ton440.wav", fs, (x.astype(np.float32)))
    # verificam - putem da load de pe disc?
    fs_loaded, x_loaded = wav.read("ex3_ton440.wav")
    sd.play(x_loaded, fs_loaded)
    sd.wait()

    # 2 b
    f = 800  # frecventa semnal
    fs = 3200  # frecventa esantioanre >= 2*f
    d = 3
    ts = np.arange(0, d, 1 / fs)  # momentele de eșantionare

    x = np.sin(2 * np.pi * f * ts)

    sd.play(x, fs)
    sd.wait()
    # salvam pe device
    wav.write("ex3_ton440.wav", fs, (x.astype(np.float32)))
    # verificam - putem da load de pe disc?
    fs_loaded, x_loaded = wav.read("ex3_ton440.wav")
    sd.play(x_loaded, fs_loaded)
    sd.wait()


    # 2 c
    f = 240  # frecventa semnal
    fs = 1200  # frecventa esantioanre >= 2*f
    dur = 0.05
    ts = np.arange(0, dur, 1 / fs)
    saw = 2 * (f * ts - np.floor(0.5 + f * ts))

    sd.play(x, fs)
    sd.wait()

    # 2 d
    f = 300  # frecventa semnal
    fs = 1200  # frecventa esantioanre >= 2*f
    dur = 0.05
    ts = np.arange(0, dur, 1 / fs)
    sq = np.sign(np.sin(2 * np.pi * f * ts))

    sd.play(x, fs)
    sd.wait()

def exe_4():
    fs = FS
    T = 0.03
    t = np.arange(0, T, 1/fs)

    A = 2.0
    f1 = 200.0
    x1 = A * np.sin(2*np.pi*f1*t)

    f2 = 300.0
    x2 = sig.sawtooth(2*np.pi*f2*t)
    s = x1 + x2

    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    axs[0].plot(t, x1); axs[0].set_title("Sinus")
    axs[1].plot(t, x2); axs[1].set_title("Sawtooth")
    axs[2].plot(t, s);  axs[2].set_title("Suma (sin + sawtooth)")
    axs[2].set_xlabel("t [s]")
    fig.suptitle("Ex.4 — Două forme de undă diferite și suma lor")
    plt.show()

def exe_5():
    fs = FS
    dur = 2.0
    f1 = 220; f2 = 320
    t = np.arange(0, dur, 1/fs)

    x1 = 0.6 * np.sin(2*np.pi*f1*t)
    x2 = 0.6 * np.sin(2*np.pi*f2*t)

    x_sum = np.concatenate([x1, x2])

    plt.figure()
    plt.plot(np.arange(len(x_sum))/fs, x_sum)
    plt.title("Ex.5 — Două sinusoide cu frecvențe diferite, puse cap la cap")
    plt.xlabel("t [s]")
    plt.show()

    sd.play(x_sum, fs)
    sd.wait()


def exe_6():
    fs = FS
    dur = 2.0
    t = np.arange(0, dur, 1/fs)

    x = lambda f: 1.0 * np.sin(2*np.pi*f*t)
    x1 = x(fs//2); x2 = x(fs//4); x3 = x(0)

    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    axs[0].plot(t, x1);
    axs[0].set_title("f = fs/2 (Nyquist)")
    axs[1].plot(t, x2);
    axs[1].set_title("f = fs/4")
    axs[2].plot(t, x3);
    axs[2].set_title("f = 0 Hz")
    axs[2].set_xlabel("t [s]")
    fig.suptitle("Ex.6 — Comportamente speciale la frecvențe particulare")
    plt.show()


def exe_7():
    fs = 1000
    dur = 0.5
    t = np.arange(0, dur, 1/fs)

    f = 220
    x = np.sin(2*np.pi*f*t)

    # (a) decim simpla - patr eșantioane 0,4,8
    x1 = x[::4]
    t1 = t[::4]   # timeline pentru decimat

    # (b) incep de la al doilea element: 1,5,9,...
    x2 = x[1::4]
    t2 = t[1::4]

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(t, x); axs[0].set_title("Semnal original (fs=1000 Hz)")
    axs[1].stem(t1, x1); axs[1].set_title("Decimare 1/4 (start index 0)")
    axs[2].stem(t2, x2); axs[2].set_title("Decimare 1/4 (start index 1)")
    axs[2].set_xlabel("t [s]")
    fig.suptitle("Ex.7 — Decimare și dependența de offset")
    plt.show()

def exe_8():
    a = np.linspace(-np.pi/2, np.pi/2, 3000)
    sin = np.sin(a)
    taylor = a                          #
    pade = (a - 7*(a**3)/60) / (1 + (a**2)/20)

    # curbele
    plt.figure()
    plt.plot(a, sin, label="sin(α)")
    plt.plot(a, taylor, label="Taylor: α")
    plt.plot(a, pade, label="Padé: (α - 7α³/60)/(1 + α²/20)")
    plt.title("Ex.8 — Aproximări pentru sin(α)")
    plt.xlabel("α [rad]")
    plt.legend()
    plt.show()

    # eroare abs
    err_taylor = np.abs(sin - taylor)
    err_pade   = np.abs(sin - pade)

    plt.figure()
    # folosim semilogy (axă y log)
    plt.semilogy(a, err_taylor, label="|sin - Taylor|")
    plt.semilogy(a, err_pade, label="|sin - Padé|")
    plt.title("Ex.8 — Eroare pe scară log (axă y)")
    plt.xlabel("α [rad]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    exe_1()
    exe_2()
    exe_3()
    exe_4()
    exe_5()
    exe_6()
    exe_7()
    exe_8()
