import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_matrix(N):
    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))
    omega = np.exp(-2j * np.pi / N)
    return omega ** (k * n)
# putere exp matrice
def on_circle(x, omegas):
    z = []

    for omega in omegas:
        zw = [x[i] * np.exp(-2 * np.pi * 1j * omega * i / len(x)) for i in range(len(x))]
        z.append(zw)

    return np.array(z)
# ex 1 - matr F

N = 8

matrix = fourier_transform_matrix(N)
h_matrix = np.transpose(np.conjugate(matrix))
# transpusa conjugatei
matr_mul = np.matmul(matrix, h_matrix)
matr_mul = np.subtract(matr_mul, np.diag(np.full(N,matr_mul[0,0])))
# prim element de pe diag * N, scadem de pe diag

const = 1e-5
# luam in considerare eroare din inmultire
print(matrix)
print(f"Norm: {np.linalg.norm(matr_mul)}")
print(f"I: {0 - const <= np.linalg.norm(matr_mul) <= 0 + const}")

fig, axs = plt.subplots(N, sharex=True, sharey=True)
fig.suptitle("Fourier Components")

components_numbered = [i + 1 for i in range(N)]
for i in range(N):
    axs[i].plot(components_numbered, matrix[i].real)
    axs[i].plot(components_numbered, matrix[i].imag, linestyle="dashed")

plt.savefig("plots/Fourier Components.pdf", format="pdf")
plt.savefig("plots/Fourier Components.png", format="png")
plt.show()


# ex 2 - matr F
nb= 3
nr = 3; nc = 2

n = 1000
t = 1000

nts = np.linspace(0, t, n)
x = 1.0 * np.sin(2 * np.pi * 3 * nts)

fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(8, 12))
fig.suptitle("Winding frequency")

for i in range(nr):
    for j in range(nc):
        axs[i, j].axhline(0, color="black")
        axs[i, j].locator_params(axis="x", nbins=nb)
        axs[i, j].locator_params(axis="y", nbins=nb)

        if i>0 or j>0:
            axs[i, j].axhline(0, color="black")
            axs[i, j].set_xlim(-1.1, 1.1)
            axs[i, j].set_ylim(-1.1, 1.1)
            axs[i, j].set_aspect("equal")
            axs[i, j].set_xlabel("Real")
            axs[i, j].set_ylabel("Imaginary")


values = [1, 4, 5, 7, 8]
#  on_circle = fuction for winding frequency on unit circle
z = on_circle(x, values)
points = np.array(list(zip(nts, x)))

axs[0][0].scatter(nts, x)
axs[0][1].scatter(z[0].real, z[0].imag)

for i in range(1, len(z)):
    axs[(i-1)//nc +1][(i-1)%nc].scatter(z[i].real, z[i].imag)

fig.tight_layout()
plt.savefig("plots/Winding Frequency.pdf", format="pdf")
plt.savefig("plots/Winding Frequency.png", format="png")
plt.show()

# plot - hold - delay -> animation in time
# ex 3

n = 1000
t = 1

nts = np.linspace(0, t, n)
x = 1.0 * np.sin(2 * np.pi * 3 * nts)

f = np.arange(start=0, stop=n, step=1)  # sampling frequency = n, DFT number of components = n => n/n = 1

s1 =  1.0 * np.sin(2 * np.pi * 3 * nts)
s2 =  2 * np.sin(2 * np.pi * 5 * nts)
s3 =  1/2 * np.sin(2 * np.pi * 2 * nts)
sgn = (s1 + s2 + s3)


values = [12, 3, 5]


def fourier_transform(x, number_of_components) -> np.ndarray:

    n = len(x) if number_of_components is None else number_of_components
    X = np.zeros(n, dtype=np.complex128)

    for m in range(n):
        component_sum = 0

        for k in range(n):
            component_sum += x[k] * np.exp(-2 * np.pi * 1j * k * m / n)

        X[m] = component_sum

    return X


def fourier_transform_using_winding_frequency(x, omegas):

    X: dict[float, np.complex128] = {}

    winding_frequencies = on_circle(x, omegas)

    for index, vector in enumerate(winding_frequencies):
        X[omegas[index]] = np.sum(vector)

    return X

ft_winding_frequency = fourier_transform_using_winding_frequency(sgn, omegas=values)
ft_over_signal = fourier_transform(sgn, None)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
fig.suptitle("Fourier Transform")

for ax in axs:
    ax.locator_params(axis="x", nbins=nb)
    ax.locator_params(axis="y", nbins=2 * nb)

axs[0].plot(nts, sgn)

stem_container = axs[1].stem(
    f,
    np.abs(ft_over_signal)
)

plt.setp(stem_container.markerline, "markersize", 3)
plt.setp(stem_container.stemlines, "linewidth", 0.3)

stem_container2 = axs[2].stem(
    list(ft_winding_frequency.keys()),
    np.abs(list(ft_winding_frequency.values()))
)

plt.setp(stem_container2.markerline, "markersize", 3)
plt.setp(stem_container2.stemlines, "linewidth", 0.3)


plt.tight_layout()
plt.savefig(f"plots/Fourier Transform.pdf", format="pdf")
plt.savefig(f"plots/Fourier Transform.png", format="png")
plt.show()

# matrix * h_matrix ar trebui sa fie multiplu I


