import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('plots_lab11', exist_ok=True)

print("Laborator 11 - Serii de timp - Partea 4\n")

# 1. Generarea seriei de timp (preluat din laboratoarele 9-10)
N = 200
t_vals = np.arange(N)

def generate_time_series():
    """Serie de timp cu trend, sezonalitate si zgomot (stil Johnson & Johnson)."""
    trend = 0.5 * t_vals + 10
    seasonal = 10 * np.sin(2 * np.pi * t_vals / 20) * (1 + 0.02 * t_vals)
    noise = np.random.normal(0, 2, N)
    return trend + seasonal + noise

y = generate_time_series()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y)
ax.set_title('Serie temporala generata (Lab 11)')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
plt.tight_layout()
plt.savefig('plots_lab11/ex1_serie.pdf')
plt.savefig('plots_lab11/ex1_serie.png')
plt.close()
print("1. Serie temporala generata\n")

# 2. Construirea matricei Hankel X

def hankel_matrix(y, L):
    """Construieste matricea Hankel X pentru fereastra de lungime L.

    X are dimensiunea L x K, unde K = N - L + 1, iar coloana k este:
        [y[k], y[k+1], ..., y[k+L-1]]^T
    """
    N = len(y)
    if not (1 <= L <= N):
        raise ValueError("L trebuie sa fie intre 1 si N")
    K = N - L + 1
    X = np.zeros((L, K))
    for k in range(K):
        X[:, k] = y[k:k + L]
    return X

# Alegem o lungime de fereastra L (de ex. ~N/2)
L = 60
X = hankel_matrix(y, L)
K = X.shape[1]

print(f"2. Matrice Hankel construita: X are dimensiunea {X.shape}\n")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(X, aspect='auto', cmap='viridis')
ax.set_title(f'Matricea Hankel X (L={L}, K={K})')
ax.set_xlabel('Coloana (k)')
ax.set_ylabel('Linie (i)')
plt.colorbar(im, ax=ax, label='Valoare')
plt.tight_layout()
plt.savefig('plots_lab11/ex2_hankel_X.pdf')
plt.savefig('plots_lab11/ex2_hankel_X.png')
plt.close()

# 3. Descompunerile: XX^T, X^TX si SVD(X)

# Matricile de covarianta (Gram)
XXT = X @ X.T   # L x L
XTX = X.T @ X   # K x K

# Valori si vectori proprii
lam_XXT, U = np.linalg.eigh(XXT)  # U: coloane = vectori proprii
lam_XTX, V = np.linalg.eigh(XTX)  # V: coloane = vectori proprii

# Sortam descrescator dupa valoarea proprie
idx1 = np.argsort(lam_XXT)[::-1]
lam_XXT = lam_XXT[idx1]
U = U[:, idx1]

idx2 = np.argsort(lam_XTX)[::-1]
lam_XTX = lam_XTX[idx2]
V = V[:, idx2]

# SVD completa a lui X
U_svd, s_svd, Vt_svd = np.linalg.svd(X, full_matrices=False)

# Relatii teoretice:
#   X = U_svd * diag(s_svd) * Vt_svd
#   XX^T = U_svd * diag(s_svd^2) * U_svd^T
#   X^TX = V_svd * diag(s_svd^2) * V_svd^T
#   valorile proprii ale XX^T si X^TX sunt aceleasi: s_svd^2

print("3. Descompuneri calculate:")
print(f"   XX^T forma: {XXT.shape}")
print(f"   X^TX forma: {XTX.shape}")
print(f"   SVD(X): U_svd {U_svd.shape}, s {s_svd.shape}, Vt {Vt_svd.shape}\n")

# Comparam spectrele valorilor proprii cu s^2
s2 = s_svd**2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(s2, 'ko-', label='s^2 din SVD')
ax.plot(lam_XXT, 'r--', marker='x', label='Valori proprii XX^T')
ax.plot(lam_XTX[:len(s2)], 'b--', marker='s', label='Valori proprii X^TX')
ax.set_title('Relatia intre spectre: s^2, eig(XX^T), eig(X^TX)')
ax.set_xlabel('Index componenta')
ax.set_ylabel('Valoare')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab11/ex3_spectre.pdf')
plt.savefig('plots_lab11/ex3_spectre.png')
plt.close()

print("   Relatiile teoretice: valorile proprii ale XX^T si X^TX coincid cu s^2 (din SVD) pana la erori numerice.\n")

# 4. Single Spectrum Analysis (SSA)

def ssa_decomposition(y, L, r=None):
    """Single Spectrum Analysis pentru o serie 1D.

    Parametri
    ---------
    y : array-like, lungime N
    L : int, lungimea ferestrei (1 < L < N)
    r : int sau None, numarul de componente principale pastrate

    Returneaza
    ---------
    components : lista de array-uri (fiecare array este o reconstructie partiala)
    s : valorile singulare
    U, Vt : componentele SVD
    """
    X = hankel_matrix(y, L)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if r is None:
        r = len(s)

    N = len(y)
    K = X.shape[1]
    components = []

    for k in range(r):
        X_k = s[k] * np.outer(U[:, k], Vt[k, :])  # matricea de rang-1

        # Diagonal averaging (Hankelizare inversa)
        y_k = np.zeros(N)
        counts = np.zeros(N)

        for i in range(L):
            for j in range(K):
                t = i + j
                if 0 <= t < N:
                    y_k[t] += X_k[i, j]
                    counts[t] += 1

        counts[counts == 0] = 1
        y_k /= counts
        components.append(y_k)

    return components, s, U, Vt

# Aplicam SSA si afisam cateva componente principale
r = 5
components, s_vals, U_ssa, Vt_ssa = ssa_decomposition(y, L, r=r)

fig, axes = plt.subplots(r+1, 1, figsize=(14, 3*(r+1)))
axes[0].plot(y, 'k')
axes[0].set_title('Serie originala')
axes[0].set_xlabel('Timp')
axes[0].set_ylabel('Valoare')

for i in range(r):
    axes[i+1].plot(components[i])
    axes[i+1].set_title(f'Componenta SSA #{i+1}, s={s_vals[i]:.2f}')
    axes[i+1].set_xlabel('Timp')
    axes[i+1].set_ylabel('Valoare')

plt.tight_layout()
plt.savefig('plots_lab11/ex4_ssa_components.pdf')
plt.savefig('plots_lab11/ex4_ssa_components.png')
plt.close()

# Exemplu de separare trend + sezonalitate + zgomot

# Trend aproximat de primele 1-2 componente
trend_approx = components[0] + components[1]
# Sezonalitate aproximata de urmatoarele 2 componente
seasonal_approx = components[2] + components[3]
# Zgomot ~ restul
noise_approx = y - (trend_approx + seasonal_approx)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
axes[0].plot(y, 'k', label='Original')
axes[0].set_title('Serie originala')
axes[0].legend()

axes[1].plot(trend_approx, 'r', label='Trend (aproximat)')
axes[1].set_title('Trend aproximat din SSA')
axes[1].legend()

axes[2].plot(seasonal_approx, 'b', label='Sezonalitate (aprox.)')
axes[2].set_title('Sezonalitate aproximata din SSA')
axes[2].legend()

axes[3].plot(noise_approx, 'g', label='Zgomot (rest)')
axes[3].set_title('Componenta de zgomot (restul)')
axes[3].legend()

for ax in axes:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Valoare')

plt.tight_layout()
plt.savefig('plots_lab11/ex4_ssa_decomposition.pdf')
plt.savefig('plots_lab11/ex4_ssa_decomposition.png')
plt.close()

print("4. SSA implementat. Primele componente pot fi interpretate ca trend, sezonalitate si zgomot.\n")

print("\nLaborator 11 finalizat")
