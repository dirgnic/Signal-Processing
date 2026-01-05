import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

np.random.seed(42)
os.makedirs('plots_lab10', exist_ok=True)

print("Laborator 10 - Serii de timp - Partea 3\n")

N = 200
t_vals = np.arange(N)

def generate_time_series():
    trend = 0.5 * t_vals + 10
    seasonal = 10 * np.sin(2 * np.pi * t_vals / 20) * (1 + 0.02 * t_vals)
    noise = np.random.normal(0, 2, N)
    return trend + seasonal + noise

y = generate_time_series()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y)
ax.set_title('Serie temporala generata')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
plt.tight_layout()
plt.savefig('plots_lab10/ex1_serie.pdf')
plt.savefig('plots_lab10/ex1_serie.png')
plt.close()
print("1. Serie temporala generata\n")

# Ex 2: AR model
def create_ar_matrix(y, p):
    n = len(y)
    X = np.zeros((n - p, p))
    for i in range(n - p):
        X[i, :] = y[i:i+p][::-1]
    Y = y[p:]
    return X, Y

def fit_ar_model(y, p):
    X, Y = create_ar_matrix(y, p)
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coeffs

def predict_ar(y, coeffs):
    p = len(coeffs)
    n = len(y)
    y_pred = np.zeros(n)
    y_pred[:p] = y[:p]
    for i in range(p, n):
        y_pred[i] = np.dot(coeffs, y[i-p:i][::-1])
    return y_pred

p = 10
ar_coeffs = fit_ar_model(y, p)
y_pred_ar = predict_ar(y, ar_coeffs)
mse_ar = np.mean((y[p:] - y_pred_ar[p:])**2)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(y, 'k--', alpha=0.5, label='Original')
axes[0].plot(y_pred_ar, label=f'AR({p})', alpha=0.8)
axes[0].set_title(f'Model AR(p={p}), MSE={mse_ar:.2f}')
axes[0].legend()
axes[0].set_xlabel('Timp')
axes[0].set_ylabel('Valoare')

axes[1].stem(range(p), ar_coeffs)
axes[1].set_title('Coeficienti AR')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Coeficient')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plots_lab10/ex2_ar.pdf')
plt.savefig('plots_lab10/ex2_ar.png')
plt.close()

print(f"2. Model AR(p={p})")
print(f"   MSE: {mse_ar:.2f}")
print(f"   Coeficienti non-zero: {np.sum(np.abs(ar_coeffs) > 1e-6)}\n")

# Visualize AR matrix X
X_ar, Y_ar = create_ar_matrix(y, p)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(X_ar[:30, :], aspect='auto', cmap='viridis')
axes[0].set_title(f'Matricea AR X (primele 30 randuri)\nDimensiune: {X_ar.shape}')
axes[0].set_xlabel('Lag (p)')
axes[0].set_ylabel('Timp (t)')
plt.colorbar(im0, ax=axes[0], label='y[t-lag]')

axes[1].bar(range(p), ar_coeffs)
axes[1].set_title('Coeficienti AR')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Coeficient')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab10/ex2_ar_matrix.pdf')
plt.savefig('plots_lab10/ex2_ar_matrix.png')
plt.close()

# Ex 3: Sparse AR models
# Greedy method
def fit_ar_greedy(y, p, max_nonzero):
    X, Y = create_ar_matrix(y, p)
    selected = []
    coeffs = np.zeros(p)
    
    for _ in range(max_nonzero):
        best_idx = -1
        best_mse = float('inf')
        
        for j in range(p):
            if j in selected:
                continue
            
            test_selected = selected + [j]
            X_sub = X[:, test_selected]
            c_sub = np.linalg.lstsq(X_sub, Y, rcond=None)[0]
            
            y_pred = X_sub @ c_sub
            mse = np.mean((Y - y_pred)**2)
            
            if mse < best_mse:
                best_mse = mse
                best_idx = j
        
        if best_idx >= 0:
            selected.append(best_idx)
    
    X_sub = X[:, selected]
    c_sub = np.linalg.lstsq(X_sub, Y, rcond=None)[0]
    
    for i, idx in enumerate(selected):
        coeffs[idx] = c_sub[i]
    
    return coeffs, selected

# L1 regularization (Lasso)
def soft_threshold(v, thresh):
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

# L1 regularization (Lasso)
def fit_ar_l1(y, p, lambda_reg, num_iters=2000):
    X, Y = create_ar_matrix(y, p)
    
    L = (np.linalg.norm(X, 2) ** 2) + 1e-12  # Lipschitz constant pt grad
    lr = 1.0 / L

    coeffs = np.zeros(p)
    for _ in range(num_iters):
        grad = X.T @ (X @ coeffs - Y)
        coeffs = soft_threshold(coeffs - lr * grad, lr * lambda_reg)

    coeffs[np.abs(coeffs) < 1e-4] = 0
    
    return coeffs

max_nonzero = 5
greedy_coeffs, selected_lags = fit_ar_greedy(y, p, max_nonzero)
y_pred_greedy = predict_ar(y, greedy_coeffs)
mse_greedy = np.mean((y[p:] - y_pred_greedy[p:])**2)

lambda_reg = 100
l1_coeffs = fit_ar_l1(y, p, lambda_reg)
y_pred_l1 = predict_ar(y, l1_coeffs)
mse_l1 = np.mean((y[p:] - y_pred_l1[p:])**2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(y, 'k--', alpha=0.5, label='Original')
axes[0, 0].plot(y_pred_greedy, label='Greedy AR', alpha=0.8)
axes[0, 0].set_title(f'Greedy AR, MSE={mse_greedy:.2f}')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Timp')
axes[0, 0].set_ylabel('Valoare')

axes[0, 1].stem(range(p), greedy_coeffs)
axes[0, 1].set_title(f'Greedy: {np.sum(np.abs(greedy_coeffs) > 1e-6)} coef. non-zero')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Coeficient')
axes[0, 1].grid(True)

axes[1, 0].plot(y, 'k--', alpha=0.5, label='Original')
axes[1, 0].plot(y_pred_l1, label='L1 AR', alpha=0.8)
axes[1, 0].set_title(f'L1 Regularized AR, MSE={mse_l1:.2f}')
axes[1, 0].legend()
axes[1, 0].set_xlabel('Timp')
axes[1, 0].set_ylabel('Valoare')

axes[1, 1].stem(range(p), l1_coeffs)
axes[1, 1].set_title(f'L1: {np.sum(np.abs(l1_coeffs) > 1e-6)} coef. non-zero')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Coeficient')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('plots_lab10/ex3_sparse.pdf')
plt.savefig('plots_lab10/ex3_sparse.png')
plt.close()

print("3. Sparse AR models")
print(f"   Greedy: {np.sum(np.abs(greedy_coeffs) > 1e-6)} coef. non-zero, MSE={mse_greedy:.2f}")
print(f"   L1: {np.sum(np.abs(l1_coeffs) > 1e-6)} coef. non-zero, MSE={mse_l1:.2f}\n")

# Visualize coefficient comparison
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(p)
width = 0.25

ax.bar(x_pos - width, ar_coeffs, width, label='AR standard', alpha=0.8)
ax.bar(x_pos, greedy_coeffs, width, label='Greedy', alpha=0.8)
ax.bar(x_pos + width, l1_coeffs, width, label='L1', alpha=0.8)

ax.set_xlabel('Lag')
ax.set_ylabel('Coeficient')
ax.set_title('Comparatie coeficienti: AR vs Greedy vs L1')
ax.set_xticks(x_pos)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('plots_lab10/ex3_coef_comparison.pdf')
plt.savefig('plots_lab10/ex3_coef_comparison.png')
plt.close()


# Ex 4: Polynomial roots using companion matrix
def ar_char_poly(coeffs):
    # 1 - phi1 z - ... - phip z^p = 0  =>  (-phip) z^p + ... + (-phi1) z + 1 = 0
    # np.roots / companion așteaptă coeficienți de la grad mare la mic.
    poly = np.r_[-coeffs[::-1], 1.0]
    poly = np.trim_zeros(poly, 'f')  # dacă cel mai mare lag e 0, reducem gradul
    return poly

def roots_via_companion(poly):
    poly = np.asarray(poly, dtype=float)
    poly = np.trim_zeros(poly, 'f')
    if len(poly) <= 1:
        return np.array([])

    m = len(poly) - 1
    a0 = poly[0]
    poly_monic = poly / a0

    companion = np.zeros((m, m))
    companion[0, :] = -poly_monic[1:]
    companion[1:, :-1] = np.eye(m - 1)

    return np.linalg.eigvals(companion)

roots_ar = roots_via_companion(ar_char_poly(ar_coeffs))
roots_greedy = roots_via_companion(ar_char_poly(greedy_coeffs))
roots_l1 = roots_via_companion(ar_char_poly(l1_coeffs))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

theta = np.linspace(0, 2*np.pi, 200)
for ax in axes:
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Cerc unitar')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')

axes[0].scatter(roots_ar.real, roots_ar.imag, s=100, marker='x')
axes[0].set_title(f'Radacini AR({p})')

axes[1].scatter(roots_greedy.real, roots_greedy.imag, s=100, marker='x')
axes[1].set_title('Radacini Greedy AR')

axes[2].scatter(roots_l1.real, roots_l1.imag, s=100, marker='x')
axes[2].set_title('Radacini L1 AR')

plt.tight_layout()
plt.savefig('plots_lab10/ex4_roots.pdf')
plt.savefig('plots_lab10/ex4_roots.png')
plt.close()

print("4. Radacini polinomiale (matrice companion)")
print(f"   AR: {len(roots_ar)} radacini")
print(f"   Greedy: {len(roots_greedy)} radacini")
print(f"   L1: {len(roots_l1)} radacini\n")

# Ex 5: Stationarity check
def check_stationarity(coeffs):
    poly = ar_char_poly(coeffs)             
    roots = np.roots(poly) if len(poly) > 1 else np.array([])
    magnitudes = np.abs(roots)
    is_stationary = np.all(magnitudes > 1) if len(roots) > 0 else True
    return is_stationary, roots

stat_ar, roots_stat_ar = check_stationarity(ar_coeffs)
stat_greedy, roots_stat_greedy = check_stationarity(greedy_coeffs)
stat_l1, roots_stat_l1 = check_stationarity(l1_coeffs)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax in axes:
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Cerc unitar')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)

colors_ar = ['green' if np.abs(r) > 1 else 'red' for r in roots_stat_ar]
axes[0].scatter(roots_stat_ar.real, roots_stat_ar.imag, c=colors_ar, s=100, marker='o')
axes[0].set_title(f'AR({p}): {"Stationar" if stat_ar else "Non-stationar"}')

colors_greedy = ['green' if np.abs(r) > 1 else 'red' for r in roots_stat_greedy]
axes[1].scatter(roots_stat_greedy.real, roots_stat_greedy.imag, c=colors_greedy, s=100, marker='o')
axes[1].set_title(f'Greedy: {"Stationar" if stat_greedy else "Non-stationar"}')

colors_l1 = ['green' if np.abs(r) > 1 else 'red' for r in roots_stat_l1]
axes[2].scatter(roots_stat_l1.real, roots_stat_l1.imag, c=colors_l1, s=100, marker='o')
axes[2].set_title(f'L1: {"Stationar" if stat_l1 else "Non-stationar"}')

plt.tight_layout()
plt.savefig('plots_lab10/ex5_stationarity.pdf')
plt.savefig('plots_lab10/ex5_stationarity.png')
plt.close()

print("5. Verificare stationaritate")
print(f"   AR({p}): {'Stationar' if stat_ar else 'Non-stationar'} (radacini in afara cercului unitar)")
print(f"   Greedy: {'Stationar' if stat_greedy else 'Non-stationar'}")
print(f"   L1: {'Stationar' if stat_l1 else 'Non-stationar'}")

print("\nLaborator 10 finalizat")
