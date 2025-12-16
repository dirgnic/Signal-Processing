import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)
os.makedirs('plots_lab9', exist_ok=True)

print("Laborator 9 - Serii de timp - Partea 2\n")

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
plt.savefig('plots_lab9/ex1_serie.pdf')
plt.savefig('plots_lab9/ex1_serie.png')
plt.close()
print("1. Serie temporala generata\n")

# Ex 2: Exponential smoothing
def exponential_smoothing_single(y, alpha):
    s = np.zeros(len(y))
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s

def exponential_smoothing_double(y, alpha):
    s = np.zeros(len(y))
    b = np.zeros(len(y))
    s[0] = y[0]
    b[0] = y[1] - y[0] if len(y) > 1 else 0
    
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * (s[i-1] + b[i-1])
        b[i] = alpha * (s[i] - s[i-1]) + (1 - alpha) * b[i-1]
    
    return s

def exponential_smoothing_triple(y, alpha, beta, gamma, period):
    s = np.zeros(len(y))
    b = np.zeros(len(y))
    seas = np.zeros(len(y))
    
    s[0] = y[0]
    b[0] = (y[1] - y[0]) if len(y) > 1 else 0
    
    for i in range(period):
        seas[i] = y[i] - s[0]
    
    for i in range(1, len(y)):
        seas_idx = i - period if i >= period else 0
        
        s[i] = alpha * (y[i] - seas[seas_idx]) + (1 - alpha) * (s[i-1] + b[i-1])
        b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1]
        seas[i] = gamma * (y[i] - s[i]) + (1 - gamma) * seas[seas_idx]
    
    return s + seas

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
errors_single = []

for alpha in alphas:
    s = exponential_smoothing_single(y, alpha)
    mse = np.mean((y - s)**2)
    errors_single.append(mse)

best_alpha = alphas[np.argmin(errors_single)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for alpha in alphas:
    s = exponential_smoothing_single(y, alpha)
    axes[0].plot(s, label=f'alpha={alpha}', alpha=0.7)

axes[0].plot(y, 'k--', alpha=0.3, label='Original')
axes[0].set_title('Mediere exponentiala simpla')
axes[0].legend()
axes[0].set_xlabel('Timp')
axes[0].set_ylabel('Valoare')

axes[1].plot(alphas, errors_single, 'o-')
axes[1].set_title(f'MSE vs alpha (optim: {best_alpha})')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('MSE')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plots_lab9/ex2_single.pdf')
plt.savefig('plots_lab9/ex2_single.png')
plt.close()

errors_double = []
for alpha in alphas:
    s = exponential_smoothing_double(y, alpha)
    mse = np.mean((y - s)**2)
    errors_double.append(mse)

best_alpha_double = alphas[np.argmin(errors_double)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for alpha in alphas:
    s = exponential_smoothing_double(y, alpha)
    axes[0].plot(s, label=f'alpha={alpha}', alpha=0.7)

axes[0].plot(y, 'k--', alpha=0.3, label='Original')
axes[0].set_title('Mediere exponentiala dubla')
axes[0].legend()
axes[0].set_xlabel('Timp')
axes[0].set_ylabel('Valoare')

axes[1].plot(alphas, errors_double, 'o-')
axes[1].set_title(f'MSE vs alpha (optim: {best_alpha_double})')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('MSE')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plots_lab9/ex2_double.pdf')
plt.savefig('plots_lab9/ex2_double.png')
plt.close()

period = 50
alpha_t, beta_t, gamma_t = 0.3, 0.1, 0.1
s_triple = exponential_smoothing_triple(y, alpha_t, beta_t, gamma_t, period)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y, 'k--', alpha=0.3, label='Original')
ax.plot(s_triple, label=f'Triple (a={alpha_t}, b={beta_t}, g={gamma_t})', alpha=0.8)
ax.set_title('Mediere exponentiala tripla')
ax.legend()
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
plt.tight_layout()
plt.savefig('plots_lab9/ex2_triple.pdf')
plt.savefig('plots_lab9/ex2_triple.png')
plt.close()

print(f"2. Mediere exponentiala")
print(f"   Simpla - alpha optim: {best_alpha}, MSE: {min(errors_single):.2f}")
print(f"   Dubla - alpha optim: {best_alpha_double}, MSE: {min(errors_double):.2f}\n")

# Ex 3: MA model
def ma_model(y, q):
    n = len(y)
    ma_pred = np.zeros(n)
    
    for i in range(n):
        if i < q:
            ma_pred[i] = np.mean(y[:i+1])
        else:
            ma_pred[i] = np.mean(y[i-q:i])
    
    errors = y - ma_pred
    return ma_pred, errors

q_values = [5, 10, 20, 50]
fig, axes = plt.subplots(len(q_values), 2, figsize=(14, 12))

for idx, q in enumerate(q_values):
    ma_pred, errors = ma_model(y, q)
    
    axes[idx, 0].plot(y, 'k--', alpha=0.3, label='Original')
    axes[idx, 0].plot(ma_pred, label=f'MA(q={q})', alpha=0.8)
    axes[idx, 0].set_title(f'MA model q={q}')
    axes[idx, 0].legend()
    axes[idx, 0].set_xlabel('Timp')
    axes[idx, 0].set_ylabel('Valoare')
    
    axes[idx, 1].plot(errors, alpha=0.7)
    axes[idx, 1].set_title(f'Erori MA(q={q}), std={np.std(errors):.2f}')
    axes[idx, 1].set_xlabel('Timp')
    axes[idx, 1].set_ylabel('Eroare')
    axes[idx, 1].axhline(0, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab9/ex3_ma.pdf')
plt.savefig('plots_lab9/ex3_ma.png')
plt.close()

print("3. Model MA implementat\n")

# Ex 4: ARMA model
def arma_model(y, p, q):
    n = len(y)
    arma_pred = np.zeros(n)
    
    for i in range(max(p, q), n):
        ar_part = 0
        for j in range(1, p+1):
            if i-j >= 0:
                ar_part += y[i-j] / p
        
        ma_part = 0
        for j in range(1, q+1):
            if i-j >= 0:
                ma_part += (y[i-j] - arma_pred[i-j]) / q
        
        arma_pred[i] = ar_part + ma_part
    
    for i in range(max(p, q)):
        arma_pred[i] = np.mean(y[:i+1])
    
    errors = y - arma_pred
    return arma_pred, errors

best_mse = float('inf')
best_p, best_q = 1, 1

results = []

for p in range(1, 21):
    for q in range(1, 21):
        try:
            arma_pred, errors = arma_model(y, p, q)
            mse = np.mean(errors**2)
            results.append((p, q, mse))
            
            if mse < best_mse:
                best_mse = mse
                best_p, best_q = p, q
        except:
            pass

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (p, q) in enumerate([(1, 1), (5, 5), (10, 10), (best_p, best_q)]):
    arma_pred, errors = arma_model(y, p, q)
    ax = axes[idx // 2, idx % 2]
    
    ax.plot(y, 'k--', alpha=0.3, label='Original')
    ax.plot(arma_pred, label=f'ARMA(p={p}, q={q})', alpha=0.8)
    ax.set_title(f'ARMA model p={p}, q={q}, MSE={np.mean(errors**2):.2f}')
    ax.legend()
    ax.set_xlabel('Timp')
    ax.set_ylabel('Valoare')

plt.tight_layout()
plt.savefig('plots_lab9/ex4_arma.pdf')
plt.savefig('plots_lab9/ex4_arma.png')
plt.close()

mse_matrix = np.zeros((20, 20))
for p, q, mse in results:
    mse_matrix[p-1, q-1] = mse

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mse_matrix, cmap='viridis', aspect='auto', origin='lower')
ax.set_xlabel('q')
ax.set_ylabel('p')
ax.set_title(f'MSE pentru diferite (p, q). Optim: p={best_p}, q={best_q}')
plt.colorbar(im, ax=ax, label='MSE')
ax.plot(best_q-1, best_p-1, 'r*', markersize=15, label='Optim')
ax.legend()
plt.tight_layout()
plt.savefig('plots_lab9/ex4_arma_optim.pdf')
plt.savefig('plots_lab9/ex4_arma_optim.png')
plt.close()

print(f"4. Model ARMA")
print(f"   Parametri optimi: p={best_p}, q={best_q}, MSE={best_mse:.2f}\n")

# ARIMA with statsmodels
try:
    print("   Fitting ARIMA model...")
    model_arima = ARIMA(y, order=(2, 0, 2))
    fitted_arima = model_arima.fit(method='yule_walker')
    
    y_pred_arima = fitted_arima.fittedvalues
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(y, 'k--', alpha=0.5, label='Original')
    axes[0].plot(y_pred_arima, label='ARIMA(2,0,2)', alpha=0.8)
    axes[0].set_title('ARIMA model (statsmodels)')
    axes[0].legend()
    axes[0].set_xlabel('Timp')
    axes[0].set_ylabel('Valoare')
    
    residuals = y - y_pred_arima
    axes[1].plot(residuals, alpha=0.7)
    axes[1].set_title(f'Reziduuri ARIMA, std={np.std(residuals):.2f}')
    axes[1].set_xlabel('Timp')
    axes[1].set_ylabel('Reziduuri')
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots_lab9/ex4_arima_statsmodels.pdf')
    plt.savefig('plots_lab9/ex4_arima_statsmodels.png')
    plt.close()
    
    print(f"   ARIMA(2,0,2) - AIC: {fitted_arima.aic:.2f}")
    print(f"   Reziduuri std: {np.std(residuals):.2f}")
    
except Exception as e:
    print(f"   ARIMA statsmodels: {str(e)[:50]}")

print("\nLaborator 9 finalizat")
