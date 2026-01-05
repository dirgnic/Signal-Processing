import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from my_module.funcs import plot

np.random.seed(42)
os.makedirs('plots_lab8', exist_ok=True)

print("Laborator 8 - Modele AR\n")



# a) - serie temporala
N = 1000
t = np.arange(N)
t = 0.01 * t**2 - 0.5 * t + 10
s = 5 * np.sin(2 * np.pi * t / 50) + 3 * np.cos(2 * np.pi * t / 100)
n = np.random.normal(0, 2, N)

y = t + s+ n # trend + sezon + zgomot

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(t)
axes[0].set_title('Trend (grad 2)')

axes[1].plot(s)
axes[1].set_title('Sezon (2 frecvente)')

axes[2].plot(n)
axes[2].set_title('Zgomot alb gaussian')

axes[3].plot(y)
axes[3].set_title('Serie temporala completa')

plt.tight_layout()
plt.savefig('plots_lab8/ex_a_componente.pdf')
plt.savefig('plots_lab8/ex_a_componente.png')
plt.close()
print("Serie generata\n")

# b) Autocorelatie (masoara similaritatea dintre serie si versiuni deplasate)
def autocorr(x, lag):
    if lag == 0:
        return 1.0
    x_mean = np.mean(x)
    c0 = np.sum((x - x_mean)**2) / len(x)
    c_lag = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / len(x)
    return c_lag / c0

max_lag = 100
acf_manual = np.array([autocorr(y, lag) for lag in range(max_lag)])

acf_numpy = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
acf_numpy = acf_numpy[len(acf_numpy)//2:len(acf_numpy)//2 + max_lag]
acf_numpy = acf_numpy / acf_numpy[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].stem(acf_manual, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('Autocorelatie (calcul manual)')
axes[0].set_xlabel('Lag')
axes[0].grid(alpha=0.3)

axes[1].stem(acf_numpy, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title('Autocorelatie (numpy.correlate)')
axes[1].set_xlabel('Lag')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots_lab8/ex_b_autocorr.pdf')
plt.savefig('plots_lab8/ex_b_autocorr.png')
plt.close()

# c) Model AR
print("c) Model AR")
p = 10
m = 50

def fit_ar(data, order):
    X = np.array([data[i-order:i][::-1] for i in range(order, len(data))])
    y_target = data[order:]
    coef = np.linalg.lstsq(X, y_target, rcond=None)[0]
    return coef

def predict_ar(data, coef, steps):
    pred = list(data[-len(coef):])
    for _ in range(steps):
        next_val = np.dot(coef, pred[-len(coef):][::-1])
        pred.append(next_val)
    return np.array(pred[-steps:])

coef = fit_ar(y, p)
train_size = N - m
y_train = y[:train_size]
y_test = y[train_size:]

pred = predict_ar(y_train, coef, m)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(N), y, 'b-', label='Original', alpha=0.6)
ax.plot(range(train_size, N), pred, 'r--', label=f'Predictie AR(p={p})', linewidth=2)
ax.axvline(train_size, color='g', linestyle='--', alpha=0.5, label='Split')
ax.set_title(f'Model AR: p={p}, orizont={m}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab8/ex_c_ar_model.pdf')
plt.savefig('plots_lab8/ex_c_ar_model.png')
plt.close()

mse = np.mean((y_test - pred)**2)
print(f"MSE: {mse:.2f}\n")

# d) Hyperparameter tuning
print("d) Optimizare hiperparametri")
p_values = range(1, 21)
m_values = [1, 5, 10, 20, 50]

results = np.zeros((len(p_values), len(m_values)))

for i, p_test in enumerate(p_values):
    for j, m_test in enumerate(m_values):
        try:
            train_size = N - m_test
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            coef = fit_ar(y_train, p_test)
            pred = predict_ar(y_train, coef, m_test)
            
            mse = np.mean((y_test - pred)**2)
            results[i, j] = mse
        except:
            results[i, j] = np.inf

best_idx = np.unravel_index(np.argmin(results), results.shape)
best_p = p_values[best_idx[0]]
best_m = m_values[best_idx[1]]
best_mse = results[best_idx]

print(f"Cel mai bun p: {best_p}")
print(f"Cel mai bun m: {best_m}")
print(f"MSE minim: {best_mse:.2f}\n")

fig, ax = plt.subplots(figsize=(12, 6))
for j, m_val in enumerate(m_values):
    ax.plot(p_values, results[:, j], marker='o', label=f'm={m_val}')
ax.set_xlabel('p (ordin AR)')
ax.set_ylabel('MSE')
ax.set_title('Optimizare hiperparametri AR')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab8/ex_d_tuning.pdf')
plt.savefig('plots_lab8/ex_d_tuning.png')
plt.close()

train_size = N - best_m
y_train = y[:train_size]
y_test = y[train_size:]
coef = fit_ar(y_train, best_p)
pred = predict_ar(y_train, coef, best_m)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(N), y, 'b-', label='Original', alpha=0.6)
ax.plot(range(train_size, N), pred, 'r--', label=f'Predictie optima AR(p={best_p})', linewidth=2)
ax.axvline(train_size, color='g', linestyle='--', alpha=0.5, label='Split')
ax.set_title(f'Model AR optimal: p={best_p}, m={best_m}, MSE={best_mse:.2f}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_lab8/ex_d_best_model.pdf')
plt.savefig('plots_lab8/ex_d_best_model.png')
plt.close()

print("Laborator 8 finalizat")
