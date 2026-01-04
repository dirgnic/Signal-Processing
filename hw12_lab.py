import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


# -------------------------
# Ex. 1: Gaussian sampling
# -------------------------

def sample_1d_gaussian(mu: float, sigma2: float, n: int = 1_000, seed: int | None = 0):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma2)
    y = rng.normal(loc=mu, scale=sigma, size=n)

    x_plot = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = (1.0 / (np.sqrt(2 * np.pi * sigma2))) * np.exp(-0.5 * (x_plot - mu) ** 2 / sigma2)

    plt.figure(figsize=(6, 4))
    plt.hist(y, bins=40, density=True, alpha=0.5, label="samples")
    plt.plot(x_plot, pdf, "r", lw=2, label="pdf")
    plt.title("1D Gaussian samples")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()


def sample_2d_gaussian(mean: np.ndarray, cov: np.ndarray, n: int = 2_000, seed: int | None = 0):
    mean = np.asarray(mean).reshape(-1)
    cov = np.asarray(cov)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(2, n))  # standard normal
    x = mean.reshape(2, 1) + L @ z

    plt.figure(figsize=(5, 5))
    plt.scatter(x[0, :], x[1, :], s=5, alpha=0.5)
    plt.axis("equal")
    plt.title("2D Gaussian samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()


# --------------------------------------
# Ex. 2: Sampling Gaussian processes
# --------------------------------------

@dataclass
class RBFKernel:
    length_scale: float = 1.0
    variance: float = 1.0

    def __call__(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).T
        x_prime = np.atleast_2d(x_prime).T
        d2 = (x - x_prime.T) ** 2
        return self.variance * np.exp(-0.5 * d2 / (self.length_scale**2))


@dataclass
class PeriodicKernel:
    length_scale: float = 1.0
    variance: float = 1.0
    period: float = 1.0

    def __call__(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).T
        x_prime = np.atleast_2d(x_prime).T
        r = np.pi * np.abs(x - x_prime.T) / self.period
        return self.variance * np.exp(-2 * (np.sin(r) ** 2) / (self.length_scale**2))


# -------------------------------------------------
# Operații care păstrează pozitiv definit (din curs)
# k1, k2 – kerneli pozitivi definiți
# -------------------------------------------------

def sum_kernel(k1, k2):
    """(1) k(x,y) = k1(x,y) + k2(x,y)."""

    def k(x, y):
        return k1(x, y) + k2(x, y)

    return k


def product_kernel(k1, k2):
    """(2) k(x,y) = k1(x,y) * k2(x,y)."""

    def k(x, y):
        return k1(x, y) * k2(x, y)

    return k


def polynomial_kernel_from_base(base_kernel, coeffs):
    """(4) p(k(x,y)) pentru un polinom cu coeficienți nenegativi.

    p(z) = c0 + c1 z + c2 z^2 + ...
    coeffs: list/array de coeficienți c0, c1, ... (toți >= 0).
    """

    coeffs = np.asarray(coeffs, dtype=float)

    def k(x, y):
        K = base_kernel(x, y)
        out = np.zeros_like(K, dtype=float)
        power = np.ones_like(K, dtype=float)
        for c in coeffs:
            out = out + c * power
            power = power * K
        return out

    return k


def exp_kernel_from_base(base_kernel):
    """(5) k(x,y) = exp(k0(x,y))."""

    def k(x, y):
        return np.exp(base_kernel(x, y))

    return k


def scaled_kernel(f, base_kernel):
    """(6) k(x,y) = f(x) k0(x,y) f(y)."""

    def k(x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        fx = np.asarray([f(xi) for xi in x])
        fy = np.asarray([f(yi) for yi in y])
        K0 = base_kernel(x, y)
        return fx[:, None] * K0 * fy[None, :]

    return k


def sample_gp(x: np.ndarray, kernel, n_samples: int = 3, seed: int | None = 0):
    x = np.asarray(x).reshape(-1)
    rng = np.random.default_rng(seed)
    K = kernel(x, x)
    # jitter for numerical stability
    K += 1e-8 * np.eye(len(x))
    L = np.linalg.cholesky(K)

    plt.figure(figsize=(7, 4))
    for _ in range(n_samples):
        z = rng.normal(size=len(x))
        f = L @ z
        plt.plot(x, f)
    plt.title("Samples from a Gaussian process")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()


# -----------------------------------------------------------
# Ex. 3: GP regression on Mauna Loa CO2 (monthly averages)
# -----------------------------------------------------------


def monthly_mean_from_daily(t_days: np.ndarray, y_daily: np.ndarray):
    """Group daily data into calendar months and return monthly mean.

    Parameters
    ----------
    t_days : array of shape (N,)
        Time in fractional years or numeric days.
    y_daily : array of shape (N,)
        CO2 concentration.

    Returns
    -------
    t_month : array
        Time of each month (use month center).
    y_month : array
        Monthly mean CO2.
    """

    # Here we assume t_days is already in fractional years, so we just
    # bucket data into bins of width ~1/12 year.
    t_days = np.asarray(t_days)
    y_daily = np.asarray(y_daily)

    # define bin edges of width 1/12 year
    dt = 1.0 / 12.0
    t_min, t_max = t_days.min(), t_days.max()
    edges = np.arange(np.floor(t_min / dt) * dt, np.ceil(t_max / dt) * dt + dt, dt)

    idx = np.digitize(t_days, edges) - 1
    n_bins = len(edges) - 1

    t_month = []
    y_month = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        t_bin = t_days[mask]
        y_bin = y_daily[mask]
        t_month.append(0.5 * (edges[b] + edges[b + 1]))
        y_month.append(np.mean(y_bin))

    return np.asarray(t_month), np.asarray(y_month)


def linear_trend(t: np.ndarray, y: np.ndarray):
    """Fit y = a * t + b using least squares and return trend and residuals."""

    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    A = np.vstack([t, np.ones_like(t)]).T
    theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = theta
    trend = a * t + b
    residuals = y - trend
    return trend, residuals, (a, b)


def gp_regression(t_train: np.ndarray, y_train: np.ndarray, t_test: np.ndarray, kernel, sigma_n: float):
    """Plain GP regression using closed-form formulas (slides 27-28)."""

    t_train = np.asarray(t_train).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)
    t_test = np.asarray(t_test).reshape(-1)

    K = kernel(t_train, t_train)
    K_s = kernel(t_train, t_test)
    K_ss = kernel(t_test, t_test)

    K_y = K + (sigma_n**2) * np.eye(len(t_train))
    L = np.linalg.cholesky(K_y)

    # Solve for alpha = K_y^{-1} y via two triangular solves
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mu_post = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    cov_post = K_ss - v.T @ v

    return mu_post, cov_post


def mauna_loa_gp_example(
    t_month: np.ndarray,
    y_month: np.ndarray,
    n_pred_last: int = 12,
    length_scale: float = 1.0,
    variance: float = 1.0,
    period: float = 1.0,
    sigma_n: float = 0.3,
):
    """Gaussian process regression on monthly CO2 (no-trend series)."""

    t = np.asarray(t_month).reshape(-1)
    y = np.asarray(y_month).reshape(-1)

    # Use all but last n_pred_last months as training
    t_train = t[:-n_pred_last]
    y_train = y[:-n_pred_last]
    t_test = t[-n_pred_last:]
    y_true = y[-n_pred_last:]

    # kernel = RBFKernel(length_scale=length_scale, variance=variance)
    kernel = PeriodicKernel(length_scale=length_scale, variance=variance, period=period)

    mu_post, cov_post = gp_regression(t_train, y_train, t_test, kernel, sigma_n=sigma_n)
    std_post = np.sqrt(np.clip(np.diag(cov_post), a_min=0.0, a_max=None))

    plt.figure(figsize=(7, 4))
    plt.plot(t, y, "k.", ms=3, label="data (no trend)")
    plt.plot(t_test, y_true, "ro", ms=4, label="true last 12 months")
    plt.plot(t_test, mu_post, "b-", lw=2, label="GP mean")
    plt.fill_between(
        t_test,
        mu_post - 2 * std_post,
        mu_post + 2 * std_post,
        color="b",
        alpha=0.2,
        label="95% CI",
    )
    plt.xlabel("time (years)")
    plt.ylabel("CO2 anomaly")
    plt.title("GP regression on Mauna Loa (last 12 months)")
    plt.legend()
    plt.tight_layout()


# -----------------------------------------------------------
# Descărcare și preprocesare date Mauna Loa (daily -> monthly)
# -----------------------------------------------------------

MAUNA_LOA_DAILY_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"
MAUNA_LOA_LOCAL_FILE = Path("co2_daily_mlo.txt")


def download_mauna_loa_daily(force: bool = False) -> Path:
    """Descarcă fișierul daily Mauna Loa, dacă nu există deja local.

    Datele sunt text cu linii comentate cu '#'.
    """

    if MAUNA_LOA_LOCAL_FILE.exists() and not force:
        return MAUNA_LOA_LOCAL_FILE

    print(f"Downloading Mauna Loa daily data from {MAUNA_LOA_DAILY_URL} ...")
    with urlopen(MAUNA_LOA_DAILY_URL) as resp:
        content = resp.read()
    MAUNA_LOA_LOCAL_FILE.write_bytes(content)
    print(f"Saved to {MAUNA_LOA_LOCAL_FILE}")
    return MAUNA_LOA_LOCAL_FILE


def load_mauna_loa_daily() -> tuple[np.ndarray, np.ndarray]:
    """Încarcă datele daily și returnează (t_decimal_year, co2).

    Structura fișierului NOAA tipic:
    year, month, day, decimal_date, co2, ...
    """

    path = download_mauna_loa_daily()
    data = np.loadtxt(path, comments="#")
    t_dec = data[:, 3]
    co2 = data[:, 4]
    # Filtrăm eventualele valori lipsă marcate cu -99.99
    mask = co2 > 0
    return t_dec[mask], co2[mask]


if __name__ == "__main__":
    # ---------------- Ex. 1: Gauss 1D și 2D ----------------
    sample_1d_gaussian(mu=0.0, sigma2=1.0)

    mean_2d = np.array([0.0, 0.0])
    cov_2d = np.array([[1.0, 0.8], [0.8, 1.0]])  # exemplu similar cu Wikipedia
    sample_2d_gaussian(mean_2d, cov_2d)

    # ---------------- Ex. 2: mostre de PG -------------------
    x_grid = np.linspace(0, 10, 200)
    sample_gp(x_grid, RBFKernel(length_scale=1.0, variance=1.0))
    sample_gp(x_grid, PeriodicKernel(length_scale=1.0, variance=1.0, period=1.0))

    # ---------------- Ex. 3: Mauna Loa ----------------------
    try:
        t_daily, y_daily = load_mauna_loa_daily()

        # (a) daily -> monthly mean
        t_month, y_month = monthly_mean_from_daily(t_daily, y_daily)
        plt.figure(figsize=(7, 4))
        plt.plot(t_month, y_month, "k-", lw=1)
        plt.xlabel("time (years)")
        plt.ylabel("CO2 (ppm)")
        plt.title("Mauna Loa CO2 – monthly mean")
        plt.tight_layout()

        # (b) trend liniar și eliminarea lui
        trend, residuals, (a, b) = linear_trend(t_month, y_month)
        plt.figure(figsize=(7, 4))
        plt.plot(t_month, y_month, "k.", ms=3, label="monthly data")
        plt.plot(t_month, trend, "r-", lw=2, label="linear trend")
        plt.xlabel("time (years)")
        plt.ylabel("CO2 (ppm)")
        plt.title("Mauna Loa – data and linear trend")
        plt.legend()
        plt.tight_layout()

        # serie fără trend
        plt.figure(figsize=(7, 4))
        plt.plot(t_month, residuals, "k-", lw=1)
        plt.xlabel("time (years)")
        plt.ylabel("CO2 anomaly (ppm)")
        plt.title("Mauna Loa – detrended monthly series")
        plt.tight_layout()

        # (c) RPG cu PG pe ultimele 12 luni (pe seria fără trend)
        mauna_loa_gp_example(
            t_month,
            residuals,
            n_pred_last=12,
            length_scale=1.0,
            variance=1.0,
            period=1.0,
            sigma_n=0.3,
        )
    except Exception as e:
        print("Mauna Loa example failed:", e)

    plt.show()
