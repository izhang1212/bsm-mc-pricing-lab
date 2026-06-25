import numpy as np
import matplotlib.pyplot as plt


def sim_one_path(S0, mu, sigma, T, n, seed=None):
    # S_t = S_{t-1} * exp((mu - 0.5σ²)dt + σ√dt·Z)
    rng = np.random.default_rng(seed)
    dt = T / n
    path = np.empty(n + 1)
    path[0] = S0
    for i in range(1, n + 1):
        Z = rng.standard_normal()
        path[i] = path[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return path


def sim_gbm(S0, mu, sigma, T, n, n_paths, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n
    Z = rng.standard_normal((n_paths, n))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def plot_paths(paths, T, title="GBM Price Paths", ax=None):
    n_paths, n_points = paths.shape
    time_grid = np.linspace(0, T, n_points)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_grid, paths.T, lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
