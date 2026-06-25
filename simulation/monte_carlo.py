import numpy as np
from .gbm import sim_gbm

# Risk-neutral Monte Carlo price for a European option via GBM
def mc_price(S0, K, r, sigma, T, n_paths=50_000, n_steps=252, seed=None, option_type='call'):
    paths = sim_gbm(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    stderr = discount * payoffs.std() / np.sqrt(n_paths)
    return price, stderr

# Run Monte Carlo pricing for increasing path counts to study convergence
def mc_convergence(S0, K, r, sigma, T, path_counts, seed=None, option_type='call'):
    prices, stderrs = [], []
    for n in path_counts:
        p, se = mc_price(S0, K, r, sigma, T, n_paths=n, seed=seed, option_type=option_type)
        prices.append(p)
        stderrs.append(se)
    return np.array(prices), np.array(stderrs)
