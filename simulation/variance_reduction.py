import numpy as np


def _terminal_price(S0, r, sigma, T, Z):
    """Exact GBM terminal price from standard normal draws Z."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _payoff(ST, K, option_type):
    if option_type == 'call':
        return np.maximum(ST - K, 0)
    return np.maximum(K - ST, 0)


def mc_naive(S0, K, r, sigma, T, n_paths=100_000, seed=None, option_type='call'):
    """
    Standard (naive) Monte Carlo with no variance reduction.
    Single-step exact GBM — correct for European options because the terminal
    distribution S_T is fully determined by one log-normal draw.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = _terminal_price(S0, r, sigma, T, Z)
    disc_payoffs = np.exp(-r * T) * _payoff(ST, K, option_type)
    price = disc_payoffs.mean()
    stderr = disc_payoffs.std() / np.sqrt(n_paths)
    return price, stderr


def mc_antithetic(S0, K, r, sigma, T, n_paths=100_000, seed=None, option_type='call'):
    """
    Antithetic variates: for each draw Z, also simulate with -Z.

    Var reduction idea: the two paths are mirror images of each other — when
    one overestimates the payoff the other underestimates it. Averaging the
    pair cancels much of that noise.

    Formally: Var((X + X') / 2) = (Var(X) + 2*Cov(X, X')) / 4.
    Since Cov(X, X') < 0, the paired average has lower variance than
    the naive estimator at the same number of random draws.

    n_paths here means n_paths pairs, so 2*n_paths terminal prices total,
    but only n_paths independent normals are drawn.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    ST_pos = _terminal_price(S0, r, sigma, T,  Z)
    ST_neg = _terminal_price(S0, r, sigma, T, -Z)

    discount = np.exp(-r * T)
    payoff_pos = discount * _payoff(ST_pos, K, option_type)
    payoff_neg = discount * _payoff(ST_neg, K, option_type)

    # Average each pair before taking the grand mean
    paired = (payoff_pos + payoff_neg) / 2
    price = paired.mean()
    stderr = paired.std() / np.sqrt(n_paths)
    return price, stderr


def mc_control_variate(S0, K, r, sigma, T, n_paths=100_000, seed=None, option_type='call'):
    """
    Control variates using the discounted terminal stock price as control.

    Key identity: under the risk-neutral measure Q,
        E[e^{-rT} * S_T] = S0   (by risk-neutral pricing of the stock itself).

    So (e^{-rT} * S_T - S0) has mean zero. Any multiple of it we add to the
    estimator keeps it unbiased. We choose the multiple (beta) to minimize
    variance:

        adjusted_i = payoff_i - beta * (control_i - S0)
        beta_opt   = Cov(payoff, control) / Var(control)

    At this beta, Var(adjusted) = Var(payoff) * (1 - rho^2), where rho is
    the correlation between the payoff and the control. The higher the
    correlation, the more variance we eliminate.

    Intuition: when S_T happens to be "lucky" (above its expectation), both
    the call payoff and the control are inflated. Subtracting beta*(control - S0)
    corrects for that luck path-by-path, tightening the estimator.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = _terminal_price(S0, r, sigma, T, Z)
    discount = np.exp(-r * T)

    disc_payoffs = discount * _payoff(ST, K, option_type)
    control = discount * ST          # e^{-rT} S_T
    control_mean = S0                # known: E[e^{-rT} S_T] = S0

    # Estimate beta from the same paths (standard practice)
    cov_mat = np.cov(disc_payoffs, control)
    beta = cov_mat[0, 1] / cov_mat[1, 1]

    adjusted = disc_payoffs - beta * (control - control_mean)
    price = adjusted.mean()
    stderr = adjusted.std() / np.sqrt(n_paths)
    return price, stderr, beta


def mc_combined(S0, K, r, sigma, T, n_paths=100_000, seed=None, option_type='call'):
    """
    Antithetic variates + control variates together.

    The antithetic step halves the variance of each paired draw; the control
    variate step then further corrects for correlated noise in those pairs.
    The two techniques are orthogonal, so their variance reductions
    approximately compound.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    discount = np.exp(-r * T)

    ST_pos = _terminal_price(S0, r, sigma, T,  Z)
    ST_neg = _terminal_price(S0, r, sigma, T, -Z)

    # Antithetic-averaged discounted payoff and control per pair
    disc_payoffs = discount * ((_payoff(ST_pos, K, option_type) +
                                _payoff(ST_neg, K, option_type)) / 2)
    control = discount * (ST_pos + ST_neg) / 2
    control_mean = S0

    cov_mat = np.cov(disc_payoffs, control)
    beta = cov_mat[0, 1] / cov_mat[1, 1]

    adjusted = disc_payoffs - beta * (control - control_mean)
    price = adjusted.mean()
    stderr = adjusted.std() / np.sqrt(n_paths)
    return price, stderr, beta


def variance_reduction_summary(S0, K, r, sigma, T, n_paths=100_000, seed=42, option_type='call'):
    """
    Run all four estimators and return a comparison dict.

    Variance reduction % is relative to the naive estimator's variance:
        VR% = (1 - SE_method^2 / SE_naive^2) * 100
    """
    p_naive, se_naive             = mc_naive(S0, K, r, sigma, T, n_paths, seed, option_type)
    p_anti,  se_anti              = mc_antithetic(S0, K, r, sigma, T, n_paths, seed, option_type)
    p_cv,    se_cv,   beta_cv     = mc_control_variate(S0, K, r, sigma, T, n_paths, seed, option_type)
    p_comb,  se_comb, beta_comb   = mc_combined(S0, K, r, sigma, T, n_paths, seed, option_type)

    var_naive = se_naive ** 2

    def vr(se):
        return (1 - se**2 / var_naive) * 100

    return {
        'naive':    {'price': p_naive, 'se': se_naive, 'vr_pct': 0.0},
        'anti':     {'price': p_anti,  'se': se_anti,  'vr_pct': vr(se_anti)},
        'cv':       {'price': p_cv,    'se': se_cv,    'vr_pct': vr(se_cv),   'beta': beta_cv},
        'combined': {'price': p_comb,  'se': se_comb,  'vr_pct': vr(se_comb), 'beta': beta_comb},
    }
