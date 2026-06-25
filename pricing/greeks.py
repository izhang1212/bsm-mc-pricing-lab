import numpy as np
from scipy.stats import norm
from .bsm import d1, d2

# Calculate delta: measures change in option price with respect to underlying asset price (S)
    # Formula for call: delta = N(d1)
    # Formula for put: delta = N(d1) - 1
def delta(S, K, r, sigma, T, option_type='call'):
    D1 = d1(S, K, r, sigma, T)
    return norm.cdf(D1) if option_type == 'call' else norm.cdf(D1) - 1

# Calculate gamma: measures delta sensitivity to changes in underlying asset price (S)
    # Identical for calls and puts
    # Formula: gamma = N'(d1) / (S * sigma * sqrt(T))
def gamma(S, K, r, sigma, T):
    # Identical for calls and puts
    D1 = d1(S, K, r, sigma, T)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))

# Calculate theta: measures change in option price with respect to time decay (t)
    # Formula for call: theta = -(S * N'(d1) * sigma)
    # Formula for put: theta = -(S * N'(d1) * sigma) + r * K * exp(-r * T) * N(-d2)
    # To get daily theta, divide by 252 (trading days in a year)
def theta(S, K, r, sigma, T, option_type='call'):
    D1, D2 = d1(S, K, r, sigma, T), d2(S, K, r, sigma, T)
    decay = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return (decay - r * K * np.exp(-r * T) * norm.cdf(D2)) / 252
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-D2)) / 252

# Calculate vega: measures change in option price with respect to volatility (sigma)
    # Identical for call and put options
    # Formula: vega = S * N'(d1) * sqrt(T)
def vega(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    return S * norm.pdf(D1) * np.sqrt(T) / 100

# Calculate rho: measures change in option price with respect to interest rate (r)
    # Formula for call: rho = K * T * exp(-r * T) * N(d2)
    # Formula for put: rho = -K * T * exp(-r * T) * N(-d2)
def rho(S, K, r, sigma, T, option_type='call'):
    D2 = d2(S, K, r, sigma, T)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(D2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-D2) / 100
