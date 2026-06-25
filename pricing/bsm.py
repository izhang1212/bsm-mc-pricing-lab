import numpy as np
from scipy.stats import norm

# Solving BSM Process:

# 1. Calculate d1 = (ln(S/K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

# 2. Calculate d2 = d1 - sigma * sqrt(T)
def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

# 3. Calculate call price = S * N(d1) - K * exp(-r * T) * N(d2)
def call_price(S, K, r, sigma, T):
    D1, D2 = d1(S, K, r, sigma, T), d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)

# 4. Calculate put price = K * exp(-r * T) * N(-d2) - S * N(-d1) 
def put_price(S, K, r, sigma, T):
    D1, D2 = d1(S, K, r, sigma, T), d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)
