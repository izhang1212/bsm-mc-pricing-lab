# Black–Scholes Monte Carlo Pricing Framework

A Python implementation of European option pricing using the Black–Scholes model and Monte Carlo simulation, with variance reduction via antithetic variates and control variates.

---

## Overview

### What does this do?

This project prices European call and put options two ways — analytically via the Black–Scholes formula, and numerically via Monte Carlo simulation. It then validates that both methods agree. It runs the simulation across thousands of paths, measures how precisely each method recovers the closed-form price, and benchmarks four estimators (naive MC, antithetic variates, control variates, and a combined approach) against each other.

### Purpose

The goal is to build an understanding of how derivatives are priced in practice. 
The Black–Scholes formula gives the exact theoretical answer; implementing Monte Carlo shows how that same answer is approximated numerically when no closed form exists (as is the case for most exotic options). 
The variance reduction work addresses a core challenge in quantitative finance: simulation is expensive, so understanding how to extract more precision from fewer paths has direct practical value. 
The Greeks analysis develops intuition for how an option's risk profile shifts with market conditions — the foundation of any hedging strategy.

**Parameters used throughout:**

| Parameter | Value | Description |
|---|---|---|
| S₀ | $50 | Initial stock price |
| K | $50 | Strike price (at-the-money) |
| r | 5% | Risk-free rate (annual) |
| σ | 25% | Volatility (annual) |
| T | 0.5 yr | Time to expiry |

---

## Methodology & Concepts

### Geometric Brownian Motion (GBM)

Stock prices are modeled as a continuous-time stochastic process where the log-return over any period is normally distributed. Under the risk-neutral measure Q:

$$dS_t = r \cdot S_t \cdot dt + \sigma \cdot S_t \cdot dW_t$$

This gives the exact terminal price:

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T} \cdot Z\right], \quad Z \sim \mathcal{N}(0,1)$$

For a European option, only the terminal price $S_T$ matters — so a single normal draw per path is exact. Multi-step simulation (used in Part 5 for the convergence study) traces the full path but converges to the same terminal distribution.

---

### Black–Scholes Model (BSM)

BSM gives a closed-form solution for European option prices under the assumption that the stock follows GBM with constant volatility and no dividends.

**Call price:**
$$C = S_0 \cdot N(d_1) - K e^{-rT} N(d_2)$$

**Put price:**
$$P = K e^{-rT} N(-d_2) - S_0 \cdot N(-d_1)$$

where:
$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Put–call parity ($C - P = S_0 - Ke^{-rT}$) provides an independent validation check.

---

### Greeks

Greeks measure how sensitive the option price is to each input. They are used in practice to hedge portfolios — a market maker who sells an option uses the Greeks to know how much stock, volatility, or time exposure they have taken on.

| Greek | Measures | Formula |
|---|---|---|
| **Delta** (Δ) | Price sensitivity to S | $N(d_1)$ for calls, $N(d_1)-1$ for puts |
| **Gamma** (Γ) | Delta's sensitivity to S | $N'(d_1) / (S\sigma\sqrt{T})$ |
| **Vega** (ν) | Sensitivity to σ | $S \cdot N'(d_1) \cdot \sqrt{T}$ |
| **Theta** (Θ) | Time decay per day | $-(S N'(d_1)\sigma)/(2\sqrt{T}) \pm r K e^{-rT} N(\pm d_2)$, divided by 252 |
| **Rho** (ρ) | Sensitivity to r | $\pm K T e^{-rT} N(\pm d_2)$ |

---

### Monte Carlo Pricing

The risk-neutral pricing formula says the option price equals the discounted expected payoff under Q:

$$C = e^{-rT} \, \mathbb{E}^Q\!\left[\max(S_T - K,\, 0)\right]$$

Monte Carlo approximates this expectation by averaging payoffs across $N$ simulated terminal prices:

$$\hat{C} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)$$

The standard error of this estimate is $\text{SE} = \sigma_{\text{payoff}} / \sqrt{N}$, so precision improves at rate $1/\sqrt{N}$ — halving the SE requires 4× more paths. Variance reduction techniques improve the constant, not the rate.

---

### Variance Reduction

#### Antithetic Variates

For each draw $Z \sim \mathcal{N}(0,1)$, also simulate the mirrored path using $-Z$. Average the two payoffs before taking the grand mean:

$$\hat{C}_{\text{anti}} = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N} \frac{f(Z_i) + f(-Z_i)}{2}$$

The key: $\text{Var}\!\left(\frac{X+X'}{2}\right) = \frac{\text{Var}(X) + 2\,\text{Cov}(X,X')}{4}$. Because high-$Z$ paths and low-$(-Z)$ paths are negatively correlated, $\text{Cov}(X,X') < 0$, and the paired average has strictly lower variance than the naive estimator at the same number of random draws.

**Result: 72.5% variance reduction on this option.**

#### Control Variates

Exploit a known identity: under the risk-neutral measure, the discounted stock price is a martingale, so:

$$\mathbb{E}^Q\!\left[e^{-rT} S_T\right] = S_0$$

This means $c_i = e^{-rT} S_T^{(i)} - S_0$ has mean zero. We can subtract any multiple of it from the estimator without introducing bias:

$$\hat{C}_{\text{cv}} = \frac{1}{N}\sum_{i=1}^{N} \left[\text{disc. payoff}_i - \beta\left(e^{-rT} S_T^{(i)} - S_0\right)\right]$$

The optimal $\beta$ that minimizes variance is:

$$\beta^* = \frac{\text{Cov}(\text{payoff},\; e^{-rT}S_T)}{\text{Var}(e^{-rT}S_T)}$$

Intuition: when $S_T$ is "lucky" (above its expectation), both the call payoff and the control are inflated. Subtracting $\beta^*(c_i)$ corrects for that luck path-by-path. The variance of the adjusted estimator is $\text{Var}(\text{payoff})(1 - \rho^2)$, where $\rho$ is the correlation between the payoff and the control.

**Result: 82.3% variance reduction, estimated $\beta \approx 0.625$.**

#### Combined (Antithetic + Control Variates)

Apply antithetic pairing first, then run the control variate correction on the paired quantities. The two techniques target different noise sources, so their reductions compound.

**Result: 97.8% variance reduction — equivalent to 44× more naive paths.**

---

## Project Structure

```
BSM_Project/
│
├── config.py                       # Central parameter store (S0, K, r, σ, T)
├── main.py                         # Entry point — runs all six parts sequentially
├── requirements.txt
│
├── pricing/
│   ├── bsm.py                      # Closed-form BSM call/put prices, d1/d2
│   └── greeks.py                   # Analytical Delta, Gamma, Vega, Theta, Rho
│
├── simulation/
│   ├── gbm.py                      # GBM path simulator (multi-step and single-path)
│   ├── monte_carlo.py              # Naive MC pricer + convergence study runner
│   └── variance_reduction.py       # Antithetic, control variates, combined, summary
│
├── visualization/
│   └── plots.py                    # All matplotlib figures (Greeks, surface, MC convergence,
│                                   #   sensitivity, variance reduction comparison)
│
└── output/                         # Timestamped PNGs written on each run
    ├── greeks_<ts>.png
    ├── vol_sensitivity_<ts>.png
    ├── price_surface_<ts>.png
    ├── mc_convergence_<ts>.png
    └── variance_reduction_<ts>.png
```

**Data flow:** `config.py` → `main.py` pulls parameters → calls `pricing/` for closed-form answers → calls `simulation/` for MC estimates → calls `visualization/` to save plots. Each module is independently importable.

---

## Running the Project

```bash
pip install -r requirements.txt
python main.py
```

Plots are saved to `output/` with a timestamp. Old plots are cleared on each run.

---

## Sample Output

```
=======================================================
  Lab: Black–Scholes and Monte Carlo Pricing
=======================================================
  Parameters: S0=50, K=50, r=5%, σ=25%, T=0.5 yr

───────────────────────────────────────────────────────
  Part 1 · Black–Scholes Pricing
───────────────────────────────────────────────────────
  Call price : $4.1300
  Put  price : $2.8955
  Put–Call parity check: C − P = 1.2345  |  S − Ke^(−rT) = 1.2345  ✓

───────────────────────────────────────────────────────
  Part 2 · Greeks (at S0)
───────────────────────────────────────────────────────
  Delta   : call = +0.59088   put = -0.40912
  Gamma   : +0.04396   (same for call & put)
  Vega    : +0.13737   (same for call & put)
  Theta   : call = -0.01867   put = -0.00899
  Rho     : call = +0.12707   put = -0.11676

───────────────────────────────────────────────────────
  Part 5 · Monte Carlo Pricing via GBM
───────────────────────────────────────────────────────
  Call:  BSM = $4.1300  |  MC (100k) = $4.1366 ± $0.0382  (95% CI)
  Put:   BSM = $2.8955  |  MC (100k) = $2.8824 ± $0.0260  (95% CI)

───────────────────────────────────────────────────────
  Part 6 · Variance Reduction (Antithetic & Control Variates)
───────────────────────────────────────────────────────
  Method            Price      Std Err    Var Reduc   Error vs BSM
  ──────────────── ────────── ────────── ──────────── ──────────────
  Naive MC         $  4.1182  $ 0.01949         0.0%       28.70 bps
  Antithetic       $  4.1402  $ 0.01023        72.5%       24.56 bps
  Control Var.     $  4.1377  $ 0.00821        82.3%       18.59 bps  β=0.625
  Combined         $  4.1239  $ 0.00292        97.8%       14.69 bps  β=2.767

  · Antithetic variates: 72.5% variance reduction by pairing each Z with −Z
  · Control variates: 82.3% reduction using E[e^{−rT}·S_T] = S₀
  · Combined: 97.8% reduction — equivalent to 44× more naive paths
```

### Plots Generated

**Greeks across stock prices** — Five subplots showing how each Greek evolves as S moves from $20 to $80, with OTM/ATM/ITM regions shaded. Delta shows the S-curve that defines hedging ratio; Gamma peaks at-the-money where the hedge changes fastest.

**Volatility sensitivity** — Call and put prices as a function of S for six volatility levels (10%–60%). Higher vol lifts all option prices and flattens the S-shape — the option becomes less "directional."

**3-D price surface** — Option price as a joint function of S and σ, rendered as a color-mapped surface. Shows that price is monotone in both dimensions for a call, and reveals the convexity in σ (Vega is always positive).

**MC convergence** — Price estimate vs. path count on a log-scale x-axis, with 95% confidence bands. The bands visibly tighten as paths increase, and the estimate converges to the BSM reference line.

**Variance reduction comparison** — Left panel overlays all four estimator convergence curves; the variance-reduced methods have visibly tighter CI bands at every path count. Right panel is a bar chart of standard errors at 100k paths with variance reduction % annotated — the combined bar is roughly 7× shorter than naive.
