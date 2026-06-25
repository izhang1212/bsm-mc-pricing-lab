import glob
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')

from config import S0, K, r, SIGMA, T
from pricing import bsm
from pricing import greeks as grk
from simulation.monte_carlo import mc_price, mc_convergence
from simulation.variance_reduction import variance_reduction_summary
from visualization import plots


def _hr(title):
    print(f'\n{"─" * 55}')
    print(f'  {title}')
    print(f'{"─" * 55}')


def _clear_output():
    removed = glob.glob(os.path.join('output', '*.png'))
    for f in removed:
        os.remove(f)
    if removed:
        print(f'  Cleared {len(removed)} old plot(s) from output/')


def main():
    _clear_output()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 55)
    print('  Lab: Black–Scholes and Monte Carlo Pricing')
    print('=' * 55)
    print(f'  Parameters: S0={S0}, K={K}, r={r:.0%}, σ={SIGMA:.0%}, T={T} yr')
    print(f'  Run ID: {ts}')

    # ── Part 1: BSM Pricing ───────────────────────────────────────
    _hr('Part 1 · Black–Scholes Pricing')
    c = bsm.call_price(S0, K, r, SIGMA, T)
    p = bsm.put_price(S0, K, r, SIGMA, T)
    print(f'  Call price : ${c:.4f}')
    print(f'  Put  price : ${p:.4f}')
    parity_lhs = c - p
    parity_rhs = S0 - K * np.exp(-r * T)
    print(f'  Put–Call parity check: C − P = {parity_lhs:.4f}  |  '
          f'S − Ke^(−rT) = {parity_rhs:.4f}  ✓')

    # ── Part 2: Greeks ────────────────────────────────────────────
    _hr('Part 2 · Greeks (at S0)')
    rows = [
        ('Delta', grk.delta(S0, K, r, SIGMA, T, 'call'), grk.delta(S0, K, r, SIGMA, T, 'put')),
        ('Gamma', grk.gamma(S0, K, r, SIGMA, T),         None),
        ('Vega',  grk.vega(S0, K, r, SIGMA, T),          None),
        ('Theta', grk.theta(S0, K, r, SIGMA, T, 'call'), grk.theta(S0, K, r, SIGMA, T, 'put')),
        ('Rho',   grk.rho(S0, K, r, SIGMA, T, 'call'),   grk.rho(S0, K, r, SIGMA, T, 'put')),
    ]
    for name, cv, pv in rows:
        if pv is not None:
            print(f'  {name:<8}: call = {cv:+.5f}   put = {pv:+.5f}')
        else:
            print(f'  {name:<8}: {cv:+.5f}   (same for call & put)')
    path = plots.plot_greeks(K, r, SIGMA, T, outfile=f'greeks_{ts}.png')
    print(f'  Saved → {path}')

    # ── Part 3: Sensitivity ───────────────────────────────────────
    _hr('Part 3 · Sensitivity Analysis')
    print('  Varying S ∈ [20, 80] across σ ∈ {10%, 20%, 30%, 40%, 50%, 60%}')
    path = plots.plot_sensitivity(K, r, T, outfile=f'vol_sensitivity_{ts}.png')
    print(f'  Saved → {path}')

    # ── Part 4: Price Surface ─────────────────────────────────────
    _hr('Part 4 · 3-D Price Surface')
    print('  Axes: stock price S × volatility σ → option price')
    path = plots.plot_price_surface(K, r, T, outfile=f'price_surface_{ts}.png')
    print(f'  Saved → {path}')

    # ── Part 5: Monte Carlo ───────────────────────────────────────
    _hr('Part 5 · Monte Carlo Pricing via GBM')
    path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    print('  Running convergence study…')
    call_prices, call_se = mc_convergence(S0, K, r, SIGMA, T, path_counts, option_type='call')
    put_prices,  put_se  = mc_convergence(S0, K, r, SIGMA, T, path_counts, option_type='put')
    mc_c, mc_c_se = call_prices[-1], call_se[-1]
    mc_p, mc_p_se = put_prices[-1],  put_se[-1]
    print(f'  Call:  BSM = ${c:.4f}  |  MC (100k) = ${mc_c:.4f} ± ${1.96*mc_c_se:.4f}  (95% CI)')
    print(f'  Put:   BSM = ${p:.4f}  |  MC (100k) = ${mc_p:.4f} ± ${1.96*mc_p_se:.4f}  (95% CI)')
    path = plots.plot_mc_convergence(
        S0, K, r, SIGMA, T, path_counts,
        (call_prices, call_se), (put_prices, put_se),
        c, p,
        outfile=f'mc_convergence_{ts}.png',
    )
    print(f'  Saved → {path}')

    # ── Part 6: Variance Reduction ────────────────────────────────────────
    _hr('Part 6 · Variance Reduction (Antithetic & Control Variates)')
    vr_path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    print('  Running all four estimators across path counts…')
    summary = variance_reduction_summary(S0, K, r, SIGMA, T, n_paths=100_000, seed=42)

    print()
    print(f'  {"Method":<16} {"Price":>10} {"Std Err":>10} {"Var Reduc":>12} {"Error vs BSM":>14}')
    print(f'  {"─"*16} {"─"*10} {"─"*10} {"─"*12} {"─"*14}')
    methods = [('naive', 'Naive MC'), ('anti', 'Antithetic'), ('cv', 'Control Var.'), ('combined', 'Combined')]
    for key, label in methods:
        m = summary[key]
        error_bps = abs(m['price'] - c) / c * 10_000
        beta_str = f"  β={m['beta']:.3f}" if 'beta' in m else ''
        print(f'  {label:<16} ${m["price"]:>8.4f}  ${m["se"]:>8.5f}  {m["vr_pct"]:>10.1f}%  {error_bps:>10.2f} bps{beta_str}')

    print()
    print(f'  Reference: BSM call = ${c:.4f}')
    print()
    print('  Interpretation:')
    print(f'  · Antithetic variates reduced estimator variance by {summary["anti"]["vr_pct"]:.1f}%')
    print(f'    by pairing each random draw Z with its mirror -Z, exploiting negative correlation.')
    print(f'  · Control variates reduced variance by {summary["cv"]["vr_pct"]:.1f}%')
    print(f'    by correcting each path\'s payoff using the known identity E[e^{{-rT}}·S_T] = S0.')
    print(f'  · Combined method achieved {summary["combined"]["vr_pct"]:.1f}% variance reduction.')
    vr_comb = summary['combined']['vr_pct'] / 100
    equiv_factor = 1 / (1 - vr_comb)
    print(f'  · The combined method is equivalent in precision to running {equiv_factor:.0f}× '
          f'more naive paths at the same compute cost.')

    path = plots.plot_variance_reduction(
        S0, K, r, SIGMA, T, c,
        vr_path_counts, summary,
        option_type='call',
        outfile=f'variance_reduction_{ts}.png',
    )
    print(f'  Saved → {path}')

    print('\n  Done.')

if __name__ == '__main__':
    main()
