import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pricing.bsm import call_price, put_price
import pricing.greeks as grk
from simulation.variance_reduction import mc_naive, mc_antithetic, mc_control_variate, mc_combined

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

_S_MIN, _S_MAX = 20, 80
_N = 300


def _out(filename):
    return os.path.join(OUTPUT_DIR, filename)


def _shade_moneyness(ax, K):
    """Shade OTM / ATM / ITM regions (call perspective)."""
    atm_lo, atm_hi = K * 0.95, K * 1.05
    ax.axvspan(_S_MIN, atm_lo,  alpha=0.07, color='tomato',    label='OTM (call)')
    ax.axvspan(atm_lo, atm_hi,  alpha=0.10, color='gold',      label='ATM')
    ax.axvspan(atm_hi, _S_MAX,  alpha=0.07, color='limegreen', label='ITM (call)')
    ax.axvline(K, color='black', linestyle='--', lw=1.0, alpha=0.5, label=f'K = {K}')


# Plot greeks for call and put options across a range of stock prices 
def plot_greeks(K, r, sigma, T, outfile='greeks.png'):
    S = np.linspace(_S_MIN, _S_MAX, _N)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Option Greeks   (K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr)',
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    specs = [
        ('Delta',            grk.delta(S, K, r, sigma, T, 'call'),  grk.delta(S, K, r, sigma, T, 'put')),
        ('Gamma',            grk.gamma(S, K, r, sigma, T),           None),
        ('Vega (per 1% σ)', grk.vega(S, K, r, sigma, T),            None),
        ('Theta – Call/day', grk.theta(S, K, r, sigma, T, 'call'),  None),
        ('Theta – Put/day',  grk.theta(S, K, r, sigma, T, 'put'),   None),
        ('Rho (per 1% r)',   grk.rho(S, K, r, sigma, T, 'call'),    grk.rho(S, K, r, sigma, T, 'put')),
    ]

    for idx, (title, call_vals, put_vals) in enumerate(specs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        _shade_moneyness(ax, K)
        ax.plot(S, call_vals, color='steelblue', lw=2.0, label='Call')
        if put_vals is not None:
            ax.plot(S, put_vals, color='tomato', lw=2.0, label='Put')
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Stock Price S')
        ax.grid(True, alpha=0.25)
        line_handles = [h for h, l in zip(*ax.get_legend_handles_labels()) if l in ('Call', 'Put')]
        line_labels  = [l for l in ax.get_legend_handles_labels()[1] if l in ('Call', 'Put')]
        if line_handles:
            ax.legend(line_handles, line_labels, fontsize=8)

    legend_elements = [
        Patch(facecolor='tomato',    alpha=0.25, label='OTM (call, S < 0.95K)'),
        Patch(facecolor='gold',      alpha=0.35, label='ATM (0.95K ≤ S ≤ 1.05K)'),
        Patch(facecolor='limegreen', alpha=0.25, label='ITM (call, S > 1.05K)'),
        Line2D([0], [0], color='black', linestyle='--', lw=1, label='Strike K'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    path = _out(outfile)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# Plot sensitivity of call and put prices to volatility across a range of stock prices
def plot_sensitivity(K, r, T, sigma_vals=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
                     outfile='vol_sensitivity.png'):
    S = np.linspace(_S_MIN, _S_MAX, _N)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(sigma_vals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Price Sensitivity to Volatility   (K={K}, r={r:.0%}, T={T} yr)',
        fontsize=13, fontweight='bold'
    )

    for sigma, color in zip(sigma_vals, colors):
        lbl = f'σ = {sigma:.0%}'
        ax1.plot(S, call_price(S, K, r, sigma, T), color=color, lw=1.8, label=lbl)
        ax2.plot(S, put_price(S, K, r, sigma, T),  color=color, lw=1.8, label=lbl)

    for ax, title, loc in zip([ax1, ax2], ['Call Price ($)', 'Put Price ($)'],
                               ['upper left', 'upper right']):
        ax.axvline(K, color='black', linestyle='--', lw=1.0, alpha=0.6, label=f'K = {K}')
        ax.set_xlabel('Stock Price S')
        ax.set_ylabel('Option Price ($)')
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, loc=loc)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = _out(outfile)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Part 4 ─────────────────────────────────────────────────────────────────────

def plot_price_surface(K, r, T,
                       S_range=(20, 80), sigma_range=(0.05, 0.80),
                       outfile='price_surface.png'):
    S_vals   = np.linspace(S_range[0],     S_range[1],     80)
    sig_vals = np.linspace(sigma_range[0], sigma_range[1], 80)
    SS, SIG  = np.meshgrid(S_vals, sig_vals)

    C = call_price(SS, K, r, SIG, T)
    P = put_price(SS,  K, r, SIG, T)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f'Option Price Surface   (K={K}, r={r:.0%}, T={T} yr)',
        fontsize=13, fontweight='bold'
    )

    for idx, (Z, label, cmap) in enumerate(
        [(C, 'Call', 'Blues'), (P, 'Put', 'Reds')], 1
    ):
        ax = fig.add_subplot(1, 2, idx, projection='3d')
        surf = ax.plot_surface(SS, SIG, Z, cmap=cmap, alpha=0.88, linewidth=0)
        ax.set_xlabel('Stock Price S', labelpad=8)
        ax.set_ylabel('Volatility σ',  labelpad=8)
        ax.set_zlabel('Price ($)',      labelpad=8)
        ax.set_title(f'{label} Price', fontsize=11)
        fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.10, label='Price ($)')

    plt.tight_layout()
    path = _out(outfile)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Part 5 ─────────────────────────────────────────────────────────────────────


def plot_mc_convergence(S0, K, r, sigma, T, path_counts,
                        call_mc, put_mc, bsm_call, bsm_put,
                        outfile='mc_convergence.png'):
    call_prices, call_se = call_mc
    put_prices,  put_se  = put_mc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Monte Carlo Convergence   (S0={S0}, K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr)',
        fontsize=13, fontweight='bold'
    )

    for ax, prices, se, bsm_val, label, color in [
        (ax1, call_prices, call_se, bsm_call, 'Call', 'steelblue'),
        (ax2, put_prices,  put_se,  bsm_put,  'Put',  'tomato'),
    ]:
        ax.semilogx(path_counts, prices, 'o-', color=color, lw=2, ms=5, label='MC estimate')
        ax.fill_between(
            path_counts,
            prices - 1.96 * se,
            prices + 1.96 * se,
            alpha=0.22, color=color, label='95% CI'
        )
        ax.axhline(bsm_val, color='black', linestyle='--', lw=1.6,
                   label=f'BSM = ${bsm_val:.4f}')
        ax.set_xlabel('Number of Simulated Paths (log scale)')
        ax.set_ylabel('Option Price ($)')
        ax.set_title(f'{label} Price Convergence', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = _out(outfile)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Part 6 ─────────────────────────────────────────────────────────────────────

def plot_variance_reduction(S0, K, r, sigma, T, bsm_price,
                            path_counts, summary, option_type='call',
                            outfile='variance_reduction.png'):
    labels = ['Naive', 'Antithetic', 'Control Var.', 'Combined']
    colors = ['#e07b54', '#5b9bd5', '#70ad47', '#9b59b6']
    seed = 42

    # collect convergence data for each method across path counts
    conv = {k: {'prices': [], 'ses': []} for k in ['naive', 'anti', 'cv', 'combined']}
    for n in path_counts:
        p, se = mc_naive(S0, K, r, sigma, T, n, seed, option_type)
        conv['naive']['prices'].append(p); conv['naive']['ses'].append(se)

        p, se = mc_antithetic(S0, K, r, sigma, T, n, seed, option_type)
        conv['anti']['prices'].append(p); conv['anti']['ses'].append(se)

        p, se, _ = mc_control_variate(S0, K, r, sigma, T, n, seed, option_type)
        conv['cv']['prices'].append(p); conv['cv']['ses'].append(se)

        p, se, _ = mc_combined(S0, K, r, sigma, T, n, seed, option_type)
        conv['combined']['prices'].append(p); conv['combined']['ses'].append(se)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f'Variance Reduction Comparison   '
        f'(S0={S0}, K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr, {option_type})',
        fontsize=13, fontweight='bold'
    )

    # left: convergence curves with 95% CI bands — tighter band = less variance
    keys = ['naive', 'anti', 'cv', 'combined']
    for key, label, color in zip(keys, labels, colors):
        prices = np.array(conv[key]['prices'])
        ses    = np.array(conv[key]['ses'])
        ax1.semilogx(path_counts, prices, 'o-', color=color, lw=2, ms=4, label=label)
        ax1.fill_between(path_counts,
                         prices - 1.96 * ses,
                         prices + 1.96 * ses,
                         alpha=0.12, color=color)

    ax1.axhline(bsm_price, color='black', linestyle='--', lw=1.5,
                label=f'BSM = ${bsm_price:.4f}')
    ax1.set_xlabel('Number of Paths (log scale)')
    ax1.set_ylabel('Estimated Price ($)')
    ax1.set_title('Convergence to BSM Price', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # right: std-error bar chart at the largest path count
    se_vals = [summary[k]['se'] for k in keys]
    vr_vals = [summary[k]['vr_pct'] for k in keys]

    x = np.arange(len(labels))
    bars = ax2.bar(x, se_vals, color=colors, width=0.55, edgecolor='white', linewidth=0.8)

    for bar, vr in zip(bars, vr_vals):
        label_text = 'baseline' if vr == 0 else f'-{vr:.1f}% var'
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(se_vals) * 0.02,
                 label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Std Error ($)')
    ax2.set_title(f'Std Error at {path_counts[-1]:,} Paths', fontsize=11)
    ax2.grid(True, alpha=0.25, axis='y')
    ax2.set_ylim(0, max(se_vals) * 1.25)

    plt.tight_layout()
    path = _out(outfile)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
