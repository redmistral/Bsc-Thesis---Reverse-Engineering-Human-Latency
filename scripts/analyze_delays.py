"""
analyze_delays.py — Plots and stratified statistics for SQ2 (Section 5.2).

Reads `model_results/delays_per_event.csv` (produced by `estimate_delays.py`)
and generates the three figures and tabular summaries the thesis cites:

  Figure 1 — `plots/delays_overall_distribution.png`
      Overall histograms of Delay A, C, B with mean and median annotations.
      This is the figure that visualises the headline 227 / 1717 / 1379 ms
      result.

  Figure 2 — `plots/delays_by_saliency.png`
      The same three delays split by high- vs low-saliency stimuli, with the
      Mann-Whitney U statistic and p-value reported on each panel. This is
      the visual support for the SQ2 contrast.

  Figure 3 — `plots/delays_a_vs_c_scatter.png`
      Scatter of Delay A against Delay C per event with a y=x reference line,
      illustrating the dissociation between the two latencies.

In addition the script prints a console-only summary table (means, medians,
Mann-Whitney U on the saliency contrasts) that is reproduced as a snippet in
`results/delays_summary.txt`.

NOTE: this script is purely *analytic* and does no preprocessing. It depends
on the per-event delays CSV but not on the raw 1000 Hz EMG.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
OUT = BASE / 'model_results'
PLOTS = BASE / 'plots'
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})

df = pd.read_csv(OUT / 'delays_per_event.csv')
print(f'Loaded {len(df)} events')

COLORS = {'high': '#d62728', 'low': '#1f77b4'}


# ============ PLOT 1: Overall delay distributions ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, vcol, name, xlim in [
    (axes[0], 'delay_a_ms', 'a_valid', 'Delay A — Psychophysiological onset', (0, 500)),
    (axes[1], 'delay_c_ms', 'c_valid', 'Delay C — Total observable delay',    (0, 5000)),
    (axes[2], 'delay_b_ms', 'b_valid', 'Delay B — Cognitive-motor (C − A)',   (0, 5000)),
]:
    vals = df.loc[df[vcol], col].dropna()
    ax.hist(vals, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
    ax.axvline(vals.mean(),   color='red',   linestyle='--', linewidth=2, label=f'mean = {vals.mean():.0f} ms')
    ax.axvline(vals.median(), color='green', linestyle='--', linewidth=2, label=f'median = {vals.median():.0f} ms')
    ax.set_xlim(xlim)
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Event count')
    ax.set_title(f'{name}\n(n={len(vals)}, {len(vals)/len(df)*100:.0f}% of events)')
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'delays_overall_distribution.png')
plt.close()
print('  plots/delays_overall_distribution.png')


# ============ PLOT 2: Delay stratified by saliency ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, vcol, name, xlim in [
    (axes[0], 'delay_a_ms', 'a_valid', 'Delay A',          (0, 500)),
    (axes[1], 'delay_c_ms', 'c_valid', 'Delay C',          (0, 5000)),
    (axes[2], 'delay_b_ms', 'b_valid', 'Delay B (C − A)',  (0, 5000)),
]:
    sub = df[df[vcol]]
    hi = sub[sub['saliency'] == 'high'][col].dropna()
    lo = sub[sub['saliency'] == 'low'][col].dropna()
    bins = np.linspace(*xlim, 40)
    ax.hist(lo, bins=bins, color=COLORS['low'],  alpha=0.55, edgecolor='black', linewidth=0.5, label=f'low  (n={len(lo)}, μ={lo.mean():.0f})')
    ax.hist(hi, bins=bins, color=COLORS['high'], alpha=0.55, edgecolor='black', linewidth=0.5, label=f'high (n={len(hi)}, μ={hi.mean():.0f})')
    if len(hi) > 10 and len(lo) > 10:
        u, p = mannwhitneyu(hi, lo, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.set_title(f'{name} by saliency\nMann-Whitney U p = {p:.3g} ({sig})')
    else:
        ax.set_title(f'{name} by saliency')
    ax.set_xlim(xlim)
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Event count')
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'delays_by_saliency.png')
plt.close()
print('  plots/delays_by_saliency.png')


# ============ PLOT 3: Boxplots (tighter summary) ============
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, col, vcol, name, ylim in [
    (axes[0], 'delay_a_ms', 'a_valid', 'Delay A',          (0, 500)),
    (axes[1], 'delay_c_ms', 'c_valid', 'Delay C',          (0, 5000)),
    (axes[2], 'delay_b_ms', 'b_valid', 'Delay B',          (0, 5000)),
]:
    sub = df[df[vcol]]
    hi = sub[sub['saliency'] == 'high'][col].dropna().values
    lo = sub[sub['saliency'] == 'low'][col].dropna().values
    bp = ax.boxplot([lo, hi], labels=['low', 'high'], patch_artist=True, widths=0.6,
                    showfliers=False, medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], [COLORS['low'], COLORS['high']]):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    u, p = mannwhitneyu(hi, lo, alternative='two-sided') if len(hi) > 10 and len(lo) > 10 else (np.nan, np.nan)
    ax.set_title(f'{name}\np = {p:.3g}' if not np.isnan(p) else name)
    ax.set_ylabel('Delay (ms)')
    ax.set_ylim(ylim)
plt.tight_layout()
plt.savefig(PLOTS / 'delays_boxplot_saliency.png')
plt.close()
print('  plots/delays_boxplot_saliency.png')


# ============ PLOT 4: Delay B scatter — A vs C ============
sub = df[df['a_valid'] & df['c_valid']]
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
for sal, marker, color in [('low', 'o', COLORS['low']), ('high', '^', COLORS['high'])]:
    s = sub[sub['saliency'] == sal]
    ax.scatter(s['delay_a_ms'], s['delay_c_ms'], c=color, alpha=0.4, s=25,
               marker=marker, edgecolor='white', linewidth=0.3, label=f'{sal} (n={len(s)})')
# Iso-lines for Delay B constants
for db in [500, 1000, 2000, 3000]:
    ax.plot([0, 500], [db, db + 500], 'k:', alpha=0.3, linewidth=1)
    ax.annotate(f'B={db}', xy=(470, db + 470), fontsize=8, alpha=0.6)
ax.set_xlabel('Delay A — EMG onset (ms)')
ax.set_ylabel('Delay C — Valence onset (ms)')
ax.set_title('Delay C vs Delay A  (dotted lines: iso-Delay-B)')
ax.set_xlim(0, 500); ax.set_ylim(0, 5000)
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'delays_a_vs_c_scatter.png')
plt.close()
print('  plots/delays_a_vs_c_scatter.png')


# ============ PLOT 5: Scene-type stratification ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, vcol, name, xlim in [
    (axes[0], 'delay_a_ms', 'a_valid', 'Delay A',          (0, 500)),
    (axes[1], 'delay_c_ms', 'c_valid', 'Delay C',          (0, 5000)),
    (axes[2], 'delay_b_ms', 'b_valid', 'Delay B',          (0, 5000)),
]:
    sub = df[df[vcol]]
    pos = sub[sub['scene'] == 'positive'][col].dropna()
    neg = sub[sub['scene'] == 'negative'][col].dropna()
    bins = np.linspace(*xlim, 40)
    ax.hist(pos, bins=bins, color='#2ca02c', alpha=0.55, edgecolor='black', linewidth=0.5, label=f'positive (n={len(pos)}, μ={pos.mean():.0f})')
    ax.hist(neg, bins=bins, color='#9467bd', alpha=0.55, edgecolor='black', linewidth=0.5, label=f'negative (n={len(neg)}, μ={neg.mean():.0f})')
    if len(pos) > 10 and len(neg) > 10:
        u, p = mannwhitneyu(pos, neg, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.set_title(f'{name} by scene\np = {p:.3g} ({sig})')
    else:
        ax.set_title(f'{name} by scene')
    ax.set_xlim(xlim)
    ax.set_xlabel('Delay (ms)')
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'delays_by_scene.png')
plt.close()
print('  plots/delays_by_scene.png')


# ============ PLOT 6: Per-channel Delay A contribution ============
ch_cols = [f'delay_a_ch{i}_ms' for i in range(9)]
ch_muscles = ['zygomaticus_1', 'zygomaticus_2', 'frontalis_1', 'frontalis_2',
              'orbicularis_oculi_1', 'orbicularis_oculi_2',
              'corrugator_1', 'corrugator_2', 'extra_channel']
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
valid_rates, medians = [], []
for c, m in zip(ch_cols, ch_muscles):
    vals = df[c].dropna()
    vals_in_range = vals[(vals >= 100) & (vals <= 500)]
    valid_rates.append(len(vals_in_range) / len(df) * 100 if len(df) else 0)
    medians.append(vals_in_range.median() if len(vals_in_range) else np.nan)
x = np.arange(9)
ax2 = ax.twinx()
b1 = ax.bar(x - 0.2, valid_rates, 0.4, color='steelblue', edgecolor='black', label='% events with crossing in 100-500ms')
b2 = ax2.bar(x + 0.2, medians, 0.4, color='coral', edgecolor='black', label='median crossing time (ms)')
ax.set_xticks(x); ax.set_xticklabels([f'ch{i}\n{ch_muscles[i][:10]}' for i in range(9)], fontsize=9)
ax.set_ylabel('% events with valid crossing', color='steelblue')
ax2.set_ylabel('median crossing time (ms)', color='coral')
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')
ax.set_title('Delay A per EMG channel — detection rate & median timing')
plt.tight_layout()
plt.savefig(PLOTS / 'delays_per_channel.png')
plt.close()
print('  plots/delays_per_channel.png')


# ============ NUMERIC SUMMARY ============
print('\n' + '=' * 80)
print('SUMMARY TABLE (for thesis)')
print('=' * 80)
lines = []
lines.append(f'{"Metric":<35s}{"n":>6s}{"mean":>8s}{"med":>8s}{"std":>8s}{"p (hi vs lo)":>14s}')
lines.append('-' * 85)
for col, vcol, name in [
    ('delay_a_ms', 'a_valid', 'Delay A (EMG onset)'),
    ('delay_c_ms', 'c_valid', 'Delay C (valence onset)'),
    ('delay_b_ms', 'b_valid', 'Delay B (cog-motor, C-A)'),
]:
    sub = df[df[vcol]]; vals = sub[col].dropna()
    hi = sub[sub['saliency']=='high'][col].dropna().values
    lo = sub[sub['saliency']=='low'][col].dropna().values
    p = mannwhitneyu(hi, lo, alternative='two-sided')[1] if len(hi)>10 and len(lo)>10 else np.nan
    lines.append(f'{name:<35s}{len(vals):>6d}{vals.mean():>8.0f}{vals.median():>8.0f}{vals.std():>8.0f}{p:>14.4g}')
print('\n'.join(lines))

# Write summary to text file
with open(OUT / 'delays_summary.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nsaved -> {OUT}/delays_summary.txt')

# ============ SALIENCY DIRECTION CHECK ============
print('\n' + '=' * 80)
print('DIRECTION OF EFFECTS (high − low, ms)')
print('=' * 80)
for col, vcol, name in [
    ('delay_a_ms', 'a_valid', 'Delay A'),
    ('delay_c_ms', 'c_valid', 'Delay C'),
    ('delay_b_ms', 'b_valid', 'Delay B'),
]:
    sub = df[df[vcol]]
    hi = sub[sub['saliency']=='high'][col].dropna().values
    lo = sub[sub['saliency']=='low'][col].dropna().values
    if len(hi) > 10 and len(lo) > 10:
        diff = hi.mean() - lo.mean()
        expect = 'faster for high' if col == 'delay_a_ms' else 'faster for high (threat prioritization)'
        observed = 'HIGH faster' if diff < 0 else 'HIGH slower'
        print(f'  {name:10s}: Δ = {diff:+6.0f} ms ({observed})')

print('\nDone.')
