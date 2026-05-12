"""
analyze_modulation.py — Per-subject delay modulation analysis (Section 5.2.4).

This is the post-supervisor-meeting addition that turns the single point
estimate "Delay B median = 1379 ms" into a structured analysis: does that gap
*vary* with who the participant is and how they engaged with the VR scene?

Pipeline
--------
1. Load per-event delays from `model_results/delays_per_event.csv` and
   aggregate to one row per participant (median over valid events). Computed
   both pooled-across-scenes and separately for the positive and negative
   scenes, so per-subject paired tests are possible.

2. Join the per-subject aggregates with the participant-metadata Excel
   released by Mavridou et al. (2025), which contains:
     - demographics : AGE, GENDER, VREXPERIENCE, EDULEVEL, PassiveActive
     - presence     : PRES_BEINGPRESENT, PRES_CAPTIVATED, PRES_IMAGES_SPACE
     - clinical     : Empathy, DASSDepression, DASSAnxiety, DASSStress,
                      alexithymia, expressivity
     - personality  : Big Five (extraversion, Neuroticism, Openness,
                      agreeableness, Conscientiousness)
     - cognition    : total_memoryscore (free-recall of scene events)

3. Run the stat battery requested by the supervisor:
     - Mann-Whitney U: positive vs negative valence on each delay (event-level
       for power, per-subject for paired interpretation).
     - Mann-Whitney U: high vs low saliency on each delay (event-level).
     - Mann-Whitney U: gender, Active vs Passive condition (per-subject Delay B).
     - Spearman rho: every continuous predictor vs per-subject Delay A and B.
     - Benjamini-Hochberg correction across the full ~44-test battery.

4. Emit five figures (paired pos/neg boxplots, scatter plots, summary bar
   plot of all rhos vs Delay B) plus two machine-readable summaries.

Outputs
-------
model_results/delays_per_subject.csv          — per-pid medians + metadata join
model_results/delays_modulation_stats.csv     — every stat test, raw and BH-corrected p
model_results/delays_modulation_summary.txt   — human-readable summary
plots/delay_by_valence_subject.png            — paired pos/neg boxplots for A, B, C
plots/delay_b_vs_{age, vr_experience, presence_being_present, empathy,
                  neuroticism, alexithymia}.png — Spearman scatters
plots/delay_b_modulation_summary.png          — bar plot of all Spearman rhos vs Delay B

The matched-questionnaire subset is small (n=35 participants with both Delay B
and full personality/clinical data); the BH-corrected battery yielded no
surviving contrasts. Reported coefficients are exploratory (see §6.4).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
DATA = BASE / 'All Data'
OUT = BASE / 'model_results'
PLOTS = BASE / 'plots'
PLOTS.mkdir(exist_ok=True)

DELAYS_PER_EVENT = OUT / 'delays_per_event.csv'
META_XLSX = DATA / 'vrstudy_datafrom_current subjects_final.xlsx'

# Continuous predictors to test against delay_b (Spearman)
CONT_PREDICTORS = [
    ('AGE',                'Age (years)'),
    ('VREXPERIENCE',       'VR experience'),
    ('PRES_BEINGPRESENT',  'Presence: being there'),
    ('PRES_CAPTIVATED',    'Presence: captivated'),
    ('PRES_IMAGES_SPACE',  'Presence: images felt as space'),
    ('Empathy',            'Empathy'),
    ('DASSDepression',     'DASS depression'),
    ('DASSAnxiety',        'DASS anxiety'),
    ('DASSStress',         'DASS stress'),
    ('extraversion',       'Extraversion'),
    ('Neuroticism',        'Neuroticism'),
    ('Openness',           'Openness'),
    ('agreeableness',      'Agreeableness'),
    ('Conscientiousness',  'Conscientiousness'),
    ('expressivity',       'Expressivity'),
    ('alexithymia',        'Alexithymia'),
    ('total_memoryscore',  'Memory score (event recall)'),
]

# Categorical predictors (Mann-Whitney) — use 1 vs 2 coding
CAT_PREDICTORS = [
    ('GENDER',       'Gender (1 vs 2)'),
    ('PassiveActive','Active vs Passive condition'),
]

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})


# ============================================================
# 1. PER-SUBJECT AGGREGATE
# ============================================================
print('Loading per-event delays...')
df_events = pd.read_csv(DELAYS_PER_EVENT)
print(f'  {len(df_events)} events, {df_events["pid"].nunique()} unique participants')

# Per-subject aggregates over VALID events only
def agg_subject(df, scene=None):
    """Aggregate per pid: median delay_{a,b,c} on valid events."""
    if scene is not None:
        df = df[df['scene'] == scene]
    rows = []
    for pid, g in df.groupby('pid'):
        a_valid = g[g['a_valid']]['delay_a_ms']
        c_valid = g[g['c_valid']]['delay_c_ms']
        b_valid = g[g['b_valid']]['delay_b_ms']
        rows.append({
            'pid': pid,
            'n_events': len(g),
            'n_a_valid': len(a_valid),
            'n_b_valid': len(b_valid),
            'n_c_valid': len(c_valid),
            'delay_a_med': a_valid.median() if len(a_valid) else np.nan,
            'delay_b_med': b_valid.median() if len(b_valid) else np.nan,
            'delay_c_med': c_valid.median() if len(c_valid) else np.nan,
        })
    return pd.DataFrame(rows)


per_sub_all = agg_subject(df_events)
per_sub_pos = agg_subject(df_events, 'positive').rename(
    columns={'delay_a_med': 'delay_a_pos', 'delay_b_med': 'delay_b_pos', 'delay_c_med': 'delay_c_pos',
             'n_events': 'n_events_pos', 'n_a_valid': 'n_a_pos', 'n_b_valid': 'n_b_pos', 'n_c_valid': 'n_c_pos'})
per_sub_neg = agg_subject(df_events, 'negative').rename(
    columns={'delay_a_med': 'delay_a_neg', 'delay_b_med': 'delay_b_neg', 'delay_c_med': 'delay_c_neg',
             'n_events': 'n_events_neg', 'n_a_valid': 'n_a_neg', 'n_b_valid': 'n_b_neg', 'n_c_valid': 'n_c_neg'})

per_sub = per_sub_all.merge(per_sub_pos[['pid', 'delay_a_pos', 'delay_b_pos', 'delay_c_pos',
                                          'n_events_pos', 'n_b_pos']], on='pid', how='left')
per_sub = per_sub.merge(per_sub_neg[['pid', 'delay_a_neg', 'delay_b_neg', 'delay_c_neg',
                                      'n_events_neg', 'n_b_neg']], on='pid', how='left')

# ============================================================
# 2. JOIN WITH METADATA
# ============================================================
print(f'Loading metadata Excel ({META_XLSX.name})...')
meta = pd.read_excel(META_XLSX)
print(f'  {len(meta)} metadata rows')

# Coerce types and drop rows without a usable ID
meta['ID'] = pd.to_numeric(meta['ID'], errors='coerce').astype('Int64')
meta = meta.dropna(subset=['ID']).copy()
meta['pid'] = meta['ID'].astype(int)

# Keep only columns we need (plus pid)
keep = ['pid'] + [c for c, _ in CONT_PREDICTORS + CAT_PREDICTORS if c in meta.columns]
keep = list(dict.fromkeys(keep))  # de-dup
meta_keep = meta[keep].copy()

# Coerce numeric where possible (PassiveActive is object)
for c, _ in CONT_PREDICTORS:
    if c in meta_keep.columns:
        meta_keep[c] = pd.to_numeric(meta_keep[c], errors='coerce')

per_sub_full = per_sub.merge(meta_keep, on='pid', how='left')
print(f'  joined {per_sub_full.shape[0]} subjects with metadata')
print(f'  with delay_b_med: {per_sub_full["delay_b_med"].notna().sum()}')
print(f'  with AGE:         {per_sub_full["AGE"].notna().sum() if "AGE" in per_sub_full.columns else 0}')

# Save the per-subject table
out_csv = OUT / 'delays_per_subject.csv'
per_sub_full.to_csv(out_csv, index=False)
print(f'Saved per-subject table -> {out_csv.name}')

# ============================================================
# 3. STAT TESTS
# ============================================================
results = []  # list of dicts


def add(test, factor, target, n, stat, p, direction):
    results.append({
        'test': test, 'factor': factor, 'target': target,
        'n': n, 'statistic': stat, 'p_raw': p, 'direction': direction,
    })


# ---- Mann-Whitney: positive vs negative valence — EVENT LEVEL (high power)
for delay in ['a', 'b', 'c']:
    valid_col = f'{delay}_valid'
    delay_col = f'delay_{delay}_ms'
    sub = df_events[df_events[valid_col]].dropna(subset=[delay_col, 'scene'])
    pos = sub[sub['scene'] == 'positive'][delay_col]
    neg = sub[sub['scene'] == 'negative'][delay_col]
    if len(pos) >= 20 and len(neg) >= 20:
        U, p = mannwhitneyu(pos, neg, alternative='two-sided')
        d = pos.mean() - neg.mean()
        add('Mann-Whitney', f'valence (pos vs neg, event-level)',
            f'delay_{delay}_ms', len(pos) + len(neg), float(U), float(p),
            f'pos μ={pos.mean():.0f} (n={len(pos)}) vs neg μ={neg.mean():.0f} (n={len(neg)}); '
            f'Δ={d:+.0f} ms')

# ---- Mann-Whitney: positive vs negative valence — per-subject (paired-by-pid)
for delay in ['a', 'b', 'c']:
    pos_col = f'delay_{delay}_pos'
    neg_col = f'delay_{delay}_neg'
    if pos_col in per_sub_full.columns and neg_col in per_sub_full.columns:
        sub = per_sub_full.dropna(subset=[pos_col, neg_col])
        if len(sub) >= 10:
            U, p = mannwhitneyu(sub[pos_col], sub[neg_col], alternative='two-sided')
            d = sub[pos_col].mean() - sub[neg_col].mean()
            add('Mann-Whitney', f'valence (pos vs neg, per-subject)',
                f'delay_{delay}_med', len(sub), float(U), float(p),
                f'{"pos > neg" if d > 0 else "neg > pos"} by {abs(d):.0f} ms')

# ---- Mann-Whitney: high vs low saliency — EVENT LEVEL
for delay in ['a', 'b', 'c']:
    valid_col = f'{delay}_valid'
    delay_col = f'delay_{delay}_ms'
    sub = df_events[df_events[valid_col]].dropna(subset=[delay_col, 'saliency'])
    hi = sub[sub['saliency'] == 'high'][delay_col]
    lo = sub[sub['saliency'] == 'low'][delay_col]
    if len(hi) >= 20 and len(lo) >= 20:
        U, p = mannwhitneyu(hi, lo, alternative='two-sided')
        d = hi.mean() - lo.mean()
        add('Mann-Whitney', f'saliency (hi vs lo, event-level)',
            f'delay_{delay}_ms', len(hi) + len(lo), float(U), float(p),
            f'hi μ={hi.mean():.0f} (n={len(hi)}) vs lo μ={lo.mean():.0f} (n={len(lo)}); '
            f'Δ={d:+.0f} ms')

# ---- Mann-Whitney: gender on overall delay_b
if 'GENDER' in per_sub_full.columns:
    sub = per_sub_full.dropna(subset=['delay_b_med', 'GENDER'])
    g_vals = sorted(sub['GENDER'].unique())
    if len(g_vals) >= 2:
        a = sub[sub['GENDER'] == g_vals[0]]['delay_b_med']
        b = sub[sub['GENDER'] == g_vals[1]]['delay_b_med']
        if len(a) >= 5 and len(b) >= 5:
            U, p = mannwhitneyu(a, b, alternative='two-sided')
            d = a.mean() - b.mean()
            add('Mann-Whitney', f'gender ({g_vals[0]} vs {g_vals[1]})',
                'delay_b_med', len(sub), float(U), float(p),
                f'{int(g_vals[0])} > {int(g_vals[1])} by {abs(d):.0f} ms' if d > 0
                else f'{int(g_vals[1])} > {int(g_vals[0])} by {abs(d):.0f} ms')

# ---- Active vs Passive
if 'PassiveActive' in per_sub_full.columns:
    sub = per_sub_full.dropna(subset=['delay_b_med', 'PassiveActive'])
    grps = sub['PassiveActive'].astype(str).str.upper().str.strip()
    is_a = grps.str.startswith('A')
    is_p = grps.str.startswith('P')
    if is_a.sum() >= 5 and is_p.sum() >= 5:
        a = sub[is_a]['delay_b_med']
        b = sub[is_p]['delay_b_med']
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        add('Mann-Whitney', 'Active vs Passive',
            'delay_b_med', len(sub), float(U), float(p),
            f'A({a.mean():.0f}) vs P({b.mean():.0f}) ms')

# ---- Spearman: continuous predictors vs delay_b_med
for col, label in CONT_PREDICTORS:
    if col not in per_sub_full.columns:
        continue
    sub = per_sub_full.dropna(subset=['delay_b_med', col])
    if len(sub) < 15:
        continue
    rho, p = spearmanr(sub[col], sub['delay_b_med'])
    add('Spearman', label, 'delay_b_med', len(sub), float(rho), float(p),
        f'ρ={rho:+.3f}')

# Also try Spearman vs delay_a (psychophysiological onset)
for col, label in CONT_PREDICTORS:
    if col not in per_sub_full.columns:
        continue
    sub = per_sub_full.dropna(subset=['delay_a_med', col])
    if len(sub) < 15:
        continue
    rho, p = spearmanr(sub[col], sub['delay_a_med'])
    add('Spearman', label, 'delay_a_med', len(sub), float(rho), float(p),
        f'ρ={rho:+.3f}')

# ============================================================
# 4. BH-CORRECT AND SAVE
# ============================================================
res_df = pd.DataFrame(results)
if len(res_df):
    # BH correction across all tests jointly
    rej, p_bh, _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
    res_df['p_bh'] = p_bh
    res_df['sig'] = res_df['p_raw'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
    res_df['sig_bh'] = res_df['p_bh'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
    res_df = res_df.sort_values('p_raw').reset_index(drop=True)
    res_df.to_csv(OUT / 'delays_modulation_stats.csv', index=False)
    print(f'\nSaved {len(res_df)} stat tests -> delays_modulation_stats.csv')

# Human-readable summary
with open(OUT / 'delays_modulation_summary.txt', 'w') as f:
    f.write('=' * 90 + '\n')
    f.write('PHASE 1 — DELAY MODULATION SUMMARY (per-subject aggregates)\n')
    f.write('=' * 90 + '\n\n')
    f.write(f'Subjects with valid delay_b_med: {per_sub_full["delay_b_med"].notna().sum()}\n')
    f.write(f'Median (across subjects):\n')
    for d in ['a', 'b', 'c']:
        col = f'delay_{d}_med'
        v = per_sub_full[col].dropna()
        f.write(f'  Delay {d.upper()} : median={v.median():.0f}  IQR={v.quantile(0.25):.0f}–{v.quantile(0.75):.0f}  n={len(v)}\n')
    f.write('\n' + '=' * 90 + '\n')
    f.write('STATISTICAL TESTS (sorted by raw p-value)\n')
    f.write('=' * 90 + '\n')
    f.write(f'{"test":12s}  {"factor":35s}  {"target":18s}  {"n":>4s}  {"stat":>9s}  {"p_raw":>9s}  {"p_bh":>9s}  sig  direction\n')
    f.write('-' * 130 + '\n')
    for _, r in res_df.iterrows():
        f.write(f'{r["test"]:12s}  {r["factor"]:35s}  {r["target"]:18s}  {int(r["n"]):>4d}  '
                f'{r["statistic"]:>9.3f}  {r["p_raw"]:>9.4f}  {r["p_bh"]:>9.4f}  {r["sig_bh"]:>3s}  {r["direction"]}\n')

print(f'Saved human summary -> delays_modulation_summary.txt')

# Echo top hits to stdout
print('\nTop 5 lowest-p tests:')
for _, r in res_df.head(5).iterrows():
    print(f'  {r["test"]:12s} {r["factor"]:35s}  target={r["target"]:18s}  '
          f'p_raw={r["p_raw"]:.4f}  p_bh={r["p_bh"]:.4f}  {r["sig_bh"]}  ({r["direction"]})')

# ============================================================
# 5. PLOTS
# ============================================================
print('\nGenerating plots...')

# 5a. Boxplot: per-subject median delay_b, positive vs negative
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, d, lbl in zip(axes, ['a', 'b', 'c'], ['A — EMG onset', 'B — cognitive-motor gap', 'C — total observable']):
    sub = per_sub_full.dropna(subset=[f'delay_{d}_pos', f'delay_{d}_neg'])
    data = [sub[f'delay_{d}_pos'].values, sub[f'delay_{d}_neg'].values]
    bp = ax.boxplot(data, tick_labels=['Positive', 'Negative'], patch_artist=True, widths=0.55)
    for patch, c in zip(bp['boxes'], ['#7fb069', '#d96b6b']):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel(f'Delay {d.upper()} (ms)')
    ax.set_title(f'Delay {lbl}')
    # significance asterisk
    row = res_df[(res_df['target'] == f'delay_{d}_med') & res_df['factor'].str.contains('valence')]
    if len(row):
        sig = row.iloc[0]['sig_bh']
        ax.text(1.5, ax.get_ylim()[1] * 0.95, sig, ha='center', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(PLOTS / 'delay_by_valence_subject.png'); plt.close()
print('  plots/delay_by_valence_subject.png')

# 5b. Spearman scatter for top continuous predictor against delay_b
def scatter_spearman(col, ylab, fname):
    if col not in per_sub_full.columns: return
    sub = per_sub_full.dropna(subset=['delay_b_med', col])
    if len(sub) < 15: return
    rho, p = spearmanr(sub[col], sub['delay_b_med'])
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(sub[col], sub['delay_b_med'], s=40, c='steelblue', edgecolor='black', alpha=0.7)
    # fit line (simple lin reg on rank-transformed for visualisation only)
    from numpy.polynomial import polynomial as P
    if sub[col].std() > 1e-6:
        m, b = np.polyfit(sub[col], sub['delay_b_med'], 1)
        xs = np.linspace(sub[col].min(), sub[col].max(), 50)
        ax.plot(xs, m * xs + b, 'k--', alpha=0.6)
    ax.set_xlabel(ylab); ax.set_ylabel('Delay B per subject (ms, median)')
    ax.set_title(f'{ylab} vs Delay B (Spearman ρ={rho:+.3f}, p={p:.3f}, n={len(sub)})')
    plt.tight_layout(); plt.savefig(PLOTS / fname); plt.close()
    print(f'  plots/{fname}')


for col, lbl in [
    ('AGE', 'Age (years)'),
    ('VREXPERIENCE', 'VR experience'),
    ('PRES_BEINGPRESENT', 'Presence: being there'),
    ('Empathy', 'Empathy'),
    ('Neuroticism', 'Neuroticism'),
    ('alexithymia', 'Alexithymia'),
]:
    fname = f'delay_b_vs_{col.lower()}.png'
    scatter_spearman(col, lbl, fname)

# 5c. Combined Spearman ρ summary plot for delay_b
sub_df = res_df[(res_df['target'] == 'delay_b_med') & (res_df['test'] == 'Spearman')].copy()
if len(sub_df):
    sub_df = sub_df.sort_values('statistic')
    fig, ax = plt.subplots(1, 1, figsize=(8, max(4, len(sub_df) * 0.32)))
    ys = np.arange(len(sub_df))
    colors = ['#d96b6b' if r > 0 else '#5e8eb8' for r in sub_df['statistic']]
    ax.barh(ys, sub_df['statistic'], color=colors, edgecolor='black', linewidth=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(sub_df['factor'])
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_xlabel('Spearman ρ vs Delay B (per subject)')
    ax.set_title('Modulation of cognitive-motor delay (Delay B) by individual factors')
    # annotate significance
    for y, (_, r) in zip(ys, sub_df.iterrows()):
        ax.text(r['statistic'] + (0.02 if r['statistic'] >= 0 else -0.02),
                y, r['sig_bh'], va='center', ha='left' if r['statistic'] >= 0 else 'right',
                fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig(PLOTS / 'delay_b_modulation_summary.png'); plt.close()
    print('  plots/delay_b_modulation_summary.png')

print('\nDone.')
