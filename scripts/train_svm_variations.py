"""
train_svm_variations.py — SVM x 4 timing windows x 3 feature sets (Section 5.3.3).

WHY THIS SCRIPT EXISTS
----------------------
Two questions came out of the May 1 supervisor meeting:

  Q1 (timing):   Where in the response is saliency information concentrated?
                 The empirical Delay B = 1379 ms suggests a **late-appraisal**
                 window may be more informative than the **event-locked**
                 window everyone defaults to.
  Q2 (model):    The supervisor's prior expectation, based on her own past
                 work with this signal, is that an SVM may equal or exceed an
                 LSTM at this task. Test that.

This script tests both questions on a single grid.

THE FOUR TIMING VARIATIONS (Mavridou's whiteboard sketch)
---------------------------------------------------------
With B = empirical median Delay B = 1379 ms:

  V1 — event-locked:    T0          ..  T0 + 500 ms
  V2 — mid-appraisal:   T0 + B/2    ..  T0 + B/2 + 500 ms
  V3 — late-appraisal:  T0 + B      ..  T0 + B + 500 ms
        (~ T1, the empirical Delay C onset)
  Baseline:             T0 + 200 ms ..  T0 + 700 ms
        (the static 200 ms alignment Mavridou et al. (2025) used)

Each window is 500 ms wide (5 samples at 10 Hz), and per channel we extract
4 simple summary statistics: mean, median, MAV (mean absolute value), std.
9 channels x 4 features = 36 features per window (broadband 20-450 Hz EMG).

THE THREE FEATURE SETS
----------------------
- broadband           : 36 features (the 4 stats over the broadband envelope only)
- multiband           : 180 features (4 stats x 9 channels x 5 sub-bands)
- multiband + scene   : 181 features (above + scene-label one-hot)

THE THREE MODELS
----------------
- SVM-RBF  (Mavridou's recommended choice; class-balanced, probability=True)
- SVM-Lin  (linear-kernel comparator)
- RF       (the workhorse from the earlier experiments)

TARGET
------
Binary saliency classification (high vs low). Subject-level 80/10/10 split,
SEED=42, identical to all other SQ1 experiments so the numbers are
directly comparable.

OUTPUT
------
model_results/metrics_window_variations.json — Acc, F1, AUC, p vs majority,
                                                 train time per cell.
plots/window_variations_comparison.png       — bar chart used in §5.3.3.
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import binomtest

warnings.filterwarnings('ignore')

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
PLOTS = BASE / 'plots'
PLOTS.mkdir(exist_ok=True)
COMBINED_CACHE = PROC / 'multimodal_windows.npz'

SEED = 42
T = 70                     # full window length at 10 Hz (-2s..+5s)
PRE = 20                   # event onset index (T_0)
N_BANDS = 5                # 5th band in v3 cache is broadband 20-450 Hz
N_CH = 9
WIN_SAMPLES = 5            # 500 ms at 10 Hz = 5 samples
B_MEDIAN_MS = 1379         # empirical Delay B median
B_MED_SAMPLES = round(B_MEDIAN_MS / 100)   # 14 samples ≈ 1.4 s
np.random.seed(SEED)

# Variation start indices (in timestep units, with 1 timestep = 100 ms)
VARIATIONS = {
    'V1_event_locked':   PRE,                                # T0
    'V2_mid_appraisal':  PRE + B_MED_SAMPLES // 2,           # T0 + B/2 ≈ T0+700ms
    'V3_late_appraisal': PRE + B_MED_SAMPLES,                # T0 + B ≈ T0+1400ms
    'Baseline_200ms':    PRE + 2,                            # T0 + 200ms
}

# Sanity: each window must fit within [0, T)
for name, start in VARIATIONS.items():
    end = start + WIN_SAMPLES
    assert 0 <= start < T and end <= T, f'{name} window [{start},{end}) out of bounds [0,{T})'

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})


# ============================================================
# LOAD
# ============================================================
print(f'Loading multimodal cache from {COMBINED_CACHE.name}...', flush=True)
z = np.load(COMBINED_CACHE, allow_pickle=True)
X_mb = z['X_emg']    # (N, 70, 45) — multi-band; reshape to (N, 70, 9, 5)
pids = z['pids']
scenes = z['scenes']
salls = z['salls']
print(f'  X_mb {X_mb.shape}  pids {len(pids)} unique={len(np.unique(pids))}', flush=True)

# Target: saliency (binary)
y_sal = (salls == 'high').astype(int)
print(f'  saliency balance: high={y_sal.sum()}  low={(1-y_sal).sum()}', flush=True)

# Reshape to (N, T, channels, bands) and pick band 4 (last = broadband 20-450 Hz)
X_4d = X_mb.reshape(-1, T, N_CH, N_BANDS)
X_broadband = X_4d[..., -1]   # (N, 70, 9)
print(f'  broadband EMG envelope: {X_broadband.shape}', flush=True)


# ============================================================
# SUBJECT SPLIT (same as v4)
# ============================================================
upids = np.array(sorted(np.unique(pids)))
rng = np.random.default_rng(SEED); rng.shuffle(upids)
n = len(upids); n_tr = int(n * 0.8); n_va = int(n * 0.1)
tr_p = set(upids[:n_tr]); va_p = set(upids[n_tr:n_tr + n_va]); te_p = set(upids[n_tr + n_va:])
tr = np.isin(pids, list(tr_p))
va = np.isin(pids, list(va_p))
te = np.isin(pids, list(te_p))
print(f'  split: train={tr.sum()}/{len(tr_p)}  val={va.sum()}/{len(va_p)}  test={te.sum()}/{len(te_p)}', flush=True)


# ============================================================
# FEATURE EXTRACTION PER VARIATION
# ============================================================
def per_window_features(seg):
    """seg: (N, W, F). Returns (N, 4*F) — mean, median, MAV, std per channel/band."""
    mean_v = seg.mean(axis=1)
    med_v = np.median(seg, axis=1)
    mav_v = np.abs(seg).mean(axis=1)
    std_v = seg.std(axis=1)
    return np.concatenate([mean_v, med_v, mav_v, std_v], axis=1)


def extract_for_variation(start_idx, feature_set='broadband'):
    """feature_set: 'broadband' (9 ch × 4 stats = 36) or 'multiband' (45 × 4 = 180)."""
    if feature_set == 'broadband':
        seg = X_broadband[:, start_idx:start_idx + WIN_SAMPLES, :]   # (N, 5, 9)
    elif feature_set == 'multiband':
        seg = X_mb[:, start_idx:start_idx + WIN_SAMPLES, :]          # (N, 5, 45)
    else:
        raise ValueError(feature_set)
    return per_window_features(seg).astype(np.float32)


# Scene context as auxiliary feature
scene_bin = (scenes == 'positive').astype(np.float32).reshape(-1, 1)


# ============================================================
# RUN GRID
# ============================================================
def fit_eval(model_name, model, X, y, tr_mask, te_mask, scale=False):
    if scale:
        sc = StandardScaler().fit(X[tr_mask])
        X = sc.transform(X)
    model.fit(X[tr_mask], y[tr_mask])
    pred = model.predict(X[te_mask])
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X[te_mask])[:, 1]
    elif hasattr(model, 'decision_function'):
        prob = model.decision_function(X[te_mask])
    else:
        prob = pred
    y_te = y[te_mask]
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    try:
        auc = roc_auc_score(y_te, prob)
    except Exception:
        auc = float('nan')
    maj = max(y_te.mean(), 1 - y_te.mean())
    bt = binomtest(int((pred == y_te).sum()), len(y_te), p=maj)
    return {'model': model_name, 'acc': float(acc), 'f1': float(f1), 'auc': float(auc),
            'maj_baseline': float(maj), 'p_vs_maj': float(bt.pvalue),
            'n_test': int(te_mask.sum())}


FEATURE_SETS = ['broadband', 'multiband', 'multiband+scene']

print('\n' + '=' * 100, flush=True)
print('GRID — saliency classification at 4 delay variations × 3 feature sets', flush=True)
print('=' * 100, flush=True)
print(f'{"Variation":<22s}{"Features":<18s}{"Model":<10s}{"Acc":>8s}{"F1":>8s}{"AUC":>8s}{"vs maj":>10s}{"p":>10s}', flush=True)
print('-' * 96, flush=True)

results = {}
for vname, start_idx in VARIATIONS.items():
    results[vname] = {
        'window_start_ms': int((start_idx - PRE) * 100),
        'window_end_ms':   int((start_idx + WIN_SAMPLES - PRE) * 100),
        'feature_sets': {},
    }
    for fset in FEATURE_SETS:
        if fset == 'broadband':
            F = extract_for_variation(start_idx, 'broadband')
        elif fset == 'multiband':
            F = extract_for_variation(start_idx, 'multiband')
        elif fset == 'multiband+scene':
            F = np.concatenate([extract_for_variation(start_idx, 'multiband'), scene_bin], axis=1)
        results[vname]['feature_sets'][fset] = {'n_features': F.shape[1], 'models': {}}
        for model_name, model, scale in [
            ('SVM-RBF',   SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                              probability=True, random_state=SEED), True),
            ('SVM-Lin',   SVC(kernel='linear', C=1.0, class_weight='balanced',
                              probability=True, random_state=SEED), True),
            ('RF',        RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=10,
                                                  max_features='sqrt', n_jobs=-1, class_weight='balanced',
                                                  random_state=SEED), False),
        ]:
            t0 = time.time()
            r = fit_eval(model_name, model, F, y_sal, tr, te, scale=scale)
            r['train_seconds'] = round(time.time() - t0, 2)
            results[vname]['feature_sets'][fset]['models'][model_name] = r
            sig = '***' if r['p_vs_maj'] < 0.001 else ('**' if r['p_vs_maj'] < 0.01 else ('*' if r['p_vs_maj'] < 0.05 else 'ns'))
            print(f'{vname:<22s}{fset:<18s}{model_name:<10s}{r["acc"]*100:>7.1f}%{r["f1"]:>8.3f}{r["auc"]:>8.3f}'
                  f'{(r["acc"] - r["maj_baseline"])*100:>+9.1f}%{r["p_vs_maj"]:>10.3g} {sig}', flush=True)
    print('-' * 96, flush=True)


# ============================================================
# BEST PER VARIATION + SUMMARY
# ============================================================
print('\nBest model per (variation, feature-set) by AUC:', flush=True)
for vname, vres in results.items():
    for fset, fres in vres['feature_sets'].items():
        best = max(fres['models'].values(), key=lambda r: r['auc'] if not np.isnan(r['auc']) else -1)
        print(f'  {vname:<22s}{fset:<18s}  best={best["model"]:<8s} AUC={best["auc"]:.3f}  Acc={best["acc"]*100:.1f}%', flush=True)

# Headline: best overall combination
print('\nOverall ranking (best AUC across all combos):', flush=True)
all_combos = []
for vname, vres in results.items():
    for fset, fres in vres['feature_sets'].items():
        for mname, r in fres['models'].items():
            all_combos.append((r['auc'], r['acc'], vname, fset, mname, r['p_vs_maj']))
for auc, acc, vname, fset, mname, p in sorted(all_combos, reverse=True)[:5]:
    print(f'  AUC={auc:.3f}  Acc={acc*100:.1f}%  {vname:<22s}{fset:<18s}{mname:<10s} p={p:.3g}', flush=True)

with open(OUT / 'metrics_window_variations.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved -> {OUT}/metrics_window_variations.json', flush=True)


# ============================================================
# PLOT
# ============================================================
print('\nGenerating comparison plot...', flush=True)
variations = list(VARIATIONS.keys())
fsets = FEATURE_SETS
# Use SVM-RBF as the canonical model (Mavridou's recommended choice)
canonical_model = 'SVM-RBF'

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

auc_matrix = np.array([[results[v]['feature_sets'][f]['models'][canonical_model]['auc']
                         for f in fsets] for v in variations])
acc_matrix = np.array([[results[v]['feature_sets'][f]['models'][canonical_model]['acc']
                         for f in fsets] for v in variations])

x = np.arange(len(variations))
width = 0.27
colors = ['#aaaaaa', '#5e8eb8', '#0a4d8c']
for i, fset in enumerate(fsets):
    axes[0].bar(x + (i - 1) * width, auc_matrix[:, i], width, label=fset,
                color=colors[i], edgecolor='black', linewidth=0.7)
axes[0].axhline(0.5, color='red', linestyle='--', linewidth=1, label='chance')
axes[0].set_xticks(x); axes[0].set_xticklabels([v.replace('_', '\n') for v in variations],
                                                 fontsize=9)
axes[0].set_ylabel('AUC'); axes[0].set_title(f'Saliency AUC — {canonical_model} across windows')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].set_ylim([0.40, max(0.85, auc_matrix.max() + 0.05)])
for i, fset in enumerate(fsets):
    for j, vname in enumerate(variations):
        axes[0].text(j + (i - 1) * width, auc_matrix[j, i] + 0.005,
                     f'{auc_matrix[j, i]:.2f}', ha='center', fontsize=8)

for i, fset in enumerate(fsets):
    axes[1].bar(x + (i - 1) * width, acc_matrix[:, i] * 100, width, label=fset,
                color=colors[i], edgecolor='black', linewidth=0.7)
maj_line = max(y_sal[te].mean(), 1 - y_sal[te].mean()) * 100
axes[1].axhline(maj_line, color='red', linestyle='--', linewidth=1,
                label=f'majority ({maj_line:.1f}%)')
axes[1].set_xticks(x); axes[1].set_xticklabels([v.replace('_', '\n') for v in variations],
                                                 fontsize=9)
axes[1].set_ylabel('Accuracy (%)'); axes[1].set_title(f'Saliency accuracy — {canonical_model} across windows')
axes[1].legend(fontsize=9, loc='lower right')

plt.tight_layout(); plt.savefig(PLOTS / 'window_variations_comparison.png'); plt.close()
print(f'  plots/window_variations_comparison.png', flush=True)

print('\nDone.', flush=True)
