"""
train_spectral.py — Spectral-feature ablation; source of the saliency AUC = 0.729 result.

WHY THIS SCRIPT
---------------
The continuous-valence experiments in `train_lstm_v4_full.py` were
underwhelming on the test set (CCC near zero across all variants). The
question this script asks is: *is the EMG signal carrying any reliable
information at all, once we score the right target?* The answer is yes for
saliency (high vs low intensity), no for signed valence -- the result that
motivates the thesis's reframing of EMG as an arousal index rather than a
valence index.

Three feature sets are compared at the **window-level** using the 459-dim
spectral feature bank built by `extract_spectral_features.py`:

  (1) EMG spectral only      — does facial-EMG content alone predict the
                                target?
  (2) Scene label only       — does just knowing whether the scene was
                                positive vs negative do the work? (the
                                confound to control for)
  (3) Scene + EMG spectral   — does EMG add anything *over* scene context?

Targets and metrics
-------------------
- Continuous valence regression : test R^2, Pearson r, per-subject r.
- Binary saliency classification: ROC-AUC (the 0.729 headline).

Random Forest is used throughout (n_estimators=400, depth-controlled) because
it handles the heterogeneous feature scales of the spectral bank without a
StandardScaler step and gives stable feature-importance rankings.

OUTPUT
------
model_results/metrics_spectral.json — every (feature-set, target) cell of the
ablation grid with all metrics + per-subject summaries.
"""
import json, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score)
from scipy.stats import pearsonr

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
OUT.mkdir(exist_ok=True)
SEED = 42

FEAT_NAMES_PER_CHANNEL = ['rms','mav','var','iemg','wl','zc','ssc','skew','kurt',
                         'mpf','mdf','pkf','bp20_80','bp80_150','bp150_250','bp250_450','sp_ent']
N_FEATS_PER_CH = 17
N_CH = 9
N_SEGS = 3  # pre, post, full
N_SPEC_FEATS = N_SEGS * N_CH * N_FEATS_PER_CH  # 459


def ccc(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mt, mp = yt.mean(), yp.mean()
    cov = ((yt-mt)*(yp-mp)).mean()
    return (2*cov) / (yt.var()+yp.var()+(mt-mp)**2 + 1e-12)


def reg_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'r': float(pearsonr(y_true, y_pred)[0]) if y_true.std()>1e-8 and y_pred.std()>1e-8 else 0.0,
        'CCC': float(ccc(y_true, y_pred)),
    }


def per_subject_r(y_true, y_pred, subs):
    rs = []
    for s in np.unique(subs):
        m = subs == s
        if m.sum() < 3: continue
        yt, yp = y_true[m], y_pred[m]
        if yt.std()>1e-6 and yp.std()>1e-6:
            rs.append(pearsonr(yt, yp)[0])
    return {'per_sub_r_mean': float(np.mean(rs)) if rs else 0.0,
            'per_sub_r_med': float(np.median(rs)) if rs else 0.0,
            'n_sub': len(rs)}


def main():
    print('Loading event_windows_spectral.pkl...', flush=True)
    with open(PROC/'event_windows_spectral.pkl','rb') as f:
        windows = pickle.load(f)

    # Filter to windows that have spec_features
    good = [w for w in windows if w.get('spec_features') is not None]
    print(f'  {len(good)}/{len(windows)} windows with spectral features', flush=True)

    # Build arrays
    X_spec = np.stack([w['spec_features'] for w in good]).astype(np.float32)
    y_post = np.array([w['valence_target'][20:].mean() for w in good], dtype=np.float32)
    y_delta = np.array([w['valence_target'][20:].mean() - w['valence_target'][:20].mean() for w in good], dtype=np.float32)
    y_sal = np.array([1 if w['saliency']=='high' else 0 for w in good], dtype=int)
    scene_bin = np.array([1 if w['scene']=='positive' else 0 for w in good], dtype=np.float32).reshape(-1,1)
    pids = np.array([w['participant_id'] for w in good])
    print(f'  X_spec={X_spec.shape}  y_post range [{y_post.min():.2f},{y_post.max():.2f}]', flush=True)
    print(f'  saliency high: {y_sal.mean()*100:.1f}%  scenes positive: {scene_bin.mean()*100:.1f}%', flush=True)

    # Clean NaN/inf
    bad_mask = ~np.isfinite(X_spec).all(axis=1)
    if bad_mask.any():
        print(f'  dropping {bad_mask.sum()} rows with NaN/Inf in spec features', flush=True)
        good_rows = ~bad_mask
        X_spec = X_spec[good_rows]; y_post = y_post[good_rows]; y_delta = y_delta[good_rows]
        y_sal = y_sal[good_rows]; scene_bin = scene_bin[good_rows]; pids = pids[good_rows]

    # Subject split
    upids = np.array(sorted(np.unique(pids)))
    rng = np.random.default_rng(SEED); rng.shuffle(upids)
    n = len(upids); n_tr = int(n*0.8); n_va = int(n*0.1)
    tr_set = set(upids[:n_tr]); va_set = set(upids[n_tr:n_tr+n_va]); te_set = set(upids[n_tr+n_va:])
    tr = np.isin(pids, list(tr_set)); va = np.isin(pids, list(va_set)); te = np.isin(pids, list(te_set))
    print(f'  split: train={len(tr_set)}/{tr.sum()}  val={len(va_set)}/{va.sum()}  test={len(te_set)}/{te.sum()}', flush=True)

    def run(X, label):
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        # regression
        rf = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                    max_features='sqrt', n_jobs=-1, random_state=SEED).fit(Xs[tr], y_post[tr])
        pred_te = rf.predict(Xs[te])
        m = reg_metrics(y_post[te], pred_te)
        ps = per_subject_r(y_post[te], pred_te, pids[te])
        # classification
        rfc = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                      max_features='sqrt', n_jobs=-1, random_state=SEED,
                                      class_weight='balanced').fit(Xs[tr], y_sal[tr])
        prob_te = rfc.predict_proba(Xs[te])[:,1]
        try: auc = roc_auc_score(y_sal[te], prob_te)
        except: auc = float('nan')
        acc = accuracy_score(y_sal[te], rfc.predict(Xs[te]))
        print(f'  {label:45s}  valence R²={m["R2"]:+.3f}  r={m["r"]:+.3f}  per-sub r={ps["per_sub_r_mean"]:+.3f}  |  saliency Acc={acc:.3f} AUC={auc:.3f}', flush=True)
        return {'label': label, 'reg': m, 'per_sub': ps, 'saliency_acc': float(acc), 'saliency_auc': float(auc),
                'rf_reg': rf, 'rf_cls': rfc, 'scaler': sc, 'feat_dim': X.shape[1]}

    print('\n' + '='*90, flush=True)
    print('FEATURE SET COMPARISON (test set)', flush=True)
    print('='*90, flush=True)

    results = {}
    t0 = time.time()
    results['scene_only'] = run(scene_bin, 'SCENE only (1 feature)')
    results['spec_only']  = run(X_spec, f'EMG spectral only ({X_spec.shape[1]} features)')
    results['spec_plus_scene'] = run(np.concatenate([X_spec, scene_bin], axis=1),
                                     f'EMG spectral + scene ({X_spec.shape[1]+1} features)')

    # Also post-only spectral (exclude pre-event)
    X_post_only = X_spec[:, N_CH*N_FEATS_PER_CH : 2*N_CH*N_FEATS_PER_CH]  # second segment = post
    results['spec_post_only'] = run(X_post_only, f'EMG spectral post-event only ({X_post_only.shape[1]} features)')
    results['spec_post_plus_scene'] = run(np.concatenate([X_post_only, scene_bin], axis=1),
                                           f'EMG post + scene ({X_post_only.shape[1]+1} features)')

    print(f'\nTOTAL TRAIN: {(time.time()-t0)/60:.1f} min', flush=True)

    # Feature importance for best EMG-only model
    print('\n' + '='*90, flush=True)
    print('TOP FEATURE IMPORTANCES (EMG spectral + scene RF regressor)', flush=True)
    print('='*90, flush=True)
    full_feat_names = []
    for seg in ['pre', 'post', 'full']:
        for ch in range(N_CH):
            for fn in FEAT_NAMES_PER_CHANNEL:
                full_feat_names.append(f'ch{ch}_{fn}_{seg}')
    full_feat_names.append('scene_positive')
    rf = results['spec_plus_scene']['rf_reg']
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1][:20]
    for rank, i in enumerate(order, 1):
        print(f'  {rank:2d}. {full_feat_names[i]:30s}  {imp[i]:.4f}', flush=True)

    # Save metrics
    serializable = {k: {kk: vv for kk, vv in v.items() if kk not in ['rf_reg','rf_cls','scaler']}
                    for k, v in results.items()}
    with open(OUT/'metrics_spectral.json','w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nmetrics -> {OUT}/metrics_spectral.json', flush=True)


if __name__ == '__main__':
    main()
