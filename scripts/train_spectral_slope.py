"""
train_spectral_slope.py — Shape-feature exploration cited in §6.4 (Future Work).

WHY THIS SCRIPT
---------------
The supervisor's key methodological point in the May 1 meeting was that
mean-based per-window features may be poor proxies for *valence*-specific
muscle activation: it is the *shape* of the envelope (slope of activation
onset, area under the curve, MAV) that distinguishes a positive smile from a
negative grimace, not the average amplitude. This script is the concrete
starting point of that future-work direction -- referenced by name in the
thesis at §6.4.

What it does
------------
For each event window, in addition to the existing 459-dim spectral feature
bank produced by `extract_spectral_features.py`, this script computes a
**linear slope** of the 10 Hz EMG envelope per channel for each of the
pre-event (-2..0 s), post-event (0..+5 s), and full (-2..+5 s) segments:

    9 channels x 3 segments = 27 slope features

Appended to the existing 459 -> 486-dim feature vector. The same RF + scene
+ saliency ablation grid as `train_spectral.py` is then re-run, so the
contribution of the slope features is read off as the marginal
delta-AUC / delta-R^2 vs the slope-free baseline.

This is reported in the thesis as exploratory / future-work, not as a primary
result -- the gain over the spectral-only baseline was modest in this small
test set, but the shape-feature direction is the recommended next step.
"""
import json, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score)
from scipy.stats import pearsonr

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
SEED = 42

PRE, POST, TOT = 20, 50, 70
N_CH = 9


def slope_per_channel(seg):
    """seg: (T, 9). Returns (9,) linear slope per channel via least-squares."""
    T = seg.shape[0]
    t = np.arange(T, dtype=np.float32)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum()
    if denom < 1e-9:
        return np.zeros(seg.shape[1], dtype=np.float32)
    num = ((t - t_mean)[:, None] * (seg - seg.mean(axis=0))).sum(axis=0)
    return (num / denom).astype(np.float32)


def compute_slopes(emg_env):
    """emg_env: (T, 9) envelope. Returns (27,) slopes for pre/post/full."""
    emg = emg_env
    if emg.shape[0] < TOT:
        pad = np.zeros((TOT - emg.shape[0], 9), dtype=emg.dtype)
        emg = np.vstack([emg, pad])
    else:
        emg = emg[:TOT]
    pre = emg[:PRE]
    post = emg[PRE:]
    full = emg
    return np.concatenate([
        slope_per_channel(pre),
        slope_per_channel(post),
        slope_per_channel(full),
    ])


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
    good = [w for w in windows if w.get('spec_features') is not None]
    print(f'  {len(good)}/{len(windows)} windows with spec features', flush=True)

    # Inspect one window
    w0 = good[0]
    print(f'  emg_features shape: {w0["emg_features"].shape}  (envelope, 10 Hz)', flush=True)
    print(f'  spec_features len : {len(w0["spec_features"])}', flush=True)

    # Build arrays
    X_spec = np.stack([w['spec_features'] for w in good]).astype(np.float32)
    X_slope = np.stack([compute_slopes(w['emg_features']) for w in good]).astype(np.float32)
    y_post = np.array([w['valence_target'][20:].mean() for w in good], dtype=np.float32)
    y_sal = np.array([1 if w['saliency']=='high' else 0 for w in good], dtype=int)
    scene_bin = np.array([1 if w['scene']=='positive' else 0 for w in good], dtype=np.float32).reshape(-1,1)
    pids = np.array([w['participant_id'] for w in good])

    # Clean NaN/inf on spec features only (slope is bounded)
    bad_mask = ~np.isfinite(X_spec).all(axis=1) | ~np.isfinite(X_slope).all(axis=1)
    if bad_mask.any():
        print(f'  dropping {bad_mask.sum()} rows with NaN/Inf', flush=True)
        good_rows = ~bad_mask
        X_spec = X_spec[good_rows]; X_slope = X_slope[good_rows]
        y_post = y_post[good_rows]; y_sal = y_sal[good_rows]
        scene_bin = scene_bin[good_rows]; pids = pids[good_rows]

    print(f'  X_spec {X_spec.shape}, X_slope {X_slope.shape}', flush=True)

    # Subject split (match train_spectral.py)
    upids = np.array(sorted(np.unique(pids)))
    rng = np.random.default_rng(SEED); rng.shuffle(upids)
    n = len(upids); n_tr = int(n*0.8); n_va = int(n*0.1)
    tr_set = set(upids[:n_tr]); te_set = set(upids[n_tr+n_va:])
    tr = np.isin(pids, list(tr_set)); te = np.isin(pids, list(te_set))
    print(f'  split: train={tr.sum()}  test={te.sum()}', flush=True)

    def run(X, label):
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        rf = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                    max_features='sqrt', n_jobs=-1, random_state=SEED).fit(Xs[tr], y_post[tr])
        pred_te = rf.predict(Xs[te])
        m = reg_metrics(y_post[te], pred_te)
        ps = per_subject_r(y_post[te], pred_te, pids[te])
        rfc = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                      max_features='sqrt', n_jobs=-1, random_state=SEED,
                                      class_weight='balanced').fit(Xs[tr], y_sal[tr])
        prob_te = rfc.predict_proba(Xs[te])[:,1]
        try: auc = roc_auc_score(y_sal[te], prob_te)
        except: auc = float('nan')
        acc = accuracy_score(y_sal[te], rfc.predict(Xs[te]))
        print(f'  {label:50s}  valence R²={m["R2"]:+.3f}  r={m["r"]:+.3f}  per-sub r={ps["per_sub_r_mean"]:+.3f}  |  saliency Acc={acc:.3f} AUC={auc:.3f}', flush=True)
        return {'label': label, 'reg': m, 'per_sub': ps, 'saliency_acc': float(acc), 'saliency_auc': float(auc),
                'rf_reg': rf, 'rf_cls': rfc, 'feat_dim': X.shape[1]}

    print('\n' + '='*100, flush=True)
    print('ABLATION WITH SLOPE FEATURES', flush=True)
    print('='*100, flush=True)

    results = {}
    t0 = time.time()

    # Baselines (match previous results)
    results['scene_only']            = run(scene_bin, 'SCENE only (1 feat)')
    results['spec_only']             = run(X_spec, f'SPEC only ({X_spec.shape[1]} feats)')
    results['spec_plus_scene']       = run(np.concatenate([X_spec, scene_bin], axis=1),
                                           f'SPEC + scene ({X_spec.shape[1]+1} feats)')
    # With slopes
    results['slope_only']            = run(X_slope, f'SLOPE only ({X_slope.shape[1]} feats)')
    results['spec_plus_slope']       = run(np.concatenate([X_spec, X_slope], axis=1),
                                           f'SPEC + slope ({X_spec.shape[1]+X_slope.shape[1]} feats)')
    results['spec_plus_slope_scene'] = run(np.concatenate([X_spec, X_slope, scene_bin], axis=1),
                                           f'SPEC + slope + scene ({X_spec.shape[1]+X_slope.shape[1]+1} feats)')

    print(f'\nTOTAL: {(time.time()-t0)/60:.1f} min', flush=True)

    # Feature importance for slopes in the best combined model
    print('\n' + '='*100, flush=True)
    print('SLOPE FEATURE RANK in SPEC+slope+scene (valence regressor)', flush=True)
    print('='*100, flush=True)
    slope_names = []
    for seg in ['pre','post','full']:
        for ch in range(N_CH):
            slope_names.append(f'ch{ch}_slope_{seg}')
    all_names = []
    for seg in ['pre','post','full']:
        for ch in range(N_CH):
            for fn in ['rms','mav','var','iemg','wl','zc','ssc','skew','kurt',
                       'mpf','mdf','pkf','bp20_80','bp80_150','bp150_250','bp250_450','sp_ent']:
                all_names.append(f'ch{ch}_{fn}_{seg}')
    all_names += slope_names + ['scene_positive']

    rf = results['spec_plus_slope_scene']['rf_reg']
    imp = rf.feature_importances_
    # rank of each slope feature
    order = np.argsort(imp)[::-1]
    ranks = {name: rank for rank, i in enumerate(order, 1) for name in [all_names[i]]}
    print('  Slope feature ranks (lower = more important):')
    for sn in slope_names:
        print(f'    {sn:25s}  rank {ranks[sn]:4d}/{len(all_names)}  imp={imp[all_names.index(sn)]:.4f}', flush=True)

    # Save
    serializable = {k: {kk: vv for kk, vv in v.items() if kk not in ['rf_reg','rf_cls']}
                    for k, v in results.items()}
    with open(OUT/'metrics_spectral_slope.json','w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nmetrics -> {OUT}/metrics_spectral_slope.json', flush=True)

    # Side-by-side summary
    print('\n' + '='*100, flush=True)
    print('SIDE-BY-SIDE: with vs without slope', flush=True)
    print('='*100, flush=True)
    print(f'{"model":40s}  {"R²":>8s}  {"per-sub r":>10s}  {"AUC":>6s}')
    for k in ['scene_only','spec_only','spec_plus_scene','slope_only','spec_plus_slope','spec_plus_slope_scene']:
        r = results[k]
        print(f'{r["label"]:40s}  {r["reg"]["R2"]:+.3f}   {r["per_sub"]["per_sub_r_mean"]:+.3f}      {r["saliency_auc"]:.3f}')


if __name__ == '__main__':
    main()
