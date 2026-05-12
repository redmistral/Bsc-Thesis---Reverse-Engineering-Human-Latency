"""
train_lstm_multiband.py — Multi-band BiLSTM (used for Figure 5 per-subject scatter).

This is the v3 variant that immediately preceded `train_lstm_v4_full.py`. It
adds multi-band EMG features (5 sub-bands x 9 channels = 45 features per
timestep) but does *not* yet add the physiology, scene context, or
baseline-corrected target that v4 introduces.

WHY KEEP IT
-----------
The thesis cites this script's per-subject CCC scatter (`plots/sq1_v3_per_
subject_scatter.png`, Figure 5) as the cleanest visualisation of "RF beats
LSTM at the per-subject level even on the richer multi-band input". v4 went
on to layer further improvements; keeping v3 in the submission lets the
supervisor reproduce that exact figure.

Hypothesis under test
---------------------
The earlier v2 LSTM underperformed because its input (single 10 Hz envelope
per channel = 9 numbers/timestep) was too impoverished for sequence learning.
v3 recomputes 5 sub-band envelopes per channel from the raw 1000 Hz EMG:

    band 1:  20- 60 Hz   (low MUAP)
    band 2:  60-120 Hz   (medium)
    band 3: 120-250 Hz   (high MUAP)
    band 4: 250-450 Hz   (very high)
    band 5:  20-450 Hz   (full broadband, matches v2 input)

Features per timestep: 9 channels x 5 bands = 45  (vs 9 in v2).
Otherwise identical setup: 70 timesteps (-2 s..+5 s @ 10 Hz), seq2seq LSTM,
subject-level 80/10/10 split, post-event evaluation.

Faithful to draft RQ: still EMG -> continuous valence; scene context only as
an ablation. If multi-band still cannot outperform RF, we report SQ1 as a
feasibility-limit finding.
"""
import json, pickle, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, sosfilt
from scipy.stats import pearsonr, wilcoxon

warnings.filterwarnings('ignore')

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
DATA = BASE / 'All Data'
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
PLOTS = BASE / 'plots'
PLOTS.mkdir(exist_ok=True)
CACHE = PROC / 'multiband_windows.npz'

SEED = 42
T = 70                   # timesteps in window  (2s pre + 5s post @ 10 Hz)
PRE = 20                 # pre-event samples
LAG = 2                  # 200 ms RF lag
FS_RAW = 1000
RMS_WIN = 100            # 100 ms RMS @ 1000 Hz
DOWNSAMPLE = 100         # 1000 Hz -> 10 Hz
PRE_MS = 2000
POST_MS = 5000
BANDS = [(20, 60), (60, 120), (120, 250), (250, 450), (20, 450)]
N_BANDS = len(BANDS)
N_CH = 9

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})

RAW_COLUMNS = (
    ['frame'] + [f'emg_ch{i}' for i in range(1, 10)] + ['heart_rate1', 'heart_rate_2']
    + [f'imu_ch{i}' for i in range(1, 10)]
    + ['accel_x', 'accel_y', 'accel_z', 'magnetometer', 'gyro', 'system_related', '_trailing']
)
EMG_COLS = [f'emg_ch{i}' for i in range(1, 10)]


def bandpass(x, low, high, fs=FS_RAW, order=4):
    nyq = fs / 2
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfilt(sos, x)


def moving_rms(x, w=RMS_WIN):
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode='same'))


def find_raw_path(pid_dir, scene):
    pat = 'POSITIVE' if scene == 'positive' else 'negative'
    for p in pid_dir.glob(f'*{pat}*_03_*_raw.txt'):
        json_p = p.with_name(p.name.replace('_raw.txt', '.json'))
        if json_p.exists():
            return p, json_p
    return None, None


def load_raw_envelopes(pid, scene):
    """Return (env_5band (T_raw, 9, 5), raw_frames (T_raw,), ref_frame) or None."""
    pid_dir = DATA / str(pid)
    raw_path, json_path = find_raw_path(pid_dir, scene)
    if raw_path is None:
        return None
    raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    raw = raw.drop(columns=['_trailing'], errors='ignore')
    raw['frame'] = raw['frame'].astype(int)

    with open(json_path) as f:
        meta = json.load(f)
    js_frames = [e['frameRef'] for e in meta['data']]
    if not js_frames:
        return None
    first_frame = raw['frame'].min()
    ref_frame = max(first_frame, min(js_frames))

    Tn = len(raw)
    env = np.zeros((Tn, N_CH, N_BANDS), dtype=np.float32)
    for c, col in enumerate(EMG_COLS):
        x = raw[col].values.astype(np.float64)
        for b, (lo, hi) in enumerate(BANDS):
            bp = bandpass(x, lo, hi)
            env[:, c, b] = moving_rms(np.abs(bp)).astype(np.float32)
    return env, raw['frame'].values.astype(np.int64), int(ref_frame)


def extract_window(env_full, raw_frames, ref_frame, event_time_s):
    """env_full: (T_raw, 9, 5) at 1000 Hz.
    Slice [-2s, +5s] around event, downsample to (70, 9, 5) by 100-sample mean.
    """
    event_frame = ref_frame + int(event_time_s * FS_RAW)
    start = event_frame - PRE_MS
    end = event_frame + POST_MS
    m = (raw_frames >= start) & (raw_frames < end)
    seg = env_full[m]
    if seg.shape[0] < (PRE_MS + POST_MS):
        return None
    seg = seg[:PRE_MS + POST_MS]
    out = seg.reshape(T, DOWNSAMPLE, N_CH, N_BANDS).mean(axis=1).astype(np.float32)
    return out


# ====================================================================
# BUILD or LOAD MULTI-BAND WINDOWS
# ====================================================================
if CACHE.exists():
    print(f'Loading cached multi-band windows from {CACHE.name}...', flush=True)
    z = np.load(CACHE, allow_pickle=True)
    X_mb = z['X_mb']; Y_seq = z['Y']; pids = z['pids']
    scenes = z['scenes']; salls = z['salls']
    print(f'  X_mb {X_mb.shape}  Y {Y_seq.shape}  ({len(pids)} windows)', flush=True)
else:
    print('Building multi-band windows from raw 1000 Hz EMG...', flush=True)
    with open(PROC / 'event_windows_spectral.pkl', 'rb') as f:
        windows = pickle.load(f)

    by_ps = defaultdict(list)
    for i, w in enumerate(windows):
        by_ps[(w['participant_id'], w['scene'])].append((i, w))
    n_groups = len(by_ps)
    print(f'  {len(windows)} windows in {n_groups} (pid, scene) groups', flush=True)

    X_list, Y_list, pids_list, scenes_list, salls_list = [], [], [], [], []
    t0 = time.time(); n_done = 0; n_skip = 0; n_bad = 0
    for k, ((pid, scene), items) in enumerate(sorted(by_ps.items()), 1):
        loaded = load_raw_envelopes(pid, scene)
        if loaded is None:
            n_skip += len(items)
            continue
        env_full, raw_frames, ref_frame = loaded
        for idx, w in items:
            val = np.asarray(w.get('valence_target'), np.float32)
            if val.ndim != 1 or val.shape[0] < T or not np.isfinite(val).all():
                n_bad += 1; continue
            mb = extract_window(env_full, raw_frames, ref_frame, w['event_time_s'])
            if mb is None or not np.isfinite(mb).all():
                n_bad += 1; continue
            X_list.append(mb.reshape(T, N_CH * N_BANDS))
            Y_list.append(val[:T])
            pids_list.append(w['participant_id'])
            scenes_list.append(w['scene'])
            salls_list.append(w['saliency'])
            n_done += 1
        if k % 20 == 0 or k == n_groups:
            el = time.time() - t0
            eta = el / k * (n_groups - k)
            print(f'  [{k}/{n_groups}] done={n_done} skip={n_skip} bad={n_bad} '
                  f'elapsed={el:.0f}s ETA={eta:.0f}s', flush=True)

    X_mb = np.stack(X_list).astype(np.float32)
    Y_seq = np.stack(Y_list).astype(np.float32)
    pids = np.array(pids_list)
    scenes = np.array(scenes_list)
    salls = np.array(salls_list)
    np.savez(CACHE, X_mb=X_mb, Y=Y_seq, pids=pids, scenes=scenes, salls=salls)
    print(f'\nSaved {n_done} windows -> {CACHE}', flush=True)
    print(f'  X_mb {X_mb.shape}  Y {Y_seq.shape}', flush=True)

scene_bin = np.array([1 if s == 'positive' else 0 for s in scenes], dtype=np.float32)
scene_feat = np.broadcast_to(scene_bin[:, None, None], (len(scene_bin), T, 1)).astype(np.float32)

# ====================================================================
# SUBJECT SPLIT (same SEED as v2 for fair comparison)
# ====================================================================
upids = np.array(sorted(np.unique(pids)))
rng = np.random.default_rng(SEED); rng.shuffle(upids)
n = len(upids); n_tr = int(n * 0.8); n_va = int(n * 0.1)
tr_p = set(upids[:n_tr]); va_p = set(upids[n_tr:n_tr + n_va]); te_p = set(upids[n_tr + n_va:])
tr = np.isin(pids, list(tr_p)); va = np.isin(pids, list(va_p)); te = np.isin(pids, list(te_p))
print(f'\n  split: train={tr.sum()}/{len(tr_p)}  val={va.sum()}/{len(va_p)}  test={te.sum()}/{len(te_p)}', flush=True)

# Standardize per feature (fit on train)
F = X_mb.shape[-1]
sc = StandardScaler().fit(X_mb[tr].reshape(-1, F))
X_mb_s = sc.transform(X_mb.reshape(-1, F)).reshape(X_mb.shape).astype(np.float32)
X_mb_scene = np.concatenate([X_mb_s, scene_feat], axis=-1)

# ====================================================================
# METRICS
# ====================================================================
def ccc(yt, yp):
    yt, yp = np.asarray(yt, np.float64), np.asarray(yp, np.float64)
    mt, mp = yt.mean(), yp.mean()
    return float((2 * ((yt-mt)*(yp-mp)).mean()) / (yt.var()+yp.var()+(mt-mp)**2 + 1e-12))


def rmse(yt, yp):
    return float(np.sqrt(((np.asarray(yt) - np.asarray(yp)) ** 2).mean()))


def pearson(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    if yt.std() < 1e-8 or yp.std() < 1e-8: return 0.0
    return float(pearsonr(yt, yp)[0])


def make_rf_pairs(X, Y, mask):
    idx = np.where(mask)[0]
    Xf, yf, widx, ts = [], [], [], []
    for i in idx:
        for t in range(T - LAG):
            Xf.append(X[i, t]); yf.append(Y[i, t + LAG])
            widx.append(i); ts.append(t)
    return (np.stack(Xf).astype(np.float32),
            np.asarray(yf, dtype=np.float32),
            np.asarray(widx), np.asarray(ts))


# ====================================================================
# MODELS
# ====================================================================
class ValenceLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


def train_lstm(X, Y, tr_mask, va_mask, tag, epochs=120, patience=20):
    print(f'\n  [LSTM-{tag}]  input_size={X.shape[-1]}  hidden=128  layers=3', flush=True)
    model = ValenceLSTM(input_size=X.shape[-1], hidden=128, layers=3, dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    tr_idx = np.where(tr_mask)[0]; va_idx = np.where(va_mask)[0]
    best_val = 1e9; best_state = None; bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        tls = []
        for k in range(0, len(tr_idx), 128):
            b = tr_idx[k:k+128]
            xb = torch.from_numpy(X[b]).to(DEVICE); yb = torch.from_numpy(Y[b]).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(loss.item())
        sched.step()
        model.eval()
        vls = []
        with torch.no_grad():
            for k in range(0, len(va_idx), 256):
                b = va_idx[k:k+256]
                xb = torch.from_numpy(X[b]).to(DEVICE); yb = torch.from_numpy(Y[b]).to(DEVICE)
                vls.append(loss_fn(model(xb), yb).item())
        tl, vl = float(np.mean(tls)), float(np.mean(vls))
        history.append({'epoch': ep, 'train_loss': tl, 'val_loss': vl})
        improved = vl < best_val - 1e-5
        if improved:
            best_val = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or ep == epochs or (improved and ep <= 5):
            print(f'    ep {ep:3d}  tr={tl:.4f}  va={vl:.4f}  {"*" if improved else ""}', flush=True)
        if bad >= patience:
            print(f'    early stop at ep {ep}  (best val {best_val:.4f})', flush=True)
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT / f'lstm_sq1_v3_{tag}_history.csv', index=False)
    return model, history


def lstm_predict(model, X, mask):
    model.eval()
    pred = np.full((len(X), T), np.nan, dtype=np.float32)
    idx = np.where(mask)[0]
    with torch.no_grad():
        for k in range(0, len(idx), 256):
            b = idx[k:k+256]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            pred[b] = model(xb).cpu().numpy()
    return pred


def rf_predict_seq(rf, X, mask):
    Xrf, _, widx, ts = make_rf_pairs(X, Y_seq, mask)
    yp = rf.predict(Xrf).astype(np.float32)
    seq = np.full((len(X), T), np.nan, dtype=np.float32)
    for p, wi, tt in zip(yp, widx, ts):
        seq[wi, tt + LAG] = p
    return seq


def agg(pred_seq, mask, eval_post=True):
    ys, ps = [], []
    tr_ = range(PRE, T) if eval_post else range(T)
    for i in np.where(mask)[0]:
        for t in tr_:
            if np.isnan(pred_seq[i, t]): continue
            ys.append(Y_seq[i, t]); ps.append(pred_seq[i, t])
    ys, ps = np.array(ys), np.array(ps)
    return {'n': int(len(ys)), 'CCC': ccc(ys, ps), 'RMSE': rmse(ys, ps), 'r': pearson(ys, ps)}


def per_subj(pred_seq, mask, eval_post=True):
    tr_ = list(range(PRE, T)) if eval_post else list(range(T))
    store = {}
    for i in np.where(mask)[0]:
        s = pids[i]
        store.setdefault(s, {'y': [], 'p': []})
        for t in tr_:
            if not np.isnan(pred_seq[i, t]):
                store[s]['y'].append(Y_seq[i, t]); store[s]['p'].append(pred_seq[i, t])
    rows = []
    for s, d in store.items():
        if len(d['y']) >= 20:
            rows.append({'subject': s, 'CCC': ccc(d['y'], d['p']),
                         'r': pearson(d['y'], d['p']),
                         'RMSE': rmse(d['y'], d['p']),
                         'n': len(d['y'])})
    return pd.DataFrame(rows)


# ====================================================================
# RUN ALL VARIANTS
# ====================================================================
results = {}

print('\n' + '=' * 80, flush=True)
print('(A) RF static 200ms lag — 45-feature multi-band')
print('=' * 80, flush=True)
Xrf_tr, yrf_tr, _, _ = make_rf_pairs(X_mb_s, Y_seq, tr)
print(f'  train frames: {Xrf_tr.shape}', flush=True)
t0 = time.time()
rf_a = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=10,
                              max_features='sqrt', n_jobs=-1, random_state=SEED).fit(Xrf_tr, yrf_tr)
print(f'  trained in {time.time()-t0:.0f}s', flush=True)
rf_a_pred = rf_predict_seq(rf_a, X_mb_s, te)
results['rf_mb'] = {'agg': agg(rf_a_pred, te), 'pred': rf_a_pred}
print(f'  agg: {results["rf_mb"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(B) LSTM seq2seq — 45-feature multi-band  [DRAFT EXPERIMENTAL]')
print('=' * 80, flush=True)
lstm_b, hist_B = train_lstm(X_mb_s, Y_seq, tr, va, tag='mb_only', epochs=120, patience=20)
lstm_b_pred = lstm_predict(lstm_b, X_mb_s, te)
results['lstm_mb'] = {'agg': agg(lstm_b_pred, te), 'pred': lstm_b_pred}
print(f'  agg: {results["lstm_mb"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(C) RF static 200ms lag — multi-band + scene')
print('=' * 80, flush=True)
Xrf_tr2, yrf_tr2, _, _ = make_rf_pairs(X_mb_scene, Y_seq, tr)
t0 = time.time()
rf_c = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=10,
                              max_features='sqrt', n_jobs=-1, random_state=SEED).fit(Xrf_tr2, yrf_tr2)
print(f'  trained in {time.time()-t0:.0f}s', flush=True)
rf_c_pred = rf_predict_seq(rf_c, X_mb_scene, te)
results['rf_mb_scene'] = {'agg': agg(rf_c_pred, te), 'pred': rf_c_pred}
print(f'  agg: {results["rf_mb_scene"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(D) LSTM seq2seq — multi-band + scene')
print('=' * 80, flush=True)
lstm_d, hist_D = train_lstm(X_mb_scene, Y_seq, tr, va, tag='mb_scene', epochs=120, patience=20)
lstm_d_pred = lstm_predict(lstm_d, X_mb_scene, te)
results['lstm_mb_scene'] = {'agg': agg(lstm_d_pred, te), 'pred': lstm_d_pred}
print(f'  agg: {results["lstm_mb_scene"]["agg"]}', flush=True)

# ====================================================================
# SUMMARY
# ====================================================================
print('\n' + '=' * 80, flush=True)
print('SQ1 v3 SUMMARY (test subjects, post-event)')
print('=' * 80, flush=True)
print(f'{"Model":<36s}{"n":>8s}{"CCC":>8s}{"RMSE":>8s}{"Pearson r":>12s}')
print('-' * 72)
for name, key in [('RF  multi-band',                'rf_mb'),
                  ('LSTM multi-band',               'lstm_mb'),
                  ('RF  multi-band + scene',        'rf_mb_scene'),
                  ('LSTM multi-band + scene',       'lstm_mb_scene')]:
    a = results[key]['agg']
    print(f'{name:<36s}{a["n"]:>8d}{a["CCC"]:>8.3f}{a["RMSE"]:>8.3f}{a["r"]:>12.3f}', flush=True)

print(f'\nΔCCC(LSTM−RF, multi-band)        = {results["lstm_mb"]["agg"]["CCC"] - results["rf_mb"]["agg"]["CCC"]:+.3f}', flush=True)
print(f'ΔCCC(LSTM−RF, multi-band+scene)  = {results["lstm_mb_scene"]["agg"]["CCC"] - results["rf_mb_scene"]["agg"]["CCC"]:+.3f}', flush=True)
print(f'ΔCCC(scene helps RF)             = {results["rf_mb_scene"]["agg"]["CCC"] - results["rf_mb"]["agg"]["CCC"]:+.3f}', flush=True)
print(f'ΔCCC(scene helps LSTM)           = {results["lstm_mb_scene"]["agg"]["CCC"] - results["lstm_mb"]["agg"]["CCC"]:+.3f}', flush=True)

# Compare against v2 (envelope-only LSTM) if available
v2_path = OUT / 'metrics_sq1_v2.json'
if v2_path.exists():
    with open(v2_path) as f:
        v2 = json.load(f)
    print('\nCOMPARISON vs v2 (envelope-only):', flush=True)
    print(f'  v2 LSTM-EMG       CCC = {v2["lstm_emg"]["CCC"]:+.3f}    -> v3 LSTM-MB       CCC = {results["lstm_mb"]["agg"]["CCC"]:+.3f}    Δ={results["lstm_mb"]["agg"]["CCC"]-v2["lstm_emg"]["CCC"]:+.3f}', flush=True)
    print(f'  v2 LSTM-EMG+scene CCC = {v2["lstm_emg_scene"]["CCC"]:+.3f}    -> v3 LSTM-MB+scene CCC = {results["lstm_mb_scene"]["agg"]["CCC"]:+.3f}    Δ={results["lstm_mb_scene"]["agg"]["CCC"]-v2["lstm_emg_scene"]["CCC"]:+.3f}', flush=True)
    print(f'  v2 RF-EMG         CCC = {v2["rf_emg"]["CCC"]:+.3f}    -> v3 RF-MB         CCC = {results["rf_mb"]["agg"]["CCC"]:+.3f}    Δ={results["rf_mb"]["agg"]["CCC"]-v2["rf_emg"]["CCC"]:+.3f}', flush=True)
    print(f'  v2 RF-EMG+scene   CCC = {v2["rf_emg_scene"]["CCC"]:+.3f}    -> v3 RF-MB+scene   CCC = {results["rf_mb_scene"]["agg"]["CCC"]:+.3f}    Δ={results["rf_mb_scene"]["agg"]["CCC"]-v2["rf_emg_scene"]["CCC"]:+.3f}', flush=True)

# Per-subject test
ps_rf_best = per_subj(rf_c_pred, te)
ps_lstm_best = per_subj(lstm_d_pred, te)
ps_rf_best.to_csv(OUT / 'sq1_v3_per_subject_rf_scene.csv', index=False)
ps_lstm_best.to_csv(OUT / 'sq1_v3_per_subject_lstm_scene.csv', index=False)
print('\nPer-subject CCC (multi-band + scene):', flush=True)
print(f'  RF   mean={ps_rf_best["CCC"].mean():+.3f}  median={ps_rf_best["CCC"].median():+.3f}  n={len(ps_rf_best)}', flush=True)
print(f'  LSTM mean={ps_lstm_best["CCC"].mean():+.3f}  median={ps_lstm_best["CCC"].median():+.3f}  n={len(ps_lstm_best)}', flush=True)
common = set(ps_rf_best['subject']) & set(ps_lstm_best['subject'])
if len(common) >= 5:
    rv = ps_rf_best.set_index('subject').loc[list(common), 'CCC'].values
    lv = ps_lstm_best.set_index('subject').loc[list(common), 'CCC'].values
    stat, p = wilcoxon(lv, rv)
    wins = int(np.sum(lv > rv))
    print(f'  Wilcoxon paired: stat={stat:.1f}  p={p:.4g}  LSTM better in {wins}/{len(common)} subjects', flush=True)

# By saliency
print('\nBY SALIENCY (multi-band + scene, test):', flush=True)
for s in ['high', 'low']:
    msk = te & (salls == s)
    if msk.sum() < 10: continue
    ar = agg(rf_c_pred, msk); al = agg(lstm_d_pred, msk)
    print(f'  sal={s:4s} n={msk.sum():>4d}  RF CCC={ar["CCC"]:+.3f}  LSTM CCC={al["CCC"]:+.3f}  Δ={al["CCC"]-ar["CCC"]:+.3f}', flush=True)

# Save metrics
summary = {k: v['agg'] for k, v in results.items()}
with open(OUT / 'metrics_sq1_v3.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nmetrics -> {OUT}/metrics_sq1_v3.json', flush=True)

# ====================================================================
# PLOTS
# ====================================================================
print('\nGenerating plots...', flush=True)

labels = ['RF\nMB', 'LSTM\nMB', 'RF\nMB+scene', 'LSTM\nMB+scene']
keys = ['rf_mb', 'lstm_mb', 'rf_mb_scene', 'lstm_mb_scene']
ccc_vals = [results[k]['agg']['CCC'] for k in keys]
rmse_vals = [results[k]['agg']['RMSE'] for k in keys]
colors = ['#8e8e8e', '#aec7e8', '#555555', '#1f77b4']

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].bar(labels, ccc_vals, color=colors, edgecolor='black', linewidth=1)
axes[0].axhline(0, color='k', linewidth=0.8)
axes[0].set_ylabel('CCC (test)'); axes[0].set_title('SQ1 v3 — Multi-band CCC')
for i, v in enumerate(ccc_vals):
    axes[0].text(i, v + 0.01 if v >= 0 else v - 0.02, f'{v:+.3f}', ha='center', fontweight='bold')
axes[1].bar(labels, rmse_vals, color=colors, edgecolor='black', linewidth=1)
axes[1].set_ylabel('RMSE (test)'); axes[1].set_title('SQ1 v3 — Multi-band RMSE')
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v3_comparison.png'); plt.close()
print('  plots/sq1_v3_comparison.png', flush=True)

# Training curves
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
hB = pd.DataFrame(hist_B); hD = pd.DataFrame(hist_D)
ax.plot(hB['epoch'], hB['val_loss'], color='#aec7e8', linewidth=2, label='LSTM MB (val)')
ax.plot(hD['epoch'], hD['val_loss'], color='#1f77b4', linewidth=2, label='LSTM MB+scene (val)')
ax.plot(hB['epoch'], hB['train_loss'], color='#aec7e8', linewidth=1, alpha=0.5, linestyle='--', label='LSTM MB (train)')
ax.plot(hD['epoch'], hD['train_loss'], color='#1f77b4', linewidth=1, alpha=0.5, linestyle='--', label='LSTM MB+scene (train)')
ax.set_xlabel('epoch'); ax.set_ylabel('MSE loss')
ax.set_title('LSTM training curves (multi-band)')
ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v3_lstm_training.png'); plt.close()
print('  plots/sq1_v3_lstm_training.png', flush=True)

# Per-subject scatter
if len(common) >= 5:
    m = ps_rf_best.merge(ps_lstm_best, on='subject', suffixes=('_rf', '_lstm'))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(m['CCC_rf'], m['CCC_lstm'], s=80, c='steelblue', edgecolor='black', alpha=0.8)
    lim = [min(m[['CCC_rf','CCC_lstm']].values.min()-0.05, -0.2),
           max(m[['CCC_rf','CCC_lstm']].values.max()+0.05, 0.9)]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='equal')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('RF CCC (multi-band + scene)')
    ax.set_ylabel('LSTM CCC (multi-band + scene)')
    wins = int((m['CCC_lstm'] > m['CCC_rf']).sum())
    ax.set_title(f'Per-subject CCC — LSTM vs RF (multi-band+scene)\nLSTM wins {wins}/{len(m)} subjects')
    ax.legend()
    plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v3_per_subject_scatter.png'); plt.close()
    print('  plots/sq1_v3_per_subject_scatter.png', flush=True)

# Example windows
te_idx = np.where(te)[0]
ex = []
for i in te_idx:
    if (not np.isnan(rf_c_pred[i, PRE:]).any() and
        not np.isnan(lstm_d_pred[i, PRE:]).any()):
        ex.append(i)
        if len(ex) >= 6: break
if ex:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    t_ax = np.arange(T) * 100 - 2000
    for ax, wi in zip(axes.flat, ex):
        ax.plot(t_ax, Y_seq[wi], 'k-', linewidth=2, label='true valence')
        ax.plot(t_ax, rf_c_pred[wi], color='#8e8e8e', linestyle='--', linewidth=1.5, label='RF MB')
        ax.plot(t_ax, lstm_d_pred[wi], color='#1f77b4', linewidth=1.5, label='LSTM MB')
        ax.axvline(0, color='red', linestyle=':', alpha=0.6)
        ax.set_xlabel('ms relative to event'); ax.set_ylabel('valence')
        ax.set_title(f'P{pids[wi]} {scenes[wi][:3]} sal={salls[wi]}', fontsize=10)
        ax.set_ylim(-1.1, 1.1); ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v3_example_predictions.png'); plt.close()
    print('  plots/sq1_v3_example_predictions.png', flush=True)

print('\nDone.', flush=True)
