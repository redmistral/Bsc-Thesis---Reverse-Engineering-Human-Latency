"""
train_lstm_v4_full.py — Multimodal BiLSTM seq2seq vs RF baseline (Section 5.3, SQ1).

PURPOSE
-------
Tests the central SQ1 hypothesis: can a deep sequence model (BiLSTM seq2seq)
learn the variable EMG-to-valence delay better than the static-lag Random
Forest baseline used by Mavridou et al. (2025)? The answer reported in
the thesis is *no*: RF wins on 12 of 13 test subjects (Wilcoxon p=0.0007).
This script is the source of that result.

ALL IMPROVEMENTS COMPARED TO THE EARLIER v2 / v3 SCRIPTS
--------------------------------------------------------
1. **EMG multi-band features** (5 bands x 9 channels = 45 features per
   timestep). Each channel is split into bands (20-60, 60-120, 120-250,
   250-450, 20-450 Hz) so the model can learn band-specific patterns.
2. **Physiology fused in**: heart rate (2 ch) + IMU (9 ch) + accelerometer
   (3 ch) = 14 extra features per timestep, on top of the EMG.
3. **Per-window baseline-corrected valence target**: rather than predicting
   absolute valence, the model predicts the *deviation from each window's
   own pre-event mean*. This removes between-subject offset bias that
   otherwise dominates the cross-subject CCC.
4. **Bidirectional LSTM with CCC loss**. CCC = concordance correlation
   coefficient = the standard metric in continuous affect recognition;
   training directly on it (rather than on MSE) maximises what we report.
5. **Subject-level 80/10/10 split** with SEED=42 so train/val/test
   participants never overlap (no within-subject leakage).
6. Reports **CCC, Pearson r, and direction accuracy** (= sign of post-pre
   change matched) for interpretability.

MODEL VARIANTS
--------------
A. RF, static 200 ms lag, multi-band EMG          — replicates Mavridou (2025).
B. BiLSTM seq2seq, multi-band EMG only            — clean hypothesis test.
C. BiLSTM seq2seq, multi-band EMG + physiology    — does HR/IMU help?
D. BiLSTM seq2seq, multi-band EMG + physiology +
   scene-label one-hot                            — ceiling LSTM.

INPUT
-----
processed_data/event_windows_spectral.pkl  : the windowed event cache
processed_data/multimodal_windows.npz      : built on first run from the
                                             above pickle, then cached so
                                             reruns skip the costly
                                             re-extraction.

OUTPUT
------
model_results/metrics_sq1_v4.json    : per-subject CCC, Pearson r, direction
                                       accuracy for every variant on val/test.
plots/sq1_v4_comparison.png          : RF vs BiLSTM bar chart used in §5.3.
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
COMBINED_CACHE = PROC / 'multimodal_windows.npz'

SEED = 42
T = 70
PRE = 20
LAG = 2
FS_RAW = 1000
RMS_WIN = 100
DOWNSAMPLE = 100
PRE_MS = 2000
POST_MS = 5000

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
HR_COLS = ['heart_rate1', 'heart_rate_2']
IMU_COLS = [f'imu_ch{i}' for i in range(1, 10)]
ACC_COLS = ['accel_x', 'accel_y', 'accel_z']
PHYS_COLS = HR_COLS + IMU_COLS + ACC_COLS  # 2+9+3 = 14
N_CH = 9
N_PHYS = len(PHYS_COLS)
BANDS = [(20, 60), (60, 120), (120, 250), (250, 450), (20, 450)]
N_BANDS = len(BANDS)


def bandpass(x, low, high, fs=FS_RAW, order=4):
    nyq = fs / 2
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfilt(sos, x)


def moving_rms(x, w=RMS_WIN):
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode='same'))


def hampel_filter(data, window_size=50, n_sigmas=3):
    """Replace isolated spike artefacts with local median (Bhowmik et al., 2017)."""
    s = pd.Series(data)
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = s.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))), raw=True
    )
    outlier_mask = np.abs(s - rolling_median) > n_sigmas * rolling_mad
    result = data.copy()
    result[outlier_mask] = rolling_median[outlier_mask].values
    return result


def find_raw_path(pid_dir, scene):
    pat = 'POSITIVE' if scene == 'positive' else 'negative'
    for p in pid_dir.glob(f'*{pat}*_03_*_raw.txt'):
        json_p = p.with_name(p.name.replace('_raw.txt', '.json'))
        if json_p.exists():
            return p, json_p
    return None, None


def load_emg_and_phys(pid, scene):
    """Return (env_5band (T_raw,9,5), phys (T_raw,14), raw_frames, ref_frame) or None."""
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

    phys = raw[PHYS_COLS].values.astype(np.float32)
    return env, phys, raw['frame'].values.astype(np.int64), int(ref_frame)


def slice_signal(sig, raw_frames, ref_frame, event_time_s):
    """Slice [-2s, +5s] around event, downsample 1000->10 Hz by mean over 100 samples.
    Works for any (T_raw, ...) shaped sig.
    """
    event_frame = ref_frame + int(event_time_s * FS_RAW)
    start = event_frame - PRE_MS
    end = event_frame + POST_MS
    m = (raw_frames >= start) & (raw_frames < end)
    seg = sig[m]
    if seg.shape[0] < (PRE_MS + POST_MS):
        return None
    seg = seg[:PRE_MS + POST_MS]
    new_shape = (T, DOWNSAMPLE) + seg.shape[1:]
    return seg.reshape(new_shape).mean(axis=1).astype(np.float32)


# ============ BUILD or LOAD COMBINED MULTIMODAL CACHE ============
if COMBINED_CACHE.exists():
    print(f'Loading combined multimodal cache from {COMBINED_CACHE.name}...', flush=True)
    z = np.load(COMBINED_CACHE, allow_pickle=True)
    X_emg = z['X_emg']
    X_phys = z['X_phys']
    Y_seq = z['Y']
    pids = z['pids']
    scenes = z['scenes']
    salls = z['salls']
    print(f'  X_emg {X_emg.shape}  X_phys {X_phys.shape}  Y {Y_seq.shape}  ({len(pids)} windows)', flush=True)
else:
    print('Building combined multimodal cache (EMG multi-band + HR/IMU/accel) from raw 1000 Hz...', flush=True)
    with open(PROC / 'event_windows_spectral.pkl', 'rb') as f:
        windows = pickle.load(f)

    by_ps = defaultdict(list)
    for i, w in enumerate(windows):
        val = np.asarray(w.get('valence_target'), np.float32)
        if val.ndim != 1 or val.shape[0] < T or not np.isfinite(val).all():
            continue
        by_ps[(w['participant_id'], w['scene'])].append((i, w))
    n_groups = len(by_ps)
    print(f'  {len(windows)} pickle windows, {n_groups} (pid, scene) groups', flush=True)

    X_emg_list, X_phys_list, Y_list = [], [], []
    pids_list, scenes_list, salls_list = [], [], []

    t0 = time.time(); n_done = 0; n_skip = 0; n_bad = 0
    for k, ((pid, scene), items) in enumerate(sorted(by_ps.items()), 1):
        loaded = load_emg_and_phys(pid, scene)
        if loaded is None:
            n_skip += len(items); continue
        env_full, phys_full, raw_frames, ref_frame = loaded
        for idx, w in items:
            emg_seg = slice_signal(env_full, raw_frames, ref_frame, w['event_time_s'])
            phys_seg = slice_signal(phys_full, raw_frames, ref_frame, w['event_time_s'])
            val = np.asarray(w['valence_target'], np.float32)[:T]
            if (emg_seg is None or phys_seg is None
                    or not np.isfinite(emg_seg).all() or not np.isfinite(phys_seg).all()):
                n_bad += 1; continue
            X_emg_list.append(emg_seg.reshape(T, N_CH * N_BANDS))
            X_phys_list.append(phys_seg)
            Y_list.append(val)
            pids_list.append(w['participant_id'])
            scenes_list.append(w['scene'])
            salls_list.append(w['saliency'])
            n_done += 1
        if k % 20 == 0 or k == n_groups:
            el = time.time() - t0
            eta = el / k * (n_groups - k)
            print(f'  [{k}/{n_groups}] done={n_done} skip={n_skip} bad={n_bad} '
                  f'elapsed={el:.0f}s ETA={eta:.0f}s', flush=True)

    X_emg = np.stack(X_emg_list).astype(np.float32)
    X_phys = np.stack(X_phys_list).astype(np.float32)
    Y_seq = np.stack(Y_list).astype(np.float32)
    pids = np.array(pids_list)
    scenes = np.array(scenes_list)
    salls = np.array(salls_list)
    np.savez(COMBINED_CACHE, X_emg=X_emg, X_phys=X_phys, Y=Y_seq,
             pids=pids, scenes=scenes, salls=salls)
    print(f'\nSaved {n_done} windows -> {COMBINED_CACHE}', flush=True)
    print(f'  X_emg {X_emg.shape}  X_phys {X_phys.shape}', flush=True)

# ============ BASELINE-CORRECT VALENCE (per window) ============
# Predict deviation from pre-event mean. Doesn't leak (uses only past).
pre_mean = Y_seq[:, :PRE].mean(axis=1, keepdims=True)
Y_centered = Y_seq - pre_mean
print(f'  pre-event mean: μ={pre_mean.mean():.3f}  σ={pre_mean.std():.3f}', flush=True)
print(f'  centered Y:     μ={Y_centered.mean():.3f}  σ={Y_centered.std():.3f}', flush=True)

scene_bin = np.array([1 if s == 'positive' else 0 for s in scenes], dtype=np.float32)
scene_feat = np.broadcast_to(scene_bin[:, None, None], (len(scene_bin), T, 1)).astype(np.float32)

# ============ SUBJECT SPLIT ============
upids = np.array(sorted(np.unique(pids)))
rng = np.random.default_rng(SEED); rng.shuffle(upids)
n = len(upids); n_tr = int(n * 0.8); n_va = int(n * 0.1)
tr_p = set(upids[:n_tr]); va_p = set(upids[n_tr:n_tr + n_va]); te_p = set(upids[n_tr + n_va:])
tr = np.isin(pids, list(tr_p)); va = np.isin(pids, list(va_p)); te = np.isin(pids, list(te_p))
print(f'\n  split: train={tr.sum()}/{len(tr_p)}  val={va.sum()}/{len(va_p)}  test={te.sum()}/{len(te_p)}', flush=True)

# Standardize EMG and physiology separately (fit on train)
F_emg = X_emg.shape[-1]; F_phys = X_phys.shape[-1]
sc_emg = StandardScaler().fit(X_emg[tr].reshape(-1, F_emg))
sc_phys = StandardScaler().fit(X_phys[tr].reshape(-1, F_phys))
X_emg_s = sc_emg.transform(X_emg.reshape(-1, F_emg)).reshape(X_emg.shape).astype(np.float32)
X_phys_s = sc_phys.transform(X_phys.reshape(-1, F_phys)).reshape(X_phys.shape).astype(np.float32)
# Replace any NaN (e.g. zero-variance HR for some subjects) with 0
X_phys_s = np.nan_to_num(X_phys_s, nan=0.0, posinf=0.0, neginf=0.0)

X_emg_phys = np.concatenate([X_emg_s, X_phys_s], axis=-1)
X_emg_phys_scene = np.concatenate([X_emg_s, X_phys_s, scene_feat], axis=-1)
print(f'  EMG-only feats: {F_emg}  EMG+phys: {X_emg_phys.shape[-1]}  EMG+phys+scene: {X_emg_phys_scene.shape[-1]}', flush=True)


# ============ METRICS ============
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


def direction_accuracy(yt, yp, eps=0.02):
    """% of timesteps where sign of (true_change) matches sign of (pred_change).
    Both compared to per-window pre-event baseline.
    eps: deadband to avoid noisy near-zero classifications.
    """
    yt = np.asarray(yt); yp = np.asarray(yp)
    mask = (np.abs(yt) > eps) & (np.abs(yp) > eps)
    if mask.sum() == 0:
        return 0.5
    return float((np.sign(yt[mask]) == np.sign(yp[mask])).mean())


# ============ CCC LOSS ============
def ccc_loss(y_pred, y_true):
    """1 - CCC. Computed over flattened batch."""
    yp = y_pred.reshape(-1)
    yt = y_true.reshape(-1)
    yp_mean = yp.mean()
    yt_mean = yt.mean()
    yp_var = yp.var(unbiased=False)
    yt_var = yt.var(unbiased=False)
    cov = ((yp - yp_mean) * (yt - yt_mean)).mean()
    cccv = (2 * cov) / (yp_var + yt_var + (yp_mean - yt_mean) ** 2 + 1e-8)
    return 1 - cccv


# ============ MODELS ============
class BiLSTMValence(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


def train_lstm(X, Y, tr_mask, va_mask, tag, epochs=120, patience=20):
    print(f'\n  [BiLSTM-{tag}]  input_size={X.shape[-1]}  hidden=128  layers=2  bidir', flush=True)
    model = BiLSTMValence(input_size=X.shape[-1], hidden=128, layers=2, dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_idx = np.where(tr_mask)[0]; va_idx = np.where(va_mask)[0]
    best_val = 1e9; best_state = None; bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        tls = []
        for k in range(0, len(tr_idx), 128):
            b = tr_idx[k:k+128]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            yb = torch.from_numpy(Y[b]).to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            # Combined loss: 0.7*CCC + 0.3*MSE for stability
            mse = ((pred - yb) ** 2).mean()
            cl = ccc_loss(pred, yb)
            loss = 0.7 * cl + 0.3 * mse
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(loss.item())
        sched.step()
        model.eval()
        vls = []; v_ccc_acc = []
        with torch.no_grad():
            for k in range(0, len(va_idx), 256):
                b = va_idx[k:k+256]
                xb = torch.from_numpy(X[b]).to(DEVICE)
                yb = torch.from_numpy(Y[b]).to(DEVICE)
                pred = model(xb)
                mse = ((pred - yb) ** 2).mean().item()
                cl = ccc_loss(pred, yb).item()
                vls.append(0.7 * cl + 0.3 * mse)
                v_ccc_acc.append(1 - cl)
        tl, vl = float(np.mean(tls)), float(np.mean(vls))
        v_ccc = float(np.mean(v_ccc_acc))
        history.append({'epoch': ep, 'train_loss': tl, 'val_loss': vl, 'val_ccc': v_ccc})
        improved = vl < best_val - 1e-5
        if improved:
            best_val = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or ep == epochs or (improved and ep <= 5):
            print(f'    ep {ep:3d}  tr={tl:.4f}  va={vl:.4f}  vCCC={v_ccc:+.3f}  {"*" if improved else ""}', flush=True)
        if bad >= patience:
            print(f'    early stop at ep {ep}  (best val {best_val:.4f})', flush=True)
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT / f'lstm_sq1_v4_{tag}_history.csv', index=False)
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


def rf_predict_seq(rf, X, Y, mask):
    Xrf, _, widx, ts = make_rf_pairs(X, Y, mask)
    yp = rf.predict(Xrf).astype(np.float32)
    seq = np.full((len(X), T), np.nan, dtype=np.float32)
    for p, wi, tt in zip(yp, widx, ts):
        seq[wi, tt + LAG] = p
    return seq


def agg(pred_seq, Y_target, mask, eval_post=True):
    """pred_seq and Y_target both in CENTERED space."""
    ys, ps = [], []
    tr_ = range(PRE, T) if eval_post else range(T)
    for i in np.where(mask)[0]:
        for t in tr_:
            if np.isnan(pred_seq[i, t]): continue
            ys.append(Y_target[i, t]); ps.append(pred_seq[i, t])
    ys, ps = np.array(ys), np.array(ps)
    return {'n': int(len(ys)),
            'CCC': ccc(ys, ps),
            'RMSE': rmse(ys, ps),
            'r': pearson(ys, ps),
            'dir_acc': direction_accuracy(ys, ps)}


def per_subj(pred_seq, Y_target, mask, eval_post=True):
    tr_ = list(range(PRE, T)) if eval_post else list(range(T))
    store = {}
    for i in np.where(mask)[0]:
        s = pids[i]
        store.setdefault(s, {'y': [], 'p': []})
        for t in tr_:
            if not np.isnan(pred_seq[i, t]):
                store[s]['y'].append(Y_target[i, t]); store[s]['p'].append(pred_seq[i, t])
    rows = []
    for s, d in store.items():
        if len(d['y']) >= 20:
            rows.append({'subject': s, 'CCC': ccc(d['y'], d['p']),
                         'r': pearson(d['y'], d['p']),
                         'RMSE': rmse(d['y'], d['p']),
                         'dir_acc': direction_accuracy(d['y'], d['p']),
                         'n': len(d['y'])})
    return pd.DataFrame(rows)


# ============================================================
# RUN ALL VARIANTS
# ============================================================
results = {}

print('\n' + '=' * 80, flush=True)
print('(A) RF static 200ms lag — EMG multi-band only')
print('=' * 80, flush=True)
Xrf_tr, yrf_tr, _, _ = make_rf_pairs(X_emg_s, Y_centered, tr)
print(f'  train frames: {Xrf_tr.shape}', flush=True)
t0 = time.time()
rf_a = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=10,
                              max_features='sqrt', n_jobs=-1, random_state=SEED).fit(Xrf_tr, yrf_tr)
print(f'  trained in {time.time()-t0:.0f}s', flush=True)
rf_a_pred = rf_predict_seq(rf_a, X_emg_s, Y_centered, te)
results['rf_emg'] = {'agg': agg(rf_a_pred, Y_centered, te), 'pred': rf_a_pred}
print(f'  agg: {results["rf_emg"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(B) BiLSTM seq2seq — EMG multi-band only  [CLEAN DRAFT HYPOTHESIS]')
print('=' * 80, flush=True)
lstm_b, hist_B = train_lstm(X_emg_s, Y_centered, tr, va, tag='emg', epochs=120, patience=20)
lstm_b_pred = lstm_predict(lstm_b, X_emg_s, te)
results['lstm_emg'] = {'agg': agg(lstm_b_pred, Y_centered, te), 'pred': lstm_b_pred}
print(f'  agg: {results["lstm_emg"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(C) BiLSTM seq2seq — EMG + physiology (HR + IMU + accel)')
print('=' * 80, flush=True)
lstm_c, hist_C = train_lstm(X_emg_phys, Y_centered, tr, va, tag='emg_phys', epochs=120, patience=20)
lstm_c_pred = lstm_predict(lstm_c, X_emg_phys, te)
results['lstm_emg_phys'] = {'agg': agg(lstm_c_pred, Y_centered, te), 'pred': lstm_c_pred}
print(f'  agg: {results["lstm_emg_phys"]["agg"]}', flush=True)

print('\n' + '=' * 80, flush=True)
print('(D) BiLSTM seq2seq — EMG + physiology + scene')
print('=' * 80, flush=True)
lstm_d, hist_D = train_lstm(X_emg_phys_scene, Y_centered, tr, va, tag='emg_phys_scene', epochs=120, patience=20)
lstm_d_pred = lstm_predict(lstm_d, X_emg_phys_scene, te)
results['lstm_full'] = {'agg': agg(lstm_d_pred, Y_centered, te), 'pred': lstm_d_pred}
print(f'  agg: {results["lstm_full"]["agg"]}', flush=True)

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '=' * 80, flush=True)
print('SQ1 v4 SUMMARY (test subjects, post-event, baseline-corrected)')
print('=' * 80, flush=True)
print(f'{"Model":<40s}{"n":>7s}{"CCC":>8s}{"r":>8s}{"DirAcc":>8s}{"RMSE":>8s}')
print('-' * 79)
for name, key in [('RF static 200ms — EMG',                'rf_emg'),
                  ('BiLSTM — EMG',                          'lstm_emg'),
                  ('BiLSTM — EMG+phys',                     'lstm_emg_phys'),
                  ('BiLSTM — EMG+phys+scene  [FULL]',       'lstm_full')]:
    a = results[key]['agg']
    print(f'{name:<40s}{a["n"]:>7d}{a["CCC"]:>8.3f}{a["r"]:>8.3f}{a["dir_acc"]*100:>7.1f}%{a["RMSE"]:>8.3f}', flush=True)

print('\nDirection accuracy (>50% = better than coin flip):', flush=True)
for name, key in [('RF EMG', 'rf_emg'), ('BiLSTM EMG', 'lstm_emg'),
                  ('BiLSTM EMG+phys', 'lstm_emg_phys'), ('BiLSTM full', 'lstm_full')]:
    print(f'  {name:25s}  {results[key]["agg"]["dir_acc"]*100:.1f}%', flush=True)

# Per-subject best model
print('\nPer-subject metrics (BiLSTM full):', flush=True)
ps_full = per_subj(lstm_d_pred, Y_centered, te)
ps_rf = per_subj(rf_a_pred, Y_centered, te)
ps_full.to_csv(OUT / 'sq1_v4_per_subject_lstm_full.csv', index=False)
ps_rf.to_csv(OUT / 'sq1_v4_per_subject_rf.csv', index=False)
print(f'  CCC      mean={ps_full["CCC"].mean():+.3f}  median={ps_full["CCC"].median():+.3f}  n={len(ps_full)}', flush=True)
print(f'  r        mean={ps_full["r"].mean():+.3f}  median={ps_full["r"].median():+.3f}', flush=True)
print(f'  dir_acc  mean={ps_full["dir_acc"].mean()*100:.1f}%  median={ps_full["dir_acc"].median()*100:.1f}%', flush=True)

common = set(ps_full['subject']) & set(ps_rf['subject'])
if len(common) >= 5:
    rv = ps_rf.set_index('subject').loc[list(common), 'CCC'].values
    lv = ps_full.set_index('subject').loc[list(common), 'CCC'].values
    stat, p = wilcoxon(lv, rv)
    wins = int(np.sum(lv > rv))
    print(f'\n  Wilcoxon paired (CCC): stat={stat:.1f}  p={p:.4g}  BiLSTM better in {wins}/{len(common)} subjects', flush=True)

# By saliency
print('\nBY SALIENCY (BiLSTM full, test):', flush=True)
for s in ['high', 'low']:
    msk = te & (salls == s)
    if msk.sum() < 10: continue
    a = agg(lstm_d_pred, Y_centered, msk)
    print(f'  sal={s:4s} n={msk.sum():>4d}  CCC={a["CCC"]:+.3f}  r={a["r"]:+.3f}  dir_acc={a["dir_acc"]*100:.1f}%', flush=True)

# Save
summary = {k: v['agg'] for k, v in results.items()}
with open(OUT / 'metrics_sq1_v4.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nmetrics -> {OUT}/metrics_sq1_v4.json', flush=True)

# ============================================================
# PLOTS
# ============================================================
print('\nGenerating plots...', flush=True)

labels = ['RF\nEMG', 'BiLSTM\nEMG', 'BiLSTM\nEMG+phys', 'BiLSTM\nfull']
keys = ['rf_emg', 'lstm_emg', 'lstm_emg_phys', 'lstm_full']
ccc_vals = [results[k]['agg']['CCC'] for k in keys]
r_vals = [results[k]['agg']['r'] for k in keys]
da_vals = [results[k]['agg']['dir_acc'] * 100 for k in keys]
colors = ['#8e8e8e', '#aec7e8', '#1f77b4', '#0a4d8c']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(labels, ccc_vals, color=colors, edgecolor='black', linewidth=1)
axes[0].axhline(0, color='k', linewidth=0.8)
axes[0].set_ylabel('CCC'); axes[0].set_title('CCC (test)')
for i, v in enumerate(ccc_vals):
    axes[0].text(i, v + 0.01 if v >= 0 else v - 0.02, f'{v:+.3f}', ha='center', fontweight='bold')

axes[1].bar(labels, r_vals, color=colors, edgecolor='black', linewidth=1)
axes[1].axhline(0, color='k', linewidth=0.8)
axes[1].set_ylabel('Pearson r'); axes[1].set_title('Pearson r (test)')
for i, v in enumerate(r_vals):
    axes[1].text(i, v + 0.01, f'{v:+.3f}', ha='center', fontweight='bold')

axes[2].bar(labels, da_vals, color=colors, edgecolor='black', linewidth=1)
axes[2].axhline(50, color='red', linewidth=1, linestyle='--', label='chance')
axes[2].set_ylabel('Direction accuracy (%)'); axes[2].set_title('Direction accuracy')
axes[2].legend()
for i, v in enumerate(da_vals):
    axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v4_comparison.png'); plt.close()
print('  plots/sq1_v4_comparison.png', flush=True)

# Training curves
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
for hist, lbl, col in [(hist_B, 'EMG only', '#aec7e8'),
                        (hist_C, 'EMG+phys', '#1f77b4'),
                        (hist_D, 'EMG+phys+scene (full)', '#0a4d8c')]:
    h = pd.DataFrame(hist)
    ax.plot(h['epoch'], h['val_ccc'], color=col, linewidth=2, label=f'{lbl} (val CCC)')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('epoch'); ax.set_ylabel('val CCC')
ax.set_title('BiLSTM training — val CCC')
ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v4_training.png'); plt.close()
print('  plots/sq1_v4_training.png', flush=True)

# Example windows
te_idx = np.where(te)[0]
ex = []
for i in te_idx:
    if (not np.isnan(rf_a_pred[i, PRE:]).any() and
        not np.isnan(lstm_d_pred[i, PRE:]).any()):
        ex.append(i)
        if len(ex) >= 6: break
if ex:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    t_ax = np.arange(T) * 100 - 2000
    for ax, wi in zip(axes.flat, ex):
        ax.plot(t_ax, Y_centered[wi], 'k-', linewidth=2, label='true Δ-valence')
        ax.plot(t_ax, rf_a_pred[wi], color='#8e8e8e', linestyle='--', linewidth=1.5, label='RF EMG')
        ax.plot(t_ax, lstm_d_pred[wi], color='#0a4d8c', linewidth=1.5, label='BiLSTM full')
        ax.axvline(0, color='red', linestyle=':', alpha=0.6)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('ms relative to event'); ax.set_ylabel('Δ valence (centered)')
        ax.set_title(f'P{pids[wi]} {scenes[wi][:3]} sal={salls[wi]}', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(PLOTS / 'sq1_v4_example_predictions.png'); plt.close()
    print('  plots/sq1_v4_example_predictions.png', flush=True)

print('\nDone.', flush=True)
