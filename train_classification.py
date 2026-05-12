"""
train_classification.py — Window-level binary valence-direction classification (§5.3.2).

WHY THIS SCRIPT EXISTS
----------------------
The continuous-valence regression results from `train_lstm_v4_full.py` are
reported in CCC (concordance correlation), which is hard to interpret outside
the affective-computing literature. To give the supervisor (and the thesis
reader) a clean, interpretable accuracy %, this script reframes SQ1 as a
binary classification:

  Per event window, predict whether the participant's joystick valence will
  RISE or FALL after T_event.

  Label rule:
    label = 1 if mean(valence[t=2..7s]) > mean(valence[t=-2..0s]) else 0
    Windows with |Δ| < `deadband` are dropped, removing the noise floor where
    the participant did not appreciably move the joystick at all.

This makes:
  - the chance baseline easy to read (= the majority-class fraction);
  - the metrics standard (accuracy / F1 / AUC).

MODEL GRID
----------
A. Majority-class baseline      — trivial floor (always predict the larger class).
B. Scene-only logistic          — "is just knowing the scene enough?" sanity check.
C. RF on EMG multi-band only    — what does facial EMG add over scene context?
D. RF on EMG + physiology       — does heart-rate / IMU add information?
E. RF on EMG + phys + scene     — ceiling RF.
F. BiLSTM, EMG only             — does temporal context help?
G. BiLSTM, EMG + phys + scene   — ceiling LSTM.

OUTPUT
------
model_results/metrics_classification.json — one entry per model with
  accuracy, F1, AUC, sensitivity, specificity, plus the binomial significance
  test against the majority-class baseline.
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import binomtest

warnings.filterwarnings('ignore')

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
PLOTS = BASE / 'plots'
PLOTS.mkdir(exist_ok=True)
COMBINED_CACHE = PROC / 'multimodal_windows.npz'

SEED = 42
T = 70
PRE = 20
DEADBAND = 0.05  # drop windows with |Δvalence| < deadband for clean labels
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})

# ============ LOAD ============
print(f'Loading multimodal cache from {COMBINED_CACHE.name}...', flush=True)
z = np.load(COMBINED_CACHE, allow_pickle=True)
X_emg = z['X_emg']     # (N, 70, 45)
X_phys = z['X_phys']   # (N, 70, 14)
Y_seq = z['Y']         # (N, 70)
pids = z['pids']
scenes = z['scenes']
salls = z['salls']
print(f'  X_emg {X_emg.shape}  X_phys {X_phys.shape}  Y {Y_seq.shape}', flush=True)

# ============ LABELS ============
pre_mean = Y_seq[:, :PRE].mean(axis=1)
post_mean = Y_seq[:, PRE:].mean(axis=1)
delta = post_mean - pre_mean
keep = np.abs(delta) > DEADBAND
print(f'  delta stats: μ={delta.mean():+.3f}  σ={delta.std():.3f}', flush=True)
print(f'  keeping {keep.sum()}/{len(keep)} windows after deadband |Δ| > {DEADBAND}', flush=True)

X_emg = X_emg[keep]; X_phys = X_phys[keep]; Y_seq = Y_seq[keep]
pids = pids[keep]; scenes = scenes[keep]; salls = salls[keep]
delta = delta[keep]
y_dir = (delta > 0).astype(int)
print(f'  class balance: rise={y_dir.sum()}/{len(y_dir)} ({y_dir.mean()*100:.1f}%)  '
      f'fall={(1-y_dir).sum()} ({(1-y_dir.mean())*100:.1f}%)', flush=True)
scene_bin = np.array([1 if s == 'positive' else 0 for s in scenes], dtype=np.float32)

# ============ SUBJECT SPLIT ============
upids = np.array(sorted(np.unique(pids)))
rng = np.random.default_rng(SEED); rng.shuffle(upids)
n = len(upids); n_tr = int(n * 0.8); n_va = int(n * 0.1)
tr_p = set(upids[:n_tr]); va_p = set(upids[n_tr:n_tr + n_va]); te_p = set(upids[n_tr + n_va:])
tr = np.isin(pids, list(tr_p)); va = np.isin(pids, list(va_p)); te = np.isin(pids, list(te_p))
print(f'\n  split: train={tr.sum()}/{len(tr_p)}  val={va.sum()}/{len(va_p)}  test={te.sum()}/{len(te_p)}', flush=True)
print(f'  train class balance: {y_dir[tr].mean()*100:.1f}% rise', flush=True)
print(f'  test  class balance: {y_dir[te].mean()*100:.1f}% rise', flush=True)


# ============ WINDOW-SUMMARY FEATURES (for RF) ============
def summarize_window(X):
    """X: (N, T, F) -> (N, 4F)  pre/post mean/std + delta of mean per feature."""
    pre = X[:, :PRE]
    post = X[:, PRE:]
    feats = np.concatenate([
        pre.mean(axis=1), pre.std(axis=1),
        post.mean(axis=1), post.std(axis=1),
        post.mean(axis=1) - pre.mean(axis=1),
    ], axis=-1)
    return feats.astype(np.float32)


F_emg = summarize_window(X_emg)              # (N, 45*5)
F_phys = summarize_window(X_phys)            # (N, 14*5)
F_emg_phys = np.concatenate([F_emg, F_phys], axis=1)
F_emg_phys_scene = np.concatenate([F_emg_phys, scene_bin[:, None]], axis=1)
print(f'  RF feature dims: emg={F_emg.shape[1]}  emg+phys={F_emg_phys.shape[1]}  '
      f'+scene={F_emg_phys_scene.shape[1]}', flush=True)

# Standardize for RF (helps tree splits but RF is invariant — for LR matters)
sc = StandardScaler().fit(F_emg[tr])
F_emg_s = sc.transform(F_emg).astype(np.float32)
sc2 = StandardScaler().fit(F_emg_phys[tr])
F_emg_phys_s = sc2.transform(F_emg_phys).astype(np.float32)
F_emg_phys_s = np.nan_to_num(F_emg_phys_s, nan=0.0, posinf=0.0, neginf=0.0)
sc3 = StandardScaler().fit(F_emg_phys_scene[tr])
F_emg_phys_scene_s = sc3.transform(F_emg_phys_scene).astype(np.float32)
F_emg_phys_scene_s = np.nan_to_num(F_emg_phys_scene_s, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize sequence features (fit on train)
sc_emg_seq = StandardScaler().fit(X_emg[tr].reshape(-1, X_emg.shape[-1]))
X_emg_s = sc_emg_seq.transform(X_emg.reshape(-1, X_emg.shape[-1])).reshape(X_emg.shape).astype(np.float32)
sc_phys_seq = StandardScaler().fit(X_phys[tr].reshape(-1, X_phys.shape[-1]))
X_phys_s = sc_phys_seq.transform(X_phys.reshape(-1, X_phys.shape[-1])).reshape(X_phys.shape).astype(np.float32)
X_phys_s = np.nan_to_num(X_phys_s, nan=0.0, posinf=0.0, neginf=0.0)
scene_feat = np.broadcast_to(scene_bin[:, None, None], (len(scene_bin), T, 1)).astype(np.float32)
X_emg_phys_scene_seq = np.concatenate([X_emg_s, X_phys_s, scene_feat], axis=-1)


# ============ METRICS ============
def report(y_true, y_pred, y_prob=None, label=''):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    cm = confusion_matrix(y_true, y_pred)
    # Binomial test against majority class baseline
    maj = max(y_true.mean(), 1 - y_true.mean())
    bt = binomtest(int((y_pred == y_true).sum()), len(y_true), p=maj)
    sig = '***' if bt.pvalue < 0.001 else ('**' if bt.pvalue < 0.01 else ('*' if bt.pvalue < 0.05 else 'ns'))
    print(f'  {label:38s}  acc={acc*100:5.1f}%  F1={f1:.3f}  AUC={auc:.3f}  '
          f'(vs maj {maj*100:.1f}%, p={bt.pvalue:.3g} {sig})  '
          f'cm=[{cm[0,0]},{cm[0,1]};{cm[1,0]},{cm[1,1]}]', flush=True)
    return {'acc': float(acc), 'f1': float(f1), 'auc': float(auc),
            'maj_baseline': float(maj), 'p_vs_maj': float(bt.pvalue),
            'cm': cm.tolist()}


# ============ BiLSTM CLASSIFIER ============
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        # use mean over time then a head -> single logit
        pooled = out.mean(dim=1)
        return self.fc(pooled).squeeze(-1)


def train_lstm_clf(X, y, tr_mask, va_mask, tag, epochs=80, patience=15):
    print(f'\n  [BiLSTM-clf-{tag}]  input_size={X.shape[-1]}', flush=True)
    model = BiLSTMClassifier(input_size=X.shape[-1], hidden=128, layers=2, dropout=0.25).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # class-balanced loss
    pos_w = torch.tensor([(1 - y[tr_mask].mean()) / max(y[tr_mask].mean(), 1e-6)],
                          dtype=torch.float32).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    tr_idx = np.where(tr_mask)[0]; va_idx = np.where(va_mask)[0]
    best_val = -1.0; best_state = None; bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        tls = []
        for k in range(0, len(tr_idx), 128):
            b = tr_idx[k:k+128]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            yb = torch.from_numpy(y[b].astype(np.float32)).to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(loss.item())
        sched.step()
        model.eval()
        v_logits = []; v_y = []
        with torch.no_grad():
            for k in range(0, len(va_idx), 256):
                b = va_idx[k:k+256]
                xb = torch.from_numpy(X[b]).to(DEVICE)
                v_logits.append(model(xb).cpu().numpy()); v_y.append(y[b])
        v_logits = np.concatenate(v_logits); v_y = np.concatenate(v_y)
        v_pred = (v_logits > 0).astype(int)
        v_acc = (v_pred == v_y).mean()
        v_auc = roc_auc_score(v_y, v_logits) if len(np.unique(v_y)) > 1 else 0.5
        tl = float(np.mean(tls))
        history.append({'epoch': ep, 'train_loss': tl, 'val_acc': float(v_acc), 'val_auc': float(v_auc)})
        improved = v_auc > best_val + 1e-4
        if improved:
            best_val = v_auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or ep == epochs or (improved and ep <= 5):
            print(f'    ep {ep:3d}  tr_loss={tl:.4f}  v_acc={v_acc*100:.1f}%  v_AUC={v_auc:.3f}  {"*" if improved else ""}', flush=True)
        if bad >= patience:
            print(f'    early stop at ep {ep}  (best val AUC {best_val:.3f})', flush=True)
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT / f'clf_lstm_{tag}_history.csv', index=False)
    return model, history


def lstm_predict_clf(model, X, mask):
    model.eval()
    idx = np.where(mask)[0]
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(idx), 256):
            b = idx[k:k+256]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            out[k:k+len(b)] = model(xb).cpu().numpy()
    return out


# ====================================================================
# RUN ALL VARIANTS
# ====================================================================
results = {}

print('\n' + '=' * 100, flush=True)
print('VALENCE DIRECTION CLASSIFICATION (test subjects)', flush=True)
print('=' * 100, flush=True)
y_te = y_dir[te]

# (A) Majority baseline
print('\n--- Baselines ---', flush=True)
maj_class = int(y_dir[tr].mean() > 0.5)
y_pred_maj = np.full_like(y_te, maj_class)
y_prob_maj = np.full(len(y_te), float(y_dir[tr].mean()))
results['majority'] = report(y_te, y_pred_maj, y_prob_maj, 'A. Majority class')

# (B) Scene-only logistic
lr = LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=200)
lr.fit(scene_bin[tr].reshape(-1, 1), y_dir[tr])
results['scene_only'] = report(y_te, lr.predict(scene_bin[te].reshape(-1, 1)),
                                lr.predict_proba(scene_bin[te].reshape(-1, 1))[:, 1], 'B. Scene only (LR)')

# (C) RF EMG
print('\n--- Random forest (window-summary features) ---', flush=True)
t0 = time.time()
rf_c = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=10,
                               max_features='sqrt', n_jobs=-1, class_weight='balanced',
                               random_state=SEED).fit(F_emg_s[tr], y_dir[tr])
print(f'  trained in {time.time()-t0:.0f}s')
results['rf_emg'] = report(y_te, rf_c.predict(F_emg_s[te]),
                            rf_c.predict_proba(F_emg_s[te])[:, 1], 'C. RF — EMG only')

# (D) RF EMG+phys
t0 = time.time()
rf_d = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=10,
                               max_features='sqrt', n_jobs=-1, class_weight='balanced',
                               random_state=SEED).fit(F_emg_phys_s[tr], y_dir[tr])
print(f'  trained in {time.time()-t0:.0f}s')
results['rf_emg_phys'] = report(y_te, rf_d.predict(F_emg_phys_s[te]),
                                 rf_d.predict_proba(F_emg_phys_s[te])[:, 1], 'D. RF — EMG + phys')

# (E) RF EMG+phys+scene
t0 = time.time()
rf_e = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=10,
                               max_features='sqrt', n_jobs=-1, class_weight='balanced',
                               random_state=SEED).fit(F_emg_phys_scene_s[tr], y_dir[tr])
print(f'  trained in {time.time()-t0:.0f}s')
results['rf_full'] = report(y_te, rf_e.predict(F_emg_phys_scene_s[te]),
                             rf_e.predict_proba(F_emg_phys_scene_s[te])[:, 1], 'E. RF — EMG + phys + scene')

# (F) BiLSTM EMG
print('\n--- BiLSTM sequence classifiers ---', flush=True)
lstm_f, hist_F = train_lstm_clf(X_emg_s, y_dir, tr, va, tag='emg', epochs=80, patience=15)
logits_f = lstm_predict_clf(lstm_f, X_emg_s, te)
results['lstm_emg'] = report(y_te, (logits_f > 0).astype(int),
                              1 / (1 + np.exp(-logits_f)), 'F. BiLSTM — EMG only')

# (G) BiLSTM full
lstm_g, hist_G = train_lstm_clf(X_emg_phys_scene_seq, y_dir, tr, va, tag='full', epochs=80, patience=15)
logits_g = lstm_predict_clf(lstm_g, X_emg_phys_scene_seq, te)
results['lstm_full'] = report(y_te, (logits_g > 0).astype(int),
                               1 / (1 + np.exp(-logits_g)), 'G. BiLSTM — EMG + phys + scene')

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '=' * 100, flush=True)
print('VALENCE DIRECTION — FINAL TABLE', flush=True)
print('=' * 100, flush=True)
print(f'{"Model":<40s}{"Acc":>10s}{"F1":>8s}{"AUC":>8s}{"vs maj":>10s}{"p-val":>10s}')
print('-' * 86)
for name, key in [('A. Majority class',          'majority'),
                  ('B. Scene only (LR)',         'scene_only'),
                  ('C. RF — EMG',                'rf_emg'),
                  ('D. RF — EMG + phys',         'rf_emg_phys'),
                  ('E. RF — EMG + phys + scene', 'rf_full'),
                  ('F. BiLSTM — EMG',            'lstm_emg'),
                  ('G. BiLSTM — full',           'lstm_full')]:
    r = results[key]
    print(f'{name:<40s}{r["acc"]*100:>9.1f}%{r["f1"]:>8.3f}{r["auc"]:>8.3f}'
          f'{(r["acc"]-r["maj_baseline"])*100:>+9.1f}%{r["p_vs_maj"]:>10.3g}', flush=True)

print(f'\nTest set: {len(y_te)} windows  (rise={y_te.sum()}, fall={(1-y_te).sum()})', flush=True)
print(f'Random chance = 50.0%   Majority baseline = {max(y_te.mean(), 1-y_te.mean())*100:.1f}%', flush=True)

# Save
with open(OUT / 'metrics_classification.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'cm'}
               for k, v in results.items()}, f, indent=2)
print(f'\nmetrics -> {OUT}/metrics_classification.json', flush=True)

# ============================================================
# PLOTS
# ============================================================
print('\nGenerating plots...', flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labels = ['Maj', 'Scene', 'RF\nEMG', 'RF\nEMG+phys', 'RF\nfull', 'BiLSTM\nEMG', 'BiLSTM\nfull']
keys = ['majority', 'scene_only', 'rf_emg', 'rf_emg_phys', 'rf_full', 'lstm_emg', 'lstm_full']
acc_vals = [results[k]['acc'] * 100 for k in keys]
auc_vals = [results[k]['auc'] for k in keys]
colors = ['#cccccc', '#ffd966', '#8e8e8e', '#5e8eb8', '#1f77b4', '#aec7e8', '#0a4d8c']

axes[0].bar(labels, acc_vals, color=colors, edgecolor='black', linewidth=1)
axes[0].axhline(50, color='red', linestyle='--', linewidth=1, label='chance (50%)')
axes[0].axhline(max(y_te.mean(), 1 - y_te.mean()) * 100, color='orange', linestyle=':',
                linewidth=1, label=f'majority ({max(y_te.mean(), 1-y_te.mean())*100:.1f}%)')
axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Valence direction — Accuracy')
axes[0].legend()
for i, v in enumerate(acc_vals):
    axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=9)

axes[1].bar(labels, auc_vals, color=colors, edgecolor='black', linewidth=1)
axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, label='chance')
axes[1].set_ylabel('AUC'); axes[1].set_title('Valence direction — AUC')
axes[1].legend()
for i, v in enumerate(auc_vals):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout(); plt.savefig(PLOTS / 'classification_valence_direction.png'); plt.close()
print('  plots/classification_valence_direction.png', flush=True)

print('\nDone.', flush=True)
