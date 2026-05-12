"""
extract_spectral_features.py — Stage 2 of the analysis pipeline (Section 3.4).

PURPOSE
-------
Augments the per-event windows produced by `run_pipeline.py` with a 459-dim
spectral + advanced-time-domain feature vector per event. This is the feature
representation that supports the saliency classification result (AUC 0.729 in
Section 5.3.2; see `train_spectral.py`) and is the standard EMG feature bank
used in the surface-EMG pattern-recognition literature.

For each event window we go back to the *raw* 1000 Hz EMG (not the cached
envelope) and compute, **per channel**, the 17-dim feature vector below:

  Time-domain (9):
    rms   -- root-mean-square (overall amplitude)
    mav   -- mean absolute value (rectified amplitude)
    var   -- variance (energy proxy)
    iemg  -- integrated EMG (sum of |x|)
    wl    -- waveform length (sum of |x[t+1] - x[t]|)
    zc    -- zero crossings (frequency proxy, with noise threshold)
    ssc   -- slope sign changes (firing-rate proxy, with noise threshold)
    skew  -- 3rd-moment asymmetry of the amplitude distribution
    kurt  -- 4th-moment heavy-tailedness of the amplitude distribution

  Frequency-domain (8) -- via Welch's PSD estimator:
    mpf       -- mean power frequency (1st spectral moment)
    mdf       -- median power frequency (frequency below which 50% of power lies)
    pkf       -- peak frequency (argmax of PSD)
    bp20_80, bp80_150, bp150_250, bp250_450 -- relative band power in 4 bands
    sp_ent    -- spectral entropy (uniform = high, peaky = low)

The full feature vector concatenates the 9 channels and is replicated three
times -- once on the *pre-event baseline* segment (-2..0 s), once on the
*post-event response* segment (0..+5 s), and once on the *full* window. This
gives 3 * 9 * 17 = 459 features per event.

OUTPUT
------
processed_data/event_windows_spectral.pkl -- the same list-of-dicts as
`event_windows_all.pkl`, but each window now has a `spec_features` key with
the 459-dim feature array (or None if extraction failed for that event).

WHY A SEPARATE STAGE
--------------------
Spectral feature extraction is computationally heavy and only some downstream
scripts need it. Keeping it separate from `run_pipeline.py` lets a fast
pipeline rerun (e.g. tweaking the saliency taxonomy) skip this stage entirely.
"""
import json, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
DATA = BASE / 'All Data'
PROC = BASE / 'processed_data'

FS = 1000
WINDOW_PRE_MS = 2000
WINDOW_POST_MS = 5000
RAW_COLUMNS = (
    ['frame'] + [f'emg_ch{i}' for i in range(1,10)] + ['heart_rate1','heart_rate_2']
    + [f'imu_ch{i}' for i in range(1,10)]
    + ['accel_x','accel_y','accel_z','magnetometer','gyro','system_related','_trailing']
)
EMG_COLS = [f'emg_ch{i}' for i in range(1,10)]

# Four EMG sub-bands used for the relative band-power features. The split
# follows standard surface-EMG practice: 20-80 Hz captures the bulk of motor-
# unit firing-rate energy; 80-150 Hz the transition band; 150-250 Hz higher-
# frequency motor-unit activity; 250-450 Hz the residual high-frequency tail.
BANDS = [(20, 80), (80, 150), (150, 250), (250, 450)]


def bandpass(data, low=20, high=450, fs=FS, order=4):
    """Same Butterworth bandpass as run_pipeline.py / estimate_delays.py.
    Reproduced here so this script can be run standalone on the raw .txt
    files without re-running the upstream pipeline.
    """
    nyq = fs/2
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sosfilt(sos, data)


def spectral_feats_channel(x, fs=FS):
    """Compute the 17-dim feature vector for one bandpassed channel segment.

    x : 1D ndarray of bandpassed (20-450 Hz) EMG samples at 1000 Hz.

    Returns a single ndarray of length 17 in the order documented at the top
    of this file.
    """
    # ============ Time-domain features ============
    n = len(x)
    rms = float(np.sqrt(np.mean(x**2)))
    mav = float(np.mean(np.abs(x)))
    var = float(np.var(x))
    iemg = float(np.sum(np.abs(x)))
    wl = float(np.sum(np.abs(np.diff(x))))
    # Zero crossings: count sign-flips in the raw waveform. A small noise
    # threshold rejects sub-quantisation jitter that would otherwise inflate
    # the count under near-zero baselines.
    thr = 1e-6
    zc = int(np.sum((x[:-1]*x[1:] < 0) & (np.abs(x[:-1]-x[1:]) >= thr)))
    # Slope sign changes: same idea but on the first difference -- a proxy for
    # the firing rate of motor units.
    d = np.diff(x)
    ssc = int(np.sum((d[:-1]*d[1:] < 0) & (np.abs(d[:-1]-d[1:]) >= thr)))
    sk = float(skew(x)) if np.std(x) > 0 else 0.0
    kt = float(kurtosis(x)) if np.std(x) > 0 else 0.0

    # ============ Frequency-domain features ============
    # Welch's method splits the signal into overlapping segments, FFTs each
    # one, and averages the squared magnitudes -- a low-variance estimate of
    # the power spectral density.
    nperseg = min(256, n)
    if nperseg < 8:
        # Segment too short for a meaningful PSD; pad the freq features with 0.
        return np.array([rms, mav, var, iemg, wl, zc, ssc, sk, kt] + [0.0]*(3 + len(BANDS) + 1))
    f, psd = welch(x, fs=fs, nperseg=nperseg)
    total_power = float(np.trapz(psd, f))
    if total_power <= 0:
        # Channel was effectively flat (electrode lost contact). Zero-fill.
        mpf = mdf = pkf = 0.0
        bands = [0.0]*len(BANDS)
        sp_ent = 0.0
    else:
        # Mean power frequency = first spectral moment (centroid).
        mpf = float(np.trapz(f*psd, f) / total_power)
        # Median power frequency = the f at which cumulative PSD reaches 50%.
        cum = np.cumsum(psd); tot = cum[-1]
        mdf_idx = int(np.searchsorted(cum, tot/2))
        mdf = float(f[min(mdf_idx, len(f)-1)])
        # Peak frequency = argmax of PSD.
        pkf = float(f[int(np.argmax(psd))])
        # Relative band power for each of the four EMG sub-bands defined above.
        bands = []
        for lo, hi in BANDS:
            m = (f >= lo) & (f <= hi)
            bands.append(float(np.trapz(psd[m], f[m]) / total_power) if m.any() else 0.0)
        # Spectral entropy on the normalised PSD (high = noise-like, low = peaky).
        p = psd / total_power + 1e-12
        sp_ent = float(-np.sum(p * np.log(p)))

    return np.array([rms, mav, var, iemg, wl, zc, ssc, sk, kt, mpf, mdf, pkf] + bands + [sp_ent])


FEAT_NAMES_PER_CHANNEL = ['rms','mav','var','iemg','wl','zc','ssc','skew','kurt',
                         'mpf','mdf','pkf','bp20_80','bp80_150','bp150_250','bp250_450','sp_ent']
N_FEATS_PER_CH = len(FEAT_NAMES_PER_CHANNEL)  # 17


def find_raw_path(pid_dir: Path, scene: str):
    """Return (raw_path, json_path) for scene in participant folder."""
    pat = 'POSITIVE' if scene == 'positive' else 'negative'
    for p in pid_dir.glob(f'*{pat}*_03_*_raw.txt'):
        json_p = p.with_name(p.name.replace('_raw.txt', '.json'))
        if json_p.exists():
            return p, json_p
    return None, None


def extract_features_for_participant(pid, scene, windows_for_scene):
    """Compute the 459-dim spectral feature vector for every event window in
    one (participant, scene) pair. Returns a list of feature arrays (or None
    for events whose window could not be extracted), in the same order as
    `windows_for_scene`.
    """
    pid_dir = DATA / str(pid)
    raw_path, json_path = find_raw_path(pid_dir, scene)
    if raw_path is None:
        return None

    # Load raw + bandpass filter all channels
    raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    raw = raw.drop(columns=['_trailing'], errors='ignore')
    raw['frame'] = raw['frame'].astype(int)

    # Load JSON for event timing consistency
    with open(json_path, 'r') as f:
        json_raw = json.load(f)
    json_frames = set(e['frameRef'] for e in json_raw['data'])
    # Align: keep only rows in raw that overlap JSON range
    first_frame = raw['frame'].min()

    # Bandpass every channel up front so each per-event slice is a cheap copy.
    filtered = np.zeros((len(raw), 9), dtype=np.float32)
    for i, ch in enumerate(EMG_COLS):
        filtered[:, i] = bandpass(raw[ch].values.astype(np.float64)).astype(np.float32)

    # Reference-frame alignment: the per-event `event_time_s` stored in
    # windows_for_scene was originally computed on the *inner-joined* (raw and
    # JSON) DataFrame, whose row 0 may not coincide with row 0 of the raw
    # stream. Using `max(first_raw_frame, min_json_frame)` reproduces that
    # join's origin within 1 sample, so absolute event frames here match the
    # ones used by run_pipeline.py.
    if len(json_frames) > 0:
        json_min = min(json_frames)
        ref_frame = max(first_frame, json_min)
    else:
        ref_frame = first_frame

    feature_rows = []
    for w in windows_for_scene:
        # Locate the event in the raw 1000 Hz frame index, then carve out the
        # canonical -2 s..+5 s window.
        event_frame_abs = ref_frame + int(w['event_time_s'] * FS)
        w_start = event_frame_abs - WINDOW_PRE_MS
        w_end = event_frame_abs + WINDOW_POST_MS
        mask = (raw['frame'] >= w_start) & (raw['frame'] <= w_end)
        idx = np.where(mask.values)[0]
        if len(idx) < FS * 5:                   # require >=5 s of data
            feature_rows.append(None)
            continue
        seg = filtered[idx]                     # (T, 9)
        # Split the window at T_event (proportional to PRE/(PRE+POST)).
        pre_end = max(1, int(len(seg) * WINDOW_PRE_MS / (WINDOW_PRE_MS + WINDOW_POST_MS)))
        pre = seg[:pre_end]                     # baseline segment
        post = seg[pre_end:]                    # response segment

        # Compute the 17-dim per-channel feature vector on each of pre/post/full,
        # then concatenate: 3 segments * 9 channels * 17 features = 459.
        feats = []
        for segment in [pre, post, seg]:
            ch_feats = []
            for c in range(9):
                ch_feats.append(spectral_feats_channel(segment[:, c]))
            feats.append(np.concatenate(ch_feats))
        feature_rows.append(np.concatenate(feats))

    return feature_rows


def main():
    print('Loading event windows...', flush=True)
    with open(PROC / 'event_windows_all.pkl', 'rb') as f:
        windows = pickle.load(f)
    print(f'  {len(windows)} windows', flush=True)

    # Group windows by (participant, scene)
    from collections import defaultdict
    by_ps = defaultdict(list)
    for i, w in enumerate(windows):
        by_ps[(w['participant_id'], w['scene'])].append((i, w))

    all_spec = [None] * len(windows)
    total_groups = len(by_ps)
    print(f'Processing {total_groups} (participant, scene) groups...', flush=True)
    t0 = time.time()
    errors = []

    for k, ((pid, scene), items) in enumerate(sorted(by_ps.items()), 1):
        t_start = time.time()
        try:
            indices = [i for i, _ in items]
            ws = [w for _, w in items]
            feats = extract_features_for_participant(pid, scene, ws)
            if feats is None:
                errors.append((pid, scene, 'raw file not found'))
                print(f'  [{k}/{total_groups}] {pid} {scene}: SKIP (no raw)', flush=True)
                continue
            n_ok = 0
            for idx, f in zip(indices, feats):
                all_spec[idx] = f
                if f is not None:
                    n_ok += 1
            elapsed_tot = time.time() - t0
            print(f'  [{k}/{total_groups}] {pid} {scene}: {n_ok}/{len(items)} windows, '
                  f'{time.time()-t_start:.1f}s (tot {elapsed_tot:.0f}s)', flush=True)
        except Exception as e:
            errors.append((pid, scene, str(e)))
            print(f'  [{k}/{total_groups}] {pid} {scene}: ERROR {e}', flush=True)

    # Attach to windows and save
    n_ok = sum(1 for f in all_spec if f is not None)
    print(f'\nOK: {n_ok}/{len(windows)} windows got spectral features', flush=True)

    for i, w in enumerate(windows):
        w['spec_features'] = all_spec[i]  # may be None

    out = PROC / 'event_windows_spectral.pkl'
    with open(out, 'wb') as f:
        pickle.dump(windows, f)
    print(f'saved -> {out}', flush=True)

    if errors:
        pd.DataFrame(errors, columns=['pid','scene','err']).to_csv(PROC/'spectral_errors.csv', index=False)
        print(f'errors -> {PROC}/spectral_errors.csv ({len(errors)} entries)', flush=True)

    print(f'TOTAL: {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'Feature names per channel ({N_FEATS_PER_CH}): {FEAT_NAMES_PER_CHANNEL}', flush=True)


if __name__ == '__main__':
    main()
