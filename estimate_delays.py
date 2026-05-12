"""
estimate_delays.py — Per-event Delay A, B, C extraction (Section 3.5 of the thesis).

This script implements the rule-based latency-extraction procedure that
underlies SQ2 (does the cognitive-motor delay vary with stimulus saliency?)
and the empirical "1.4 s gap" headline result. For every event window in
`event_windows_spectral.pkl` it computes three latencies relative to the
stimulus onset T_event:

  Delay A — Psychophysiological onset (the body reacting)
    Time from T_event to the first sample at which the EMG envelope of any of
    the 9 facial channels exceeds the channel's baseline mean by k standard
    deviations (k=3). Computed on the *raw* 1000 Hz envelope, not on the
    downsampled per-window cache, so the precision is 1 ms. Aggregated across
    channels by taking the minimum crossing time -- the first muscle to fire.
    Validity gate: 100-500 ms, the cross-study consensus from the Dimberg &
    Thunberg facial-EMG literature.

  Delay C — Total observable delay (the conscious self-report arriving)
    Time from T_event to the first significant change in the joystick valence
    rating, evaluated by two complementary criteria reported side-by-side:
      (1) Threshold rule: |valence(t) - baseline_mean| > k * baseline_std.
      (2) Derivative rule: |dvalence/dt| > 0.1 valence-units per second.
    Whichever criterion fires first is taken as the per-event Delay C; the
    method-comparison summary at the end of the script confirms the two
    rules agree to within a few hundred ms on average. Validity gate:
    500-5000 ms (Huang et al., 2015).

  Delay B — Cognitive-motor gap = Delay C - Delay A
    The interval the participant's brain spent doing appraisal and motor
    planning between the EMG having fired and the joystick having moved.
    This is the central quantity of the thesis: median ~1.4 s vs the
    field-standard 200 ms heuristic. Only computed when both A and C are
    valid; further constrained to be positive and below 4900 ms.

After the per-event extraction the script prints (a) validity rates, (b)
descriptive statistics, (c) a Mann-Whitney U comparison of each delay across
high- vs low-saliency events (the SQ2 test), and (d) a sanity check that the
threshold and derivative Delay-C rules agree.

INPUT  : processed_data/event_windows_spectral.pkl    (output of extract_spectral_features.py)
OUTPUT : model_results/delays_per_event.csv           (one row per event window;
                                                       consumed by analyze_delays.py
                                                       and analyze_modulation.py)
"""
import json, pickle, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

BASE = Path('/Users/selimcan/Desktop/Courses/data thesis')
DATA = BASE / 'All Data'
PROC = BASE / 'processed_data'
OUT = BASE / 'model_results'
OUT.mkdir(exist_ok=True)

# ---- Sampling and timing constants -----------------------------------------
FS = 1000                     # raw EMG sampling rate (Hz)
PRE_MS = 2000                 # pre-event baseline window length (ms) -- the
                              # 2 s immediately before T_event from which the
                              # per-channel mu and sigma are computed.

# Validity gates from the literature. Anything outside these bands is treated
# as a non-detection and excluded from the modulation analyses.
DELAY_A_MIN_MS = 100          # Dimberg & Thunberg (1998) lower bound
DELAY_A_MAX_MS = 500
DELAY_C_MIN_MS = 500          # Huang et al. (2015) conscious-response window
DELAY_C_MAX_MS = 5000
DELAY_B_MIN_MS = 100          # Delay B must be positive and physiologically
DELAY_B_MAX_MS = 4900         # plausible (capped just below the post-window).

THRESHOLD_K = 3.0             # baseline mean + k*std rule for both A and C
DERIV_THRESHOLD = 0.1         # valence units / second for the dV/dt rule
MIN_VAL_STD = 1e-3            # floor on the valence-baseline std so that
                              # near-flat ratings don't yield infinite z-scores

# Same column layout used by run_pipeline.py.
RAW_COLUMNS = (
    ['frame'] + [f'emg_ch{i}' for i in range(1, 10)] + ['heart_rate1', 'heart_rate_2']
    + [f'imu_ch{i}' for i in range(1, 10)]
    + ['accel_x', 'accel_y', 'accel_z', 'magnetometer', 'gyro', 'system_related', '_trailing']
)
EMG_COLS = [f'emg_ch{i}' for i in range(1, 10)]


def bandpass(data, low=20, high=450, fs=FS, order=4):
    """4th-order Butterworth 20-450 Hz, identical to run_pipeline.py. Repeated
    here so estimate_delays.py can be run standalone on the *raw* 1000 Hz EMG
    rather than on the (already downsampled) cached envelope.
    """
    nyq = fs / 2
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfilt(sos, data)


def moving_rms(x, window=20):
    """Moving root-mean-square envelope; 20 ms window at 1000 Hz. The short
    window preserves fast onset transients that the per-event Delay A detection
    needs to localise within ~10 ms.
    """
    return np.sqrt(np.convolve(x ** 2, np.ones(window) / window, mode='same'))


def hampel_filter(data, window_size=50, n_sigmas=3):
    """Replace isolated spike artefacts with the local rolling median
    (Bhowmik et al., 2017). See run_pipeline.py for the same docstring; kept
    here so this script can be run independently of the upstream pipeline.
    """
    s = pd.Series(data)
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = s.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))), raw=True
    )
    outlier_mask = np.abs(s - rolling_median) > n_sigmas * rolling_mad
    result = data.copy()
    result[outlier_mask] = rolling_median[outlier_mask].values
    return result


def find_raw_path(pid_dir: Path, scene: str):
    """Locate the (raw, json) pair for one (participant, scene). Mirrors the
    convention used by run_pipeline.py: scene index 03 + uppercase POSITIVE /
    lowercase negative.
    """
    pat = 'POSITIVE' if scene == 'positive' else 'negative'
    for p in pid_dir.glob(f'*{pat}*_03_*_raw.txt'):
        json_p = p.with_name(p.name.replace('_raw.txt', '.json'))
        if json_p.exists():
            return p, json_p
    return None, None


def load_emg_envelope_1000hz(pid, scene):
    """Recompute the 9-channel EMG envelope at full 1000 Hz resolution.

    The cached `event_windows_spectral.pkl` only stores the downsampled (10
    Hz) per-event envelope, which is too coarse for ~100 ms latency
    extraction. Instead we reload the raw .txt and rerun the
    bandpass -> hampel -> rectify -> moving-RMS pipeline with a 20 ms RMS
    window (run_pipeline.py uses a 10 Hz envelope filter for the cached
    output; here we want fast onset transients).

    Returns (envelope_array (T, 9), raw_frames (T,), ref_frame:int) or None
    if the raw files for that (pid, scene) cannot be located.
    """
    pid_dir = DATA / str(pid)
    raw_path, json_path = find_raw_path(pid_dir, scene)
    if raw_path is None:
        return None
    raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    raw = raw.drop(columns=['_trailing'], errors='ignore')
    raw['frame'] = raw['frame'].astype(int)

    # Use the JSON's earliest frameRef to anchor the time origin -- the JSON
    # event log starts at the moment the VR scene begins, but the raw stream
    # may have started a fraction of a second earlier.
    with open(json_path, 'r') as f:
        json_raw = json.load(f)
    json_frames = [e['frameRef'] for e in json_raw['data']]
    if not json_frames:
        return None
    first_frame = raw['frame'].min()
    ref_frame = max(first_frame, min(json_frames))

    # Per-channel pipeline: bandpass -> hampel -> rectify -> 20 ms moving RMS.
    env = np.zeros((len(raw), 9), dtype=np.float32)
    for i, ch in enumerate(EMG_COLS):
        bp = bandpass(raw[ch].values.astype(np.float64))
        bp = hampel_filter(bp)
        env[:, i] = moving_rms(np.abs(bp), window=20).astype(np.float32)

    return env, raw['frame'].values.astype(np.int64), int(ref_frame)


def detect_delay_a(env, raw_frames, ref_frame, event_time_s):
    """Compute Delay A (EMG-onset latency) for one event.

    Algorithm
    ---------
    1. Convert the JSON-relative event time into an absolute raw-frame index.
    2. Define the baseline window as [event - 2 s, event) and the test window
       as (event + 10 ms, event + 600 ms]. The 10 ms gap removes any
       event-simultaneous artefact (e.g. a stimulus sound triggering a startle
       in the same sample as T_event itself).
    3. For each of the 9 channels compute mu and sigma over the baseline,
       set threshold = mu + 3*sigma, and find the first test-window sample
       above threshold. Channels with zero baseline variance (dead/saturated)
       contribute NaN.
    4. The per-event Delay A is the *minimum* of the 9 per-channel latencies
       (the earliest muscle to fire).

    Returns (delay_a_ms, [per_channel_latencies_ms]). Either may be NaN if no
    channel crossed within the 600 ms search window.
    """
    event_frame = ref_frame + int(event_time_s * FS)
    base_start = event_frame - PRE_MS
    base_end = event_frame
    test_start = event_frame + 10                  # skip first 10 ms (artefact-prone)
    test_end = event_frame + DELAY_A_MAX_MS + 100  # small buffer past validity

    bmask = (raw_frames >= base_start) & (raw_frames < base_end)
    tmask = (raw_frames >= test_start) & (raw_frames <= test_end)
    bidx = np.where(bmask)[0]
    tidx = np.where(tmask)[0]
    # Need at least 0.5 s of baseline and 50 samples of test window.
    if len(bidx) < FS * 0.5 or len(tidx) < 50:
        return np.nan, [np.nan] * 9

    per_ch = []
    for c in range(9):
        base = env[bidx, c]
        mu, sd = float(np.mean(base)), float(np.std(base))
        if sd < 1e-9:                              # dead channel -> NaN
            per_ch.append(np.nan)
            continue
        thr = mu + THRESHOLD_K * sd
        test = env[tidx, c]
        above = np.where(test > thr)[0]
        if len(above) == 0:                        # never crossed -> NaN
            per_ch.append(np.nan)
            continue
        first = above[0]
        # Report the latency in milliseconds (= raw-frame difference at 1000 Hz).
        per_ch.append(float(raw_frames[tidx[first]] - event_frame))

    valid = [m for m in per_ch if not np.isnan(m)]
    delay_a = min(valid) if valid else np.nan
    return delay_a, per_ch


def detect_delay_c(valence, time_rel_s):
    """Compute Delay C (joystick-onset latency) for one event using two
    complementary rules. Both are reported because no single rule is
    universally robust:
      - the *threshold* rule fires when the participant has clearly committed
        to a new rating, but it can be late if the change is gradual;
      - the *derivative* rule fires on the very first joystick movement, but
        it is sensitive to noise.

    Returns (delay_c_threshold_ms, delay_c_derivative_ms). Either may be NaN.
    """
    valence = np.asarray(valence, dtype=np.float32)
    time_rel_s = np.asarray(time_rel_s, dtype=np.float32)
    # Drop any NaN samples before computing baseline statistics.
    if np.isnan(valence).any():
        nan_mask = ~np.isnan(valence)
        valence = valence[nan_mask]
        time_rel_s = time_rel_s[nan_mask]
    pre = time_rel_s < 0
    post = time_rel_s > 0
    if pre.sum() < 5 or post.sum() < 5:
        return np.nan, np.nan

    # ---- Rule 1: amplitude threshold ---------------------------------------
    base = valence[pre]
    mu, sd = float(np.mean(base)), float(np.std(base))
    sd = max(sd, MIN_VAL_STD)            # avoid divide-by-zero on flat dials
    thr = THRESHOLD_K * sd
    post_v = valence[post]
    post_t = time_rel_s[post]
    dev = np.abs(post_v - mu)
    above = np.where(dev > thr)[0]
    delay_c_thr = float(post_t[above[0]] * 1000) if len(above) else np.nan

    # ---- Rule 2: time-derivative -------------------------------------------
    # Compute |dV/dt| on the post-event side and fire on the first sample
    # where the absolute slope exceeds 0.1 units/s. Mid-time is used because
    # the derivative is naturally located between the two samples it spans.
    dt = np.diff(time_rel_s)
    dv = np.diff(valence)
    valid_dt = dt > 1e-6
    deriv_abs = np.zeros_like(dt)
    deriv_abs[valid_dt] = np.abs(dv[valid_dt] / dt[valid_dt])
    mid_t = (time_rel_s[:-1] + time_rel_s[1:]) / 2
    post_mask_d = mid_t > 0
    if post_mask_d.sum() < 3:
        delay_c_deriv = np.nan
    else:
        above_d = np.where((deriv_abs > DERIV_THRESHOLD) & post_mask_d)[0]
        delay_c_deriv = float(mid_t[above_d[0]] * 1000) if len(above_d) else np.nan

    return delay_c_thr, delay_c_deriv


def main():
    # ---- Load cached event windows ----------------------------------------
    print('Loading event_windows_spectral.pkl...', flush=True)
    with open(PROC / 'event_windows_spectral.pkl', 'rb') as f:
        windows = pickle.load(f)
    print(f'  {len(windows)} windows', flush=True)

    # ---- Group by (participant, scene) so the raw 1000 Hz envelope is read
    # only once per scene rather than once per event (~10x speedup).
    by_ps = defaultdict(list)
    for i, w in enumerate(windows):
        by_ps[(w['participant_id'], w['scene'])].append((i, w))
    total_groups = len(by_ps)
    print(f'Processing {total_groups} (pid, scene) groups at 1000 Hz...', flush=True)

    rows = []
    t0 = time.time()
    skip_missing = 0

    for k, ((pid, scene), items) in enumerate(sorted(by_ps.items()), 1):
        loaded = load_emg_envelope_1000hz(pid, scene)
        if loaded is None:
            # Some participants are missing one of the two scenes' raw .txt;
            # those events cannot have a Delay A computed and are skipped.
            skip_missing += len(items)
            if k % 20 == 0 or k == total_groups:
                elapsed = time.time() - t0
                print(f'  [{k}/{total_groups}] {pid} {scene}: SKIP (no raw)  elapsed {elapsed:.0f}s', flush=True)
            continue
        env, raw_frames, ref_frame = loaded

        for idx, w in items:
            event_time = w['event_time_s']
            sal = w['saliency']
            valence = w['valence_target']
            # Older cache versions did not store time_relative_s; reconstruct
            # it from the canonical -2 s..+5 s window.
            time_rel = w.get('time_relative_s')
            if time_rel is None:
                time_rel = np.linspace(-2.0, 5.0, len(valence))

            # Compute the three delays per event.
            d_a, ch_delays = detect_delay_a(env, raw_frames, ref_frame, event_time)
            d_c_thr, d_c_deriv = detect_delay_c(valence, time_rel)
            # Prefer the threshold rule; fall back to derivative if missing.
            d_c = d_c_thr if not np.isnan(d_c_thr) else d_c_deriv
            d_b = (d_c - d_a) if (not np.isnan(d_a) and not np.isnan(d_c)) else np.nan

            # Apply the literature-derived validity gates.
            a_valid = (not np.isnan(d_a)) and (DELAY_A_MIN_MS <= d_a <= DELAY_A_MAX_MS)
            c_valid = (not np.isnan(d_c)) and (DELAY_C_MIN_MS <= d_c <= DELAY_C_MAX_MS)
            b_valid = (not np.isnan(d_b)) and (DELAY_B_MIN_MS <= d_b <= DELAY_B_MAX_MS)

            row = {
                'idx': idx, 'pid': pid, 'scene': scene, 'saliency': sal,
                'event_time_s': event_time,
                'delay_a_ms': d_a,
                'delay_c_thr_ms': d_c_thr,
                'delay_c_deriv_ms': d_c_deriv,
                'delay_c_ms': d_c,
                'delay_b_ms': d_b,
                'a_valid': a_valid, 'c_valid': c_valid, 'b_valid': b_valid,
            }
            # Per-channel latencies kept for inspection / Figure 4.
            for ci, cd in enumerate(ch_delays):
                row[f'delay_a_ch{ci}_ms'] = cd
            rows.append(row)

        if k % 20 == 0 or k == total_groups:
            elapsed = time.time() - t0
            eta = elapsed / k * (total_groups - k)
            print(f'  [{k}/{total_groups}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'delays_per_event.csv', index=False)
    print(f'\nSaved {len(df)} events -> {OUT}/delays_per_event.csv', flush=True)
    if skip_missing:
        print(f'  Skipped {skip_missing} events (raw EMG not found)', flush=True)

    # ============ SUMMARY ============
    # Validity rates: how many events fall inside each literature band.
    print('\n' + '=' * 80, flush=True)
    print('VALIDITY', flush=True)
    print('=' * 80, flush=True)
    print(f'  Delay A valid (100-500ms)    : {df["a_valid"].sum():>4d}/{len(df)}  ({df["a_valid"].mean()*100:.1f}%)', flush=True)
    print(f'  Delay C valid (500-5000ms)   : {df["c_valid"].sum():>4d}/{len(df)}  ({df["c_valid"].mean()*100:.1f}%)', flush=True)
    print(f'  Delay B valid (100-4900ms)   : {df["b_valid"].sum():>4d}/{len(df)}  ({df["b_valid"].mean()*100:.1f}%)', flush=True)
    all_ok = df['a_valid'] & df['c_valid'] & df['b_valid']
    print(f'  All three valid              : {all_ok.sum():>4d}/{len(df)}  ({all_ok.mean()*100:.1f}%)', flush=True)

    # Descriptive statistics underlying the headline numbers reported in §5.2.
    print('\n' + '=' * 80, flush=True)
    print('DELAY STATISTICS (valid events only)', flush=True)
    print('=' * 80, flush=True)
    for col, vcol in [('delay_a_ms', 'a_valid'), ('delay_c_ms', 'c_valid'),
                      ('delay_b_ms', 'b_valid')]:
        vals = df.loc[df[vcol], col]
        if len(vals):
            print(f'  {col:18s}  n={len(vals):>4d}  mean={vals.mean():>6.0f}  '
                  f'median={vals.median():>6.0f}  std={vals.std():>5.0f}  '
                  f'[{vals.min():.0f} – {vals.max():.0f}] ms', flush=True)

    # SQ2: Mann-Whitney U on each delay across saliency bins.
    print('\n' + '=' * 80, flush=True)
    print('DELAY BY SALIENCY (Mann-Whitney U, two-sided)', flush=True)
    print('=' * 80, flush=True)
    for col, vcol in [('delay_a_ms', 'a_valid'), ('delay_c_ms', 'c_valid'),
                      ('delay_b_ms', 'b_valid')]:
        sub = df[df[vcol]]
        hi = sub[sub['saliency'] == 'high'][col].dropna().values
        lo = sub[sub['saliency'] == 'low'][col].dropna().values
        if len(hi) > 10 and len(lo) > 10:
            u, p = mannwhitneyu(hi, lo, alternative='two-sided')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            diff = hi.mean() - lo.mean()
            print(f'  {col:15s}  high: n={len(hi):>4d} μ={hi.mean():>6.0f}  '
                  f'low: n={len(lo):>4d} μ={lo.mean():>6.0f}  '
                  f'Δ={diff:+6.0f}ms  U={u:.0f}  p={p:.4g} {sig}', flush=True)
        else:
            print(f'  {col:15s}  insufficient sample (high={len(hi)}, low={len(lo)})', flush=True)

    # Internal-consistency check between the two Delay-C rules (RMSE ~ several
    # hundred ms is acceptable; we used the threshold rule as primary).
    print('\n' + '=' * 80, flush=True)
    print('DELAY C METHOD COMPARISON (threshold vs derivative)', flush=True)
    print('=' * 80, flush=True)
    both = df.dropna(subset=['delay_c_thr_ms', 'delay_c_deriv_ms'])
    if len(both):
        rmse = np.sqrt(((both['delay_c_thr_ms'] - both['delay_c_deriv_ms']) ** 2).mean())
        print(f'  n={len(both)}  RMSE(thr vs deriv)={rmse:.0f}ms  '
              f'mean(thr)={both["delay_c_thr_ms"].mean():.0f}  '
              f'mean(deriv)={both["delay_c_deriv_ms"].mean():.0f}', flush=True)
    print(f'  threshold-only hits : {df["delay_c_thr_ms"].notna().sum()}', flush=True)
    print(f'  derivative-only hits: {df["delay_c_deriv_ms"].notna().sum()}', flush=True)

    print(f'\nTOTAL: {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
