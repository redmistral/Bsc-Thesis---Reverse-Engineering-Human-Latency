"""
run_pipeline.py — Stage 1 of the analysis pipeline (Section 3.4 of the thesis).

PURPOSE
-------
This script ingests the raw recordings released by Mavridou et al. (2025) and
produces the cached event-window data structure that every subsequent script in
this submission depends on. Specifically, for each participant and each of the
two affective VR scenes (positive, negative) it:

  1. Loads the three raw streams that were recorded simultaneously:
       - `*_raw.txt`     : 1000 Hz multichannel sensor stream
                            (9 facial-EMG channels, 2 heart-rate channels, IMU,
                             accelerometer/magnetometer/gyroscope).
       - `*.json`        : per-frame event annotations exported by the VR engine,
                            with continuously-computed valence/arousal estimates,
                            facial-expression intensities, and the EventID list
                            of every active stimulus.
       - `Rating_Scene_*` : the participant's continuous self-report of valence
                            from the joystick-controlled rating dial.
  2. Aligns and merges the three streams on the shared `frame` index.
  3. Detects per-event onsets (the sample where a new EventID enters the active
     event-set) and classifies each event as high- or low-saliency using the
     stimulus taxonomy provided by the data collectors.
  4. Computes the per-channel EMG envelope using the four-stage pipeline
     described in Section 3.4: Butterworth bandpass (20-450 Hz) -> Hampel
     outlier replacement -> full-wave rectification -> low-pass envelope.
  5. Cuts a +/- window around every event onset (-2 s pre, +5 s post),
     downsamples 1000 Hz -> 10 Hz by simple decimation (every 100th sample),
     and stores the resulting per-event window dict.
  6. In addition, builds the two feature matrices used by the lag-fixed Random
     Forest (X_rf with the 200 ms lag from Mavridou et al. 2025) and the
     sequence model (X_lstm with a 50-sample = 5 s context window).

OUTPUT
------
processed_data/event_windows_all.pkl  : list[dict], one entry per event window.
                                        Consumed by extract_spectral_features.py
                                        and (transitively) by every training
                                        script in this submission.
processed_data/features_all.npz       : compact (X_rf, y_rf, X_lstm, y_lstm)
                                        arrays for legacy comparison purposes.
processed_data/meta_*.csv             : participant/scene/saliency labels per
                                        sample, kept aligned with the npz arrays.
processed_data/processing_stats.csv   : per-(participant, scene) diagnostic row
                                        with frame counts, rating coverage, and
                                        any dead EMG channels detected.

WHY THIS LIVES IN A SEPARATE STAGE
----------------------------------
The raw 1000 Hz streams are large (>1 GB across the cohort) and the bandpass +
Hampel + envelope pipeline is the most expensive step in the whole project.
Caching the per-event windows once and reusing them across model variants makes
each downstream experiment reproducible in seconds rather than hours.
"""
import pandas as pd
import numpy as np
import json
import pickle
import time
import sys
from pathlib import Path
from scipy.signal import butter, sosfilt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
# All raw sensor channels in the released dataset are sampled at 1000 Hz.
FS = 1000

# Absolute paths. The dataset is not redistributed with this submission; the
# `All Data/` directory must be placed alongside this script as described in
# README.md.
BASE_DIR = Path('/Users/selimcan/Desktop/Courses/data thesis')
ALL_DATA_DIR = BASE_DIR / 'All Data'
OUTPUT_DIR = BASE_DIR / 'processed_data'
OUTPUT_DIR.mkdir(exist_ok=True)

# Event-window definition (Section 3.4): -2 s baseline + 5 s response window.
# The 2 s pre-event interval is reused later in estimate_delays.py to compute
# the per-channel EMG baseline (mean + 3*std threshold for Delay A).
WINDOW_PRE_MS = 2000
WINDOW_POST_MS = 5000
DOWNSAMPLE_HZ = 10                    # post-pipeline working rate (100 ms steps)

# The 200 ms RF lag replicates the alignment that Mavridou et al. (2025) used
# in their original IEEE Access paper -- it is the exact assumption that this
# thesis empirically tests against the measured ~1.4 s cognitive-motor gap.
RF_LAG_MS = 200

# 50 samples at 10 Hz = 5 s of context, matching the post-event response window.
LSTM_SEQ_LEN = 50

# Column layout of the released `*_raw.txt` files. The trailing column is a
# residual whitespace artefact of the original CSV export and is dropped.
RAW_COLUMNS = (
    ['frame'] +
    [f'emg_ch{i}' for i in range(1, 10)] +
    ['heart_rate1', 'heart_rate_2'] +
    [f'imu_ch{i}' for i in range(1, 10)] +
    ['accel_x', 'accel_y', 'accel_z', 'magnetometer', 'gyro', 'system_related', '_trailing']
)
EMG_COLS = [f'emg_ch{i}' for i in range(1, 10)]
RATING_COLUMNS = ['frame', 'valence_self', 'arousal_self', 'time_s']

# ----- Saliency taxonomy (Mavridou et al., 2025 supplementary) ---------------
# The released dataset annotates every stimulus with a numeric EventID. The
# four sets below partition those IDs by *scene* and *intended salience*.
# "High-saliency" stimuli are emotionally intense set-pieces (e.g. fire,
# spider, dance robot); "low-saliency" stimuli are ambient props (e.g.
# scenery flowers, wall textures). This taxonomy drives the binary saliency
# classification target used in Sections 5.2.3 and 5.3.3.
NEGATIVE_HIGH_SALIENCY = {32, 33, 34, 35, 37, 38, 39, 52, 53, 54, 56}
NEGATIVE_LOW_SALIENCY = {30, 31, 36, 40, 41, 43, 44, 46, 47, 48, 49, 51, 55, 62}
POSITIVE_HIGH_SALIENCY = {1, 2, 3, 6, 7, 202}
POSITIVE_LOW_SALIENCY = {4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23}

# ============================================================
# FUNCTIONS
# ============================================================
def classify_saliency(event_ids_set, scene):
    """Map a set of newly-active EventIDs to {'high', 'low', 'none'} for one
    scene. If any of the new IDs intersect the high-saliency taxonomy for that
    scene, the event is labelled high; otherwise low. Empty sets -> 'none'.
    """
    if not event_ids_set:
        return 'none'
    if scene == 'positive':
        return 'high' if event_ids_set & POSITIVE_HIGH_SALIENCY else 'low'
    elif scene == 'negative':
        return 'high' if event_ids_set & NEGATIVE_HIGH_SALIENCY else 'low'
    return 'unknown'

def butterworth_bandpass(data, lowcut=20, highcut=450, fs=FS, order=4):
    """Stage 1 of the EMG pipeline (Section 3.4).

    A 4th-order Butterworth bandpass implemented as cascaded second-order
    sections (the SOS form is numerically stable at high orders). The 20-450 Hz
    range follows the Blumenthal et al. (2005) committee report on surface EMG:
    20 Hz removes movement artefacts and DC drift; 450 Hz cuts mains-related
    high-frequency noise while preserving the bulk of motor-unit activity.
    """
    nyq = fs / 2
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfilt(sos, data)

def hampel_filter(data, window_size=50, n_sigmas=3):
    """Stage 2 of the EMG pipeline (Section 3.4).

    Hampel outlier replacement (Bhowmik et al., 2017). For each sample we
    compute the local median and the local median-absolute-deviation (MAD,
    scaled by 1.4826 to make it a consistent estimator of the Gaussian std).
    Samples that lie more than `n_sigmas` MADs from the local median are
    flagged as transient spike artefacts (e.g. cable-tug, electrode contact
    noise) and replaced by the local median, preserving the surrounding
    waveform.

    A 50-sample (50 ms at 1000 Hz) window is short enough to follow real EMG
    bursts but long enough to give a stable median estimate.
    """
    s = pd.Series(data)
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = s.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))), raw=True
    )
    outlier_mask = np.abs(s - rolling_median) > n_sigmas * rolling_mad
    result = data.copy()
    result[outlier_mask] = rolling_median[outlier_mask].values
    # Returning the count of replacements lets the caller log per-channel noise.
    return result, outlier_mask.sum()

def emg_envelope(data, cutoff=10, fs=FS, order=2):
    """Stages 3-4 of the EMG pipeline (Section 3.4).

    Full-wave rectification (np.abs) converts the bipolar EMG signal to a
    non-negative activation magnitude; a 10 Hz low-pass Butterworth then
    smooths the rectified train into the slow muscle-activation envelope that
    psychophysiology actually scores. This is the "moving-RMS-equivalent"
    envelope referenced in the thesis.
    """
    rectified = np.abs(data)
    nyq = fs / 2
    sos = butter(order, cutoff / nyq, btype='low', output='sos')
    return sosfilt(sos, rectified)

def get_event_set(event_ids):
    """Coerce the JSON's per-frame `EventID` field (which may be a list, a
    NaN, or None depending on whether any stimulus is active in that frame)
    into a frozenset. Frozensets allow set-difference per-frame to detect
    *newly-arrived* events (see process_scene below).
    """
    if event_ids is None or (isinstance(event_ids, float) and np.isnan(event_ids)):
        return frozenset()
    return frozenset(event_ids)

def flatten_json_entry(entry):
    """One row of the JSON event log -> one flat record. The nested
    `Expression` and `Event` objects are unwrapped so the result can become a
    DataFrame and merge with the raw EMG by frame index.
    """
    return {
        'frame': entry['frameRef'],
        'arousal_computed': entry['Arousal'],
        'valence_computed': entry['Valence'],
        'smile_int': entry['Expression']['smileInt'],
        'frown_int': entry['Expression']['frownInt'],
        'surprise_int': entry['Expression']['surpriseInt'],
        'event_ids': entry['Event']['EventID'],
    }

def process_scene(raw_path, json_path, rating_path, scene_name, participant_id):
    """Ingest one (participant, scene) pair end-to-end.

    Returns
    -------
    merged       : DataFrame with rows = 1000 Hz samples and columns spanning
                   raw EMG, EMG envelopes, the joystick valence (forward-filled
                   to 1000 Hz), and the per-frame event metadata.
    event_onsets : the subset of `merged` rows where at least one *new* EventID
                   appears, with a `saliency` column attached.
    stats        : per-scene diagnostic dict (see end of function).
    """
    # ---- 1. Load the three raw streams ------------------------------------
    raw = pd.read_csv(raw_path, header=None, names=RAW_COLUMNS)
    raw = raw.drop(columns=['_trailing'], errors='ignore')
    raw['frame'] = raw['frame'].astype(int)

    with open(json_path, 'r') as f:
        json_raw = json.load(f)
    json_df = pd.DataFrame([flatten_json_entry(e) for e in json_raw['data']])
    json_df['frame'] = json_df['frame'].astype(int)

    # The rating file is whitespace-separated and headerless.
    rating = pd.read_csv(rating_path, sep=r'\s+', header=None, names=RATING_COLUMNS)
    rating['frame'] = rating['frame'].astype(int)

    # ---- 2. Align streams on the shared `frame` index ---------------------
    # Inner-join raw with the JSON event log (only frames with an event entry
    # survive). Then merge_asof attaches the nearest joystick rating within a
    # 50-frame (= 50 ms) tolerance, which handles the small jitter between the
    # 1000 Hz raw stream and the lower-rate rating sampler.
    merged = pd.merge(raw, json_df, on='frame', how='inner').sort_values('frame')
    merged = pd.merge_asof(
        merged,
        rating.sort_values('frame')[['frame', 'valence_self', 'arousal_self', 'time_s']],
        on='frame', direction='nearest', tolerance=50
    )
    merged['time_sec'] = (merged['frame'] - merged['frame'].min()) / 1000.0

    # ---- 3. Detect event onsets -------------------------------------------
    # An "event onset" is the first frame in which a given EventID enters the
    # active event-set, defined as set-difference against the previous frame.
    # This is more robust than equality checks because a stimulus may overlap
    # with neighbours, but its appearance is unique.
    merged['event_set'] = merged['event_ids'].apply(get_event_set)
    prev_sets = merged['event_set'].shift(1, fill_value=frozenset())
    merged['new_events'] = [c - p for c, p in zip(merged['event_set'], prev_sets)]
    merged['is_event_onset'] = merged['new_events'].apply(lambda s: len(s) > 0)
    event_onsets = merged[merged['is_event_onset']].copy()
    event_onsets['saliency'] = event_onsets['new_events'].apply(
        lambda s: classify_saliency(s, scene_name)
    )

    # ---- 4. Compute per-channel EMG envelopes -----------------------------
    # bandpass -> hampel -> rectify+lowpass, applied to each of the 9 channels.
    for ch in EMG_COLS:
        filtered = butterworth_bandpass(merged[ch].values)
        cleaned, _ = hampel_filter(filtered)
        merged[f'{ch}_envelope'] = emg_envelope(cleaned)

    # ---- 5. Flag dead channels --------------------------------------------
    # A channel whose envelope has near-zero std for an entire scene is almost
    # always an electrode that lost contact for the whole recording. We only
    # report it; removal happens downstream.
    dead_channels = []
    for ch in EMG_COLS:
        if merged[f'{ch}_envelope'].std() < 1e-8:
            dead_channels.append(ch)

    return merged, event_onsets, {
        'participant': participant_id,
        'scene': scene_name,
        'n_frames': len(merged),
        'duration_s': merged['time_sec'].max(),
        'n_events': len(event_onsets),
        'rating_coverage': merged['valence_self'].notna().mean(),
        'dead_channels': dead_channels,
    }

def extract_windows(merged, event_onsets, scene_name, participant_id):
    """Cut +/- windows around every event onset and store the per-event dict.

    Each window is 7 s wide (-2 s pre, +5 s post) and downsampled to 10 Hz by
    taking every 100th sample. We discard windows that are >20% incomplete in
    either time-coverage or rating-coverage: such windows usually correspond
    to events near the very start or end of a scene.
    """
    envelope_cols = [f'{ch}_envelope' for ch in EMG_COLS]
    windows = []
    for _, onset_row in event_onsets.iterrows():
        event_frame = onset_row['frame']
        event_time = onset_row['time_sec']
        new_events = sorted(onset_row['new_events'])
        saliency = onset_row['saliency']

        # Bound the window in raw-frame units (1000 Hz).
        w_start = event_frame - WINDOW_PRE_MS
        w_end = event_frame + WINDOW_POST_MS
        window = merged[(merged['frame'] >= w_start) & (merged['frame'] <= w_end)]

        # Reject events too close to scene boundaries (incomplete window).
        if len(window) < (WINDOW_PRE_MS + WINDOW_POST_MS) * 0.8:
            continue

        # 1000 Hz -> 10 Hz: take every 100th sample (simple decimation; the
        # envelope is already low-pass-smoothed below 10 Hz, so anti-alias is
        # redundant here).
        window_ds = window.iloc[::100].copy()
        emg_features = window_ds[envelope_cols].values
        # Forward+backward fill the joystick valence to repair short gaps that
        # arise from the rating sampler running asynchronously to the raw
        # stream.
        valence_target = window_ds['valence_self'].ffill().bfill().values
        time_relative = (window_ds['frame'].values - event_frame) / 1000.0

        # If even after ffill/bfill more than 20% of the window has no valid
        # rating, drop the window outright -- the target is too sparse to use.
        if np.isnan(valence_target).sum() > len(valence_target) * 0.2:
            continue

        windows.append({
            'participant_id': participant_id,
            'scene': scene_name,
            'event_ids': str(new_events),
            'saliency': saliency,
            'event_time_s': event_time,
            'emg_features': emg_features,
            'valence_target': valence_target,
            'time_relative_s': time_relative,
            'n_samples': len(window_ds),
        })
    return windows

def build_rf_features(windows, lag_ms=RF_LAG_MS, sr=DOWNSAMPLE_HZ):
    """Per-time-step (X_t, y_{t+lag}) pairs for the static-200 ms RF baseline.

    This *exactly* reproduces the alignment used by Mavridou et al. (2025):
    each row is a 9-channel EMG envelope sample, predicting the joystick
    valence 200 ms later (= 2 samples ahead at 10 Hz). No temporal context.
    """
    lag_samples = int(lag_ms / 1000 * sr)
    X_list, y_list, meta_list = [], [], []
    for w in windows:
        emg, val = w['emg_features'], w['valence_target']
        for t in range(len(val) - lag_samples):
            X_list.append(emg[t])
            y_list.append(val[t + lag_samples])
            meta_list.append({'participant': w['participant_id'], 'scene': w['scene'], 'saliency': w['saliency']})
    return np.array(X_list), np.array(y_list), meta_list

def build_lstm_features(windows, seq_len=LSTM_SEQ_LEN):
    """Sliding-window (X_{t-seq+1..t}, y_t) pairs for sequence models.

    Each X is a (seq_len, 9) tensor of 9-channel EMG envelope, and each y is
    the *current* joystick valence. Unlike the RF baseline, no fixed lag is
    imposed -- the model is free to learn its own variable alignment, which is
    the central question of SQ1.
    """
    X_list, y_list, meta_list = [], [], []
    for w in windows:
        emg, val = w['emg_features'], w['valence_target']
        for t in range(len(val) - seq_len):
            X_list.append(emg[t:t + seq_len])
            y_list.append(val[t + seq_len])
            meta_list.append({'participant': w['participant_id'], 'scene': w['scene'], 'saliency': w['saliency']})
    return np.array(X_list), np.array(y_list), meta_list

def find_scene_files(participant_dir, participant_id):
    """Locate the (raw, json, rating) triple for each scene in one
    participant's directory. Returns a dict keyed by 'positive'/'negative';
    scenes with any missing file are silently omitted.

    Naming convention used by the data collectors:
      - positive scene = scene index 4 (filename contains 'POSITIVE')
      - negative scene = scene index 5 (filename contains 'negative')
    The third scene (neutral, index 3) was not redistributed as raw EMG for
    most participants, which is why the per-event delay analysis in this
    thesis is limited to positive vs negative.
    """
    scenes = {}
    for scene_key, pattern, rating_scene in [
        ('positive', '*POSITIVE*_03_*', '4'),
        ('negative', '*negative*_03_*', '5'),
    ]:
        raw_files = list(participant_dir.glob(f'{pattern}_raw.txt'))
        json_files = list(participant_dir.glob(f'{pattern}.json'))
        rating_file = participant_dir / f'Rating_Scene_{rating_scene}_P_{participant_id}_.txt'

        if raw_files and json_files and rating_file.exists():
            scenes[scene_key] = {
                'raw': raw_files[0],
                'json': json_files[0],
                'rating': rating_file,
            }
    return scenes

# ============================================================
# MAIN
# ============================================================
def main():
    """Walk every participant directory, process all scenes, and persist the
    cached event-window list plus the legacy RF/LSTM feature matrices.
    """
    t_start = time.time()
    print('=' * 60, flush=True)
    print('Multi-Participant EMG Pipeline', flush=True)
    print('=' * 60, flush=True)

    # The cohort lives under `All Data/<pid>/`. One pilot participant (100172)
    # was kept in a sibling directory during development; we include it if
    # present so the script reproduces from either layout.
    participant_dirs = {}
    p172_dir = BASE_DIR / '100172'
    if p172_dir.exists():
        participant_dirs['100172'] = p172_dir

    if ALL_DATA_DIR.exists():
        for d in sorted(ALL_DATA_DIR.iterdir()):
            if d.is_dir() and d.name.isdigit():
                participant_dirs[d.name] = d

    total_p = len(participant_dirs)
    print(f'Found {total_p} participant directories', flush=True)

    all_windows = []
    all_stats = []
    errors = []
    skipped = []

    # Per-participant loop. Failures on a single (pid, scene) pair are logged
    # and skipped rather than aborting the whole run -- with ~150+ participants
    # one bad file should not block the cohort.
    for idx, (pid, pdir) in enumerate(sorted(participant_dirs.items()), 1):
        scenes = find_scene_files(pdir, pid)
        if not scenes:
            skipped.append(pid)
            continue

        for scene_name, files in scenes.items():
            try:
                t0 = time.time()
                merged, event_onsets, stats = process_scene(
                    files['raw'], files['json'], files['rating'],
                    scene_name, pid
                )
                windows = extract_windows(merged, event_onsets, scene_name, pid)
                dt = time.time() - t0

                all_windows.extend(windows)
                all_stats.append(stats)

                print(f'[{idx}/{total_p}] {pid} {scene_name:>8s}: '
                      f'{stats["n_events"]:>2d} events, {len(windows):>2d} windows, '
                      f'{dt:.1f}s', flush=True)

            except Exception as e:
                errors.append({'participant': pid, 'scene': scene_name, 'error': str(e)})
                print(f'[{idx}/{total_p}] {pid} {scene_name:>8s}: ERROR — {e}', flush=True)

    elapsed = time.time() - t_start

    # ---- Cohort-level summary ---------------------------------------------
    print(f'\n{"=" * 60}', flush=True)
    print(f'SUMMARY ({elapsed/60:.1f} min)', flush=True)
    print(f'{"=" * 60}', flush=True)
    participants_ok = sorted(set(s['participant'] for s in all_stats))
    print(f'Processed: {len(participants_ok)}/{total_p}  |  Skipped: {len(skipped)}  |  Errors: {len(errors)}', flush=True)
    if errors:
        for e in errors[:10]:
            print(f'  ERR {e["participant"]} {e["scene"]}: {e["error"][:80]}', flush=True)

    n_pos = sum(1 for w in all_windows if w['scene'] == 'positive')
    n_neg = sum(1 for w in all_windows if w['scene'] == 'negative')
    n_hi = sum(1 for w in all_windows if w['saliency'] == 'high')
    n_lo = sum(1 for w in all_windows if w['saliency'] == 'low')
    print(f'Windows: {len(all_windows)} (pos={n_pos}, neg={n_neg}, high={n_hi}, low={n_lo})', flush=True)

    if not all_windows:
        print('ERROR: No windows extracted.', flush=True)
        return

    # ---- Build the legacy RF/LSTM feature matrices ------------------------
    # Kept for backwards compatibility with earlier notebooks. The training
    # scripts in this submission read `event_windows_all.pkl` directly and
    # do not depend on these npz arrays.
    print('Building feature matrices...', flush=True)
    X_rf, y_rf, meta_rf = build_rf_features(all_windows)
    X_lstm, y_lstm, meta_lstm = build_lstm_features(all_windows)
    print(f'  RF: {X_rf.shape}  LSTM: {X_lstm.shape}', flush=True)
    print(f'  NaN: RF={np.isnan(X_rf).sum()+np.isnan(y_rf).sum()}, LSTM={np.isnan(X_lstm).sum()+np.isnan(y_lstm).sum()}', flush=True)

    # ---- Persist all artefacts --------------------------------------------
    print('Saving...', flush=True)
    np.savez_compressed(str(OUTPUT_DIR / 'features_all.npz'), X_rf=X_rf, y_rf=y_rf, X_lstm=X_lstm, y_lstm=y_lstm)
    pd.DataFrame(meta_rf).to_csv(str(OUTPUT_DIR / 'meta_rf_all.csv'), index=False)
    pd.DataFrame(meta_lstm).to_csv(str(OUTPUT_DIR / 'meta_lstm_all.csv'), index=False)
    # event_windows_all.pkl is the *primary* output: every downstream script
    # reads it (or its spectral-augmented descendant).
    with open(str(OUTPUT_DIR / 'event_windows_all.pkl'), 'wb') as f:
        pickle.dump(all_windows, f)
    pd.DataFrame(all_stats).to_csv(str(OUTPUT_DIR / 'processing_stats.csv'), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(str(OUTPUT_DIR / 'processing_errors.csv'), index=False)

    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f'  {f.name} ({f.stat().st_size/1024:.1f} KB)', flush=True)

    print(f'\nDONE — {len(participants_ok)} participants, {len(all_windows)} windows', flush=True)

if __name__ == '__main__':
    main()
