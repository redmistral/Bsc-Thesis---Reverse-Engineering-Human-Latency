# Bachelor Thesis Submission — Reverse-Engineering Human Latency

**Author:** Selim Can Mutlu (2119179)
**Programme:** BSc Cognitive Science & Artificial Intelligence — Tilburg University
**Supervisor:** dr. Ifigeneia Mavridou
**Submission target:** May 22, 2026

---

## ▶ Start here

The entry point for the code submission is the Jupyter notebook **[`code/pipeline_walkthrough.ipynb`](code/pipeline_walkthrough.ipynb)**. It walks through every `.py` file in the order in which they run, explains *what* each script does and *why* it is at that point in the pipeline, and loads the pre-computed results so the narrative is fast to read end-to-end. Open it in JupyterLab / VS Code / Colab.

If you only want a static text overview without the notebook, the **"Reproducing the results"** section below lists the same execution order with a one-line description per script.

Every `.py` file in `code/` carries a full header docstring + inline comments tying each step to the thesis section it implements; the notebook is the connective tissue that explains how the scripts compose.

---

## Folder layout

```
submission/
├── README.md                  ← this file
├── thesis/                    ← LaTeX source for the thesis report
│   ├── main.tex               ← main document (compile with pdflatex + bibtex)
│   ├── references.bib         ← bibliography (BibTeX)
│   ├── frontmatter.tex        ← title-page macros (do not edit)
│   ├── logo.eps               ← Tilburg University logo
│   ├── logo-eps-converted-to.pdf
│   ├── Example_plot.png       ← unused template asset (kept for compile compatibility)
│   ├── spam.png               ← unused template asset (kept for compile compatibility)
│   └── plots/                 ← seven figures referenced from main.tex
│       ├── delays_overall_distribution.png
│       ├── delays_a_vs_c_scatter.png
│       ├── delays_by_saliency.png
│       ├── delay_b_modulation_summary.png
│       ├── sq1_v4_comparison.png
│       ├── sq1_v3_per_subject_scatter.png
│       └── window_variations_comparison.png
├── code/                      ← analysis and training scripts
│   ├── pipeline_walkthrough.ipynb ← ★ START HERE: notebook that documents the full pipeline
│   ├── run_pipeline.py        ← raw .txt + .json preprocessing → event_windows_all.pkl (Section 3.4)
│   ├── extract_spectral_features.py ← spectral features per event → event_windows_spectral.pkl (Section 3.4)
│   ├── estimate_delays.py     ← Delay A / B / C extraction (Section 3.5)
│   ├── analyze_delays.py      ← per-event delay distributions and saliency tests (Section 5.2)
│   ├── analyze_modulation.py  ← per-subject delay modulation by demographics + personality (Section 5.2.4)
│   ├── train_lstm_v4_full.py  ← BiLSTM seq2seq with multimodal features and CCC loss (Section 5.3)
│   ├── train_lstm_multiband.py← multi-band BiLSTM, used for Figure 5 (per-subject scatter)
│   ├── train_classification.py← window-level binary valence-direction classifier (Section 5.3.2)
│   ├── train_spectral.py      ← spectral feature ablation, source of saliency AUC = 0.729
│   ├── train_spectral_slope.py← shape-feature (slope/AUC) starting point cited in §6.4 future work
│   ├── train_svm_variations.py← SVM × 4 delay-variation × 3 feature-set ablation (Section 5.3.3)
│   └── results/               ← machine-readable summaries that back the tables in the thesis
│       ├── delays_per_event.csv
│       ├── delays_per_subject.csv
│       ├── delays_summary.txt
│       ├── delays_modulation_stats.csv
│       ├── delays_modulation_summary.txt
│       ├── metrics_spectral.json
│       ├── metrics_sq1_v{2,3,4}.json
│       ├── metrics_classification.json
│       └── metrics_window_variations.json
```

---

## Compiling the thesis

Open `thesis/main.tex` in any LaTeX editor (Overleaf is the simplest) and compile with `pdflatex → bibtex → pdflatex → pdflatex`. The document depends on the `apacite` and `classicthesis` packages; both are standard on Overleaf and TeX Live.

To compile locally:

```bash
cd submission/thesis
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The output `main.pdf` is the submitted thesis.

---

## Reproducing the results

The dataset (Mavridou et al., 2025, *IEEE Access*) is **not** included in this submission for ethical and storage reasons. To reproduce the analyses, place the released raw data under `All Data/<participant_id>/` (one folder per participant, each containing `*_raw.txt` and `*.json` per scene as in Mavridou et al., 2025).

Pipeline order. Steps 1–2 build the cached event-window files that all subsequent scripts read; run them once before any of the analysis or training scripts.

1. **`run_pipeline.py`** — reads raw 1000 Hz EMG `.txt`, event-annotation `.json`, and continuous-rating files for every participant; applies Butterworth bandpass (20–450 Hz), Hampel outlier removal, full-wave rectification, and 100 ms moving-RMS envelope; merges streams, classifies per-event saliency, and extracts $[-2, +5]$ s event windows → `processed_data/event_windows_all.pkl`.
2. **`extract_spectral_features.py`** — reads `event_windows_all.pkl`; computes per-channel spectral and shape features per event window → `processed_data/event_windows_spectral.pkl`. Required input for `estimate_delays.py`, `train_spectral.py`, `train_lstm_multiband.py`, and `train_lstm_v4_full.py`.
3. **`estimate_delays.py`** — reads raw 1000 Hz EMG and JSON event annotations; outputs per-event Delay A/B/C → `results/delays_per_event.csv`.
4. **`analyze_delays.py`** — reads `delays_per_event.csv`; produces overall delay histograms, A-vs-C scatter, and saliency-stratified plots; runs Mann-Whitney $U$ tests on saliency contrasts.
5. **`analyze_modulation.py`** — joins per-event delays with the participant-metadata Excel from Mavridou et al. (2025); produces per-subject aggregates and the 44-test modulation battery (Spearman $\rho$ + Mann-Whitney) → `results/delays_modulation_*.{csv,txt}`.
6. **`train_lstm_v4_full.py`** — constructs the multimodal feature cache (`processed_data/multimodal_windows.npz`) from raw EMG + HR + IMU + accelerometer; trains BiLSTM seq2seq for continuous valence with combined CCC + MSE loss → metrics in `results/metrics_sq1_v4.json`.
7. **`train_lstm_multiband.py`** — earlier variant; used for Figure 5 (per-subject CCC scatter, multi-band + scene condition).
8. **`train_classification.py`** — window-level binary classifier for valence direction → `results/metrics_classification.json`.
9. **`train_spectral.py`** — spectral-feature ablation (459 features per window) for the saliency AUC = 0.729 result → `results/metrics_spectral.json`.
10. **`train_spectral_slope.py`** — exploratory shape-feature analysis (slope, area-under-curve) cited as a future-work starting point in §6.4.
11. **`train_svm_variations.py`** — SVM × 4 timing windows × 3 feature sets for the supervisor-driven Section 5.3.3 ablation → `results/metrics_window_variations.json`.

All trained models use a deterministic subject-level 80/10/10 split with `SEED = 42`. Re-running any script with the same data and seed reproduces the numbers reported in the thesis.

---

## Headline findings (one-line summary)

- **Delay A** (psychophysiological onset) median **227 ms**.
- **Delay C** (total observable delay to self-report) median **1,717 ms**.
- **Delay B** (cognitive-motor gap, $C - A$) median **1,379 ms** ≈ **1.4 s**.
- The static **200 ms** alignment used in prior work captures only **∼12 %** of the empirical lag.
- BiLSTM seq2seq did not exceed the static-lag Random Forest on continuous valence (Wilcoxon $p = 0.0007$, RF won in 12/13 test subjects).
- Saliency classification with EMG + scene reaches **AUC 0.729**; SVM and Random Forest are tied at AUC 0.705 across four 500 ms timing windows.

---

## Software environment

- Python 3.11
- NumPy 1.26, SciPy 1.11, pandas 2.1
- scikit-learn 1.4, statsmodels (for BH multiple-comparison correction)
- PyTorch 2.2 with Apple Metal Performance Shaders (`mps`) device backend
- Matplotlib 3.8

---

## Data, ethics, and AI use

The Science Museum London dataset (Mavridou et al., 2025) was obtained through a data-sharing agreement with the original collectors and is anonymised. This thesis project did not involve new data collection. The author retains the analysis code; the dataset remains the property of the original collectors.

Generative-AI assistance (Claude, Anthropic) was used during drafting for paraphrasing, code review, and English-language polishing. The author reviewed and edited all AI-assisted content and takes full responsibility for the final thesis.
