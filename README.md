# QRG-EEG: Full model suite

Standalone EEG experiment code for anonymous review. **All datasets must be supplied separately as local files.** The program contains no dataset acquisition, remote dataset-client or package-installation code.

## Files

- `run_full.py`: experiment implementation, local dataset readers, evaluation and exports.
- `requirements_full.txt`: this edition's Python dependencies.
- `README.md`: environment, data layout and execution instructions.

## Environment

Use Linux, Python 3.11 and an NVIDIA GPU for model training. Install dependencies in a dedicated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements_full.txt
python -m pip check
```

An existing compatible CUDA-enabled PyTorch environment can also be used. Training selects physical GPU 0. Dependencies are installed explicitly before execution; the experiment program does not install packages.

## Supply the datasets

Place the recordings under `data/` beside the script, or pass another local directory using `--data-root`. Upload/extract the actual files before execution. Repository pointer files, archives and external cache-pointer text files are not recordings.

| Local folder | Participants | Required inputs |
|---|---:|---|
| `data/BNCI2014_008/` | 8 | Original Santa Lucia-format MAT recordings; put each participant's files in subfolders `1/` through `8/`. |
| `data/UCL_ALS_MI/` | 8 | `S1.mat`, `S2.mat`, `S5.mat`, `S9.mat`, `S21.mat`, `S31.mat`, `S34.mat`, `S39.mat`. Subfolders are supported. |
| `data/BIGP3BCI_ALS/` | 8 | Study N EDF recordings, retaining the original `N_<id>` participant folders and session hierarchy. |
| `data/EEGET_ALS_MI/` | 6 | ALS participant directories containing `EEG.edf` and the corresponding `eeg.json` or `scenario.json`. Retain the original sample/scenario hierarchy. |

BNCI MAT files must contain the original `data` run structs, including `X`, `y`, `channels` and `classes`. The reader uses the recorded Target/NonTarget class mapping, continuous 256 Hz EEG and dense stimulus codes. EEG is interpreted in the original microvolt units, filtered, epoched over 0–0.8 s and resampled to 128 Hz. Participant folders identify subjects 1–8.

UCL retains the original `L`/`R`/`Re` trial structures and 19 EEG channels. bigP3BCI EDFs must retain their `EEG_` channels, `StimulusBegin` and `StimulusType` channels. EEGET requires its original Thinking/Resting annotations and JSON scenario labels for left/right-hand recordings. The same trial and class eligibility rules as the experiment implementation apply.

### Existing prepared bundles

Each dataset folder may instead contain `bundle.npz`. When present, that file takes precedence over raw files. This supports the existing non-pickle prepared-cache format and retains the supplied preprocessed EEG arrays.

The NPZ file contains `X`, `y`, `subjects`, `sessions` and `header`. `X` is `[trials, channels, samples]`; `y` is an integer vector. `subjects` and `sessions` are one-dimensional string arrays or JSON-encoded string-list scalars. `header` is a JSON string containing `task`, `sfreq`, `ch_names`, `class_names` and optional `metadata`. Object/pickle arrays are rejected.

Prepared bundles must already be at 128 Hz, with 102 samples for P300, 512 for UCL imagery, and 256 for EEGET imagery. P300 class order is `['NonTarget', 'Target']` with labels 0/1; imagery order is `['Left', 'Right', 'Rest']` with labels 0/1/2. Supply all eight BNCI, eight bigP3BCI, eight UCL and six EEGET participants. Prepared bundles bypass repeated preprocessing and undergo shape, finiteness, participant and class checks.

## Run on GPU 0

Print the schedule:

```bash
python run_full.py --plan
```

Check local folder/file presence without loading training libraries:

```bash
python run_full.py --data-root ./data --check-inputs
```

Read and validate dataset contents without training; this check also works with CPU PyTorch:

```bash
python run_full.py --data-root ./data --check-only
```

Run the experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python run_full.py --data-root ./data --output-root ./outputs_full_local
```

Missing or invalid data produce a local-file/schema error. The program does not contact a dataset service or retrieve replacement files. To resume, repeat the same command with the same data, software, configuration and output directory. Completed evaluations are reused after their identities and saved artifacts pass validation. Source-validation fits are additional to the final-evaluation counts below.

## Experiment specification

Each configuration covers **30 participants × three seeds (11, 22, 33) = 90 evaluations**. This edition schedules **50 configuration entries and 4,500 final evaluations**. Each participant is held out once. Preprocessing fits, model selection, epoch selection and calibration use source participants. Held-out labels are used for scoring.

The full suite has 4,050 fitted evaluations and 450 test-time shuffle-control evaluations. The latter reuse the corresponding trained main QRG checkpoint. The main model runs first for each participant; frozen-encoder surrogates also use its checkpoint. EA + EEGNet estimates alignment from unlabeled evaluation data and is marked transductive. Task-dependent comparators use the implemented P300/imagery variants, as named below.

| Configuration ID | Model or experiment | Evaluations |
|---|---|---:|
| `B8_QRG_EEG` | QRG-EEG | 90 |
| `B1_RIEMANNIAN_TS_LR` | Riemannian TS-LR | 90 |
| `B8_XDAWNCOV_FBCSP` | xDAWN covariance + TS-LR / FBCSP + LDA | 90 |
| `B9_ERPCOV_MDM` | ERP covariance MDM / covariance MDM | 90 |
| `B3_TASK_STRONG` | Shrinkage LDA / filter-bank network | 90 |
| `B5_GRAPH_CSPNET` | xDAWN BLDA / SPD time-frequency graph network | 90 |
| `B2_EEGNET` | EEGNet | 90 |
| `B10_EEG_CONFORMER` | EEG Conformer-style | 90 |
| `CRGNET` | Convolutional + Riemannian fusion | 90 |
| `B4_QEEGNET` | QEEGNet-style | 90 |
| `B6_SAME_GRAPH_GAT` | Same-graph GAT | 90 |
| `B7_MATCHED_CLASSICAL_INTERACTION` | Parameter-matched classical interaction | 90 |
| `B12_TRIG_SURROGATE` | Trigonometric surrogate | 90 |
| `C1_STAGE1_HEAD` | Stage-one linear head | 90 |
| `C2_EEGNET_LATENT` | EEGNet on latent components | 90 |
| `C3_NODE_ONLY_EQ` | Node-only circuit, matched readout | 90 |
| `EDGE_ONLY_VQC` | Edge-only circuit | 90 |
| `C5_FIXED_SAME_E` | Fixed entanglement on the same edges | 90 |
| `RIEMANNIAN_VQC` | Riemannian VQC | 90 |
| `PLV_GAT` | PLV GAT | 90 |
| `C6_RANDOM_TOPOLOGY_G0` | Random topology 1 | 90 |
| `C6_RANDOM_TOPOLOGY_G1` | Random topology 2 | 90 |
| `C6_RANDOM_TOPOLOGY_G2` | Random topology 3 | 90 |
| `C7_SHUFFLED_TT_P0` | Edges shuffled at training and testing, permutation 1 | 90 |
| `C7_SHUFFLED_TT_P1` | Edges shuffled at training and testing, permutation 2 | 90 |
| `C7_SHUFFLED_TT_P2` | Edges shuffled at training and testing, permutation 3 | 90 |
| `C7_SHUFFLED_TT_P3` | Edges shuffled at training and testing, permutation 4 | 90 |
| `C7_SHUFFLED_TT_P4` | Edges shuffled at training and testing, permutation 5 | 90 |
| `C8_SHUFFLED_TEST_P0` | Edges shuffled only at testing, permutation 1 | 90 |
| `C8_SHUFFLED_TEST_P1` | Edges shuffled only at testing, permutation 2 | 90 |
| `C8_SHUFFLED_TEST_P2` | Edges shuffled only at testing, permutation 3 | 90 |
| `C8_SHUFFLED_TEST_P3` | Edges shuffled only at testing, permutation 4 | 90 |
| `C8_SHUFFLED_TEST_P4` | Edges shuffled only at testing, permutation 5 | 90 |
| `B7_FROZEN` | Frozen-encoder classical interaction | 90 |
| `RFF_FROZEN` | Frozen-encoder random Fourier features | 90 |
| `SVM_FROZEN` | Frozen-encoder RBF SVM | 90 |
| `RFF_E2E` | End-to-end random Fourier features | 90 |
| `EA_EEGNET` | Euclidean alignment + EEGNet | 90 |
| `QRG_DIAG_NOMIX` | No mixers | 90 |
| `QRG_DIAG_RYONLY` | RY mixers only | 90 |
| `QRG_DIAG_Q6` | Six qubits, source-selected structure | 90 |
| `QRG_DIAG_Q8` | Eight qubits, source-selected structure | 90 |
| `QRG_DIAG_L2` | Two circuit layers | 90 |
| `QRG_DIAG_L4` | Four circuit layers | 90 |
| `X1_NNCONV_ECC` | NNConv/ECC | 90 |
| `X2_GINE` | GINE | 90 |
| `X3_PAIRWISE_MLP` | Learned pairwise MLP | 90 |
| `REPRESENTATION_QRG` | QRG-EEG, alternative representation | 90 |
| `REPRESENTATION_B7` | Matched classical model, alternative representation | 90 |
| `QRG_Q6_FIXED` | QRG-EEG, six-qubit configuration | 90 |

Quantum calculations use differentiable PyTorch state-vector simulation. The full suite retains nested source-only configuration selection. The six-entry suite retains its fixed reference structure: Variant A, q=8, two layers, graph multiplier 1.5 and average pooling, with the indicated representation or q=6 override.

## Outputs

Outputs are written under `outputs_full_local/experiments/` when using the command above.

| Output | Contents |
|---|---|
| `per_seed_results.csv` | Successful configuration/participant/seed evaluations and metrics. |
| `per_subject_mean_sd.csv` | Mean and sample SD across three seeds for each participant. |
| `dataset_subject_mean_sd.csv` | Dataset mean and sample SD across participant means. |
| `dataset_per_seed_means.csv` | Mean across participants for each dataset and seed. |
| `dataset_seed_mean_sd.csv` | Mean and sample SD across the three dataset-level seed means. |
| `job_status.csv`, `model_coverage.csv`, `completion.json` | Status and coverage, including all 90 required evaluations per configuration. |
| `results.md` | Tables calculated from measured results. |
| `participant_registry.json`, `model_registry.csv` | Participant and configuration registries. |
| `experiment_manifest.json` | Scientific settings, source identity and actual software/device versions. |
| `evaluations/` | Per-evaluation labels, probabilities, decisions, result records and fitted artifacts. |
| `configurations/` | Source-selection plans, validation caches and runtime records. |

Metrics include Macro-F1, accuracy, balanced accuracy, Cohen's kappa, AUROC and target average precision for P300. Target average precision is not applicable to imagery. Uncalibrated Macro-F1/balanced accuracy and class-specific results are also retained. Fractional metrics use a 0–1 scale. All sample SDs use `ddof=1`; seed SD and participant SD are exported separately.

Each successful evaluation commits its artifacts and result record before aggregate exports are refreshed. Neural checkpoints are saved when supplied by the fitter. Retain the output directory for resuming. Incomplete evaluations are reported explicitly and are not replaced by numeric scores.

## Optional analyses

`--finite-shots` enables exact and sampled readout at 64, 256, 1024 and 4096 shots, with five sampled repeats per budget. `--efficiency` enables separate controlled simulator/projection benchmarks. Both default to disabled and add analyses beyond the final-evaluation count.

The full suite uses its generated main-model checkpoints for finite-shot inference.

`--reference-selections` accepts local `source_selection_v13__<dataset>__test_<participant>.json` records. Optional inputs are read from local paths as well.

## Reproducibility

The dependency file gives supported ranges rather than a historical environment lock. For numerical replication, retain the exact input arrays, scientific settings, selected configurations and software versions recorded by the reference experiment. Existing prepared bundles preserve the reference preprocessed arrays when changing the surrounding file layout.

This distribution contains code and documentation only. Dataset files are supplied separately by the repository maintainer. The source and archive metadata contain no author/contact fields or private workstation paths.
