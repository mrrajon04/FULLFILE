#!/usr/bin/env python3
"""QRG-EEG six suite using datasets supplied separately as local files."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import zipfile
from pathlib import Path

SUITE = 'six'
REQUIREMENTS_FILE = 'requirements_six.txt'
EXPECTED_SUBJECTS = {'BNCI2014_008':8, 'UCL_ALS_MI':8, 'BIGP3BCI_ALS':8, 'EEGET_ALS_MI':6}
DATASET_CHOICES = tuple(EXPECTED_SUBJECTS)
SEEDS = (11, 22, 33)
MODEL_REGISTRY = [{'id': 'X1_NNCONV_ECC', 'part': 'baselines', 'model': 'X1_NNCONV_ECC', 'label': 'NNConv/ECC'}, {'id': 'X2_GINE', 'part': 'baselines', 'model': 'X2_GINE', 'label': 'GINE'}, {'id': 'X3_PAIRWISE_MLP', 'part': 'baselines', 'model': 'X3_PAIRWISE_MLP', 'label': 'Learned pairwise MLP'}, {'id': 'REPRESENTATION_QRG', 'part': 'representation', 'model': 'B8_QRG_EEG', 'label': 'QRG-EEG, alternative representation'}, {'id': 'REPRESENTATION_B7', 'part': 'representation', 'model': 'X0_B7_MATCHED', 'label': 'Matched classical model, alternative representation'}, {'id': 'QRG_Q6_FIXED', 'part': 'qubits_q6', 'model': 'B8_QRG_EEG', 'label': 'QRG-EEG, six-qubit configuration'}]
EXPORTED_METRICS = ('macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'auroc',
                    'target_auprc', 'macro_f1_uncal', 'balanced_accuracy_uncal')
SOURCE_SHA256 = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'


def parse_arguments(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='QRG-EEG experiments using user-supplied local datasets.')
    parser.add_argument('--data-root', type=Path, default=base/'data', help='Local directory containing the four dataset folders.')
    parser.add_argument('--output-root', type=Path, default=base/'outputs_six_local')
    parser.add_argument('--plan', action='store_true', help='Print the schedule without importing training libraries.')
    parser.add_argument('--check-inputs', action='store_true', help='Check local file/folder presence using the standard library only.')
    parser.add_argument('--check-only', action='store_true', help='Read and validate all local EEG inputs without model training; CUDA is optional.')
    parser.add_argument('--reference-checkpoints', type=Path)
    parser.add_argument('--reference-selections', type=Path)
    parser.add_argument('--finite-shots', action='store_true')
    parser.add_argument('--efficiency', action='store_true')
    args = parser.parse_args(argv)
    args.scope = 'all'
    args.datasets = list(DATASET_CHOICES)
    args.skip_shots = not args.finite_shots
    args.skip_efficiency = not args.efficiency
    return args


def experiment_plan(args=None):
    test_only = sum(spec['part'] == 'test_only' for spec in MODEL_REGISTRY)
    return {'suite': SUITE, 'datasets': EXPECTED_SUBJECTS, 'participants':30,
            'seeds':list(SEEDS), 'configurations':len(MODEL_REGISTRY),
            'evaluations_per_configuration':90, 'final_evaluations':90*len(MODEL_REGISTRY),
            'final_fitted_evaluations':90*(len(MODEL_REGISTRY)-test_only),
            'test_only_evaluations':90*test_only, 'input_policy':'local files supplied separately',
            'source_validation_fits':'additional; held-out participant excluded from selection',
            'configuration_ids':[spec['id'] for spec in MODEL_REGISTRY]}


def inspect_local_inputs(root):
    root = Path(root).expanduser().resolve()
    report, missing = {}, []
    for name, count in EXPECTED_SUBJECTS.items():
        folder = root/name
        if not folder.is_dir():
            missing.append(str(folder))
            continue
        bundle = folder/'bundle.npz'
        if bundle.is_file():
            try:
                with zipfile.ZipFile(bundle) as archive:
                    required = {'X.npy','y.npy','subjects.npy','sessions.npy','header.npy'}
                    if not required <= set(archive.namelist()):
                        raise ValueError('required prepared-bundle arrays are missing')
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                missing.append(str(bundle) + ': ' + str(error))
                continue
            report[name] = {'format':'prepared NPZ','file':'bundle.npz','expected_participants':count}
            continue
        files = [p for p in folder.rglob('*') if p.is_file()]
        if name == 'BNCI2014_008':
            mats = [p for p in files if p.suffix.lower()=='.mat']
            if len(mats)<8:
                missing.append(str(folder) + ': supply MAT recordings for all eight participants, or bundle.npz')
            else:
                report[name] = {'format':'MAT','files':len(mats),'expected_participants':count}
        elif name == 'UCL_ALS_MI':
            found = set()
            for p in files:
                if p.suffix.lower()=='.mat':
                    match = re.fullmatch(r'(?:S|Subject)0*([0-9]+)', p.stem, re.I)
                    if match:
                        found.add(int(match.group(1)))
            absent = {1,2,5,9,21,31,34,39} - found
            if absent:
                missing.append(str(folder) + ': missing UCL participant MAT files ' + str(sorted(absent)))
            else:
                report[name] = {'format':'MAT','participants':sorted(found),'expected_participants':count}
        elif name == 'BIGP3BCI_ALS':
            people = {part.upper() for p in files if p.suffix.lower()=='.edf'
                      for part in p.relative_to(folder).parts if re.fullmatch(r'N_[0-9]+',part,re.I)}
            if len(people)!=8:
                missing.append(str(folder) + f': expected EDF recordings under eight N_<id> participant folders; found {len(people)}')
            else:
                report[name] = {'format':'EDF','participants':sorted(people),'expected_participants':count}
        else:
            edfs = [p for p in files if p.name.lower()=='eeg.edf']
            people = {int(re.fullmatch(r'ALS[_-]?([0-9]+)',part,re.I).group(1))
                      for p in edfs for part in p.relative_to(folder).parts
                      if re.fullmatch(r'ALS[_-]?([0-9]+)',part,re.I)}
            sidecars = [p for p in edfs if (p.parent/'eeg.json').is_file() or (p.parent/'scenario.json').is_file()]
            if len(people)!=6 or not sidecars:
                missing.append(str(folder) + ': supply six ALS participant folders with EEG.edf and eeg.json or scenario.json metadata')
            else:
                report[name] = {'format':'EDF plus JSON','participants':sorted(people),'expected_participants':count}
    if missing:
        raise FileNotFoundError('Local dataset inputs are incomplete:\n  ' + '\n  '.join(missing) +
                                '\nPlace the files under --data-root as described in README.md.')
    return report


ARGS = parse_arguments() if __name__=='__main__' else parse_arguments([])
if __name__=='__main__':
    if ARGS.plan:
        print(json.dumps(experiment_plan(ARGS),indent=2))
        raise SystemExit(0)
    try:
        INPUT_FILES = inspect_local_inputs(ARGS.data_root)
    except FileNotFoundError as error:
        raise SystemExit(str(error))
    if ARGS.check_inputs:
        print(json.dumps(INPUT_FILES,indent=2))
        raise SystemExit(0)
    required = ('numpy','scipy','pandas','sklearn','torch','mne','pyriemann',
                'networkx','h5py','mat73','matplotlib','seaborn','tqdm','joblib')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Install ' + REQUIREMENTS_FILE + ' in the active Python environment. Missing: ' + ', '.join(missing))

os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
RUN_FINITE_SHOTS = ARGS.finite_shots
RUN_EFFICIENCY = ARGS.efficiency

SOURCE_SHA256 = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'
import collections
import contextlib
import csv
import faulthandler
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
FIXED_ACTION = 'full'
FIXED_RUNTIME_VERSION = 'experiment_runtime'
_FIXED_ROOT = None
_FIXED_CONTEXT = {}
_FIXED_EVENTS = []
_FIXED_EVENT_LOCK = threading.RLock()
_FIXED_SESSION_START = time.monotonic()
_FIXED_CONSOLE = sys.stdout
_FIXED_LOCK_HANDLE = None
_FIXED_RUNTIME_MANIFEST = {}
_FIXED_LAST_CONSOLE = time.monotonic()
_FIXED_LAST_RESULT = time.monotonic()
_FIXED_PREFLIGHT_INJECT = False
RELEASE_IDENTITY_SHA256 = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'
SOURCE_IDENTITY_SHA256 = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'

def fixed_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def fixed_json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)

def fixed_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=fixed_json_value)

def fixed_digest(value):
    return hashlib.sha256(fixed_canonical(value).encode()).hexdigest()

def fixed_atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def fixed_atomic_json(path, value):
    fixed_atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, default=fixed_json_value, allow_nan=False) + '\n').encode())

def fixed_emit(event, **fields):
    global _FIXED_LAST_CONSOLE
    with _FIXED_EVENT_LOCK:
        row = {k: v for k, v in _FIXED_CONTEXT.items() if not k.startswith('_')}
        row.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), event=event, elapsed_s=round(time.monotonic() - _FIXED_SESSION_START, 3), pid=os.getpid())
        row.update(fields)
        if '_unit_start' in _FIXED_CONTEXT:
            row['unit_elapsed_s'] = round(time.monotonic() - _FIXED_CONTEXT['_unit_start'], 3)
        _FIXED_EVENTS.append(row)
        if len(_FIXED_EVENTS) > 5000:
            del _FIXED_EVENTS[:1000]
        if _FIXED_ROOT is not None:
            with (_FIXED_ROOT / 'progress.jsonl').open('a', buffering=1) as stream:
                stream.write(json.dumps(row, default=fixed_json_value, allow_nan=False) + '\n')
            fixed_atomic_json(_FIXED_ROOT / 'current_stage.json', row)
        if event in ('RUNTIME_READY', 'DATASET_READY', 'MODEL_FAILED', 'DATASET_FAILED', 'STARTUP_CHECKS_PASS', 'RUN_COMPLETE'):
            details = ' '.join((str(row[k]) for k in ('dataset', 'test_subject', 'model', 'detail') if k in row))
            print(event, details, file=_FIXED_CONSOLE, flush=True)
        elif event == 'HEARTBEAT' and time.monotonic() - _FIXED_LAST_RESULT >= 300 and (time.monotonic() - _FIXED_LAST_CONSOLE >= 300):
            details = ' '.join((str(row[k]) for k in ('dataset', 'test_subject', 'model', 'stage') if k in row))
            config = f" config {row['config']}/{row.get('total_configs', '?')}" if 'config' in row else ''
            validator = f" validator {row['validator']}" if 'validator' in row else ''
            print(f"V12 STATUS: {details}{config}{validator} Device: {row.get('device', 'cpu')} elapsed {row.get('unit_elapsed_s', 0):.0f}s", file=_FIXED_CONSOLE, flush=True)
            _FIXED_LAST_CONSOLE = time.monotonic()

@contextlib.contextmanager
def fixed_operation(stage, **fields):
    previous = dict(_FIXED_CONTEXT)
    _FIXED_CONTEXT.update(fields, stage=stage, _unit_start=time.monotonic())
    fixed_emit('START')
    try:
        yield
    except BaseException as error:
        fixed_emit('INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'ERROR', detail=repr(error))
        raise
    else:
        fixed_emit('DONE')
    finally:
        _FIXED_CONTEXT.clear()
        _FIXED_CONTEXT.update(previous)

@contextlib.contextmanager
def fixed_quiet():
    if _FIXED_ROOT is None:
        yield
        return
    with (_FIXED_ROOT / 'details.log').open('a', buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield

def fixed_early_stack_request(signum, frame):
    faulthandler.dump_traceback(file=sys.__stderr__, all_threads=True)
if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, fixed_early_stack_request)

@contextlib.contextmanager
def fixed_watchdog():
    stopped = threading.Event()
    stack = (_FIXED_ROOT / f'stacks_{os.getpid()}.log').open('a', buffering=1)
    faulthandler.enable(file=stack, all_threads=True)
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)

    def beat():
        while not stopped.wait(60):
            fixed_emit('HEARTBEAT')
    worker = threading.Thread(target=beat, name='qrg-heartbeat', daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=1)
        if hasattr(signal, 'SIGUSR1'):
            faulthandler.unregister(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, fixed_early_stack_request)
        faulthandler.disable()
        stack.close()

def fixed_release_lock():
    global _FIXED_LOCK_HANDLE
    if _FIXED_LOCK_HANDLE is not None:
        fcntl.flock(_FIXED_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
        _FIXED_LOCK_HANDLE.close()
        _FIXED_LOCK_HANDLE = None

def fixed_live_code_digest(namespace):
    seen = set()

    def encode(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, bytes):
            return ['bytes', value.hex()]
        if isinstance(value, Path):
            return ['path', str(value)]
        if isinstance(value, types.CodeType):
            return ['code', value.co_code.hex(), list(value.co_names), list(value.co_varnames), list(value.co_freevars), list(value.co_cellvars), [encode(x) for x in value.co_consts], value.co_argcount, value.co_kwonlyargcount, value.co_posonlyargcount, value.co_flags]
        if isinstance(value, types.FunctionType):
            key = id(value)
            if key in seen:
                return ['function_ref', value.__qualname__]
            seen.add(key)
            closure = []
            for cell in value.__closure__ or ():
                try:
                    closure.append(encode(cell.cell_contents))
                except ValueError:
                    closure.append(['empty_cell'])
            return ['function', value.__qualname__, encode(value.__code__), encode(value.__defaults__), encode(value.__kwdefaults__), closure]
        if isinstance(value, (tuple, list)):
            return [encode(x) for x in value]
        if isinstance(value, (set, frozenset)):
            return sorted([encode(x) for x in value], key=repr)
        if isinstance(value, dict):
            return [[encode(k), encode(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
        if isinstance(value, type):
            return ['class_ref', value.__module__, value.__qualname__]
        return ['object_type', type(value).__module__, type(value).__qualname__]
    records = []
    for name, value in sorted(list(namespace.items())):
        if isinstance(value, types.FunctionType) and (value.__globals__ is namespace or value.__module__ == namespace.get('__name__')):
            records.append([name, encode(value)])
        elif isinstance(value, type) and value.__module__ == namespace.get('__name__'):
            methods = []
            for member, fn in sorted(value.__dict__.items()):
                if isinstance(fn, (staticmethod, classmethod)):
                    fn = fn.__func__
                if isinstance(fn, property):
                    fn = fn.fget
                if isinstance(fn, types.FunctionType):
                    methods.append([member, encode(fn)])
            records.append([name, methods])
    return fixed_digest(records)
LOADER_IDENTITY_SHA256 = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'
import os
EXECUTION_PROFILE = 'full'
if EXECUTION_PROFILE not in ('check', 'full'):
    raise ValueError('QRG_EXECUTION_PROFILE must be check or full')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import h5py
import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sps
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh, expm, logm
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

@dataclass
class Config:
    EXECUTION_PROFILE: str = EXECUTION_PROFILE
    CODE_VERSION: str = 'graph_conditioned_eeg'
    BUILD_ID: str = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'
    RESULT_SCHEMA_VERSION: str = 'v8_public_31_columns'
    TRAINING_RECIPE: str = 'patient_risk_0p65_plus_patient_class_risk_0p35_plus_ovr_rank_0p08'
    QRG_NATURAL_RISK_WEIGHT: float = 0.65
    QRG_SUBJECT_CLASS_RISK_WEIGHT: float = 0.35
    QRG_RANK_RELATIVE_WEIGHT: float = 0.08
    QRG_EMA_DECAY: float = 0.99
    QRG_LR_STAGE2_CIRCUIT: float = 0.0008
    QRG_LR_STAGE2_READOUT: float = 0.0015
    QRG_LR_FINETUNE_BACKBONE: float = 8e-05
    QRG_LR_FINETUNE_CIRCUIT: float = 0.0003
    QRG_LR_FINETUNE_READOUT: float = 0.0008
    QRG_SELECTION_VERSION: str = 'v11_fixed_A_L2_g1p5_hybrid65_35_raw_macro_proxy12_12_6_top2_oof'
    QRG_IMPLEMENTATION_ID: str = 'graph_conditioned_eeg'
    QRG_BUILD_FINGERPRINT: str = '417b84b01c71060a199bbd3ec584ca2f25cf36f2ac8cef8cbde74f2a3016ca1c'
    QRG_PROXY_SHORTLIST: int = 2
    QRG_PROXY_PRETRAIN_EPOCHS: int = 12
    QRG_PROXY_QUANTUM_EPOCHS: int = 12
    QRG_PROXY_FINETUNE_EPOCHS: int = 6
    QRG_PROXY_PATIENCE: int = 4
    QRG_EEGET_USE_ALL_SOURCE_VALIDATORS: bool = True
    PRIMARY_DECISION_POLICY: str = 'uniform_source_only_model_specific'
    REQUIRE_CUDA_PAPER: bool = not ARGS.check_only
    RUN_MODE: str = 'paper'
    SEEDS: Tuple[int, ...] = (11, 22, 33)
    PRIMARY_SEED: int = 11
    PROJECT_DIR: Path = field(default_factory=lambda: Path('.'))
    DATA_ROOT: Path = field(default_factory=lambda: Path(ARGS.data_root))
    RESULTS_ROOT: Path = field(default_factory=lambda: Path(ARGS.output_root))
    RUN_NAME: Optional[str] = 'experiment'
    RUN_SCOPE: str = 'full_submission'
    RUN_SUPPLEMENTARY_CONTROLS: bool = True
    DATASETS_PAPER: Tuple[str, ...] = ('BNCI2014_008', 'UCL_ALS_MI', 'BIGP3BCI_ALS', 'EEGET_ALS_MI')
    BIGP3_STUDY: str = 'N'
    P300_FMIN: float = 0.5
    P300_FMAX: float = 20.0
    P300_TMIN: float = 0.0
    P300_TMAX: float = 0.8
    MI_FMIN: float = 4.0
    MI_FMAX: float = 40.0
    MI_SECONDS: float = 4.0
    UCL_MI_SECONDS: float = 4.0
    EEGET_MI_SECONDS: float = 2.0
    RESAMPLE: int = 128
    ARTIFACT_UV: float = 120.0
    Q_DEFAULT: int = 8
    Q_ARCH: Tuple[int, ...] = (6, 8)
    Q_SENSITIVITY: Tuple[int, ...] = (4, 6, 8, 12, 16)
    PROJECTION_MI: str = 'csp'
    PROJECTION_P300: str = 'xdawn'
    MI_PROJECTION_CANDIDATES: Tuple[str, ...] = ('csp', 'pca')
    P300_EDGE_MODE: str = 'erp_residual'
    P300_EDGE_CANDIDATES: Tuple[str, ...] = ('erp_residual', 'plain')
    GRAPH_MULTIPLIER: float = 1.5
    GRAPH_MULTIPLIERS: Tuple[float, ...] = (1.0, 1.5, 2.0)
    EDGE_READOUT: str = 'all_graph_edges'
    ROBUST_CLIP: float = 3.0
    THETA_INIT: float = 0.35
    THETA_INIT_STD: float = 0.02
    MIXER_X_INIT: float = math.pi / 4
    MIXER_Y_INIT: float = 0.15
    MIXER_INIT_STD: float = 0.05
    VARIANT_DEFAULT: str = 'A'
    VARIANTS: Tuple[str, ...] = ('A', 'B', 'C')
    L_DEFAULT: int = 2
    L_SENSITIVITY: Tuple[int, ...] = (2, 4, 6, 8, 10)
    ALPHA_MAX: float = math.pi
    BETA_MAX: float = math.pi / 2
    NODE_FEATURE_DIM: int = 12
    TEMPORAL_FILTERS: int = 12
    DROPOUT: float = 0.25
    SHARED_NODE_EDGE_ACROSS_BLOCKS: bool = True
    PARAM_MATCH_MAX_HIDDEN: int = 64
    PARAM_MATCH_TOLERANCE: float = 0.1
    PRETRAIN_EPOCHS: int = 60
    QUANTUM_EPOCHS: int = 60
    FINETUNE_EPOCHS: int = 30
    PATIENCE: int = 12
    BATCH_SIZE_GPU: int = 96
    BATCH_SIZE_CPU: int = 48
    Q_BATCH_SIZE_GPU: int = 64
    Q_BATCH_SIZE_CPU: int = 32
    LR_PRETRAIN: float = 0.001
    LR_QUANTUM: float = 0.002
    LR_HEAD: float = 0.001
    LR_BACKBONE_FINETUNE: float = 0.0001
    WEIGHT_DECAY: float = 0.0001
    GRAD_CLIP: float = 5.0
    USE_RISK_VARIANCE: bool = False
    LAMBDA_RV: float = 0.0
    QRG_USE_RISK_VARIANCE: bool = False
    QRG_LAMBDA_RV: float = 0.0
    QRG_RANK_LAMBDA: float = 0.08
    QRG_RANK_MARGIN: float = 0.25
    QRG_MACRO_TOLERANCE: float = 0.01
    QRG_CLASS_WEIGHT_POWER: float = 0.5
    QRG_CLASS_WEIGHT_MIN: float = 0.25
    QRG_CLASS_WEIGHT_MAX: float = 4.0
    QRG_MIN_PRETRAIN_BEST_EPOCH: int = 5
    QRG_MIN_STAGE2_BEST_EPOCH: int = 6
    QRG_MIN_FINETUNE_BEST_EPOCH: int = 3
    QRG_PRIOR_TAU_GRID: Tuple[float, ...] = (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5)
    INNER_MAX_SUBJECTS_SERVER: int = 1
    INNER_MAX_SUBJECTS_ARCH: int = 2
    INNER_MAX_SUBJECTS_PAPER: Optional[int] = 3
    PAPER_TUNE_SEEDS: Tuple[int, ...] = (11,)
    PAPER_Q_CANDIDATES: Tuple[int, ...] = (6, 8)
    PAPER_VARIANT_CANDIDATES: Tuple[str, ...] = ('A',)
    PAPER_REFIT_ALL_SOURCE: bool = True
    TUNE_GRID: Tuple[Dict[str, Any], ...] = ({'q': 6, 'layers': 2, 'graph_multiplier': 1.0, 'lr_quantum': 0.002}, {'q': 8, 'layers': 2, 'graph_multiplier': 1.0, 'lr_quantum': 0.002}, {'q': 8, 'layers': 2, 'graph_multiplier': 1.5, 'lr_quantum': 0.001})
    MAIN_MODELS: Tuple[str, ...] = ('B1_RIEMANNIAN_TS_LR', 'B2_EEGNET', 'B3_TASK_STRONG', 'B4_QEEGNET', 'B5_GRAPH_CSPNET', 'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B8_QRG_EEG')
    SUPPLEMENTARY_MODELS: Tuple[str, ...] = ('RIEMANNIAN_VQC', 'CRGNET', 'PLV_GAT')
    RQ2_MODELS: Tuple[str, ...] = ('NODE_ONLY_VQC', 'FIXED_ENTANGLEMENT_VQC', 'B8_QRG_EEG', 'SHUFFLED_EDGE_QRG', 'EDGE_ONLY_VQC')
    MAX_SUBJECTS_SERVER: int = 3
    MAX_OUTER_FOLDS_SERVER: int = 1
    MAX_TRIALS_PER_SUBJECT_SERVER: int = 60
    MAX_OUTER_FOLDS_ARCH: int = 2
    MAX_TRIALS_PER_SUBJECT_ARCH: int = 320
    MAX_OUTER_FOLDS_PAPER: Optional[int] = None
    MAX_TRIALS_PER_SUBJECT_PAPER: Optional[int] = None
    NUM_WORKERS: int = 0
    STRATIFIED_SBCE_SAMPLER: bool = True
    ARCH_SCREEN_DATASETS: Tuple[str, ...] = ('BNCI2014_008', 'UCL_ALS_MI')
    ARCH_BASELINE_MODELS: Tuple[str, ...] = ('B1_RIEMANNIAN_TS_LR', 'B2_EEGNET', 'B3_TASK_STRONG', 'B4_QEEGNET', 'B5_GRAPH_CSPNET', 'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION')
    TORCH_THREADS: int = 4
    SAVE_CHECKPOINTS: bool = True
    RESUME: bool = True
    V12_CALIBRATION_RULE: str = 'balanced_accuracy'
    V12_BASELINE_TRAINING: str = 'ema_checkpoint'
    V12_BASELINE_LR_GRID: Tuple[float, ...] = (0.0003, 0.001, 0.003)
    V12_BASELINE_DROPOUT_GRID: Tuple[float, ...] = (0.25, 0.5)
    V12_LINEAR_C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0)
    V12_B8_NFILTER: int = 4
    V12_B8_CSP_FILTERS: int = 4
    V12_B8_MI_SELECT: int = 8
    V12_N_PERMUTATIONS: int = 5
    V12_N_RANDOM_TOPOLOGIES: int = 3
    V12_RFF_D_GRID: Tuple[int, ...] = (256, 1024, 4096)
    V12_RFF_SIGMA_GRID: Tuple[float, ...] = (math.pi / 4, math.pi / 2, math.pi)
    V12_SVM_C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    V12_SVM_GAMMA_GRID: Tuple[Any, ...] = ('scale', 0.01, 0.1, 1.0)
    V12_RFF_E2E_FEATURES: int = 1024
    V12_B7_FROZEN_HIDDEN_GRID: Tuple[int, ...] = (8, 16, 32)
    V12_DIAG_SUBJECTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    TEST_SUBJECTS_PER_DATASET: Tuple[Tuple[str, int], ...] = ()
    DIAG_SUBJECTS_PER_DATASET: Tuple[Tuple[str, int], ...] = ()
    V12_MAIN_MODELS: Tuple[str, ...] = ('B1_RIEMANNIAN_TS_LR', 'B8_XDAWNCOV_FBCSP', 'B9_ERPCOV_MDM', 'B3_TASK_STRONG', 'B5_GRAPH_CSPNET', 'B2_EEGNET', 'B10_EEG_CONFORMER', 'CRGNET', 'B4_QEEGNET', 'B8_QRG_EEG', 'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE')
    V12_RQ2_MODELS: Tuple[str, ...] = ('C1_STAGE1_HEAD', 'C2_EEGNET_LATENT', 'C3_NODE_ONLY_EQ', 'EDGE_ONLY_VQC', 'C5_FIXED_SAME_E', 'RIEMANNIAN_VQC', 'PLV_GAT')
    V12_SURROGATE_MODELS: Tuple[str, ...] = ('B7_FROZEN', 'RFF_FROZEN', 'SVM_FROZEN', 'RFF_E2E')
    V12_CHANCE_PERMUTATIONS: int = 1000
    V12_FD_STEP: float = 0.05
    V12_BOOTSTRAP: int = 10000
    V12_V11_RUN_NAME: Optional[str] = 'reference'
    V13_DETERMINISTIC: bool = True
    V13_QRG_SEARCH: bool = True
    V13_L_GRID: Tuple[int, ...] = (2, 4)
    V13_G_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0)
    V13_P300_POOL_GRID: Tuple[str, ...] = ('avg', 'max')
    V13_EQUAL_BUDGET: bool = True
    V13_SEARCH_K: Optional[int] = None
cfg = Config()
if cfg.RUN_MODE == 'server_test':
    cfg.SEEDS = (cfg.PRIMARY_SEED,)
    cfg.ACTIVE_DATASETS = ('BNCI2014_008',)
    cfg.ACTIVE_MODELS = ('B1_RIEMANNIAN_TS_LR', 'B2_EEGNET', 'B8_QRG_EEG')
    cfg.MAX_SUBJECTS = cfg.MAX_SUBJECTS_SERVER
    cfg.MAX_OUTER_FOLDS = cfg.MAX_OUTER_FOLDS_SERVER
    cfg.MAX_TRIALS_PER_SUBJECT = cfg.MAX_TRIALS_PER_SUBJECT_SERVER
    cfg.PRETRAIN_EPOCHS = 2
    cfg.QUANTUM_EPOCHS = 2
    cfg.FINETUNE_EPOCHS = 1
    cfg.PATIENCE = 2
elif cfg.RUN_MODE == 'architecture':
    cfg.SEEDS = (11, 22)
    cfg.ACTIVE_DATASETS = cfg.DATASETS_PAPER
    cfg.ACTIVE_MODELS = cfg.MAIN_MODELS
    cfg.MAX_SUBJECTS = None
    cfg.MAX_OUTER_FOLDS = cfg.MAX_OUTER_FOLDS_ARCH
    cfg.MAX_TRIALS_PER_SUBJECT = cfg.MAX_TRIALS_PER_SUBJECT_ARCH
    cfg.PRETRAIN_EPOCHS = 50
    cfg.QUANTUM_EPOCHS = 50
    cfg.FINETUNE_EPOCHS = 25
    cfg.PATIENCE = 10
elif cfg.RUN_MODE == 'paper':
    cfg.ACTIVE_DATASETS = ('BNCI2014_008',) if EXECUTION_PROFILE == 'check' else cfg.DATASETS_PAPER
    if cfg.RUN_SCOPE != 'full_submission':
        raise ValueError("This configuration requires RUN_SCOPE='full_submission'.")
    cfg.ACTIVE_MODELS = cfg.MAIN_MODELS
    cfg.MAX_SUBJECTS = None
    cfg.MAX_OUTER_FOLDS = 1 if EXECUTION_PROFILE == 'check' else cfg.MAX_OUTER_FOLDS_PAPER
    cfg.MAX_TRIALS_PER_SUBJECT = cfg.MAX_TRIALS_PER_SUBJECT_PAPER
else:
    raise ValueError('RUN_MODE must be server_test, architecture, or paper')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cfg.RUN_MODE == 'paper' and cfg.REQUIRE_CUDA_PAPER and (DEVICE.type != 'cuda'):
    raise RuntimeError('CUDA is required. Use a CUDA-enabled PyTorch installation and select GPU 0.')
torch.set_num_threads(cfg.TORCH_THREADS)
cfg.DATA_ROOT = Path(ARGS.data_root)
cfg.RESULTS_ROOT = Path(ARGS.output_root)
RUN_DIR = cfg.RESULTS_ROOT / 'runtime_state'

def _atomic_write_text(path, text):
    fixed_atomic_bytes(path, text.encode('utf-8'))
try:
    from IPython.display import display
except ImportError:
    display = print

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(cfg.PRIMARY_SEED)
V8_PUBLIC_COLUMNS = ['dataset', 'model', 'seed', 'test_subject', 'variant', 'q', 'layers', 'graph_multiplier', 'projection_mode', 'edge_mode', 'val_subject', 'training_recipe', 'n_edges', 'best_epoch', 'decision_threshold', 'prior_tau', 'trainable_params', 'quantum_params', 'two_qubit_gates', 'circuit_depth', 'circuit_evaluations_per_epoch', 'parameter_match_target', 'parameter_match_gap', 'wall_seconds', 'class_f1', 'macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'auroc', 'target_auprc']
LOGICAL_KEY_COLUMNS = ['dataset', 'model', 'seed', 'test_subject']

@dataclass
class EEGDatasetBundle:
    name: str
    task: str
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    sfreq: float
    ch_names: List[str]
    class_names: List[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def audit(self) -> pd.DataFrame:
        rows = []
        for subject in sorted(np.unique(self.subjects), key=str):
            idx = np.flatnonzero(self.subjects == subject)
            counts = {self.class_names[int(k)]: int(v) for k, v in zip(*np.unique(self.y[idx], return_counts=True))}
            rows.append({'subject': str(subject), 'n_trials': len(idx), 'class_counts': counts})
        return pd.DataFrame(rows)

@dataclass
class FoldArrays:
    x_raw: np.ndarray
    x_deep: np.ndarray
    x_riem: np.ndarray
    x_latent_deep: np.ndarray
    x_latent_riem: np.ndarray
    edges: np.ndarray
    tangent_vec: np.ndarray
    y: np.ndarray
    subject_codes: np.ndarray
    subject_names: np.ndarray
    sessions: np.ndarray

@dataclass
class PreparedFold:
    dataset_name: str
    task: str
    class_names: List[str]
    sfreq: float
    q: int
    graph_edges: List[Tuple[int, int]]
    readout_edges: List[Tuple[int, int]]
    projector_matrix: np.ndarray
    train: FoldArrays
    val: FoldArrays
    test: FoldArrays
    fit_audit: Dict[str, Any]

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

class ResultStore:

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / 'fold_results.csv'
        self.record_dir = self.run_dir / 'records'
        self.pred_dir = self.run_dir / 'fold_artifacts' / 'predictions'
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.records = {}
        self._load_records()

    @staticmethod
    def _key(key: Mapping[str, Any]) -> Tuple[str, str, int, str]:
        return (str(key['dataset']), str(key['model']), int(key['seed']), str(key['test_subject']))

    @staticmethod
    def _token(key: Mapping[str, Any]) -> str:
        logical = ResultStore._key(key)
        return hashlib.sha256(json.dumps(logical).encode()).hexdigest()[:24]

    def artifact_path(self, key: Mapping[str, Any]) -> Path:
        safe = re.sub('[^A-Za-z0-9_.-]+', '_', '__'.join(map(str, self._key(key))))
        return self.pred_dir / f'{safe}__{self._token(key)}.npz'

    def _load_records(self) -> None:
        for path in sorted(self.record_dir.glob('*.json')):
            payload = json.loads(path.read_text())
            if payload.get('experiment_fingerprint') != EXPERIMENT_FINGERPRINT:
                raise RuntimeError(f'Stale record fingerprint: {path}')
            row = payload.get('row', {})
            if set(row.keys()) != set(V8_PUBLIC_COLUMNS):
                raise RuntimeError(f'Public row schema mismatch in {path}')
            row = {column: row.get(column) for column in V8_PUBLIC_COLUMNS}
            payload['row'] = row
            key = self._key(row)
            if tuple(payload.get('logical_key', ())) != key:
                raise RuntimeError(f'Record logical key does not match its row: {path}')
            if key in self.records:
                raise RuntimeError(f'Duplicate committed logical key: {key}')
            artifact = Path(payload['artifact_path'])
            if not artifact.exists():
                raise RuntimeError(f'Committed row has no prediction artifact: {artifact}')
            actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if payload.get('artifact_sha256') != actual_sha:
                raise RuntimeError(f'Prediction-artifact checksum mismatch: {artifact}')
            checkpoint = payload.get('checkpoint_path')
            if checkpoint:
                checkpoint_path = Path(checkpoint)
                if not checkpoint_path.exists():
                    raise RuntimeError(f'Committed row has no declared checkpoint: {checkpoint}')
                checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                if payload.get('checkpoint_sha256') != checkpoint_sha:
                    raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
            self.records[key] = payload
        self.rows = [payload['row'] for _, payload in sorted(self.records.items())]

    def _rebuild_csv(self) -> None:
        frame = pd.DataFrame(self.rows, columns=V8_PUBLIC_COLUMNS)
        tmp = self.csv_path.with_suffix('.csv.tmp')
        with open(tmp, 'w', encoding='utf-8', newline='') as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.csv_path)

    def completed(self, key: Mapping[str, Any]) -> bool:
        logical = self._key(key)
        if logical not in self.records:
            return False
        artifact = Path(self.records[logical]['artifact_path'])
        if not artifact.exists():
            raise RuntimeError(f'Committed job is missing its artifact: {logical}')
        checkpoint = self.records[logical].get('checkpoint_path')
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise RuntimeError(f'Committed job is missing its checkpoint: {logical}')
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if self.records[logical].get('checkpoint_sha256') != checkpoint_sha:
                raise RuntimeError(f'Committed job has a corrupt checkpoint: {logical}')
        return True

    def add(self, row: Dict[str, Any], artifact: Dict[str, Any], internal: Optional[Dict[str, Any]]=None, checkpoint_path: Optional[Path]=None) -> None:
        internal = dict(internal or {})
        internal['runtime_identity'] = {'version': FIXED_RUNTIME_VERSION, 'scientific_source_sha256': SOURCE_IDENTITY_SHA256, 'scientific_fingerprint': EXPERIMENT_FINGERPRINT}
        ordered = {column: row.get(column, np.nan) for column in V8_PUBLIC_COLUMNS}
        logical = self._key(ordered)
        if logical in self.records:
            raise RuntimeError(f'Refusing duplicate commit: {logical}')
        artifact_path = self.artifact_path(ordered)
        artifact_tmp = artifact_path.with_suffix('.npz.tmp')
        with open(artifact_tmp, 'wb') as handle:
            np.savez_compressed(handle, **artifact)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(artifact_tmp, artifact_path)
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        payload = {'experiment_fingerprint': EXPERIMENT_FINGERPRINT, 'logical_key': list(logical), 'row': _jsonable(ordered), 'artifact_path': str(artifact_path), 'artifact_sha256': artifact_sha, 'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None, 'checkpoint_sha256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() if checkpoint_path is not None else None, 'internal': _jsonable(internal or {})}
        record_path = self.record_dir / f'{self._token(ordered)}.json'
        _atomic_write_text(record_path, json.dumps(payload, indent=2, allow_nan=False))
        self.records[logical] = payload
        self.rows.append(payload['row'])
        self._rebuild_csv()

def count_trainable(model: nn.Module) -> int:
    return int(sum((p.numel() for p in model.parameters() if p.requires_grad)))

def _prediction_from_probability(prob: np.ndarray, decision_threshold: Optional[float]=None) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim == 2 and prob.shape[1] == 2 and (decision_threshold is not None) and np.isfinite(decision_threshold):
        return (prob[:, 1] >= float(decision_threshold)).astype(int)
    return prob.argmax(axis=1)

def multiclass_metrics(y_true: np.ndarray, prob: np.ndarray, task: str, decision_threshold: Optional[float]=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or len(prob) != len(y_true) or (not np.isfinite(prob).all()):
        raise ValueError('Metric probabilities must be finite [N,K] aligned with labels.')
    pred = _prediction_from_probability(prob, decision_threshold)
    labels = np.arange(prob.shape[1], dtype=int)
    out = {'macro_f1': float(f1_score(y_true, pred, labels=labels, average='macro', zero_division=0)), 'accuracy': float(accuracy_score(y_true, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)), 'kappa': float(cohen_kappa_score(y_true, pred)), 'auroc': np.nan, 'target_auprc': np.nan}
    try:
        if prob.shape[1] == 2:
            out['auroc'] = float(roc_auc_score(y_true, prob[:, 1]))
            if task == 'p300':
                out['target_auprc'] = float(average_precision_score(y_true, prob[:, 1]))
        else:
            out['auroc'] = float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro'))
    except ValueError:
        pass
    return out

def per_class_f1(y_true: np.ndarray, prob: np.ndarray, decision_threshold: Optional[float]=None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = _prediction_from_probability(prob, decision_threshold)
    return f1_score(y_true, pred, labels=np.arange(prob.shape[1]), average=None, zero_division=0).astype(float)

def apply_prior_correction(prob: np.ndarray, class_priors: np.ndarray, tau: float) -> np.ndarray:
    probability = np.asarray(prob, dtype=float)
    priors = np.asarray(class_priors, dtype=float)
    if probability.ndim != 2 or len(priors) != probability.shape[1]:
        return probability
    eps = 1e-08
    score = np.log(np.clip(probability, eps, 1.0)) - float(tau) * np.log(np.clip(priors, eps, 1.0))
    score -= score.max(axis=1, keepdims=True)
    adjusted = np.exp(score)
    adjusted /= np.clip(adjusted.sum(axis=1, keepdims=True), eps, None)
    return adjusted

def best_binary_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[1] != 2 or len(y) == 0 or (len(np.unique(y)) < 2):
        return (0.5, np.nan, np.nan)
    scores = prob[:, 1]
    candidates = np.unique(np.clip(np.r_[0.5, scores, np.quantile(scores, np.linspace(0.01, 0.99, 99))], 0.01, 0.99))
    rows = []
    for threshold in candidates:
        metrics = multiclass_metrics(y, prob, 'p300', float(threshold))
        rows.append((float(threshold), metrics))
    best_macro = max((float(metrics['macro_f1']) for _, metrics in rows))
    eligible = [item for item in rows if float(item[1]['macro_f1']) >= best_macro - float(cfg.QRG_MACRO_TOLERANCE) - 1e-12]
    threshold, metrics = max(eligible, key=lambda item: (qrg_secondary_score(item[1], 'p300'), float(item[1]['macro_f1']), -abs(item[0] - 0.5)))
    return (float(threshold), float(metrics['macro_f1']), float(metrics['balanced_accuracy']))

def best_prior_tau(y_true: np.ndarray, prob: np.ndarray, class_priors: np.ndarray, tau_grid: Sequence[float]) -> Tuple[float, float, float]:
    rows = []
    for tau in tau_grid:
        adjusted = apply_prior_correction(prob, class_priors, float(tau))
        rows.append((float(tau), multiclass_metrics(y_true, adjusted, 'mi')))
    if not rows:
        return (0.0, np.nan, np.nan)
    best_macro = max((float(metrics['macro_f1']) for _, metrics in rows))
    eligible = [item for item in rows if float(item[1]['macro_f1']) >= best_macro - float(cfg.QRG_MACRO_TOLERANCE) - 1e-12]
    tau, metrics = max(eligible, key=lambda item: (qrg_secondary_score(item[1], 'mi'), float(item[1]['macro_f1']), -abs(item[0])))
    return (float(tau), float(metrics['macro_f1']), float(metrics['balanced_accuracy']))

def _finite_mean(values: Sequence[float], default: float=-1000000000.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float(default)

def qrg_secondary_score(metrics: Mapping[str, float], task: str) -> float:
    vals = [metrics.get('accuracy', np.nan), metrics.get('balanced_accuracy', np.nan)]
    k = metrics.get('kappa', np.nan)
    if np.isfinite(k):
        vals.append(0.5 * (float(k) + 1.0))
    a = metrics.get('auroc', np.nan)
    if np.isfinite(a):
        vals.append(float(a))
    if task == 'p300':
        p = metrics.get('target_auprc', np.nan)
        if np.isfinite(p):
            vals.append(float(p))
    return _finite_mean(vals)

class DatasetSchemaError(RuntimeError):
    pass

def extract_archive(path: Path, destination: Path) -> Path:
    destination = Path(destination)
    marker = destination / '.extracted'
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(destination)
    else:
        shutil.unpack_archive(str(path), str(destination))
    marker.write_text('ok\n')
    return destination

def _coerce_trial_matrix(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or min(arr.shape) < 2:
        return None
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return np.asarray(arr, dtype=np.float32)

def _split_trial_container(value: Any) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            out.extend(_split_trial_container(item))
        return out
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        trial = _coerce_trial_matrix(arr)
        return [] if trial is None else [trial]
    if arr.ndim == 3:
        shapes = arr.shape
        if shapes[0] <= 64 and shapes[1] > 64:
            trial_axis = 2
        elif shapes[1] <= 64 and shapes[0] > 64:
            trial_axis = 2
        else:
            trial_axis = int(np.argmin(shapes))
        for k in range(shapes[trial_axis]):
            item = np.take(arr, k, axis=trial_axis)
            trial = _coerce_trial_matrix(item)
            if trial is not None:
                out.append(trial)
    return out

def _load_mat_any(path: Path) -> Dict[str, Any]:
    try:
        return {k: v for k, v in sio.loadmat(path, squeeze_me=True, struct_as_record=False).items() if not k.startswith('__')}
    except NotImplementedError:
        try:
            import mat73
            return mat73.loadmat(path)
        except Exception:
            with h5py.File(path, 'r') as h5:
                return {k: np.array(v) for k, v in h5.items()}

def _mat_to_mapping(obj: Any) -> Any:
    if hasattr(obj, '_fieldnames'):
        return {name: _mat_to_mapping(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, dict):
        return {k: _mat_to_mapping(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_mat_to_mapping(x) for x in obj.flat]
    return obj

def _find_lrre(mapping: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(mapping, dict):
        lower = {str(k).lower(): k for k in mapping}
        if 'l' in lower and 'r' in lower and ('re' in lower or 'rest' in lower):
            found.append(mapping)
        for value in mapping.values():
            found.extend(_find_lrre(value))
    elif isinstance(mapping, list):
        for value in mapping:
            found.extend(_find_lrre(value))
    return found

def stack_common_trials(trials: Sequence[np.ndarray], labels: Sequence[int], subjects: Sequence[str], sessions: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not trials:
        raise DatasetSchemaError('No valid EEG trials were parsed.')
    c_counts = pd.Series([x.shape[0] for x in trials]).value_counts()
    t_counts = pd.Series([x.shape[1] for x in trials]).value_counts()
    C = int(c_counts.index[0])
    T = int(t_counts.index[0])
    keep = [i for i, x in enumerate(trials) if x.shape == (C, T)]
    if not keep:
        raise DatasetSchemaError('No common channel/time shape survived parsing.')
    X = np.stack([trials[i] for i in keep]).astype(np.float32)
    return (X, np.asarray(labels)[keep].astype(np.int64), np.asarray(subjects, dtype=object)[keep], np.asarray(sessions, dtype=object)[keep])

def _ucl_subject_id(path: Path, raw: Mapping[str, Any]) -> Optional[int]:
    for key in raw:
        m = re.fullmatch('Subject(\\d+)', str(key), flags=re.I)
        if m:
            return int(m.group(1))
    m = re.search('(?:OGF?S|S)[_-]?(\\d+)', path.stem, flags=re.I)
    return int(m.group(1)) if m else None

def _ucl_clean_segments(trial: np.ndarray, raw_samples: int, class_name: str) -> List[np.ndarray]:
    x = _coerce_trial_matrix(trial)
    if x is None or x.shape[0] < 19:
        return []
    x = x[:19].astype(np.float32, copy=False)
    if x.shape[1] < raw_samples or not np.isfinite(x).all():
        return []
    if class_name != 'Rest' or x.shape[1] < 2 * raw_samples:
        start = (x.shape[1] - raw_samples) // 2
        return [x[:, start:start + raw_samples]]
    n_windows = x.shape[1] // raw_samples
    used = n_windows * raw_samples
    offset = max(0, (x.shape[1] - used) // 2)
    return [x[:, offset + k * raw_samples:offset + (k + 1) * raw_samples] for k in range(n_windows)]

def load_ucl_als_mi() -> EEGDatasetBundle:
    folder = local_dataset_folder('UCL_ALS_MI')
    expected_ids = [1, 2, 5, 9, 21, 31, 34, 39]
    raw_sfreq = 256.0
    raw_samples = int(round(cfg.UCL_MI_SECONDS * raw_sfreq))
    eeg_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    files_by_subject: Dict[int, Path] = {}
    raw_cache: Dict[Path, Dict[str, Any]] = {}
    for path in sorted(folder.rglob('*.mat')):
        try:
            raw = _load_mat_any(path)
        except Exception as exc:
            print('Skipping unreadable UCL MAT:', path.name, repr(exc))
            continue
        sid = _ucl_subject_id(path, raw)
        if sid in expected_ids and sid not in files_by_subject:
            files_by_subject[sid] = path
            raw_cache[path] = raw
    missing = [sid for sid in expected_ids if sid not in files_by_subject]
    if missing:
        raise DatasetSchemaError(f'UCL cache is missing expected ALS subject files: {missing}. Found={list(files_by_subject)}')
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[str] = []
    sessions: List[str] = []
    per_subject = {}
    for sid in expected_ids:
        path = files_by_subject[sid]
        raw = raw_cache[path]
        mapping = _mat_to_mapping(raw)
        groups = _find_lrre(mapping)
        if not groups:
            raise DatasetSchemaError(f'{path.name}: no L/R/Re structure was found.')
        subject_name = f'UCL_S{sid}'
        counts = {'Left': 0, 'Right': 0, 'Rest': 0}
        seen_arrays = set()
        for group_idx, group in enumerate(groups):
            lower = {str(k).lower(): k for k in group}
            if not {'l', 'r'}.issubset(lower) or not {'re', 'rest'} & set(lower):
                continue
            class_specs = [(lower['l'], 0, 'Left'), (lower['r'], 1, 'Right'), (lower['re'] if 're' in lower else lower['rest'], 2, 'Rest')]
            for actual_key, label, class_name in class_specs:
                for item in _split_trial_container(group[actual_key]):
                    for x in _ucl_clean_segments(item, raw_samples, class_name):
                        fingerprint = hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()
                        dedup_key = (class_name, fingerprint)
                        if dedup_key in seen_arrays:
                            continue
                        seen_arrays.add(dedup_key)
                        trials.append(x)
                        labels.append(label)
                        subjects.append(subject_name)
                        sessions.append(f'{subject_name}_longitudinal')
                        counts[class_name] += 1
        if min(counts.values()) == 0:
            raise DatasetSchemaError(f'{path.name}: all Left/Right/Rest classes are required; extracted {counts}.')
        per_subject[subject_name] = counts
        ratio = counts['Rest'] / max(1, min(counts['Left'], counts['Right']))
        if ratio < 0.2:
            print(f'WARNING {subject_name}: Rest remains sparse after real-window segmentation: {counts}')
        print(subject_name, counts)
    X = np.stack(trials).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subj = np.asarray(subjects, dtype=object)
    sess = np.asarray(sessions, dtype=object)
    parsed_subjects = sorted(np.unique(subj).tolist(), key=str)
    if set(parsed_subjects) != {f'UCL_S{s}' for s in expected_ids}:
        raise DatasetSchemaError(f'UCL subject gate failed: {parsed_subjects}')
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise DatasetSchemaError(f'UCL class gate failed: {np.unique(y).tolist()}')
    if X.shape[1] != 19:
        raise DatasetSchemaError(f'UCL EEG-channel gate failed: expected 19, got {X.shape[1]}')
    return EEGDatasetBundle(name='UCL_ALS_MI', task='mi', X=X, y=y, subjects=subj, sessions=sess, sfreq=raw_sfreq, ch_names=eeg_names, class_names=['Left', 'Right', 'Rest'], source='local UCL ALS imagery MAT files', metadata={'n_subjects': 8, 'subject_ids': expected_ids, 'trial_schema': 'L/R/Re', 'raw_channels': 22, 'eeg_channels_used': 19, 'signal_unit': 'unknown', 'epoch_seconds': cfg.UCL_MI_SECONDS, 'subject_class_counts': per_subject})

def load_bigp3bci_als() -> EEGDatasetBundle:
    folder = local_dataset_folder('BIGP3BCI_ALS')
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[str] = []
    sessions: List[str] = []
    ch_names_final: Optional[List[str]] = None
    sfreq_final: Optional[float] = None
    edf_files = sorted(folder.rglob('*.edf'))
    if not edf_files:
        raise DatasetSchemaError(f'No EDF files found under {folder}.')
    for path in edf_files:
        subject = _study_subject_from_path(path, cfg.BIGP3_STUDY)
        if subject is None:
            continue
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        eeg_names = [name for name in raw.ch_names if name.upper().startswith('EEG_')]
        begin_name = next((n for n in raw.ch_names if n.lower() == 'stimulusbegin'), None)
        type_name = next((n for n in raw.ch_names if n.lower() == 'stimulustype'), None)
        if not eeg_names or begin_name is None or type_name is None:
            continue
        eeg = raw.get_data(picks=eeg_names)
        begin = raw.get_data(picks=[begin_name])[0]
        stim_type = raw.get_data(picks=[type_name])[0]
        onset = np.flatnonzero((begin[1:] > 0.5) & (begin[:-1] <= 0.5)) + 1
        n_samp = int(round((cfg.P300_TMAX - cfg.P300_TMIN) * raw.info['sfreq']))
        session = next((p for p in path.parts if re.fullmatch('SE\\d+', p, flags=re.I)), 'SE000')
        for sample in onset:
            if sample + n_samp > eeg.shape[1]:
                continue
            trials.append(eeg[:, sample:sample + n_samp].astype(np.float32))
            labels.append(int(stim_type[sample] > 0.5))
            subjects.append(subject)
            sessions.append(session)
        ch_names_final = eeg_names
        sfreq_final = float(raw.info['sfreq'])
    X, y, subj, sess = stack_common_trials(trials, labels, subjects, sessions)
    unique_subjects = sorted(np.unique(subj).tolist(), key=str)
    if len(unique_subjects) != 8:
        raise DatasetSchemaError(f'bigP3BCI Study{cfg.BIGP3_STUDY} should provide 8 ALS participants; parsed {len(unique_subjects)}: {unique_subjects}')
    if set(np.unique(y).tolist()) != {0, 1}:
        raise DatasetSchemaError(f'bigP3BCI Study{cfg.BIGP3_STUDY} needs Target/NonTarget labels; parsed {np.unique(y).tolist()}')
    return EEGDatasetBundle(name='BIGP3BCI_ALS', task='p300', X=X, y=y, subjects=subj, sessions=sess, sfreq=sfreq_final or 256.0, ch_names=ch_names_final or [f'EEG{i + 1}' for i in range(X.shape[1])], class_names=['NonTarget', 'Target'], source=f'PhysioNet bigP3BCI Study{cfg.BIGP3_STUDY}', metadata={'study': cfg.BIGP3_STUDY, 'clinical_population': 'ALS', 'study_level_als_selection': True, 'n_subjects': 8, 'signal_unit': 'V', 'epoch_seconds': cfg.P300_TMAX - cfg.P300_TMIN})

def _eeget_subject_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        m = re.fullmatch('ALS[_-]?(\\d+)', str(part), flags=re.I)
        if m:
            return f'ALS{int(m.group(1)):02d}'
    return None

def _normalize_eeget_scenario(text: str) -> Optional[str]:
    import unicodedata
    s = ' '.join(str(text).strip().lower().replace('_', ' ').split())
    if not s:
        return None
    ascii_s = ''.join((ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)))
    left_aliases = {'nang tay trai', 'lift left hand', 'lifting left hand', 'left hand', 'left hand movement'}
    right_aliases = {'nang tay phai', 'lift right hand', 'lifting right hand', 'right hand', 'right hand movement'}
    if ascii_s in left_aliases or 'nang tay trai' in ascii_s:
        return 'left_hand'
    if ascii_s in right_aliases or 'nang tay phai' in ascii_s:
        return 'right_hand'
    return None

def _read_eeget_scenario(sample_dir: Path) -> Optional[str]:
    for meta_name in ('eeg.json', 'scenario.json'):
        meta_path = sample_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            obj = json.loads(meta_path.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            continue
        if isinstance(obj, dict):
            for key in ('Scenario', 'scenario', 'Task', 'task', 'Name', 'name'):
                if key in obj and str(obj[key]).strip():
                    return str(obj[key]).strip()
    return None

def _eeget_event_kind(description: str) -> Optional[str]:
    s = ' '.join(str(description).strip().replace('_', ' ').split()).lower()
    if s == 'thinking':
        return 'thinking'
    if s == 'resting' or s == 'rest':
        return 'resting'
    if s in {'thinking and acting', 'think acting', 'acting', 'typing'}:
        return None
    return None

def _eeget_eeg_picks(raw: mne.io.BaseRaw) -> Tuple[List[int], List[str]]:
    official = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
    lower_to_idx = {str(name).lower(): i for i, name in enumerate(raw.ch_names)}
    if all((name.lower() in lower_to_idx for name in official)):
        picks = [lower_to_idx[name.lower()] for name in official]
        return (picks, official)
    picks_arr = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude=[])
    if len(picks_arr) >= 8:
        picks = [int(i) for i in picks_arr]
        return (picks, [raw.ch_names[i] for i in picks])
    if len(raw.ch_names) >= 8:
        n = min(32, len(raw.ch_names))
        picks = list(range(n))
        return (picks, [raw.ch_names[i] for i in picks])
    raise DatasetSchemaError(f'EEGET EDF exposes too few EEG-like signals: {len(raw.ch_names)} channels.')

def _crop_eeget_annotation(eeg: np.ndarray, sfreq: float, onset: float, duration: float, seconds: float) -> Optional[np.ndarray]:
    n_samp = int(round(float(seconds) * float(sfreq)))
    if n_samp < 2:
        return None
    event_start = int(round(float(onset) * float(sfreq)))
    event_len = int(round(max(float(duration), 0.0) * float(sfreq)))
    if event_len < n_samp:
        return None
    start = event_start + (event_len - n_samp) // 2
    stop = start + n_samp
    if start < 0 or stop > eeg.shape[1]:
        return None
    return eeg[:, start:stop].astype(np.float32, copy=False)

def load_eeget_als_mi() -> EEGDatasetBundle:
    folder = local_dataset_folder('EEGET_ALS_MI')
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[str] = []
    sessions: List[str] = []
    sfreqs: List[float] = []
    channel_counts: List[int] = []
    channel_name_sets: List[Tuple[str, ...]] = []
    scenario_audit: List[Dict[str, Any]] = []
    edf_files = sorted((p for p in folder.rglob('*.edf') if p.name.lower() == 'eeg.edf' and _eeget_subject_from_path(p) is not None))
    if not edf_files:
        raise DatasetSchemaError(f'EEGET-ALS local directory contains no ALS EEG.edf files under {folder}.')
    for path in edf_files:
        subject = _eeget_subject_from_path(path)
        sample_dir = path.parent
        scenario_raw = _read_eeget_scenario(sample_dir)
        scenario = _normalize_eeget_scenario(scenario_raw or '')
        if scenario not in {'left_hand', 'right_hand'}:
            continue
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        picks, picked_names = _eeget_eeg_picks(raw)
        eeg = raw.get_data(picks=picks).astype(np.float32, copy=False)
        sfreq = float(raw.info['sfreq'])
        local_counts = {'Left': 0, 'Right': 0, 'Rest': 0}
        for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            event = _eeget_event_kind(str(desc))
            if event is None:
                continue
            if event == 'thinking':
                label = 0 if scenario == 'left_hand' else 1
            elif event == 'resting':
                label = 2
            else:
                continue
            epoch = _crop_eeget_annotation(eeg, sfreq, float(onset), float(duration), cfg.EEGET_MI_SECONDS)
            if epoch is None:
                continue
            try:
                rel_session = str(sample_dir.relative_to(folder))
            except Exception:
                rel_session = sample_dir.name
            trials.append(epoch)
            labels.append(label)
            subjects.append(subject)
            sessions.append(rel_session)
            sfreqs.append(sfreq)
            channel_counts.append(epoch.shape[0])
            channel_name_sets.append(tuple(picked_names))
            if label == 0:
                local_counts['Left'] += 1
            elif label == 1:
                local_counts['Right'] += 1
            else:
                local_counts['Rest'] += 1
        scenario_audit.append({'subject': subject, 'sample': str(sample_dir), 'scenario_raw': scenario_raw, 'scenario': scenario, **local_counts})
    if not trials:
        audit_path = RUN_DIR / 'EEGET_LR0_scenario_audit.csv'
        pd.DataFrame(scenario_audit).to_csv(audit_path, index=False)
        raise DatasetSchemaError('EEGET-ALS local recordings were found, but no LR0 epochs survived the official scenario+event parser. Inspect ' + str(audit_path))
    C = int(pd.Series(channel_counts).mode().iloc[0])
    sfreq = float(pd.Series(sfreqs).mode().iloc[0])
    kept = [i for i, (x, sf) in enumerate(zip(trials, sfreqs)) if x.shape[0] == C and abs(float(sf) - sfreq) < 1e-06]
    if not kept:
        raise DatasetSchemaError('EEGET-ALS has no consistent EEG acquisition schema.')
    X = np.stack([trials[i] for i in kept]).astype(np.float32)
    y = np.asarray([labels[i] for i in kept], dtype=np.int64)
    subj = np.asarray([subjects[i] for i in kept], dtype=object)
    sess = np.asarray([sessions[i] for i in kept], dtype=object)
    ch_names = list(channel_name_sets[kept[0]])
    parsed_subjects = sorted(np.unique(subj).tolist(), key=str)
    if len(parsed_subjects) != 6:
        audit_path = RUN_DIR / 'EEGET_LR0_scenario_audit.csv'
        pd.DataFrame(scenario_audit).to_csv(audit_path, index=False)
        raise DatasetSchemaError(f'EEGET primary LR0 experiment requires 6 ALS patients; parsed {len(parsed_subjects)}: {parsed_subjects}. Inspect {audit_path}')
    per_subject_rows = []
    for s in parsed_subjects:
        idx = subj == s
        counts = {int(k): int(v) for k, v in zip(*np.unique(y[idx], return_counts=True))}
        per_subject_rows.append({'subject': s, 'Left': counts.get(0, 0), 'Right': counts.get(1, 0), 'Rest': counts.get(2, 0)})
        if set(counts) != {0, 1, 2}:
            print(f'WARNING EEGET {s}: available LR0 classes are {counts}; patient retained exactly as recorded.')
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise DatasetSchemaError(f'EEGET-ALS dataset-level LR0 gate requires Left/Right/Rest; parsed {np.unique(y).tolist()}.')
    pd.DataFrame(scenario_audit).to_csv(RUN_DIR / 'EEGET_LR0_scenario_audit.csv', index=False)
    pd.DataFrame(per_subject_rows).to_csv(RUN_DIR / 'EEGET_LR0_subject_counts.csv', index=False)
    print('EEGET LR0 subject counts:')
    display(pd.DataFrame(per_subject_rows))
    return EEGDatasetBundle(name='EEGET_ALS_MI', task='mi', X=X, y=y, subjects=subj, sessions=sess, sfreq=sfreq, ch_names=ch_names, class_names=['Left', 'Right', 'Rest'], source='local EEGET ALS imagery EDF files', metadata={'strict_schema': 'official LR0 scenario+event parser', 'lr0_mapping': {'left_hand+Thinking': 'Left', 'right_hand+Thinking': 'Right', 'Resting': 'Rest'}, 'excluded_events': ['Thinking and Acting', 'Typing'], 'signal_unit': 'V', 'absolute_amplitude_gate': False, 'amplitude_policy_reason': 'fixed EEGET rule: retain official parsed epochs; log amplitude QC without generic peak exclusion', 'epoch_seconds': cfg.EEGET_MI_SECONDS, 'n_subjects': 6, 'native_channels': C, 'raw_sfreq': sfreq})
'Read datasets supplied alongside the experiment code.'

def local_dataset_folder(name):
    folder = Path(cfg.DATA_ROOT) / name
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing local dataset directory: {folder}. Supply the dataset as described in README.md.')
    return folder

def _bnci_field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    if isinstance(value, np.void) and value.dtype.names and (name in value.dtype.names):
        return value[name]
    return getattr(value, name, None)

def _bnci_runs(value):
    if all((_bnci_field(value, field) is not None for field in ('X', 'y', 'channels', 'classes'))):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _bnci_runs(item)
    elif isinstance(value, np.ndarray) and (value.dtype.hasobject or value.dtype.names):
        for item in value.ravel():
            yield from _bnci_runs(item)
    elif isinstance(value, dict) and 'data' in value:
        yield from _bnci_runs(value['data'])

def _bnci_subject_file(path, root):
    for part in path.relative_to(root).parts[:-1]:
        if part in {str(i) for i in range(1, 9)}:
            return int(part)
        match = re.fullmatch('(?:subject|sub|participant)[_-]?0*([1-8])', part, flags=re.I)
        if match:
            return int(match.group(1))
    match = re.fullmatch('[A-Za-z_-]*0*([1-8])', path.stem)
    return int(match.group(1)) if match else None

def _bnci_run_arrays(run, filename):
    channels = [str(x).strip() for x in np.asarray(_bnci_field(run, 'channels'), dtype=object).ravel()]
    if len(channels) != 8:
        raise DatasetSchemaError(f'{filename}: BNCI2014_008 requires eight EEG channels, found {len(channels)}.')
    signal = np.asarray(_bnci_field(run, 'X'), dtype=np.float64)
    if signal.ndim != 2:
        raise DatasetSchemaError(f'{filename}: run.X must be a samples-by-channels matrix.')
    if signal.shape[1] != len(channels) and signal.shape[0] == len(channels):
        signal = signal.T
    events = np.asarray(_bnci_field(run, 'y')).reshape(-1)
    if signal.shape != (len(events), len(channels)) or not np.isfinite(signal).all():
        raise DatasetSchemaError(f'{filename}: continuous EEG and dense stimulus labels are inconsistent.')
    if not np.isfinite(events).all() or not np.equal(events, np.rint(events)).all():
        raise DatasetSchemaError(f'{filename}: stimulus labels must be finite integer codes.')
    classes = [str(x) for x in np.asarray(_bnci_field(run, 'classes'), dtype=object).ravel()]
    event_map = {}
    for event_id, label in enumerate(classes, 1):
        label = re.sub('[^a-z]', '', label.lower())
        if label == 'target':
            event_map[event_id] = 1
        elif label in {'nontarget', 'nonattended', 'unattended'}:
            event_map[event_id] = 0
    if set(event_map.values()) != {0, 1}:
        raise DatasetSchemaError(f'{filename}: run.classes must identify Target and NonTarget.')
    observed = set(events.astype(np.int64)) - {0}
    if not observed <= set(event_map):
        raise DatasetSchemaError(f'{filename}: unexpected target-stimulus codes {sorted(observed - set(event_map))}.')
    return (signal, events.astype(np.int64), channels, event_map)

def load_bnci2014_008():
    folder = local_dataset_folder('BNCI2014_008')
    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.mat']
    participants = {}
    for path in sorted(files):
        subject = _bnci_subject_file(path, folder)
        if subject is None:
            raise DatasetSchemaError(f'{path.name}: put the participant MAT file in a numbered subdirectory 1 through 8.')
        participants.setdefault(subject, []).append(path)
    if set(participants) != set(range(1, 9)):
        raise DatasetSchemaError(f'BNCI2014_008 requires participants 1 through 8; found {sorted(participants)}.')
    epochs_all, labels, subjects, sessions = ([], [], [], [])
    channel_names = None
    for subject, paths in sorted(participants.items()):
        for path in paths:
            content = _load_mat_any(path)
            runs = list(_bnci_runs(content))
            if not runs:
                raise DatasetSchemaError(f'{path.name}: expected Santa Lucia data structs with X, y, channels and classes.')
            for run in runs:
                signal, stimulus, channels, event_map = _bnci_run_arrays(run, path.name)
                if channel_names is not None and channels != channel_names:
                    raise DatasetSchemaError('BNCI channel identities/order differ between recordings.')
                channel_names = channels
                info = mne.create_info(channels + ['STI_TARGET'], sfreq=256.0, ch_types=['eeg'] * len(channels) + ['stim'])
                raw = mne.io.RawArray(np.vstack([signal.T * 1e-06, stimulus[None, :]]), info, verbose='ERROR')
                events = mne.find_events(raw, stim_channel='STI_TARGET', shortest_event=0, initial_event=True, verbose='ERROR')
                if not len(events):
                    continue
                raw.filter(cfg.P300_FMIN, cfg.P300_FMAX, picks='eeg', verbose='ERROR')
                event_id = {'Target' if label else 'NonTarget': code for code, label in event_map.items()}
                epochs = mne.Epochs(raw, events, event_id=event_id, tmin=cfg.P300_TMIN, tmax=cfg.P300_TMAX, baseline=None, picks='eeg', preload=True, reject=None, proj=False, on_missing='ignore', verbose='ERROR')
                if not len(epochs):
                    continue
                epochs.resample(float(cfg.RESAMPLE), npad='auto', verbose='ERROR')
                X = (epochs.get_data(copy=True) * 1000000.0).astype(np.float32)
                y = np.asarray([event_map[int(code)] for code in epochs.events[:, 2]], np.int64)
                epochs_all.append(X)
                labels.extend(y.tolist())
                subjects.extend([str(subject)] * len(y))
                sessions.extend(['0'] * len(y))
    if not epochs_all:
        raise DatasetSchemaError('BNCI2014_008: no valid target/non-target epochs found in the local MAT files.')
    return EEGDatasetBundle(name='BNCI2014_008', task='p300', X=np.concatenate(epochs_all), y=np.asarray(labels, np.int64), subjects=np.asarray(subjects), sessions=np.asarray(sessions), sfreq=float(cfg.RESAMPLE), ch_names=channel_names, class_names=['NonTarget', 'Target'], source='local BNCI2014_008 Santa Lucia MAT files', metadata={'clinical_population': 'ALS', 'signal_unit': 'uV', 'preprocessed_bandpass': [cfg.P300_FMIN, cfg.P300_FMAX], 'preprocessed_resample': cfg.RESAMPLE, 'absolute_amplitude_gate': False, 'amplitude_policy_reason': 'No additional absolute amplitude gate for the BNCI P300 paradigm.', 'epoch_seconds': float(cfg.P300_TMAX - cfg.P300_TMIN)})

def read_local_prepared_bundle(path, name):
    """Read the original non-pickle prepared-bundle format."""
    with np.load(path, allow_pickle=False) as data:
        required = {'X', 'y', 'subjects', 'sessions', 'header'}
        if not required <= set(data.files):
            raise DatasetSchemaError(f'{path}: prepared bundle requires {sorted(required)}.')
        header = json.loads(str(data['header'].item()))

        def text_array(field):
            values = data[field]
            if values.ndim == 0:
                values = np.asarray(json.loads(str(values.item()))).astype(str)
            return np.asarray(values).astype(str)
        bundle = EEGDatasetBundle(name, header['task'], data['X'].astype(np.float32), data['y'].copy(), text_array('subjects'), text_array('sessions'), float(header['sfreq']), list(header['ch_names']), list(header['class_names']), 'local prepared ' + name, dict(header.get('metadata', {})))
    task = 'p300' if name in ('BNCI2014_008', 'BIGP3BCI_ALS') else 'mi'
    classes = ['NonTarget', 'Target'] if task == 'p300' else ['Left', 'Right', 'Rest']
    if bundle.task != task or bundle.class_names != classes:
        raise DatasetSchemaError(f'{path}: the prepared task/class order must be {task}: {classes}.')
    if bundle.X.ndim != 3 or bundle.y.ndim != 1 or bundle.y.dtype.kind not in 'iu':
        raise DatasetSchemaError(f'{path}: X must be [trials, channels, samples] and y integer [trials].')
    seconds = 0.8 if task == 'p300' else 4.0 if name == 'UCL_ALS_MI' else 2.0
    expected_samples = int(round(seconds * cfg.RESAMPLE))
    if bundle.sfreq != float(cfg.RESAMPLE) or bundle.X.shape[-1] != expected_samples:
        raise DatasetSchemaError(f'{path}: prepared data must already be {cfg.RESAMPLE} Hz with {expected_samples} samples per trial.')
    if len(bundle.ch_names) != bundle.X.shape[1] or bundle.X.shape[1] < 8:
        raise DatasetSchemaError(f'{path}: channel metadata do not match the prepared EEG.')
    if name == 'BNCI2014_008' and bundle.X.shape[1] != 8:
        raise DatasetSchemaError(f'{path}: BNCI2014_008 requires eight EEG channels.')
    if name == 'UCL_ALS_MI' and bundle.X.shape[1] != 19:
        raise DatasetSchemaError(f'{path}: UCL requires 19 EEG channels.')
    if bundle.subjects.ndim != 1 or bundle.sessions.ndim != 1:
        raise DatasetSchemaError(f'{path}: subjects and sessions must be one-dimensional.')
    validate_final_bundle(bundle)
    return bundle

def fixed_load_dataset(name):
    folder = local_dataset_folder(name)
    prepared = folder / 'bundle.npz'
    with fixed_operation('load_local_dataset', dataset=name, device='cpu'):
        if prepared.is_file():
            bundle = read_local_prepared_bundle(prepared, name)
        else:
            with fixed_quiet():
                bundle = fixed_preprocess(DATASET_LOADERS[name]())
            validate_final_bundle(bundle)
        identity = fixed_bundle_identity(bundle)
        identity_path = RUN_DIR / 'dataset_identities' / f'{name}.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != identity:
            raise RuntimeError('Local EEG differs from the completed work in this run: ' + name)
        fixed_atomic_json(identity_path, {'dataset': name, 'identity': identity, 'shape': list(bundle.X.shape), 'input_policy': 'local files supplied by the user'})
        audit = bundle.audit()
        audit.insert(0, 'dataset', name)
        _atomic_csv(RUN_DIR / (name + '__dataset_audit.csv'), audit)
        fixed_emit('DATASET_READY', dataset=name, detail=str(bundle.X.shape) + ' subjects=' + str(len(np.unique(bundle.subjects))))
        return bundle

def _study_subject_from_path(path, study_letter):
    pattern = re.compile(re.escape(study_letter) + '_[0-9]+', flags=re.I)
    for part in Path(path).parts:
        if pattern.fullmatch(part):
            return part.upper()
    return None
DATASET_LOADERS = {'BNCI2014_008': load_bnci2014_008, 'UCL_ALS_MI': load_ucl_als_mi, 'BIGP3BCI_ALS': load_bigp3bci_als, 'EEGET_ALS_MI': load_eeget_als_mi}

def _bandpass_array(X: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    if fmax >= sfreq / 2:
        fmax = sfreq / 2 - 0.5
    sos = sps.butter(4, [fmin, fmax], btype='bandpass', fs=sfreq, output='sos')
    return sps.sosfiltfilt(sos, X, axis=-1).astype(np.float32)

def _resample_array(X: np.ndarray, old_sfreq: float, new_sfreq: float) -> np.ndarray:
    if abs(old_sfreq - new_sfreq) < 1e-06:
        return X.astype(np.float32)
    new_t = int(round(X.shape[-1] * new_sfreq / old_sfreq))
    return sps.resample(X, new_t, axis=-1).astype(np.float32)

def _crop_or_pad(X: np.ndarray, length: int) -> np.ndarray:
    if X.shape[-1] > length:
        start = (X.shape[-1] - length) // 2
        return X[..., start:start + length]
    if X.shape[-1] < length:
        left = (length - X.shape[-1]) // 2
        right = length - X.shape[-1] - left
        mode = 'reflect' if X.shape[-1] > 1 else 'edge'
        return np.pad(X, [(0, 0), (0, 0), (left, right)], mode=mode)
    return X

def deterministic_subject_cap(bundle: EEGDatasetBundle, cap: Optional[int], seed: int) -> EEGDatasetBundle:
    if cap is None:
        return bundle
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for subject in sorted(np.unique(bundle.subjects), key=str):
        idx = np.flatnonzero(bundle.subjects == subject)
        if len(idx) <= cap:
            keep.extend(idx.tolist())
            continue
        cls, cnt = np.unique(bundle.y[idx], return_counts=True)
        raw = cap * cnt / cnt.sum()
        quota = np.floor(raw).astype(int)
        if cap >= len(cls):
            quota = np.maximum(quota, 1)
        while quota.sum() > cap:
            j = int(np.argmax(quota - 1))
            quota[j] -= 1
        remainder = cap - quota.sum()
        order = np.argsort(-(raw - np.floor(raw)))
        for j in order[:remainder]:
            quota[j] += 1
        selected = []
        for c, n in zip(cls, quota):
            cidx = idx[bundle.y[idx] == c]
            selected.extend(rng.choice(cidx, size=min(int(n), len(cidx)), replace=False).tolist())
        if len(selected) < cap:
            remaining = np.setdiff1d(idx, np.asarray(selected, dtype=int))
            selected.extend(rng.choice(remaining, size=min(cap - len(selected), len(remaining)), replace=False).tolist())
        keep.extend(selected[:cap])
    keep = np.asarray(sorted(set(keep)), dtype=int)
    return EEGDatasetBundle(name=bundle.name, task=bundle.task, X=bundle.X[keep], y=bundle.y[keep], subjects=bundle.subjects[keep], sessions=bundle.sessions[keep], sfreq=bundle.sfreq, ch_names=bundle.ch_names, class_names=bundle.class_names, source=bundle.source, metadata={**bundle.metadata, 'trial_cap': cap, 'trial_cap_policy': 'proportional-with-minimum-class-presence'})

def fixed_preprocess(bundle: EEGDatasetBundle) -> EEGDatasetBundle:
    X = np.asarray(bundle.X, dtype=np.float32)
    if len(X) == 0:
        raise DatasetSchemaError(f'{bundle.name}: loader returned zero trials before preprocessing.')
    if bundle.task == 'p300':
        already_filtered = bundle.metadata.get('preprocessed_bandpass') == [cfg.P300_FMIN, cfg.P300_FMAX]
        if not already_filtered:
            X = _bandpass_array(X, bundle.sfreq, cfg.P300_FMIN, cfg.P300_FMAX)
        X = _resample_array(X, bundle.sfreq, cfg.RESAMPLE)
        target_seconds = float(bundle.metadata.get('epoch_seconds', cfg.P300_TMAX - cfg.P300_TMIN))
        X = _crop_or_pad(X, int(round(target_seconds * cfg.RESAMPLE)))
    else:
        X = _bandpass_array(X, bundle.sfreq, cfg.MI_FMIN, cfg.MI_FMAX)
        X = _resample_array(X, bundle.sfreq, cfg.RESAMPLE)
        target_seconds = float(bundle.metadata.get('epoch_seconds', cfg.MI_SECONDS))
        X = _crop_or_pad(X, int(round(target_seconds * cfg.RESAMPLE)))
    if bool(bundle.metadata.get('artifact_already_rejected', False)):
        good = np.ones(len(X), dtype=bool)
        artifact_policy = 'dataset-preprocessed; no second amplitude gate'
    elif bundle.metadata.get('absolute_amplitude_gate', True) is False:
        good = np.ones(len(X), dtype=bool)
        artifact_policy = str(bundle.metadata.get('amplitude_policy_reason', 'fixed dataset rule; no absolute amplitude gate'))
    else:
        unit = str(bundle.metadata.get('signal_unit', 'unknown')).lower()
        if unit in {'v', 'volt', 'volts'}:
            peak_uv = np.max(np.abs(X), axis=(1, 2)) * 1000000.0
            good = peak_uv <= cfg.ARTIFACT_UV
            artifact_policy = f'fixed {cfg.ARTIFACT_UV:g} uV peak gate'
        elif unit in {'uv', 'µv', 'microvolt', 'microvolts'}:
            peak_uv = np.max(np.abs(X), axis=(1, 2))
            good = peak_uv <= cfg.ARTIFACT_UV
            artifact_policy = f'fixed {cfg.ARTIFACT_UV:g} uV peak gate'
        else:
            good = np.ones(len(X), dtype=bool)
            artifact_policy = 'unknown units; no absolute amplitude gate'
        if not np.any(good):
            raise DatasetSchemaError(f'{bundle.name}: fixed amplitude gate rejected every trial; verify the declared signal unit.')
    out = EEGDatasetBundle(name=bundle.name, task=bundle.task, X=X[good], y=bundle.y[good], subjects=bundle.subjects[good], sessions=bundle.sessions[good], sfreq=float(cfg.RESAMPLE), ch_names=bundle.ch_names[:X.shape[1]], class_names=bundle.class_names, source=bundle.source, metadata={**bundle.metadata, 'artifact_keep_rate': float(good.mean()), 'artifact_policy': artifact_policy, 'final_samples': int(X.shape[-1]), 'final_sfreq': cfg.RESAMPLE})
    if len(out.X) == 0:
        raise DatasetSchemaError(f'{bundle.name}: zero trials survived fixed preprocessing.')
    return out

def trialwise_zscore(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-06)).astype(np.float32)

class RobustChannelScaler:

    def fit(self, X: np.ndarray) -> 'RobustChannelScaler':
        centered = X - X.mean(axis=-1, keepdims=True)
        self.scale_ = np.median(np.abs(centered), axis=(0, 2)) / 0.67448975
        positive = self.scale_[self.scale_ > 0]
        floor = np.median(positive) * 0.001 if len(positive) else 1.0
        self.scale_ = np.maximum(self.scale_, max(floor, 1e-08))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        centered = X - X.mean(axis=-1, keepdims=True)
        return (centered / self.scale_[None, :, None]).astype(np.float32)

def outer_subjects(bundle, max_folds=None, seed=None):
    return sorted(set(map(str, bundle.subjects)), key=str)

def choose_inner_subjects(source_subjects: Sequence[str], mode: str, seed: int) -> List[str]:
    source_subjects = sorted(map(str, source_subjects))
    if mode == 'server_test':
        limit = cfg.INNER_MAX_SUBJECTS_SERVER
    elif mode == 'architecture':
        limit = cfg.INNER_MAX_SUBJECTS_ARCH
    else:
        limit = cfg.INNER_MAX_SUBJECTS_PAPER
    if limit is None or len(source_subjects) <= limit:
        return source_subjects
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(source_subjects, size=limit, replace=False).tolist())

def split_indices(bundle: EEGDatasetBundle, test_subject: str, val_subject: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj = bundle.subjects.astype(str)
    test = np.flatnonzero(subj == str(test_subject))
    if val_subject is None:
        train = np.flatnonzero(subj != str(test_subject))
        val = np.array([], dtype=int)
    else:
        val = np.flatnonzero(subj == str(val_subject))
        train = np.flatnonzero((subj != str(test_subject)) & (subj != str(val_subject)))
    if len(test) == 0 or len(train) == 0:
        raise RuntimeError('Invalid subject split.')
    return (train, val, test)

def assert_no_subject_overlap(train_subjects: np.ndarray, val_subjects: np.ndarray, test_subjects: np.ndarray) -> None:
    a, b, c = (set(map(str, train_subjects)), set(map(str, val_subjects)), set(map(str, test_subjects)))
    if a & b or a & c or b & c:
        raise RuntimeError('Subject leakage detected between train/validation/test.')

def mean_trial_covariance(X: np.ndarray, eps: float=1e-06, trace_normalize: bool=False) -> np.ndarray:
    covs = []
    for trial in X:
        z = trial - trial.mean(axis=1, keepdims=True)
        cov = z @ z.T / max(1, z.shape[1] - 1)
        if trace_normalize:
            cov = cov / max(float(np.trace(cov)), eps)
        covs.append(cov)
    C = np.mean(covs, axis=0)
    ridge = eps * max(float(np.trace(C)) / C.shape[0], np.finfo(np.float64).tiny)
    return ((C + C.T) / 2 + ridge * np.eye(C.shape[0])).astype(np.float64)

def _trial_covariances_for_csp(X: np.ndarray) -> np.ndarray:
    covs = []
    for trial in X:
        z = trial - trial.mean(axis=1, keepdims=True)
        C = z @ z.T / max(1, z.shape[1] - 1)
        C = C / max(float(np.trace(C)), 1e-10)
        C = 0.5 * (C + C.T) + 1e-06 * np.eye(C.shape[0]) / C.shape[0]
        covs.append(C)
    return np.stack(covs)

def _fisher_score_1d(values: np.ndarray, y: np.ndarray) -> float:
    values = np.asarray(values, float)
    y = np.asarray(y)
    mu = float(values.mean())
    between = 0.0
    within = 0.0
    for cls in np.unique(y):
        z = values[y == cls]
        if len(z) == 0:
            continue
        m = float(z.mean())
        between += len(z) * (m - mu) ** 2
        within += float(np.sum((z - m) ** 2))
    return between / max(within, 1e-12)

def _stable_sign_columns(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float).copy()
    for j in range(P.shape[1]):
        i = int(np.argmax(np.abs(P[:, j])))
        if P[i, j] < 0:
            P[:, j] *= -1
    return P

class SpatialProjector:

    def __init__(self, q: int, mode: str):
        self.q = int(q)
        self.mode = str(mode).lower()

    def _fit_csp(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        C = X.shape[1]
        covs = _trial_covariances_for_csp(X)
        candidates = []
        n_each = max(2, math.ceil(self.q / max(1, len(np.unique(y)))))
        for cls in np.unique(y):
            in_cls = covs[y == cls]
            out_cls = covs[y != cls]
            if len(in_cls) == 0 or len(out_cls) == 0:
                continue
            A = np.mean(in_cls, axis=0)
            B = np.mean(out_cls, axis=0)
            composite = A + B + 1e-06 * np.eye(C)
            vals, vecs = eigh(A, composite)
            order_hi = np.argsort(vals)[::-1][:n_each]
            order_lo = np.argsort(vals)[:n_each]
            for j in np.unique(np.concatenate([order_hi, order_lo])):
                w = vecs[:, int(j)].astype(float)
                w /= max(np.linalg.norm(w), 1e-12)
                feat = np.log(np.maximum(np.einsum('i,nij,j->n', w, covs, w), 1e-12))
                candidates.append((_fisher_score_1d(feat, y), w))
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for _, w0 in candidates:
            w = w0.copy()
            for b in selected:
                w -= b * np.dot(b, w)
            n = np.linalg.norm(w)
            if n < 1e-06:
                continue
            w /= n
            selected.append(w)
            if len(selected) >= self.q:
                break
        if len(selected) < self.q:
            overall = mean_trial_covariance(X, trace_normalize=True)
            vals, vecs = eigh(overall)
            for j in np.argsort(vals)[::-1]:
                w = vecs[:, j].astype(float)
                for b in selected:
                    w -= b * np.dot(b, w)
                n = np.linalg.norm(w)
                if n < 1e-06:
                    continue
                selected.append(w / n)
                if len(selected) >= self.q:
                    break
        if len(selected) < self.q:
            raise RuntimeError(f'CSP projection produced only {len(selected)} independent components for q={self.q}.')
        return np.stack(selected[:self.q], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SpatialProjector':
        C = X.shape[1]
        if self.q > C:
            raise ValueError(f'q={self.q} exceeds C={C}.')
        if self.mode == 'csp':
            P = self._fit_csp(X, y)
        elif self.mode == 'xdawn' and len(np.unique(y)) == 2:
            overall = mean_trial_covariance(X)
            target = X[y == 1]
            if len(target) < 2:
                raise RuntimeError('xDAWN requires source target trials.')
            erp = target.mean(axis=0)
            signal_cov = erp @ erp.T / max(1, erp.shape[1] - 1)
            covariance_scale = max(float(np.trace(overall)) / C, np.finfo(np.float64).tiny)
            vals, vecs = eigh(signal_cov + 1e-06 * covariance_scale * np.eye(C), overall + 1e-06 * covariance_scale * np.eye(C))
            P = vecs[:, np.argsort(vals)[::-1][:self.q]]
            P = np.linalg.qr(P)[0][:, :self.q]
        elif self.mode == 'pca':
            overall = mean_trial_covariance(X)
            vals, vecs = eigh(overall)
            P = vecs[:, np.argsort(vals)[::-1][:self.q]]
            P = np.linalg.qr(P)[0][:, :self.q]
        else:
            raise ValueError(f'Unknown/invalid projection mode {self.mode!r}.')
        self.P_ = _stable_sign_columns(P).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return np.empty((0, self.q, X.shape[-1]), dtype=np.float32)
        return np.einsum('cq,nct->nqt', self.P_, X).astype(np.float32)

def shrink_covariance(trial: np.ndarray) -> np.ndarray:
    X = trial.T
    est = OAS(store_precision=False, assume_centered=False).fit(X)
    C = np.asarray(est.covariance_, dtype=np.float64)
    C += 1e-07 * max(np.trace(C) / C.shape[0], 1e-08) * np.eye(C.shape[0])
    return 0.5 * (C + C.T)

def spd_sqrt_and_inv(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = eigh((C + C.T) / 2)
    vals = np.maximum(vals, 1e-10)
    return (vecs * np.sqrt(vals) @ vecs.T, vecs * (1.0 / np.sqrt(vals)) @ vecs.T)

def spd_log(C: np.ndarray) -> np.ndarray:
    vals, vecs = eigh((C + C.T) / 2)
    vals = np.maximum(vals, 1e-10)
    return vecs * np.log(vals) @ vecs.T

def spd_exp(S: np.ndarray) -> np.ndarray:
    vals, vecs = eigh((S + S.T) / 2)
    return vecs * np.exp(vals) @ vecs.T

def riemannian_mean(covs: np.ndarray, max_iter: int=30, tol: float=1e-08) -> np.ndarray:
    M = np.mean(covs, axis=0)
    for _ in range(max_iter):
        sqrt, invsqrt = spd_sqrt_and_inv(M)
        delta = np.mean([spd_log(invsqrt @ C @ invsqrt) for C in covs], axis=0)
        norm = np.linalg.norm(delta, ord='fro')
        M = sqrt @ spd_exp(delta) @ sqrt
        M = 0.5 * (M + M.T)
        if norm < tol:
            break
    return M

def tangent_matrix(C: np.ndarray, reference: np.ndarray) -> np.ndarray:
    _, invsqrt = spd_sqrt_and_inv(reference)
    H = spd_log(invsqrt @ C @ invsqrt)
    return (0.5 * (H + H.T)).astype(np.float32)

def sym_to_vec(H: np.ndarray) -> np.ndarray:
    q = H.shape[-1]
    idx = np.triu_indices(q)
    vec = H[..., idx[0], idx[1]].copy()
    off = idx[0] != idx[1]
    vec[..., off] *= math.sqrt(2.0)
    return vec.astype(np.float32)

class RobustFeatureScaler:

    def fit(self, X: np.ndarray) -> 'RobustFeatureScaler':
        X = np.asarray(X, float)
        self.center_ = np.median(X, axis=0)
        self.scale_ = np.median(np.abs(X - self.center_), axis=0) / 0.67448975
        fallback = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ > 1e-08, self.scale_, np.maximum(fallback, 1e-06))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if len(X) == 0:
            return np.empty_like(X, dtype=np.float32)
        Z = (X - self.center_) / self.scale_
        return np.clip(Z, -cfg.ROBUST_CLIP, cfg.ROBUST_CLIP).astype(np.float32)

def mst_plus_strong_edges(weight_matrix: np.ndarray, n_edges: int) -> List[Tuple[int, int]]:
    q = weight_matrix.shape[0]
    n_edges = int(np.clip(n_edges, q - 1, q * (q - 1) // 2))
    graph = nx.Graph()
    graph.add_nodes_from(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            graph.add_edge(i, j, weight=float(weight_matrix[i, j]))
    tree = nx.maximum_spanning_tree(graph, weight='weight')
    edges = {tuple(sorted(e)) for e in tree.edges()}
    ranked = sorted([(float(weight_matrix[i, j]), i, j) for i in range(q) for j in range(i + 1, q)], reverse=True)
    for _, i, j in ranked:
        if len(edges) >= n_edges:
            break
        edges.add((i, j))
    return sorted(edges)

class RiemannianGraphBuilder:

    def __init__(self, q: int, graph_multiplier: float, edge_mode: str='plain'):
        self.q = int(q)
        self.graph_multiplier = float(graph_multiplier)
        self.edge_mode = str(edge_mode).lower()

    def _cov_input(self, x: np.ndarray) -> np.ndarray:
        if self.edge_mode == 'plain':
            return x
        if self.edge_mode == 'erp_residual':
            if not hasattr(self, 'erp_template_'):
                raise RuntimeError('ERP template not fitted.')
            residual = x - self.erp_template_
            return np.concatenate([x, residual], axis=1)
        raise ValueError(f'Unknown edge_mode {self.edge_mode!r}.')

    def fit(self, X_latent: np.ndarray, y: Optional[np.ndarray]=None) -> 'RiemannianGraphBuilder':
        if self.edge_mode == 'erp_residual':
            if y is None or 1 not in set(np.asarray(y).tolist()):
                raise RuntimeError('ERP-residual covariance requires source P300 target labels.')
            self.erp_template_ = np.asarray(X_latent)[np.asarray(y) == 1].mean(axis=0)
        covs = np.stack([shrink_covariance(self._cov_input(x)) for x in X_latent])
        self.reference_ = riemannian_mean(covs)
        H = np.stack([tangent_matrix(C, self.reference_) for C in covs])
        mean_abs = np.mean(np.abs(H), axis=0)
        np.fill_diagonal(mean_abs, 0.0)
        self.edges_ = mst_plus_strong_edges(mean_abs, int(round(self.graph_multiplier * self.q)))
        raw_edges = np.stack([[h[i, j] for i, j in self.edges_] for h in H])
        self.edge_scaler_ = RobustFeatureScaler().fit(raw_edges)
        self.tangent_scaler_ = StandardScaler().fit(sym_to_vec(H))
        self.mean_abs_ = mean_abs.astype(np.float32)
        return self

    def transform(self, X_latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_latent = np.asarray(X_latent)
        if len(X_latent) == 0:
            E = len(self.edges_)
            D = self.q * (self.q + 1) // 2
            return (np.empty((0, E), np.float32), np.empty((0, D), np.float32), np.empty((0, self.q, self.q), np.float32))
        covs = np.stack([shrink_covariance(self._cov_input(x)) for x in X_latent])
        H = np.stack([tangent_matrix(C, self.reference_) for C in covs])
        raw_edges = np.stack([[h[i, j] for i, j in self.edges_] for h in H])
        return (self.edge_scaler_.transform(raw_edges), self.tangent_scaler_.transform(sym_to_vec(H)).astype(np.float32), H)

def make_subject_codes(subjects: np.ndarray, source_subjects: Sequence[str]) -> np.ndarray:
    mapping = {str(s): i for i, s in enumerate(sorted(map(str, source_subjects)))}
    return np.asarray([mapping.get(str(s), -1) for s in subjects], dtype=np.int64)

def _fold_arrays_from_split(bundle: EEGDatasetBundle, idx: np.ndarray, x_deep: np.ndarray, x_riem: np.ndarray, latent_deep: np.ndarray, latent_riem: np.ndarray, edges: np.ndarray, tangent: np.ndarray, source_subjects: Sequence[str]) -> FoldArrays:
    idx = np.asarray(idx, dtype=int)
    return FoldArrays(x_raw=bundle.X[idx].astype(np.float32), x_deep=x_deep.astype(np.float32), x_riem=x_riem.astype(np.float32), x_latent_deep=latent_deep.astype(np.float32), x_latent_riem=latent_riem.astype(np.float32), edges=edges.astype(np.float32), tangent_vec=tangent.astype(np.float32), y=bundle.y[idx].astype(np.int64), subject_codes=make_subject_codes(bundle.subjects[idx], source_subjects), subject_names=bundle.subjects[idx].astype(object), sessions=bundle.sessions[idx].astype(object))

def prepare_fold(bundle: EEGDatasetBundle, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, q: int, graph_multiplier: float, projection_mode: Optional[str]=None, edge_mode: Optional[str]=None) -> PreparedFold:
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    train_subjects = sorted(np.unique(bundle.subjects[train_idx].astype(str)).tolist())
    val_subjects = bundle.subjects[val_idx].astype(str) if len(val_idx) else np.array([], dtype=str)
    test_subjects = bundle.subjects[test_idx].astype(str) if len(test_idx) else np.array([], dtype=str)
    assert_no_subject_overlap(bundle.subjects[train_idx], val_subjects, test_subjects)
    projection_mode = str(projection_mode or (cfg.PROJECTION_P300 if bundle.task == 'p300' else cfg.PROJECTION_MI))
    edge_mode = str(edge_mode or (cfg.P300_EDGE_MODE if bundle.task == 'p300' else 'plain'))
    Xtr = bundle.X[train_idx].astype(np.float32)
    Xv = bundle.X[val_idx].astype(np.float32)
    Xte = bundle.X[test_idx].astype(np.float32)
    projector = SpatialProjector(q=q, mode=projection_mode).fit(Xtr, bundle.y[train_idx])
    Ltr = projector.transform(Xtr)
    Lv = projector.transform(Xv)
    Lte = projector.transform(Xte)
    latent_deep_scaler = RobustChannelScaler().fit(Ltr)
    Ld_tr = latent_deep_scaler.transform(Ltr)
    Ld_v = latent_deep_scaler.transform(Lv)
    Ld_te = latent_deep_scaler.transform(Lte)
    latent_riem_scaler = RobustChannelScaler().fit(Ltr)
    Lr_tr = latent_riem_scaler.transform(Ltr)
    Lr_v = latent_riem_scaler.transform(Lv)
    Lr_te = latent_riem_scaler.transform(Lte)
    deep_scaler = RobustChannelScaler().fit(Xtr)
    riem_scaler = RobustChannelScaler().fit(Xtr)
    Dtr = deep_scaler.transform(Xtr)
    Dv = deep_scaler.transform(Xv)
    Dte = deep_scaler.transform(Xte)
    Rtr = riem_scaler.transform(Xtr)
    Rv = riem_scaler.transform(Xv)
    Rte = riem_scaler.transform(Xte)
    builder = RiemannianGraphBuilder(q=q, graph_multiplier=graph_multiplier, edge_mode=edge_mode).fit(Lr_tr, bundle.y[train_idx])
    Etr, Ttr, Htr = builder.transform(Lr_tr)
    Ev, Tv, Hv = builder.transform(Lr_v)
    Ete, Tte, Hte = builder.transform(Lr_te)
    fit_audit = {'fit_subjects': train_subjects, 'validation_subjects': sorted(np.unique(val_subjects).tolist()), 'test_subjects': sorted(np.unique(test_subjects).tolist()), 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'q': int(q), 'graph_multiplier': float(graph_multiplier), 'graph_edges': builder.edges_, 'reference_hash': hashlib.sha256(builder.reference_.tobytes()).hexdigest()[:16], 'projection_hash': hashlib.sha256(projector.P_.tobytes()).hexdigest()[:16], 'alignment_rule': 'fit P on source-train -> same latent components -> separate source-fitted branch scalers', 'latent_identity': 'component i = deep node i = covariance row/column i = graph node i = qubit i', 'deep_normalization': 'source-fitted robust component scaling; trial amplitude preserved', 'riemannian_normalization': 'independent source-fitted robust component scaling; trial amplitude preserved', 'graph_mean_abs': builder.mean_abs_.tolist()}
    fold_out = PreparedFold(dataset_name=bundle.name, task=bundle.task, class_names=bundle.class_names, sfreq=bundle.sfreq, q=q, graph_edges=builder.edges_, readout_edges=builder.edges_, projector_matrix=projector.P_, train=_fold_arrays_from_split(bundle, train_idx, Dtr, Rtr, Ld_tr, Lr_tr, Etr, Ttr, train_subjects), val=_fold_arrays_from_split(bundle, val_idx, Dv, Rv, Ld_v, Lr_v, Ev, Tv, train_subjects), test=_fold_arrays_from_split(bundle, test_idx, Dte, Rte, Ld_te, Lr_te, Ete, Tte, train_subjects), fit_audit=fit_audit)
    fold_out.v12_H = {'train': Htr, 'val': Hv, 'test': Hte}
    return fold_out

class SharedTemporalNodeEncoder(nn.Module):

    def __init__(self, q: int, feature_dim: int, filters: int, dropout: float):
        super().__init__()
        self.q = q
        self.feature_dim = feature_dim
        self.block1 = nn.Sequential(nn.Conv1d(1, filters, kernel_size=31, padding=15, bias=False), nn.BatchNorm1d(filters), nn.ELU(), nn.AvgPool1d(4), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Conv1d(filters, filters, kernel_size=15, padding=7, groups=filters, bias=False), nn.Conv1d(filters, feature_dim, kernel_size=1, bias=False), nn.BatchNorm1d(feature_dim), nn.ELU(), nn.AvgPool1d(4), nn.Dropout(dropout), nn.Conv1d(feature_dim, feature_dim, kernel_size=7, padding=3, groups=feature_dim, bias=False), nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False), nn.BatchNorm1d(feature_dim), nn.ELU(), nn.AdaptiveAvgPool1d(1))
        self.node_projection = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, q, T = x.shape
        z = x.reshape(B * q, 1, T)
        z = self.block2(self.block1(z)).squeeze(-1)
        d = z.reshape(B, q, self.feature_dim)
        v = self.node_projection(d).squeeze(-1)
        return (d, v)

    def freeze_temporal(self) -> None:
        for module in [self.block1, self.block2, self.node_projection]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_final(self) -> None:
        for p in self.block1.parameters():
            p.requires_grad = False
        for p in self.block2.parameters():
            p.requires_grad = True
        for p in self.node_projection.parameters():
            p.requires_grad = True

    def set_stage_mode(self, stage: str) -> None:
        stage = str(stage)
        if stage == 'frozen':
            self.block1.eval()
            self.block2.eval()
            self.node_projection.eval()
        elif stage == 'finetune':
            self.block1.eval()
            self.block2.train()
            self.node_projection.train()
        elif stage == 'pretrain':
            self.block1.train()
            self.block2.train()
            self.node_projection.train()
        else:
            raise ValueError(stage)

class NodeBackbonePretrainer(nn.Module):

    def __init__(self, encoder: SharedTemporalNodeEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.q, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, v = self.encoder(batch['x_latent'])
        return self.classifier(v)

class EEGNetClassifier(nn.Module):

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float=0.25, F1: int=8, D: int=2, F2: int=16):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv2d(1, F1, (1, 63), padding=(0, 31), bias=False), nn.BatchNorm2d(F1))
        self.spatial = nn.Sequential(nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7), groups=F1 * D, bias=False), nn.Conv2d(F1 * D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(F2, n_classes)

    def features(self, x):
        z = self.temporal(x.unsqueeze(1))
        weight = self.spatial[0].weight
        z = F.conv2d(z, weight.mean(dim=2, keepdim=True).expand_as(weight), groups=self.spatial[0].groups)
        z = self.spatial[1:](z)
        z = z.mean(dim=-1, keepdim=True).expand_as(z)
        z = z - z.mean(dim=-1, keepdim=True)
        return self.separable(z).flatten(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.features(batch['x_deep']))

class CRGNetBaseline(nn.Module):

    def __init__(self, n_channels: int, n_times: int, tangent_dim: int, n_classes: int):
        super().__init__()
        self.eeg = EEGNetClassifier(n_channels, n_times, n_classes=16)
        self.eeg.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Linear(16 + tangent_dim, 32), nn.ELU(), nn.Dropout(0.25), nn.Linear(32, n_classes))

    def forward(self, batch):
        tangent = batch['tangent']
        tangent = tangent.mean(dim=1, keepdim=True).expand_as(tangent)
        tangent = F.layer_norm(tangent, (tangent.shape[1],))
        return self.head(torch.cat([self.eeg.features(batch['x_deep']), tangent], dim=1))

def _as_complex(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.complex64)

def gate_rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = (torch.cos(theta / 2), torch.sin(theta / 2))
    out = torch.zeros(theta.shape + (2, 2), dtype=torch.complex64, device=theta.device)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = -1j * s
    out[..., 1, 0] = -1j * s
    return out

def gate_ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = (torch.cos(theta / 2), torch.sin(theta / 2))
    out = torch.zeros(theta.shape + (2, 2), dtype=torch.complex64, device=theta.device)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    return out

def gate_rz(theta: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(theta.shape + (2, 2), dtype=torch.complex64, device=theta.device)
    out[..., 0, 0] = torch.exp(-0.5j * theta)
    out[..., 1, 1] = torch.exp(0.5j * theta)
    return out

def pauli_matrix(kind: str, device: torch.device) -> torch.Tensor:
    if kind == 'XX':
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        return torch.kron(X, X)
    if kind == 'YY':
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        return torch.kron(Y, Y)
    if kind == 'ZZ':
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        return torch.kron(Z, Z)
    raise ValueError(kind)

def gate_ising(theta: torch.Tensor, kind: str) -> torch.Tensor:
    I = torch.eye(4, dtype=torch.complex64, device=theta.device)
    P = pauli_matrix(kind, theta.device)
    c = torch.cos(theta / 2)[..., None, None]
    s = torch.sin(theta / 2)[..., None, None]
    return c * I - 1j * s * P

def apply_single(state: torch.Tensor, gate: torch.Tensor, wire: int, q: int) -> torch.Tensor:
    B = state.shape[0]
    shape = [B] + [2] * q
    axis = wire + 1
    perm = [0] + [a for a in range(1, q + 1) if a != axis] + [axis]
    inv = np.argsort(perm).tolist()
    psi = state.reshape(shape).permute(perm).reshape(B, -1, 2)
    if gate.ndim == 2:
        gate = gate.unsqueeze(0).expand(B, -1, -1)
    psi = torch.einsum('bij,bnj->bni', gate, psi)
    return psi.reshape([B] + [2] * (q - 1) + [2]).permute(inv).reshape(B, -1)

def apply_two(state: torch.Tensor, gate: torch.Tensor, wire_a: int, wire_b: int, q: int) -> torch.Tensor:
    if wire_a == wire_b:
        raise ValueError('Two-qubit gate requires different wires.')
    B = state.shape[0]
    axes = [wire_a + 1, wire_b + 1]
    perm = [0] + [a for a in range(1, q + 1) if a not in axes] + axes
    inv = np.argsort(perm).tolist()
    psi = state.reshape([B] + [2] * q).permute(perm).reshape(B, -1, 4)
    if gate.ndim == 2:
        gate = gate.unsqueeze(0).expand(B, -1, -1)
    psi = torch.einsum('bij,bnj->bni', gate, psi)
    return psi.reshape([B] + [2] * (q - 2) + [2, 2]).permute(inv).reshape(B, -1)

def cnot_gate(device: torch.device) -> torch.Tensor:
    return torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64, device=device)

class BatchedStatevectorBackend(nn.Module):

    def __init__(self, q: int, readout_edges: Sequence[Tuple[int, int]]):
        super().__init__()
        self.q = q
        self.readout_edges = list(readout_edges)
        n = 2 ** q
        idx = torch.arange(n, dtype=torch.long)
        z_signs = []
        for wire in range(q):
            bit = idx >> q - 1 - wire & 1
            z_signs.append(1.0 - 2.0 * bit.float())
        self.register_buffer('z_signs', torch.stack(z_signs), persistent=False)
        zz = [self.z_signs[i] * self.z_signs[j] for i, j in self.readout_edges]
        self.register_buffer('zz_signs', torch.stack(zz) if zz else torch.empty((0, n)), persistent=False)

    def initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros((batch, 2 ** self.q), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0j
        return state

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        prob = state.abs().square().real
        z = prob @ self.z_signs.T
        if len(self.readout_edges):
            zz = prob @ self.zz_signs.T
            return torch.cat([z, zz], dim=1)
        return z

class RobustTorchScaler(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('center', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.fitted = False

    def fit_numpy(self, X: np.ndarray) -> None:
        scaler = RobustFeatureScaler().fit(X)
        self.center.copy_(torch.tensor(scaler.center_, dtype=torch.float32))
        self.scale.copy_(torch.tensor(scaler.scale_, dtype=torch.float32))
        self.fitted = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.clamp((X - self.center) / self.scale, -cfg.ROBUST_CLIP, cfg.ROBUST_CLIP)

class GraphConditionedCircuit(nn.Module):

    def __init__(self, q: int, edges: Sequence[Tuple[int, int]], readout_edges: Sequence[Tuple[int, int]], layers: int, variant: str, deep_dim: int):
        super().__init__()
        self.q = q
        self.edges = list(edges)
        self.layers = layers
        self.variant = variant.upper()
        self.deep_dim = deep_dim
        self.backend = BatchedStatevectorBackend(q, readout_edges)
        E = len(self.edges)
        self.theta_node = nn.Parameter(cfg.THETA_INIT + cfg.THETA_INIT_STD * torch.randn(q))
        self.theta_edge = nn.Parameter(cfg.THETA_INIT + cfg.THETA_INIT_STD * torch.randn(E))
        self.mixer_x = nn.Parameter(cfg.MIXER_X_INIT + cfg.MIXER_INIT_STD * torch.randn(layers, q))
        self.mixer_y = nn.Parameter(cfg.MIXER_Y_INIT + cfg.MIXER_INIT_STD * torch.randn(layers, q))
        if self.variant == 'B':
            self.node_a = nn.Parameter(0.1 * torch.randn(deep_dim))
            self.node_b = nn.Parameter(0.1 * torch.randn(deep_dim))
            self.pauli_scales = nn.Parameter(cfg.THETA_INIT + cfg.THETA_INIT_STD * torch.randn(3))
        elif self.variant == 'C':
            self.reupload_scale = nn.Parameter(torch.full((layers,), 1.0 / math.sqrt(max(1, layers))))
            self.gate_a = nn.Parameter(torch.tensor(1.0))
            self.gate_b = nn.Parameter(torch.tensor(0.0))
        elif self.variant != 'A':
            raise ValueError(f'Unknown QRG variant {variant}')

    def forward(self, node_values: torch.Tensor, edge_values: torch.Tensor, deep_vectors: torch.Tensor) -> torch.Tensor:
        B = node_values.shape[0]
        state = self.backend.initial_state(B, node_values.device)
        for layer in range(self.layers):
            if self.variant == 'B':
                d = F.layer_norm(deep_vectors, (self.deep_dim,))
                alpha = torch.einsum('bqd,d->bq', d, self.node_a)
                gamma = torch.einsum('bqd,d->bq', d, self.node_b)
                for i in range(self.q):
                    state = apply_single(state, gate_ry(alpha[:, i]), i, self.q)
                    state = apply_single(state, gate_rz(gamma[:, i]), i, self.q)
                for edge_idx, (i, j) in enumerate(self.edges):
                    base = edge_values[:, edge_idx]
                    state = apply_two(state, gate_ising(self.pauli_scales[0] * base, 'XX'), i, j, self.q)
                    state = apply_two(state, gate_ising(self.pauli_scales[1] * base, 'YY'), i, j, self.q)
                    state = apply_two(state, gate_ising(self.pauli_scales[2] * base, 'ZZ'), i, j, self.q)
            elif self.variant == 'C':
                alpha = self.reupload_scale[layer] * self.theta_node * node_values
                for i in range(self.q):
                    state = apply_single(state, gate_ry(alpha[:, i]), i, self.q)
                soft = torch.sigmoid(self.gate_a * edge_values.abs() + self.gate_b)
                beta = soft * self.theta_edge * edge_values
                for edge_idx, (i, j) in enumerate(self.edges):
                    state = apply_two(state, gate_ising(beta[:, edge_idx], 'ZZ'), i, j, self.q)
            else:
                alpha = cfg.ALPHA_MAX * torch.tanh(node_values * self.theta_node)
                for i in range(self.q):
                    state = apply_single(state, gate_ry(alpha[:, i]), i, self.q)
                beta = cfg.BETA_MAX * torch.tanh(edge_values * self.theta_edge)
                for edge_idx, (i, j) in enumerate(self.edges):
                    state = apply_two(state, gate_ising(beta[:, edge_idx], 'ZZ'), i, j, self.q)
            for i in range(self.q):
                state = apply_single(state, gate_rx(self.mixer_x[layer, i].expand(B)), i, self.q)
                state = apply_single(state, gate_ry(self.mixer_y[layer, i].expand(B)), i, self.q)
        return self.backend.measure(state)

class QRGEEG(nn.Module):

    def __init__(self, q: int, graph_edges: Sequence[Tuple[int, int]], readout_edges: Sequence[Tuple[int, int]], n_classes: int, layers: int, variant: str):
        super().__init__()
        self.node_encoder = SharedTemporalNodeEncoder(q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
        self.node_scaler = RobustTorchScaler(q)
        self.circuit = GraphConditionedCircuit(q, graph_edges, readout_edges, layers, variant, cfg.NODE_FEATURE_DIM)
        measurement_dim = q + len(readout_edges)
        self.readout = nn.Linear(measurement_dim, n_classes)
        self.quantum_parameter_names = ('circuit',)

    def fit_node_scaler(self, loader: DataLoader, device: torch.device) -> None:
        self.eval()
        values = []
        with torch.no_grad():
            for batch in loader:
                x = batch['x_latent'].to(device)
                _, v = self.node_encoder(x)
                values.append(v.cpu().numpy())
        self.node_scaler.fit_numpy(np.concatenate(values, axis=0))

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool=False):
        d, v = self.node_encoder(batch['x_latent'])
        v = self.node_scaler(v)
        m = self.circuit(v, batch['edges'], d)
        logits = self.readout(m)
        return (logits, m) if return_features else logits

    def quantum_parameter_count(self) -> int:
        return int(sum((p.numel() for p in self.circuit.parameters() if p.requires_grad)))

    def two_qubit_gates_per_trial(self) -> int:
        multiplier = 3 if self.circuit.variant == 'B' else 1
        return self.circuit.layers * len(self.circuit.edges) * multiplier

class GenericQEEGCircuit(nn.Module):

    def __init__(self, q: int, layers: int):
        super().__init__()
        self.q = q
        self.layers = layers
        self.weights = nn.Parameter(0.05 * torch.randn(layers, q))
        self.backend = BatchedStatevectorBackend(q, [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        state = self.backend.initial_state(B, x.device)
        for i in range(self.q):
            state = apply_single(state, gate_rz(math.pi * torch.tanh(x[:, i])), i, self.q)
        for l in range(self.layers):
            for i in range(self.q):
                state = apply_two(state, cnot_gate(x.device), i, (i + 1) % self.q, self.q)
            for i in range(self.q):
                state = apply_single(state, gate_ry(self.weights[l, i].expand(B)), i, self.q)
        return self.backend.measure(state).square()

class QEEGNetStyle(nn.Module):

    def __init__(self, n_channels: int, n_times: int, n_classes: int, q: int, layers: int):
        super().__init__()
        self.eeg = EEGNetClassifier(n_channels, n_times, n_classes=q)
        self.compress = nn.Linear(16, q)
        self.eeg.classifier = nn.Identity()
        self.circuit = GenericQEEGCircuit(q, layers)
        self.readout = nn.Linear(q, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.compress(self.eeg.features(batch['x_deep']))
        return self.readout(self.circuit(x))

class RiemannianVQC(nn.Module):

    def __init__(self, tangent_dim: int, n_classes: int, q: int, layers: int):
        super().__init__()
        self.compress = nn.Linear(tangent_dim, q)
        self.circuit = GenericQEEGCircuit(q, layers)
        self.readout = nn.Linear(q, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.readout(self.circuit(self.compress(batch['tangent'])))

class WeightedGATLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(out_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(out_dim) * 0.1)
        self.att_edge = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_values: torch.Tensor, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        h = self.linear(x)
        B, q, D = h.shape
        outputs = torch.zeros_like(h)
        incoming: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(q)}
        for e, (i, j) in enumerate(edges):
            incoming[j].append((e, i))
            incoming[i].append((e, j))
        for dst in range(q):
            neighbors = incoming[dst] + [(-1, dst)]
            scores = []
            messages = []
            for e, src in neighbors:
                edge = torch.zeros(B, device=x.device) if e < 0 else edge_values[:, e]
                score = F.leaky_relu((h[:, src] * self.att_src).sum(-1) + (h[:, dst] * self.att_dst).sum(-1) + self.att_edge * edge, 0.2)
                scores.append(score)
                messages.append(h[:, src])
            att = torch.softmax(torch.stack(scores, dim=1), dim=1)
            msg = torch.stack(messages, dim=1)
            outputs[:, dst] = (att.unsqueeze(-1) * msg).sum(1) + self.bias
        return F.elu(outputs)

class SameGraphGAT(nn.Module):

    def __init__(self, node_encoder: SharedTemporalNodeEncoder, edges: Sequence[Tuple[int, int]], n_classes: int, hidden: int=8):
        super().__init__()
        self.node_encoder = node_encoder
        self.edges = list(edges)
        self.node_scaler = RobustTorchScaler(node_encoder.q)
        self.gat1 = WeightedGATLayer(1, hidden)
        self.gat2 = WeightedGATLayer(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def fit_node_scaler(self, loader, device):
        self.eval()
        vals = []
        with torch.no_grad():
            for batch in loader:
                _, v = self.node_encoder(batch['x_latent'].to(device))
                vals.append(v.cpu().numpy())
        self.node_scaler.fit_numpy(np.concatenate(vals))

    def forward(self, batch):
        _, v = self.node_encoder(batch['x_latent'])
        v = self.node_scaler(v)
        v = v.mean(dim=1, keepdim=True).expand_as(v)
        v = F.layer_norm(v, (v.shape[1],)).unsqueeze(-1)
        edge_values = batch['edges'].square()
        h = self.gat2(self.gat1(v, edge_values, self.edges), edge_values, self.edges)
        return self.head(h.mean(1))

class MatchedClassicalInteraction(nn.Module):

    def __init__(self, node_encoder: SharedTemporalNodeEncoder, edges: Sequence[Tuple[int, int]], n_classes: int, hidden: int=12):
        super().__init__()
        self.node_encoder = node_encoder
        self.edges = list(edges)
        self.node_scaler = RobustTorchScaler(node_encoder.q)
        input_dim = node_encoder.q + len(edges)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_classes))

    def fit_node_scaler(self, loader, device):
        self.eval()
        vals = []
        with torch.no_grad():
            for batch in loader:
                _, v = self.node_encoder(batch['x_latent'].to(device))
                vals.append(v.cpu().numpy())
        self.node_scaler.fit_numpy(np.concatenate(vals))

    def forward(self, batch):
        _, v = self.node_encoder(batch['x_latent'])
        v = self.node_scaler(v)
        v = v.mean(dim=1, keepdim=True).expand_as(v)
        v = F.layer_norm(v, (v.shape[1],))
        interactions = []
        for e, (i, j) in enumerate(self.edges):
            interactions.append((batch['edges'][:, e].square() * (v[:, i] - v[:, j]).square()).unsqueeze(1))
        return self.net(torch.cat([v] + interactions, dim=1))

def filter_bank_numpy(X: np.ndarray, sfreq: float, bands: Sequence[Tuple[float, float]]) -> np.ndarray:
    outs = []
    for lo, hi in bands:
        outs.append(_bandpass_array(X, sfreq, lo, hi))
    return np.stack(outs, axis=1).astype(np.float32)

class FBCNetLite(nn.Module):

    def __init__(self, n_bands: int, n_channels: int, n_classes: int, spatial_filters: int=4):
        super().__init__()
        self.spatial = nn.Conv2d(n_bands, n_bands * spatial_filters, (n_channels, 1), groups=n_bands, bias=False)
        self.bn = nn.BatchNorm2d(n_bands * spatial_filters)
        self.head = nn.Linear(n_bands * spatial_filters, n_classes)

    def forward(self, batch):
        z = self.bn(self.spatial(batch['filter_bank'])).squeeze(2)
        z = F.normalize(z, p=2.0, dim=-1)
        z = z.mean(dim=-1, keepdim=True).expand_as(z)
        var = z.var(dim=-1, unbiased=False).clamp_min(1e-06).log()
        return self.head(var)

class SPDTimeFrequencyGraphNet(nn.Module):

    def __init__(self, n_channels: int, n_bands: int, n_classes: int, hidden: int=16, spd_dim: Optional[int]=None):
        super().__init__()
        self.n_bands = n_bands
        self.spd_dim = int(spd_dim or min(8, n_channels))
        self.raw_W = nn.Parameter(torch.randn(n_channels, self.spd_dim) * 0.05)
        vec_dim = self.spd_dim * (self.spd_dim + 1) // 2
        self.norm = nn.LayerNorm(vec_dim)
        self.node = nn.Linear(vec_dim, hidden)
        self.self_lin = nn.Linear(hidden, hidden)
        self.neigh_lin = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def _logeig_vector(self, C: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.raw_W, dim=0)[:, :self.spd_dim]
        mapped = torch.einsum('cr,bfcd,ds->bfrs', W, C, W)
        eye = torch.eye(self.spd_dim, device=C.device, dtype=C.dtype)
        mapped = 0.5 * (mapped + mapped.transpose(-1, -2)) + 1e-05 * eye
        vals, vecs = torch.linalg.eigh(mapped)
        vals = vals.clamp_min(1e-06).log()
        logc = torch.diag_embed(vals)
        idx = torch.triu_indices(self.spd_dim, self.spd_dim, device=C.device)
        out = logc[..., idx[0], idx[1]]
        off = idx[0] != idx[1]
        out[..., off] = out[..., off] * math.sqrt(2.0)
        return out

    def forward(self, batch):
        x = self.norm(self._logeig_vector(batch['spd_band_matrices']))
        x = x.mean(dim=1, keepdim=True).expand_as(x)
        x = x - x.mean(dim=1, keepdim=True)
        x = self.node(x)
        left = torch.roll(x, 1, 1)
        right = torch.roll(x, -1, 1)
        h = F.elu(self.self_lin(x) + 0.5 * self.neigh_lin(left + right))
        return self.head(h.mean(1))

class PLVGATBaseline(nn.Module):

    def __init__(self, n_channels: int, n_classes: int, hidden: int=8):
        super().__init__()
        self.node_proj = nn.Linear(1, hidden)
        self.gat = WeightedGATLayer(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.complete_edges = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

    def forward(self, batch):
        node = batch['plv_node']
        node = node.mean(dim=1, keepdim=True).expand_as(node)
        node = F.layer_norm(node, (node.shape[1],)).unsqueeze(-1)
        h = self.node_proj(node)
        edge_values = batch['plv_edges']
        edge_values = edge_values.mean(dim=1, keepdim=True).expand_as(edge_values)
        return self.head(self.gat(h, edge_values, self.complete_edges).mean(1))

def fit_riemannian_classical(train: FoldArrays, val: FoldArrays, test: FoldArrays, kind: str='LR'):
    Xtr, Xv, Xte = (train.tangent_vec, val.tangent_vec, test.tangent_vec)
    if kind == 'SVM':
        model = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=cfg.PRIMARY_SEED)
    else:
        model = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=cfg.PRIMARY_SEED)
    model.fit(Xtr, train.y)
    pv = model.predict_proba(Xv) if len(Xv) else np.empty((0, len(np.unique(train.y))), dtype=float)
    pt = model.predict_proba(Xte) if len(Xte) else np.empty((0, len(np.unique(train.y))), dtype=float)
    return (pv, pt, model)

class BinaryBLDA:

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = BayesianRidge().fit(Xs, 2 * np.asarray(y) - 1)
        return self

    def predict_proba(self, X):
        score = self.model.predict(self.scaler.transform(X))
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -20, 20)))
        return np.column_stack([1 - prob, prob])

def fit_xdawn_blda(train: FoldArrays, val: FoldArrays, test: FoldArrays):
    feature_dim = int(np.prod(train.x_latent_riem.shape[1:]))
    Xtr = train.x_latent_riem.reshape(len(train.y), feature_dim)
    Xv = val.x_latent_riem.reshape(len(val.y), feature_dim)
    Xte = test.x_latent_riem.reshape(len(test.y), feature_dim)
    model = BinaryBLDA().fit(Xtr, train.y)
    pv = model.predict_proba(Xv) if len(Xv) else np.empty((0, 2), dtype=float)
    pt = model.predict_proba(Xte) if len(Xte) else np.empty((0, 2), dtype=float)
    return (pv, pt, model)

class FoldTorchDataset(Dataset):

    def __init__(self, arrays: FoldArrays, sfreq: float, task: str, need_filter_bank: bool=False, need_spd_bands: bool=False, need_plv: bool=False):
        self.a = arrays
        self.sfreq = sfreq
        self.task = task
        self.filter_bank = None
        self.spd_band_matrices = None
        self.plv_node = None
        self.plv_edges = None
        if need_filter_bank or need_spd_bands:
            bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 28), (28, 40)] if task == 'mi' else [(1, 4), (4, 8), (8, 12), (12, 20)]
            self.filter_bank = filter_bank_numpy(arrays.x_raw, sfreq, bands)
            if need_spd_bands:
                self.spd_band_matrices = np.asarray([[shrink_covariance(b).astype(np.float32) for b in trial] for trial in self.filter_bank], dtype=np.float32)
        if need_plv:
            nodes = []
            edges = []
            complete = [(i, j) for i in range(arrays.x_raw.shape[1]) for j in range(i + 1, arrays.x_raw.shape[1])]
            for trial in arrays.x_raw:
                phase = np.angle(sps.hilbert(trial, axis=-1))
                nodes.append(np.log(np.var(trial, axis=-1) + 1e-06))
                edges.append([np.abs(np.mean(np.exp(1j * (phase[i] - phase[j])))) for i, j in complete])
            self.plv_node = np.asarray(nodes, np.float32)
            self.plv_edges = np.asarray(edges, np.float32)
        self._compute_sbce_weights()

    def _compute_sbce_weights(self):
        y = np.asarray(self.a.y, dtype=int)
        s = np.asarray(self.a.subject_codes, dtype=int)
        N = len(y)
        valid_subjects = sorted((x for x in np.unique(s).tolist() if x >= 0))
        S = max(1, len(valid_subjects))
        subject_n = {u: int(np.sum(s == u)) for u in valid_subjects}
        self.uniform_sbce_weight = np.ones(N, dtype=np.float32)
        for u, n in subject_n.items():
            self.uniform_sbce_weight[s == u] = N / (S * max(1, n))
        cells = {(u, int(c)): int(np.sum((s == u) & (y == c))) for u in valid_subjects for c in np.unique(y[s == u])}
        M = max(1, len(cells))
        self.stratified_sample_weight = np.ones(N, dtype=np.float64)
        self.stratified_sbce_weight = np.ones(N, dtype=np.float32)
        for (u, c), nsc in cells.items():
            mask = (s == u) & (y == c)
            self.stratified_sample_weight[mask] = 1.0 / max(1, nsc)
            self.stratified_sbce_weight[mask] = M * max(1, nsc) / (S * max(1, subject_n[u]))

    def __len__(self):
        return len(self.a.y)

    def __getitem__(self, idx):
        out = {'x_raw': torch.from_numpy(self.a.x_raw[idx]), 'x_deep': torch.from_numpy(self.a.x_deep[idx]), 'x_latent': torch.from_numpy(self.a.x_latent_deep[idx]), 'edges': torch.from_numpy(self.a.edges[idx]), 'tangent': torch.from_numpy(self.a.tangent_vec[idx]), 'y': torch.tensor(self.a.y[idx], dtype=torch.long), 'subject': torch.tensor(self.a.subject_codes[idx], dtype=torch.long), 'uniform_sbce_weight': torch.tensor(self.uniform_sbce_weight[idx], dtype=torch.float32), 'stratified_sbce_weight': torch.tensor(self.stratified_sbce_weight[idx], dtype=torch.float32)}
        if self.filter_bank is not None:
            out['filter_bank'] = torch.from_numpy(self.filter_bank[idx])
        if self.spd_band_matrices is not None:
            out['spd_band_matrices'] = torch.from_numpy(self.spd_band_matrices[idx])
        if self.plv_node is not None:
            out['plv_node'] = torch.from_numpy(self.plv_node[idx])
            out['plv_edges'] = torch.from_numpy(self.plv_edges[idx])
        return out

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class SubjectBalancedLoss(nn.Module):

    def __init__(self, use_rv: bool=False, lambda_rv: float=0.0, class_weight: Optional[torch.Tensor]=None, rank_lambda: float=0.0, rank_margin: float=0.25):
        super().__init__()
        self.use_rv = bool(use_rv)
        self.lambda_rv = float(lambda_rv)
        self.rank_lambda = float(rank_lambda)
        self.rank_margin = float(rank_margin)
        if class_weight is None:
            self.class_weight = None
        else:
            self.register_buffer('class_weight', torch.as_tensor(class_weight, dtype=torch.float32))

    def _ranking_loss(self, logits: torch.Tensor, y: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.rank_lambda <= 0.0 or logits.ndim != 2:
            return logits.sum() * 0.0
        if logits.shape[1] == 2:
            score = logits[:, 1] - logits[:, 0]
            pieces = []
            for u in torch.unique(subjects):
                if int(u.item()) < 0:
                    continue
                m = subjects == u
                pos = score[m & (y == 1)]
                neg = score[m & (y == 0)]
                if pos.numel() and neg.numel():
                    pieces.append(F.softplus(self.rank_margin - (pos[:, None] - neg[None, :])).mean())
            if not pieces:
                pos = score[y == 1]
                neg = score[y == 0]
                if pos.numel() and neg.numel():
                    pieces.append(F.softplus(self.rank_margin - (pos[:, None] - neg[None, :])).mean())
            return torch.stack(pieces).mean() if pieces else logits.sum() * 0.0
        true = logits.gather(1, y[:, None])
        diff = true - logits
        mask = F.one_hot(y, num_classes=logits.shape[1]).bool()
        violations = F.softplus(self.rank_margin - diff)
        return violations.masked_select(~mask).mean()

    def forward(self, logits, y, subjects, sample_weight):
        ce = F.cross_entropy(logits, y, reduction='none', weight=self.class_weight)
        sb = (sample_weight * ce).mean()
        total = sb
        if self.use_rv:
            sl = []
            for u in torch.unique(subjects):
                if int(u.item()) < 0:
                    continue
                m = subjects == u
                if m.any():
                    sl.append(ce[m].mean())
            if len(sl) > 1:
                total = total + self.lambda_rv * torch.stack(sl).var(unbiased=False)
        if self.rank_lambda > 0.0:
            total = total + self.rank_lambda * self._ranking_loss(logits, y, subjects)
        return total

def _make_training_criterion(class_weight: Optional[torch.Tensor]=None, qrg_objective: bool=False) -> SubjectBalancedLoss:
    if qrg_objective:
        return SubjectBalancedLoss(use_rv=cfg.QRG_USE_RISK_VARIANCE, lambda_rv=cfg.QRG_LAMBDA_RV, class_weight=class_weight, rank_lambda=cfg.QRG_RANK_LAMBDA, rank_margin=cfg.QRG_RANK_MARGIN)
    return SubjectBalancedLoss(use_rv=cfg.USE_RISK_VARIANCE, lambda_rv=cfg.LAMBDA_RV, class_weight=class_weight, rank_lambda=0.0)

def make_loaders(fold: PreparedFold, batch_size: int, need_filter_bank=False, need_spd_bands=False, need_plv=False):
    datasets = [FoldTorchDataset(a, fold.sfreq, fold.task, need_filter_bank, need_spd_bands, need_plv) for a in [fold.train, fold.val, fold.test]]
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(torch.initial_seed() % (2 ** 63 - 1)))
    if cfg.STRATIFIED_SBCE_SAMPLER and len(datasets[0]):
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(datasets[0].stratified_sample_weight, dtype=torch.double), num_samples=len(datasets[0]), replacement=True, generator=loader_generator)
        train = DataLoader(datasets[0], batch_size=batch_size, sampler=sampler, num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        setattr(train, 'uses_stratified_sbce', True)
    else:
        train = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, generator=loader_generator, num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        setattr(train, 'uses_stratified_sbce', False)
    val = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test = DataLoader(datasets[2], batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)
    return (train, val, test)

def _batch_sbce_weight(batch, loader):
    key = 'stratified_sbce_weight' if getattr(loader, 'uses_stratified_sbce', False) else 'uniform_sbce_weight'
    return batch[key]

def _source_class_priors(loader) -> np.ndarray:
    y = np.asarray(loader.dataset.a.y, dtype=np.int64)
    if len(y) == 0:
        return np.ones(1, dtype=np.float64)
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()

def _source_class_weights(loader, device) -> torch.Tensor:
    priors = _source_class_priors(loader)
    w = np.power(np.maximum(priors, 1e-12), -float(cfg.QRG_CLASS_WEIGHT_POWER))
    w /= max(1e-12, float(w.mean()))
    w = np.clip(w, cfg.QRG_CLASS_WEIGHT_MIN, cfg.QRG_CLASS_WEIGHT_MAX)
    w /= max(1e-12, float(w.mean()))
    return torch.as_tensor(w, dtype=torch.float32, device=device)

def _set_node_stage_mode(model, stage: str):
    if not hasattr(model, 'node_encoder'):
        return
    if hasattr(model.node_encoder, 'set_stage_mode'):
        model.node_encoder.set_stage_mode(stage)

@torch.no_grad()
def predict_torch(model, loader, device):
    model.eval()
    ys = []
    probs = []
    feats = []
    for batch in loader:
        batch = move_batch(batch, device)
        try:
            logits, feature = model(batch, return_features=True)
            feats.append(feature.detach().cpu().numpy())
        except TypeError:
            logits = model(batch)
        ys.append(batch['y'].cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not ys:
        n_classes = getattr(getattr(model, 'readout', None), 'out_features', 0) or 0
        return (np.empty((0,), int), np.empty((0, n_classes), float), None)
    return (np.concatenate(ys), np.concatenate(probs), np.concatenate(feats) if feats else None)

def _qrg_validation_calibration(yv: np.ndarray, pv: np.ndarray, train_loader, task: str) -> Tuple[np.ndarray, Optional[float], float]:
    if task == 'p300' and pv.ndim == 2 and (pv.shape[1] == 2):
        threshold, _, _ = best_binary_threshold(yv, pv)
        return (pv, float(threshold), 0.0)
    if task == 'mi' and pv.ndim == 2 and (pv.shape[1] > 1):
        priors = _source_class_priors(train_loader)
        tau, _, _ = best_prior_tau(yv, pv, priors, cfg.QRG_PRIOR_TAU_GRID)
        adj = apply_prior_correction(pv, priors, tau)
        return (adj, None, float(tau))
    return (pv, None, 0.0)

def train_plain_model(model, train_loader, val_loader, epochs, lr, device, patience=None, node_stage: Optional[str]=None, class_weight: Optional[torch.Tensor]=None, min_best_epoch: int=1, calibrate_qrg: bool=False, qrg_objective: bool=False):
    model = model.to(device)
    patience = patience or cfg.PATIENCE
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=qrg_objective)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=max(2, patience // 3))
    best = None
    best_f1 = -1.0
    selected_f1 = -1.0
    best_secondary = -1000000000.0
    best_epoch = max(1, int(min_best_epoch))
    wait = 0
    history = []
    _fixed_epoch_counter = 0
    for epoch in range(max(1, int(epochs))):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, max(1, int(epochs)), 'train_plain_model')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            fixed_training_step(model)
            losses.append(float(loss.detach()))
        if len(val_loader.dataset):
            yv, pv, _ = predict_torch(model, val_loader, device)
            decision_threshold = None
            prior_tau = 0.0
            if calibrate_qrg:
                pv_eval, decision_threshold, prior_tau = _qrg_validation_calibration(yv, pv, train_loader, train_loader.dataset.task)
            else:
                pv_eval = pv
            metrics = multiclass_metrics(yv, pv_eval, train_loader.dataset.task, decision_threshold=decision_threshold)
            f1 = metrics['macro_f1']
            bacc = metrics['balanced_accuracy']
            secondary = qrg_secondary_score(metrics, train_loader.dataset.task) if qrg_objective else float(bacc)
        else:
            decision_threshold = None
            prior_tau = 0.0
            f1 = -float(np.mean(losses))
            bacc = np.nan
            secondary = -1000000000.0
            metrics = {'accuracy': np.nan, 'balanced_accuracy': np.nan, 'kappa': np.nan, 'auroc': np.nan, 'target_auprc': np.nan}
        eligible = epoch + 1 >= max(1, int(min_best_epoch))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_macro_f1': float(f1), 'val_accuracy': float(metrics.get('accuracy', np.nan)), 'val_balanced_accuracy': float(bacc) if np.isfinite(bacc) else np.nan, 'val_kappa': float(metrics.get('kappa', np.nan)), 'val_auroc': float(metrics.get('auroc', np.nan)), 'val_target_auprc': float(metrics.get('target_auprc', np.nan)), 'val_secondary_score': float(secondary), 'decision_threshold': float(decision_threshold) if decision_threshold is not None else np.nan, 'prior_tau': float(prior_tau)})
        fixed_epoch_end()
        if eligible:
            scheduler.step(f1)
            if not qrg_objective:
                improved = f1 > best_f1 + 1e-12
                if improved:
                    best_f1 = float(f1)
            else:
                tol = float(cfg.QRG_MACRO_TOLERANCE)
                best_f1 = max(best_f1, float(f1))
                if selected_f1 < best_f1 - tol - 1e-12:
                    improved = True
                elif f1 >= best_f1 - tol - 1e-12:
                    improved = secondary > best_secondary + 1e-12 or (abs(secondary - best_secondary) <= 1e-12 and f1 > selected_f1 + 1e-12)
                else:
                    improved = False
            if improved:
                selected_f1 = float(f1)
                best_secondary = float(secondary)
                best_epoch = epoch + 1
                wait = 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best is None:
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return (model, pd.DataFrame(history), best_epoch)

def pretrain_node_encoder(model: QRGEEG, train_loader, val_loader, n_classes: int, device):
    wrapper = NodeBackbonePretrainer(model.node_encoder, n_classes).to(device)
    class_weight = _source_class_weights(train_loader, device)
    wrapper, history, best_epoch = train_plain_model(wrapper, train_loader, val_loader, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device, node_stage=None, class_weight=class_weight, min_best_epoch=cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, calibrate_qrg=True, qrg_objective=True)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    return (history, best_epoch)

def _staged_finetune_optimizer(model):
    prefixes = ('node_encoder.block2', 'node_encoder.node_projection')
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(prefixes)]
    interaction = [p for n, p in model.named_parameters() if p.requires_grad and (not n.startswith(prefixes))]
    groups = []
    if backbone:
        groups.append({'params': backbone, 'lr': cfg.LR_BACKBONE_FINETUNE})
    if interaction:
        groups.append({'params': interaction, 'lr': cfg.LR_QUANTUM})
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)

def _refresh_node_scaler(model, train_loader, device):
    if hasattr(model, 'fit_node_scaler'):
        model.fit_node_scaler(train_loader, device)

def _run_finetune_with_validation(model, train_loader, val_loader, device, class_weight: Optional[torch.Tensor]=None, min_best_epoch: int=1, calibrate_qrg: bool=False, qrg_objective: bool=False):
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=qrg_objective)
    opt = _staged_finetune_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=max(2, cfg.PATIENCE // 3))
    best = None
    best_f1 = -1.0
    selected_f1 = -1.0
    best_secondary = -1000000000.0
    best_epoch = max(1, int(min_best_epoch))
    wait = 0
    hist = []
    _fixed_epoch_counter = 0
    for epoch in range(cfg.FINETUNE_EPOCHS):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, cfg.FINETUNE_EPOCHS, '_run_finetune_with_validation')
        model.train()
        _set_node_stage_mode(model, 'finetune')
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            fixed_training_step(model)
            losses.append(float(loss.detach()))
        _refresh_node_scaler(model, train_loader, device)
        yv, pv, _ = predict_torch(model, val_loader, device)
        decision_threshold = None
        prior_tau = 0.0
        if calibrate_qrg:
            pv_eval, decision_threshold, prior_tau = _qrg_validation_calibration(yv, pv, train_loader, train_loader.dataset.task)
        else:
            pv_eval = pv
        metrics = multiclass_metrics(yv, pv_eval, train_loader.dataset.task, decision_threshold=decision_threshold)
        f1 = metrics['macro_f1']
        bacc = metrics['balanced_accuracy']
        secondary = qrg_secondary_score(metrics, train_loader.dataset.task) if qrg_objective else float(bacc)
        eligible = epoch + 1 >= max(1, int(min_best_epoch))
        hist.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_macro_f1': float(f1), 'val_accuracy': float(metrics.get('accuracy', np.nan)), 'val_balanced_accuracy': float(bacc), 'val_kappa': float(metrics.get('kappa', np.nan)), 'val_auroc': float(metrics.get('auroc', np.nan)), 'val_target_auprc': float(metrics.get('target_auprc', np.nan)), 'val_secondary_score': float(secondary), 'decision_threshold': float(decision_threshold) if decision_threshold is not None else np.nan, 'prior_tau': float(prior_tau)})
        if eligible:
            scheduler.step(f1)
            if not qrg_objective:
                improved = f1 > best_f1 + 1e-12
                if improved:
                    best_f1 = float(f1)
            else:
                tol = float(cfg.QRG_MACRO_TOLERANCE)
                best_f1 = max(best_f1, float(f1))
                if selected_f1 < best_f1 - tol - 1e-12:
                    improved = True
                elif f1 >= best_f1 - tol - 1e-12:
                    improved = secondary > best_secondary + 1e-12 or (abs(secondary - best_secondary) <= 1e-12 and f1 > selected_f1 + 1e-12)
                else:
                    improved = False
            if improved:
                selected_f1 = float(f1)
                best_secondary = float(secondary)
                best_epoch = epoch + 1
                wait = 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.PATIENCE:
                    break
    if best is None:
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    _refresh_node_scaler(model, train_loader, device)
    return (model, pd.DataFrame(hist), best_epoch)

def train_qrg_staged(model: QRGEEG, train_loader, val_loader, n_classes: int, device):
    class_weight = _source_class_weights(train_loader, device)
    pre_hist, pre_epoch = pretrain_node_encoder(model, train_loader, val_loader, n_classes, device)
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model, stage2_hist, best2 = train_plain_model(model, train_loader, val_loader, cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, node_stage='frozen', class_weight=class_weight, min_best_epoch=cfg.QRG_MIN_STAGE2_BEST_EPOCH, calibrate_qrg=True, qrg_objective=True)
    model.node_encoder.unfreeze_final()
    model, ft_hist, best3 = _run_finetune_with_validation(model, train_loader, val_loader, device, class_weight=class_weight, min_best_epoch=cfg.QRG_MIN_FINETUNE_BEST_EPOCH, calibrate_qrg=True, qrg_objective=True)
    return (model, {'pretrain': pre_hist, 'stage2': stage2_hist, 'finetune': ft_hist}, {'pretrain': max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, int(pre_epoch)), 'stage2': max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, int(best2)), 'finetune': max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, int(best3))})

def train_graph_control_staged(model, train_loader, val_loader, n_classes: int, device):
    wrapper = NodeBackbonePretrainer(model.node_encoder, n_classes).to(device)
    wrapper, pre_hist, pre_epoch = train_plain_model(wrapper, train_loader, val_loader, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model, stage2_hist, best2 = train_plain_model(model, train_loader, val_loader, cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, node_stage='frozen')
    model.node_encoder.unfreeze_final()
    model, ft_hist, best3 = _run_finetune_with_validation(model, train_loader, val_loader, device)
    return (model, {'pretrain': pre_hist, 'stage2': stage2_hist, 'finetune': ft_hist}, {'pretrain': max(1, int(pre_epoch)), 'stage2': max(1, int(best2)), 'finetune': max(1, int(best3))})

def train_plain_fixed(model, train_loader, epochs: int, lr: float, device, node_stage: Optional[str]=None, class_weight: Optional[torch.Tensor]=None, qrg_objective: bool=False):
    model = model.to(device)
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=qrg_objective)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    _fixed_epoch_counter = 0
    for _ in range(max(1, int(epochs))):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, max(1, int(epochs)), 'train_plain_fixed')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            fixed_training_step(model)
    return model

def _finetune_fixed(model, train_loader, epochs: int, device, class_weight: Optional[torch.Tensor]=None, qrg_objective: bool=False):
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=qrg_objective)
    opt = _staged_finetune_optimizer(model)
    for _ in range(max(1, int(epochs))):
        model.train()
        _set_node_stage_mode(model, 'finetune')
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
        _refresh_node_scaler(model, train_loader, device)
    return model

def train_qrg_fixed(model: QRGEEG, train_loader, n_classes: int, device, plan: Mapping[str, int]):
    class_weight = _source_class_weights(train_loader, device)
    pretrain_epochs = max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, int(plan.get('pretrain', cfg.PRETRAIN_EPOCHS)))
    stage2_epochs = max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, int(plan.get('stage2', cfg.QUANTUM_EPOCHS)))
    finetune_epochs = max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, int(plan.get('finetune', cfg.FINETUNE_EPOCHS)))
    wrapper = NodeBackbonePretrainer(model.node_encoder, n_classes).to(device)
    wrapper = train_plain_fixed(wrapper, train_loader, pretrain_epochs, cfg.LR_PRETRAIN, device, class_weight=class_weight, qrg_objective=True)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model = train_plain_fixed(model, train_loader, stage2_epochs, cfg.LR_QUANTUM, device, node_stage='frozen', class_weight=class_weight, qrg_objective=True)
    model.node_encoder.unfreeze_final()
    model = _finetune_fixed(model, train_loader, finetune_epochs, device, class_weight=class_weight, qrg_objective=True)
    _refresh_node_scaler(model, train_loader, device)
    return model

def train_graph_control_fixed(model, train_loader, n_classes: int, device, plan: Mapping[str, int]):
    wrapper = NodeBackbonePretrainer(model.node_encoder, n_classes).to(device)
    wrapper = train_plain_fixed(wrapper, train_loader, plan.get('pretrain', cfg.PRETRAIN_EPOCHS), cfg.LR_PRETRAIN, device)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model = train_plain_fixed(model, train_loader, plan.get('stage2', cfg.QUANTUM_EPOCHS), cfg.LR_QUANTUM, device, node_stage='frozen')
    model.node_encoder.unfreeze_final()
    model = _finetune_fixed(model, train_loader, plan.get('finetune', cfg.FINETUNE_EPOCHS), device)
    _refresh_node_scaler(model, train_loader, device)
    return model

@torch.no_grad()
def qrg_angle_diagnostics(model: QRGEEG, loader, device) -> Dict[str, float]:
    model.eval()
    node_args = []
    edge_args = []
    for k, batch in enumerate(loader):
        batch = move_batch(batch, device)
        d, v = model.node_encoder(batch['x_latent'])
        v = model.node_scaler(v)
        node_args.append((v * model.circuit.theta_node).cpu().numpy())
        edge_args.append((batch['edges'] * model.circuit.theta_edge).cpu().numpy())
        if k >= 7:
            break
    if not node_args:
        return {}
    n = np.concatenate(node_args)
    e = np.concatenate(edge_args) if edge_args else np.zeros((1, 1))
    return {'node_arg_abs_mean': float(np.mean(np.abs(n))), 'node_sat_frac_gt2': float(np.mean(np.abs(n) > 2.0)), 'edge_arg_abs_mean': float(np.mean(np.abs(e))), 'edge_sat_frac_gt2': float(np.mean(np.abs(e) > 2.0))}

def clone_encoder(encoder: SharedTemporalNodeEncoder) -> SharedTemporalNodeEncoder:
    new = SharedTemporalNodeEncoder(encoder.q, encoder.feature_dim, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
    new.load_state_dict(copy.deepcopy(encoder.state_dict()))
    return new

def build_qrg_model(fold: PreparedFold, n_classes: int, variant: str='A', ablation: Optional[str]=None, layers: Optional[int]=None) -> QRGEEG:
    model = QRGEEG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, int(layers or cfg.L_DEFAULT), variant)
    model.ablation = ablation
    original_forward = model.forward

    def forward(batch, return_features=False):
        altered = dict(batch)
        if ablation == 'node_only':
            altered['edges'] = torch.zeros_like(batch['edges'])
        elif ablation == 'fixed_entanglement':
            altered['edges'] = torch.ones_like(batch['edges'])
        elif ablation == 'shuffled_edge':
            altered['edges'] = torch.roll(batch['edges'], 1, dims=1)
        elif ablation == 'edge_only':
            altered['x_latent'] = torch.zeros_like(batch['x_latent'])
        return original_forward(altered, return_features=return_features)
    model.forward = forward
    return model

def _effective_final_trainable(model: nn.Module) -> int:
    candidate = copy.deepcopy(model)
    if hasattr(candidate, 'node_encoder'):
        candidate.node_encoder.freeze_temporal()
        candidate.node_encoder.unfreeze_final()
    return count_trainable(candidate)

def _closest_hidden(builder, target: int, max_hidden: Optional[int]=None):
    max_hidden = max_hidden or cfg.PARAM_MATCH_MAX_HIDDEN
    choices = []
    for hidden in range(1, max_hidden + 1):
        model = builder(hidden)
        count = _effective_final_trainable(model)
        choices.append((abs(count - target), count, hidden, model))
    _, count, hidden, model = min(choices, key=lambda x: (x[0], x[2]))
    model.parameter_match_target = int(target)
    model.parameter_match_actual = int(count)
    model.parameter_match_gap = float(abs(count - target) / max(1, target))
    return model

def model_requirements(model_name: str, task: Optional[str]=None) -> Dict[str, bool]:
    is_mi = str(task).lower() == 'mi'
    return {'filter_bank': model_name == 'B3_TASK_STRONG' and is_mi, 'spd_bands': model_name == 'B5_GRAPH_CSPNET' and is_mi, 'plv': model_name == 'PLV_GAT'}

def build_torch_model(model_name: str, fold: PreparedFold, n_classes: int, variant: str='A', layers: Optional[int]=None):
    C, T = fold.train.x_deep.shape[1:]
    layers = int(layers or cfg.L_DEFAULT)
    if model_name == 'B2_EEGNET':
        return EEGNetClassifier(C, T, n_classes)
    if model_name == 'B3_TASK_STRONG' and fold.task == 'mi':
        return FBCNetLite(6, C, n_classes)
    if model_name == 'B4_QEEGNET':
        return QEEGNetStyle(C, T, n_classes, fold.q, layers)
    if model_name == 'B5_GRAPH_CSPNET' and fold.task == 'mi':
        return SPDTimeFrequencyGraphNet(C, 6, n_classes)
    if model_name in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION'}:
        target_model = build_qrg_model(fold, n_classes, variant, layers=layers)
        target = _effective_final_trainable(target_model)
        if model_name == 'B6_SAME_GRAPH_GAT':
            return _closest_hidden(lambda h: SameGraphGAT(SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT), fold.graph_edges, n_classes, hidden=h), target)
        return _closest_hidden(lambda h: MatchedClassicalInteraction(SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT), fold.graph_edges, n_classes, hidden=h), target)
    if model_name == 'B8_QRG_EEG':
        return build_qrg_model(fold, n_classes, variant, layers=layers)
    if model_name == 'RIEMANNIAN_VQC':
        return RiemannianVQC(fold.train.tangent_vec.shape[1], n_classes, fold.q, layers)
    if model_name == 'CRGNET':
        return CRGNetBaseline(C, T, fold.train.tangent_vec.shape[1], n_classes)
    if model_name == 'PLV_GAT':
        return PLVGATBaseline(C, n_classes)
    if model_name == 'NODE_ONLY_VQC':
        return build_qrg_model(fold, n_classes, 'A', 'node_only', layers=layers)
    if model_name == 'FIXED_ENTANGLEMENT_VQC':
        return build_qrg_model(fold, n_classes, 'A', 'fixed_entanglement', layers=layers)
    if model_name == 'SHUFFLED_EDGE_QRG':
        return build_qrg_model(fold, n_classes, 'A', 'shuffled_edge', layers=layers)
    if model_name == 'EDGE_ONLY_VQC':
        return build_qrg_model(fold, n_classes, 'A', 'edge_only', layers=layers)
    raise KeyError(model_name)

def run_single_model(model_name: str, fold: PreparedFold, seed: int, variant: str='A', layers: Optional[int]=None, evaluate_test: bool=True):
    seed_everything(seed)
    n_classes = len(fold.class_names)
    req = model_requirements(model_name, fold.task)
    is_quantum = 'VQC' in model_name or 'QRG' in model_name or model_name == 'B4_QEEGNET' or (model_name in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION'})
    batch_size = (cfg.Q_BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.Q_BATCH_SIZE_CPU) if is_quantum else cfg.BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.BATCH_SIZE_CPU
    train_loader, val_loader, test_loader = make_loaders(fold, batch_size, req['filter_bank'], req['spd_bands'], req['plv'])
    layers = int(layers or cfg.L_DEFAULT)
    start = time.perf_counter()
    statevector_batches_per_epoch = 0
    circuit_evaluations_per_epoch = 0
    circuit_depth = 0
    stage_plan = {}
    if model_name == 'B1_RIEMANNIAN_TS_LR':
        pv, pt, model = fit_riemannian_classical(fold.train, fold.val, fold.test, 'LR')
        yv, yt = (fold.val.y, fold.test.y)
        trainable = sum((np.size(x) for x in [model.coef_, model.intercept_]))
        feature_test = None
        history = None
        best_epoch = np.nan
        quantum_params = 0
        two_q = 0
    elif model_name == 'B5_GRAPH_CSPNET' and fold.task == 'p300':
        pv, pt, model = fit_xdawn_blda(fold.train, fold.val, fold.test)
        yv, yt = (fold.val.y, fold.test.y)
        trainable = int(np.size(model.model.coef_) + 1)
        feature_test = None
        history = None
        best_epoch = np.nan
        quantum_params = 0
        two_q = 0
    elif model_name == 'B3_TASK_STRONG' and fold.task == 'p300':
        feature_dim = int(np.prod(fold.train.x_latent_riem.shape[1:]))
        Xtr = fold.train.x_latent_riem.reshape(len(fold.train.y), feature_dim)
        Xv = fold.val.x_latent_riem.reshape(len(fold.val.y), feature_dim)
        Xte = fold.test.x_latent_riem.reshape(len(fold.test.y), feature_dim)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, fold.train.y)
        pv = lda.predict_proba(Xv)
        pt = lda.predict_proba(Xte) if evaluate_test else np.empty((0, len(fold.class_names)))
        yv = fold.val.y
        yt = fold.test.y
        trainable = int(np.size(lda.coef_) + np.size(lda.intercept_))
        model = lda
        feature_test = None
        history = None
        best_epoch = np.nan
        quantum_params = 0
        two_q = 0
    else:
        model = build_torch_model(model_name, fold, n_classes, variant, layers=layers)
        if isinstance(model, QRGEEG):
            model, history, stage_plan = train_qrg_staged(model, train_loader, val_loader, n_classes, DEVICE)
            best_epoch = sum(stage_plan.values())
        elif isinstance(model, (SameGraphGAT, MatchedClassicalInteraction)):
            model, history, stage_plan = train_graph_control_staged(model, train_loader, val_loader, n_classes, DEVICE)
            best_epoch = sum(stage_plan.values())
        else:
            max_epochs = cfg.QUANTUM_EPOCHS if is_quantum else cfg.PRETRAIN_EPOCHS
            lr = cfg.LR_QUANTUM if is_quantum else cfg.LR_PRETRAIN
            model, history, best_epoch = train_plain_model(model, train_loader, val_loader, max_epochs, lr, DEVICE)
            stage_plan = {'plain': max(1, int(best_epoch))}
        yv, pv, _ = predict_torch(model, val_loader, DEVICE)
        if evaluate_test:
            yt, pt, feature_test = predict_torch(model, test_loader, DEVICE)
        else:
            yt = np.empty((0,), dtype=int)
            pt = np.empty((0, n_classes))
            feature_test = None
        trainable = count_trainable(model)
        quantum_params = model.quantum_parameter_count() if hasattr(model, 'quantum_parameter_count') else int(sum((p.numel() for n, p in model.named_parameters() if 'circuit' in n and p.requires_grad)))
        two_q = model.two_qubit_gates_per_trial() if hasattr(model, 'two_qubit_gates_per_trial') else 0
        if isinstance(model, (QRGEEG, QEEGNetStyle, RiemannianVQC)):
            statevector_batches_per_epoch = len(train_loader)
            circuit_evaluations_per_epoch = len(train_loader.dataset)
            if isinstance(model, QRGEEG):
                edge_channels = 3 if model.circuit.variant == 'B' else 1
                node_axes = 2 if model.circuit.variant == 'B' else 1
                circuit_depth = model.circuit.layers * (node_axes + edge_channels * len(model.circuit.edges) + 2)
            else:
                circuit_depth = layers * (fold.q + fold.q)
    elapsed = time.perf_counter() - start
    decision_threshold = None
    prior_tau = 0.0
    pv_eval = pv
    pt_eval = pt
    if isinstance(model, QRGEEG):
        if fold.task == 'p300' and pv.ndim == 2 and (pv.shape[1] == 2):
            decision_threshold, _, _ = best_binary_threshold(yv, pv)
        elif fold.task == 'mi' and pv.ndim == 2 and (pv.shape[1] > 1):
            priors = _source_class_priors(train_loader)
            prior_tau, _, _ = best_prior_tau(yv, pv, priors, cfg.QRG_PRIOR_TAU_GRID)
            pv_eval = apply_prior_correction(pv, priors, prior_tau)
            if evaluate_test and len(pt):
                pt_eval = apply_prior_correction(pt, priors, prior_tau)
    val_metrics = multiclass_metrics(yv, pv_eval, fold.task, decision_threshold=decision_threshold)
    val_class_f1 = per_class_f1(yv, pv_eval, decision_threshold=decision_threshold)
    test_metrics = multiclass_metrics(yt, pt_eval, fold.task, decision_threshold=decision_threshold) if evaluate_test and len(yt) else {k: np.nan for k in ['macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'auroc', 'target_auprc']}
    return {'model_object': model, 'val_metrics': val_metrics, 'val_class_f1': val_class_f1.tolist(), 'test_metrics': test_metrics, 'val_y': yv, 'val_prob_raw': pv, 'source_priors': _source_class_priors(train_loader), 'y_test': yt, 'prob_test': pt_eval, 'features_test': feature_test, 'trainable_params': int(trainable), 'quantum_params': int(quantum_params), 'two_qubit_gates': int(two_q), 'circuit_depth': int(circuit_depth), 'circuit_evaluations_per_epoch': int(circuit_evaluations_per_epoch), 'statevector_batches_per_epoch': int(statevector_batches_per_epoch), 'wall_seconds': float(elapsed), 'parameter_match_target': getattr(model, 'parameter_match_target', np.nan), 'parameter_match_gap': getattr(model, 'parameter_match_gap', np.nan), 'history': history, 'best_epoch': best_epoch, 'stage_plan': stage_plan, 'decision_threshold': float(decision_threshold) if decision_threshold is not None else np.nan, 'prior_tau': float(prior_tau), 'qrg_diagnostics': qrg_angle_diagnostics(model, val_loader, DEVICE) if isinstance(model, QRGEEG) else {}}

def source_validation_indices(bundle: EEGDatasetBundle, reserved_subject: str, val_subject: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj = bundle.subjects.astype(str)
    tr = np.flatnonzero((subj != str(reserved_subject)) & (subj != str(val_subject)))
    va = np.flatnonzero(subj == str(val_subject))
    te = np.array([], dtype=int)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError('Invalid source-only validation split.')
    return (tr, va, te)

def _default_representation(bundle: EEGDatasetBundle) -> Tuple[str, str]:
    return (cfg.PROJECTION_P300, cfg.P300_EDGE_MODE) if bundle.task == 'p300' else (cfg.PROJECTION_MI, 'plain')

def run_outer_fold(bundle: EEGDatasetBundle, test_subject: str, seed: int, models: Sequence[str], variant: str='A', q: Optional[int]=None, graph_multiplier: Optional[float]=None, layers: Optional[int]=None, projection_mode: Optional[str]=None, edge_mode: Optional[str]=None):
    q = q or min(cfg.Q_DEFAULT, bundle.X.shape[1])
    graph_multiplier = graph_multiplier or cfg.GRAPH_MULTIPLIER
    layers = int(layers or cfg.L_DEFAULT)
    projection_mode, edge_mode = (projection_mode, edge_mode) if projection_mode else _default_representation(bundle)
    source = [s for s in sorted(np.unique(bundle.subjects.astype(str)).tolist()) if s != str(test_subject)]
    val_subject = choose_inner_subjects(source, cfg.RUN_MODE, seed)[0]
    tr, va, te = split_indices(bundle, test_subject, val_subject)
    fold = prepare_fold(bundle, tr, va, te, q, graph_multiplier, projection_mode, edge_mode)
    for model_name in models:
        out = run_single_model(model_name, fold, seed, variant, layers=layers, evaluate_test=True)
        reported = 'B5_XDAWN_BLDA' if model_name == 'B5_GRAPH_CSPNET' and bundle.task == 'p300' else model_name
        row = {'dataset': bundle.name, 'model': reported, 'seed': seed, 'test_subject': test_subject, 'variant': variant, 'q': q, 'layers': layers, 'graph_multiplier': graph_multiplier, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'val_subject': val_subject, 'n_edges': len(fold.graph_edges), 'best_epoch': out['best_epoch'], 'trainable_params': out['trainable_params'], 'quantum_params': out['quantum_params'], 'two_qubit_gates': out['two_qubit_gates'], 'circuit_depth': out['circuit_depth'], 'circuit_evaluations_per_epoch': out['circuit_evaluations_per_epoch'], 'statevector_batches_per_epoch': out['statevector_batches_per_epoch'], 'parameter_match_target': out['parameter_match_target'], 'parameter_match_gap': out['parameter_match_gap'], 'wall_seconds': out['wall_seconds'], **out['test_metrics']}
        key = {k: row[k] for k in ['dataset', 'model', 'seed', 'test_subject', 'variant', 'q', 'layers', 'graph_multiplier', 'projection_mode', 'edge_mode']}
        if not store.completed(key):
            store.add(row, {'y': out['y_test'], 'prob': out['prob_test'], 'features': out['features_test'] if out['features_test'] is not None else np.empty((0,))})
        print(row)

def _representation_candidates(bundle: EEGDatasetBundle):
    if bundle.task == 'mi':
        return [(m, 'plain') for m in cfg.MI_PROJECTION_CANDIDATES]
    return [(cfg.PROJECTION_P300, e) for e in cfg.P300_EDGE_CANDIDATES]

def architecture_screen(bundle: EEGDatasetBundle):
    rows = []
    reserved = outer_subjects(bundle, cfg.MAX_OUTER_FOLDS, cfg.PRIMARY_SEED)
    for reserved_subject in reserved:
        source = [s for s in sorted(np.unique(bundle.subjects.astype(str)).tolist()) if s != reserved_subject]
        vals = choose_inner_subjects(source, 'architecture', cfg.PRIMARY_SEED)
        for val_subject in vals:
            tr, va, te = source_validation_indices(bundle, reserved_subject, val_subject)
            for projection_mode, edge_mode in _representation_candidates(bundle):
                for q in [x for x in cfg.Q_ARCH if x <= bundle.X.shape[1]]:
                    fold = prepare_fold(bundle, tr, va, te, q, cfg.GRAPH_MULTIPLIER, projection_mode, edge_mode)
                    for model_name in cfg.ARCH_BASELINE_MODELS:
                        for seed in cfg.SEEDS:
                            try:
                                out = run_single_model(model_name, fold, seed, 'A', layers=cfg.L_DEFAULT, evaluate_test=False)
                                reported = 'B5_XDAWN_BLDA' if model_name == 'B5_GRAPH_CSPNET' and bundle.task == 'p300' else model_name
                                rows.append({'dataset': bundle.name, 'model': reported, 'reserved_subject_not_evaluated': reserved_subject, 'val_subject': val_subject, 'seed': seed, 'variant': 'baseline', 'q': q, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'graph_multiplier': cfg.GRAPH_MULTIPLIER, **out['val_metrics'], 'params': out['trainable_params'], 'time': out['wall_seconds'], 'stage_plan': json.dumps(out['stage_plan'])})
                            except Exception as exc:
                                rows.append({'dataset': bundle.name, 'model': model_name, 'reserved_subject_not_evaluated': reserved_subject, 'val_subject': val_subject, 'seed': seed, 'variant': 'baseline', 'q': q, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'graph_multiplier': cfg.GRAPH_MULTIPLIER, 'error': repr(exc)})
                    for variant in cfg.VARIANTS:
                        for seed in cfg.SEEDS:
                            out = run_single_model('B8_QRG_EEG', fold, seed, variant, layers=cfg.L_DEFAULT, evaluate_test=False)
                            rows.append({'dataset': bundle.name, 'model': 'B8_QRG_EEG', 'reserved_subject_not_evaluated': reserved_subject, 'val_subject': val_subject, 'seed': seed, 'variant': variant, 'q': q, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'graph_multiplier': cfg.GRAPH_MULTIPLIER, **out['val_metrics'], 'params': out['trainable_params'], 'time': out['wall_seconds'], 'stage_plan': json.dumps(out['stage_plan']), **out.get('qrg_diagnostics', {})})
    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / f'architecture_diagnostic_{bundle.name}.csv', index=False)
    return df

def run_final_model(model_name: str, fold: PreparedFold, seed: int, selected: Mapping[str, Any], plan: Mapping[str, int]):
    seed_everything(seed)
    n_classes = len(fold.class_names)
    req = model_requirements(model_name, fold.task)
    is_quantum = 'VQC' in model_name or 'QRG' in model_name or model_name == 'B4_QEEGNET' or (model_name in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION'})
    batch_size = (cfg.Q_BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.Q_BATCH_SIZE_CPU) if is_quantum else cfg.BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.BATCH_SIZE_CPU
    train_loader, _, test_loader = make_loaders(fold, batch_size, req['filter_bank'], req['spd_bands'], req['plv'])
    start = time.perf_counter()
    model = None
    feature_test = None
    quantum_params = two_q = circuit_depth = circuit_evals = 0
    if model_name == 'B1_RIEMANNIAN_TS_LR':
        model = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=seed).fit(fold.train.tangent_vec, fold.train.y)
        pt = model.predict_proba(fold.test.tangent_vec)
        yt = fold.test.y
        trainable = int(model.coef_.size + model.intercept_.size)
    elif model_name == 'B3_TASK_STRONG' and fold.task == 'p300':
        Xtr = fold.train.x_latent_riem.reshape(len(fold.train.y), -1)
        Xte = fold.test.x_latent_riem.reshape(len(fold.test.y), -1)
        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, fold.train.y)
        pt = model.predict_proba(Xte)
        yt = fold.test.y
        trainable = int(np.size(model.coef_) + np.size(model.intercept_))
    elif model_name == 'B5_GRAPH_CSPNET' and fold.task == 'p300':
        Xtr = fold.train.x_latent_riem.reshape(len(fold.train.y), -1)
        Xte = fold.test.x_latent_riem.reshape(len(fold.test.y), -1)
        model = BinaryBLDA().fit(Xtr, fold.train.y)
        pt = model.predict_proba(Xte)
        yt = fold.test.y
        trainable = int(model.model.coef_.size + 1)
    else:
        model = build_torch_model(model_name, fold, n_classes, selected['variant'], selected['layers'])
        if isinstance(model, QRGEEG):
            model = train_qrg_fixed(model, train_loader, n_classes, DEVICE, plan)
        elif isinstance(model, (SameGraphGAT, MatchedClassicalInteraction)):
            model = train_graph_control_fixed(model, train_loader, n_classes, DEVICE, plan)
        else:
            epochs = int(plan.get('plain', cfg.QUANTUM_EPOCHS if is_quantum else cfg.PRETRAIN_EPOCHS))
            lr = cfg.LR_QUANTUM if is_quantum else cfg.LR_PRETRAIN
            model = train_plain_fixed(model, train_loader, epochs, lr, DEVICE)
        yt, pt, feature_test = predict_torch(model, test_loader, DEVICE)
        trainable = count_trainable(model)
        quantum_params = model.quantum_parameter_count() if hasattr(model, 'quantum_parameter_count') else int(sum((p.numel() for n, p in model.named_parameters() if 'circuit' in n and p.requires_grad)))
        two_q = model.two_qubit_gates_per_trial() if hasattr(model, 'two_qubit_gates_per_trial') else 0
        if isinstance(model, QRGEEG):
            edge_channels = 3 if model.circuit.variant == 'B' else 1
            node_axes = 2 if model.circuit.variant == 'B' else 1
            circuit_depth = model.circuit.layers * (node_axes + edge_channels * len(model.circuit.edges) + 2)
            circuit_evals = len(train_loader.dataset)
        elif isinstance(model, (QEEGNetStyle, RiemannianVQC)):
            circuit_depth = selected['layers'] * (fold.q + fold.q)
            circuit_evals = len(train_loader.dataset)
    decision_threshold = None
    prior_tau = 0.0
    pt_eval = pt
    if isinstance(model, QRGEEG):
        if fold.task == 'p300':
            t = selected.get('decision_threshold', np.nan)
            if t is not None and np.isfinite(t):
                decision_threshold = float(t)
        elif fold.task == 'mi':
            prior_tau = float(selected.get('prior_tau', 0.0) or 0.0)
            priors = _source_class_priors(train_loader)
            pt_eval = apply_prior_correction(pt, priors, prior_tau)
    metrics = multiclass_metrics(yt, pt_eval, fold.task, decision_threshold=decision_threshold)
    class_f1 = per_class_f1(yt, pt_eval, decision_threshold=decision_threshold).tolist()
    return {'model_object': model, 'metrics': metrics, 'class_f1': class_f1, 'decision_threshold': float(decision_threshold) if decision_threshold is not None else np.nan, 'prior_tau': float(prior_tau), 'y': yt, 'prob': pt_eval, 'features': feature_test, 'trainable_params': int(trainable), 'quantum_params': int(quantum_params), 'two_qubit_gates': int(two_q), 'circuit_depth': int(circuit_depth), 'circuit_evaluations_per_epoch': int(circuit_evals), 'wall_seconds': float(time.perf_counter() - start), 'parameter_match_target': getattr(model, 'parameter_match_target', np.nan), 'parameter_match_gap': getattr(model, 'parameter_match_gap', np.nan)}
if '_V11_BASE_TRAIN_PLAIN_MODEL' not in globals():
    _V11_BASE_TRAIN_PLAIN_MODEL = train_plain_model
if '_V11_BASE_TRAIN_PLAIN_FIXED' not in globals():
    _V11_BASE_TRAIN_PLAIN_FIXED = train_plain_fixed
if '_V11_BASE_FINETUNE_WITH_VALIDATION' not in globals():
    _V11_BASE_FINETUNE_WITH_VALIDATION = _run_finetune_with_validation
if '_V11_BASE_RUN_FINAL_MODEL' not in globals():
    _V11_BASE_RUN_FINAL_MODEL = run_final_model
if '_V11_BASE_BUILD_TORCH_MODEL' not in globals():
    _V11_BASE_BUILD_TORCH_MODEL = build_torch_model
cfg.PAPER_VARIANT_CANDIDATES = ('A',)
cfg.PAPER_Q_CANDIDATES = (6, 8)
cfg.L_DEFAULT = 2
cfg.GRAPH_MULTIPLIER = 1.5
cfg.QRG_NATURAL_SUBJECT_RISK = float(getattr(cfg, 'QRG_NATURAL_RISK_WEIGHT', 0.65))
cfg.QRG_SUBJECT_CLASS_RISK = float(getattr(cfg, 'QRG_SUBJECT_CLASS_RISK_WEIGHT', 0.35))
cfg.QRG_RANK_LAMBDA = float(getattr(cfg, 'QRG_RANK_RELATIVE_WEIGHT', getattr(cfg, 'QRG_RANK_LAMBDA', 0.05)))
cfg.QRG_RANK_MARGIN = 0.25
cfg.QRG_USE_RISK_VARIANCE = False
cfg.QRG_LAMBDA_RV = 0.0
cfg.QRG_EMA_DECAY = float(getattr(cfg, 'QRG_EMA_DECAY', 0.995))
cfg.QRG_LR_STAGE2_CIRCUIT = float(getattr(cfg, 'QRG_LR_STAGE2_CIRCUIT', 0.0008))
cfg.QRG_LR_STAGE2_READOUT = float(getattr(cfg, 'QRG_LR_STAGE2_READOUT', 0.0015))
cfg.QRG_LR_FINETUNE_BACKBONE = float(getattr(cfg, 'QRG_LR_FINETUNE_BACKBONE', 8e-05))
cfg.QRG_LR_FINETUNE_CIRCUIT = float(getattr(cfg, 'QRG_LR_FINETUNE_CIRCUIT', 0.0003))
cfg.QRG_LR_FINETUNE_READOUT = float(getattr(cfg, 'QRG_LR_FINETUNE_READOUT', 0.0008))
cfg.QRG_SELECTION_VERSION = str(getattr(cfg, 'QRG_SELECTION_VERSION', 'v11_fixed_A_L2_g1p5_hybrid65_35_raw_macro_proxy12_12_6_top2_oof'))

def _v11_total_architectural_parameters(model: nn.Module) -> int:
    return int(sum((parameter.numel() for parameter in model.parameters())))

def _v11_closest_hidden_total(builder, target: int) -> nn.Module:
    choices = []
    for hidden in range(1, int(cfg.PARAM_MATCH_MAX_HIDDEN) + 1):
        candidate = builder(hidden)
        count = _v11_total_architectural_parameters(candidate)
        choices.append((abs(count - int(target)), hidden, count, candidate))
    _, _, actual, model = min(choices, key=lambda item: (item[0], item[1]))
    model.parameter_match_target = int(target)
    model.parameter_match_actual = int(actual)
    model.parameter_match_gap = float(abs(actual - target) / max(1, int(target)))
    return model

def build_torch_model(model_name: str, fold: PreparedFold, n_classes: int, variant: str='A', layers: Optional[int]=None):
    if model_name not in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION'}:
        return _V11_BASE_BUILD_TORCH_MODEL(model_name, fold, n_classes, variant, layers)
    resolved_layers = int(layers or cfg.L_DEFAULT)
    target_model = build_qrg_model(fold, n_classes, variant, layers=resolved_layers)
    target = _v11_total_architectural_parameters(target_model)
    if model_name == 'B6_SAME_GRAPH_GAT':
        return _v11_closest_hidden_total(lambda hidden: SameGraphGAT(SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT), fold.graph_edges, n_classes, hidden=hidden), target)
    return _v11_closest_hidden_total(lambda hidden: MatchedClassicalInteraction(SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT), fold.graph_edges, n_classes, hidden=hidden), target)

def _source_subject_balanced_priors(loader) -> np.ndarray:
    y = np.asarray(loader.dataset.a.y, dtype=np.int64)
    s = np.asarray(loader.dataset.a.subject_codes, dtype=np.int64)
    if len(y) == 0:
        return np.ones(1, dtype=np.float64)
    n_classes = int(y.max()) + 1
    subject_priors = []
    for subject in sorted((int(x) for x in np.unique(s) if int(x) >= 0)):
        ys = y[s == subject]
        if len(ys):
            counts = np.bincount(ys, minlength=n_classes).astype(np.float64)
            subject_priors.append(counts / max(1.0, counts.sum()))
    if not subject_priors:
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        return counts / max(1.0, counts.sum())
    prior = np.mean(np.stack(subject_priors, axis=0), axis=0)
    prior = np.maximum(prior, 1e-08)
    return prior / prior.sum()

def _source_class_priors(loader) -> np.ndarray:
    return _source_subject_balanced_priors(loader)

def _source_class_weights(loader, device):
    return None

def _finite_metric(value: Any, default: float=-1000000000.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)

def _v11_json_ready(value: Any):
    if isinstance(value, Mapping):
        return {str(key): _v11_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_v11_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _v11_json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def _v11_secondary_score(metrics: Mapping[str, float], task: str) -> float:
    values = [metrics.get('accuracy', np.nan), metrics.get('balanced_accuracy', np.nan)]
    kappa = metrics.get('kappa', np.nan)
    if np.isfinite(kappa):
        values.append(np.clip(0.5 * (float(kappa) + 1.0), 0.0, 1.0))
    auroc = metrics.get('auroc', np.nan)
    if np.isfinite(auroc):
        values.append(float(auroc))
    if task == 'p300':
        auprc = metrics.get('target_auprc', np.nan)
        if np.isfinite(auprc):
            values.append(float(auprc))
    values = [float(x) for x in values if np.isfinite(x)]
    return float(np.mean(values)) if values else -1000000000.0

def _v11_joint_score(metrics: Mapping[str, float], task: str) -> float:
    values = [metrics.get('macro_f1', np.nan), metrics.get('accuracy', np.nan), metrics.get('balanced_accuracy', np.nan), metrics.get('auroc', np.nan)]
    kappa = metrics.get('kappa', np.nan)
    if np.isfinite(kappa):
        values.append(np.clip(0.5 * (float(kappa) + 1.0), 0.0, 1.0))
    if task == 'p300':
        auprc = metrics.get('target_auprc', np.nan)
        if np.isfinite(auprc):
            values.append(float(auprc))
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return -1000000000.0
    return float(np.exp(np.mean(np.log(np.clip(values, 1e-06, 1.0)))))

def qrg_secondary_score(metrics: Mapping[str, float], task: str) -> float:
    return _v11_secondary_score(metrics, task)

class SubjectBalancedLoss(nn.Module):

    def __init__(self, use_rv: bool=False, lambda_rv: float=0.0, class_weight: Optional[torch.Tensor]=None, rank_lambda: float=0.0, rank_margin: float=0.25, qrg_objective: bool=False):
        super().__init__()
        self.use_rv = bool(use_rv)
        self.lambda_rv = float(lambda_rv)
        self.rank_lambda = float(rank_lambda)
        self.rank_margin = float(rank_margin)
        self.qrg_objective = bool(qrg_objective)
        if class_weight is None:
            self.class_weight = None
        else:
            self.register_buffer('class_weight', torch.as_tensor(class_weight, dtype=torch.float32))

    @staticmethod
    def _valid_subjects(subjects: torch.Tensor) -> List[torch.Tensor]:
        return [u for u in torch.unique(subjects) if int(u.item()) >= 0]

    def _natural_subject_risk(self, ce: torch.Tensor, subjects: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
        pieces = []
        weight = sample_weight.float().clamp_min(0.0)
        for subject in self._valid_subjects(subjects):
            mask = subjects == subject
            if mask.any():
                denom = weight[mask].sum().clamp_min(1e-08)
                pieces.append((weight[mask] * ce[mask]).sum() / denom)
        return torch.stack(pieces).mean() if pieces else ce.mean()

    def _subject_class_risk(self, ce: torch.Tensor, y: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        subject_pieces = []
        for subject in self._valid_subjects(subjects):
            subject_mask = subjects == subject
            class_pieces = []
            for klass in torch.unique(y[subject_mask]):
                cell = subject_mask & (y == klass)
                if cell.any():
                    class_pieces.append(ce[cell].mean())
            if class_pieces:
                subject_pieces.append(torch.stack(class_pieces).mean())
        return torch.stack(subject_pieces).mean() if subject_pieces else ce.mean()

    def _within_subject_ovr_rank(self, logits: torch.Tensor, y: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.rank_lambda <= 0.0 or logits.ndim != 2:
            return logits.sum() * 0.0
        n_classes = int(logits.shape[1])
        classes = [1] if n_classes == 2 else list(range(n_classes))
        subject_losses = []
        for subject in self._valid_subjects(subjects):
            subject_mask = subjects == subject
            class_losses = []
            for klass in classes:
                subject_logits = logits[subject_mask]
                other = torch.ones(n_classes, dtype=torch.bool, device=logits.device)
                other[klass] = False
                score = subject_logits[:, klass] - torch.logsumexp(subject_logits[:, other], dim=1)
                labels = y[subject_mask]
                positive = score[labels == klass]
                negative = score[labels != klass]
                if positive.numel() and negative.numel():
                    differences = positive[:, None] - negative[None, :]
                    class_losses.append(F.softplus(self.rank_margin - differences).mean())
            if class_losses:
                subject_losses.append(torch.stack(class_losses).mean())
        if not subject_losses:
            return logits.sum() * 0.0
        return torch.stack(subject_losses).mean()

    def forward(self, logits: torch.Tensor, y: torch.Tensor, subjects: torch.Tensor, sample_weight: torch.Tensor, features: Optional[torch.Tensor]=None) -> torch.Tensor:
        del features
        ce = F.cross_entropy(logits, y, reduction='none', weight=self.class_weight)
        if not self.qrg_objective:
            base = (sample_weight.float() * ce).mean()
            if self.use_rv:
                risks = [ce[subjects == u].mean() for u in self._valid_subjects(subjects) if (subjects == u).any()]
                if len(risks) > 1:
                    base = base + self.lambda_rv * torch.stack(risks).var(unbiased=False)
            return base
        natural = self._natural_subject_risk(ce, subjects, sample_weight)
        subject_class = self._subject_class_risk(ce, y, subjects)
        rank = self._within_subject_ovr_rank(logits, y, subjects)
        return float(cfg.QRG_NATURAL_SUBJECT_RISK) * natural + float(cfg.QRG_SUBJECT_CLASS_RISK) * subject_class + float(cfg.QRG_RANK_LAMBDA) * rank

def _make_training_criterion(class_weight=None, qrg_objective: bool=False):
    return SubjectBalancedLoss(use_rv=False if qrg_objective else bool(cfg.USE_RISK_VARIANCE), lambda_rv=0.0 if qrg_objective else float(cfg.LAMBDA_RV), class_weight=None if qrg_objective else class_weight, rank_lambda=float(cfg.QRG_RANK_LAMBDA) if qrg_objective else 0.0, rank_margin=float(cfg.QRG_RANK_MARGIN), qrg_objective=bool(qrg_objective))

class _V11EMA:

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.steps = 0
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        self.steps += 1
        for key, value in state.items():
            is_fitted_buffer = key.endswith('node_scaler.center') or key.endswith('node_scaler.scale')
            if self.steps == 1 or is_fitted_buffer or (not (torch.is_floating_point(value) or torch.is_complex(value))):
                self.shadow[key].copy_(value.detach())
            else:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.shadow.items()}

def _v11_cosine_scheduler(optimizer, max_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(max_epochs)), eta_min=1e-05)

def _v11_parameter_group(named_parameters: Sequence[Tuple[str, nn.Parameter]], predicate) -> List[nn.Parameter]:
    return [parameter for name, parameter in named_parameters if parameter.requires_grad and parameter.numel() > 0 and predicate(name)]

def _v11_optimizer(model: nn.Module, stage: str, fallback_lr: float):
    named = list(model.named_parameters())
    if stage == 'stage2' and isinstance(model, QRGEEG):
        circuit = _v11_parameter_group(named, lambda name: name.startswith('circuit'))
        readout = _v11_parameter_group(named, lambda name: name.startswith('readout'))
        other = _v11_parameter_group(named, lambda name: not (name.startswith('circuit') or name.startswith('readout')))
        groups = []
        if circuit:
            groups.append({'params': circuit, 'lr': float(cfg.QRG_LR_STAGE2_CIRCUIT)})
        if readout:
            groups.append({'params': readout, 'lr': float(cfg.QRG_LR_STAGE2_READOUT)})
        if other:
            groups.append({'params': other, 'lr': float(cfg.QRG_LR_STAGE2_CIRCUIT)})
    elif stage == 'finetune' and isinstance(model, QRGEEG):
        backbone_prefixes = ('node_encoder.block2', 'node_encoder.node_projection')
        backbone = _v11_parameter_group(named, lambda name: name.startswith(backbone_prefixes))
        circuit = _v11_parameter_group(named, lambda name: name.startswith('circuit'))
        readout = _v11_parameter_group(named, lambda name: name.startswith('readout'))
        other = _v11_parameter_group(named, lambda name: not (name.startswith(backbone_prefixes) or name.startswith('circuit') or name.startswith('readout')))
        groups = []
        if backbone:
            groups.append({'params': backbone, 'lr': float(cfg.QRG_LR_FINETUNE_BACKBONE)})
        if circuit:
            groups.append({'params': circuit, 'lr': float(cfg.QRG_LR_FINETUNE_CIRCUIT)})
        if readout:
            groups.append({'params': readout, 'lr': float(cfg.QRG_LR_FINETUNE_READOUT)})
        if other:
            groups.append({'params': other, 'lr': float(cfg.QRG_LR_FINETUNE_CIRCUIT)})
    elif stage == 'stage2' and isinstance(model, (SameGraphGAT, MatchedClassicalInteraction)):
        parameters = [parameter for _, parameter in named if parameter.requires_grad and parameter.numel() > 0]
        groups = [{'params': parameters, 'lr': float(cfg.QRG_LR_STAGE2_READOUT)}]
    elif stage == 'finetune' and isinstance(model, (SameGraphGAT, MatchedClassicalInteraction)):
        backbone_prefixes = ('node_encoder.block2', 'node_encoder.node_projection')
        backbone = _v11_parameter_group(named, lambda name: name.startswith(backbone_prefixes))
        interaction = _v11_parameter_group(named, lambda name: not name.startswith(backbone_prefixes))
        groups = []
        if backbone:
            groups.append({'params': backbone, 'lr': float(cfg.QRG_LR_FINETUNE_BACKBONE)})
        if interaction:
            groups.append({'params': interaction, 'lr': float(cfg.QRG_LR_FINETUNE_READOUT)})
    else:
        parameters = [parameter for _, parameter in named if parameter.requires_grad and parameter.numel() > 0]
        groups = [{'params': parameters, 'lr': float(fallback_lr)}]
    if not groups or not any((group['params'] for group in groups)):
        raise RuntimeError(f'No trainable parameters for V11 stage {stage!r}')
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)

def _v11_full_source_loader(train_loader):
    return DataLoader(train_loader.dataset, batch_size=max(1, int(train_loader.batch_size or 64)), shuffle=False, num_workers=0, pin_memory=False)

def _refresh_node_scaler(model, train_loader, device):
    if not hasattr(model, 'fit_node_scaler'):
        return
    if getattr(model, 'ablation', None) == 'edge_only':
        with torch.no_grad():
            model.node_scaler.center.zero_()
            model.node_scaler.scale.fill_(1.0)
            model.node_scaler.fitted = True
        return
    model.fit_node_scaler(_v11_full_source_loader(train_loader), device)

def _v11_sync_scaler(model: nn.Module, ema: _V11EMA) -> None:
    state = model.state_dict()
    for key in ('node_scaler.center', 'node_scaler.scale'):
        if key in state and key in ema.shadow:
            ema.shadow[key].copy_(state[key].detach())

def _v11_checkpoint_key(y_true: np.ndarray, probability: np.ndarray, task: str) -> Tuple[float, float, float]:
    metrics = multiclass_metrics(y_true, probability, task)
    class_scores = per_class_f1(y_true, probability)
    return (_finite_metric(metrics.get('macro_f1')), _finite_metric(np.min(class_scores) if len(class_scores) else np.nan), _finite_metric(_v11_joint_score(metrics, task)))

def _v11_evaluate_ema(model: nn.Module, ema: _V11EMA, train_loader, val_loader, device, refresh_scaler: bool):
    live_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(ema.snapshot())
    if refresh_scaler and hasattr(model, 'fit_node_scaler'):
        _refresh_node_scaler(model, train_loader, device)
    candidate_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    y_val, p_val, _ = predict_torch(model, val_loader, device)
    model.load_state_dict(live_state)
    return (candidate_state, y_val, p_val)

def _v11_train_validation_stage(model: nn.Module, train_loader, val_loader, max_epochs: int, fallback_lr: float, device, stage: str, min_best_epoch: int):
    model = model.to(device)
    criterion = _make_training_criterion(qrg_objective=True)
    optimizer = _v11_optimizer(model, stage, fallback_lr)
    scheduler = _v11_cosine_scheduler(optimizer, max_epochs)
    ema = _V11EMA(model, cfg.QRG_EMA_DECAY)
    candidate_records = []
    best_macro_seen = -1000000000.0
    wait = 0
    history = []
    _fixed_epoch_counter = 0
    for epoch in range(max(1, int(max_epochs))):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, max(1, int(max_epochs)), '_v11_train_validation_stage')
        model.train()
        if stage == 'stage2':
            _set_node_stage_mode(model, 'frozen')
        elif stage == 'finetune':
            _set_node_stage_mode(model, 'finetune')
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
            ema.update(model)
            losses.append(float(loss.detach()))
        scheduler.step()
        if stage == 'finetune' and hasattr(model, 'fit_node_scaler'):
            _refresh_node_scaler(model, train_loader, device)
            _v11_sync_scaler(model, ema)
        candidate, y_val, p_val = _v11_evaluate_ema(model, ema, train_loader, val_loader, device, refresh_scaler=stage == 'finetune')
        metrics = multiclass_metrics(y_val, p_val, train_loader.dataset.task)
        key = _v11_checkpoint_key(y_val, p_val, train_loader.dataset.task)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)) if losses else np.nan, 'val_macro_f1_raw': float(metrics['macro_f1']), 'val_accuracy_raw': float(metrics['accuracy']), 'val_balanced_accuracy_raw': float(metrics['balanced_accuracy']), 'val_kappa_raw': float(metrics['kappa']), 'val_auroc_raw': float(metrics['auroc']), 'val_target_auprc_raw': float(metrics['target_auprc']), 'val_joint_geometric_raw': float(_v11_joint_score(metrics, train_loader.dataset.task))})
        fixed_epoch_end()
        if epoch + 1 < max(1, int(min_best_epoch)):
            continue
        candidate_records.append({'state': candidate, 'epoch': epoch + 1, 'macro_f1': key[0], 'min_class_f1': key[1], 'joint_score': key[2]})
        if key[0] > best_macro_seen + 1e-12:
            best_macro_seen = key[0]
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.PATIENCE):
                break
    if candidate_records:
        macro_floor = best_macro_seen - float(cfg.QRG_MACRO_TOLERANCE)
        eligible = [item for item in candidate_records if item['macro_f1'] >= macro_floor - 1e-12]
        selected_checkpoint = max(eligible, key=lambda item: (item['joint_score'], item['min_class_f1'], item['macro_f1'], -item['epoch']))
        best_state = selected_checkpoint['state']
        best_epoch = int(selected_checkpoint['epoch'])
    else:
        best_state = ema.snapshot()
        best_epoch = max(1, int(min_best_epoch))
    model.load_state_dict(best_state)
    if hasattr(model, 'fit_node_scaler'):
        _refresh_node_scaler(model, train_loader, device)
    return (model, pd.DataFrame(history), int(best_epoch))

def _v11_train_baseline_validation_stage(model, train_loader, val_loader, epochs, lr, device, patience, node_stage, class_weight, min_best_epoch):
    model = model.to(device)
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(lr), weight_decay=cfg.WEIGHT_DECAY)
    maximum = max(1, int(epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum, eta_min=float(lr) * 0.05)
    best_state = None
    best_score = -float('inf')
    best_epoch = max(1, int(min_best_epoch))
    wait = 0
    history = []
    patience = int(patience or cfg.PATIENCE)
    _fixed_epoch_counter = 0
    for epoch in range(maximum):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, maximum, '_v11_train_baseline_validation_stage')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
            losses.append(float(loss.detach()))
        scheduler.step()
        if len(val_loader.dataset):
            y_val, p_val, _ = predict_torch(model, val_loader, device)
            metrics = multiclass_metrics(y_val, p_val, train_loader.dataset.task)
            score = _finite_metric(metrics['macro_f1'])
        else:
            metrics = {'accuracy': np.nan, 'balanced_accuracy': np.nan, 'kappa': np.nan, 'auroc': np.nan, 'target_auprc': np.nan}
            score = -float(np.mean(losses)) if losses else -float('inf')
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)) if losses else np.nan, 'val_macro_f1': float(score), 'val_accuracy': float(metrics.get('accuracy', np.nan)), 'val_balanced_accuracy': float(metrics.get('balanced_accuracy', np.nan)), 'val_kappa': float(metrics.get('kappa', np.nan)), 'val_auroc': float(metrics.get('auroc', np.nan)), 'val_target_auprc': float(metrics.get('target_auprc', np.nan)), 'learning_rate': float(scheduler.get_last_lr()[0])})
        fixed_epoch_end()
        if epoch + 1 < max(1, int(min_best_epoch)):
            continue
        if score > best_score + 1e-12:
            best_score = float(score)
            best_epoch = epoch + 1
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return (model, pd.DataFrame(history), int(best_epoch))

def train_plain_model(model, train_loader, val_loader, epochs, lr, device, patience=None, node_stage=None, class_weight=None, min_best_epoch=1, calibrate_qrg=False, qrg_objective=False):
    if not qrg_objective:
        if calibrate_qrg:
            raise RuntimeError('Baseline training cannot use QRG-specific validation calibration.')
        return _v11_train_baseline_validation_stage(model, train_loader, val_loader, epochs, lr, device, patience, node_stage, class_weight, min_best_epoch)
    del patience, node_stage, class_weight, calibrate_qrg
    stage = 'stage2' if isinstance(model, QRGEEG) else 'pretrain'
    return _v11_train_validation_stage(model, train_loader, val_loader, int(epochs), float(lr), device, stage, int(min_best_epoch))

def train_plain_fixed(model, train_loader, epochs: int, lr: float, device, node_stage=None, class_weight=None, qrg_objective: bool=False):
    if qrg_objective:
        return _V11_BASE_TRAIN_PLAIN_FIXED(model, train_loader, epochs, lr, device, node_stage=node_stage, class_weight=class_weight, qrg_objective=True)
    model = model.to(device)
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(lr), weight_decay=cfg.WEIGHT_DECAY)
    is_quantum = isinstance(model, (QEEGNetStyle, RiemannianVQC))
    maximum = int(cfg.QUANTUM_EPOCHS if is_quantum else cfg.PRETRAIN_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, maximum), eta_min=float(lr) * 0.05)
    _fixed_epoch_counter = 0
    for _ in range(max(1, int(epochs))):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, max(1, int(epochs)), 'train_plain_fixed')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
        scheduler.step()
    return model

def _v11_canonical_node_pretrainer(model: nn.Module, n_classes: int) -> NodeBackbonePretrainer:
    q = int(model.node_encoder.q)
    base_seed = int(torch.initial_seed() % (2 ** 31 - 1))
    canonical_seed = int((base_seed + 104729 + 1009 * q + 9176 * n_classes) % (2 ** 31 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(canonical_seed)
        encoder = SharedTemporalNodeEncoder(q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
        wrapper = NodeBackbonePretrainer(encoder, n_classes)
    torch.manual_seed(canonical_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(canonical_seed + 1)
    return wrapper

def pretrain_node_encoder(model: QRGEEG, train_loader, val_loader, n_classes: int, device):
    if getattr(model, 'ablation', None) == 'edge_only':
        _refresh_node_scaler(model, train_loader, device)
        return (pd.DataFrame(), 0)
    wrapper = _v11_canonical_node_pretrainer(model, n_classes).to(device)
    wrapper, history, best_epoch = _v11_train_validation_stage(wrapper, train_loader, val_loader, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device, 'pretrain', cfg.QRG_MIN_PRETRAIN_BEST_EPOCH)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    return (history, int(best_epoch))

def _run_finetune_with_validation(model, train_loader, val_loader, device, class_weight=None, min_best_epoch=1, calibrate_qrg=False, qrg_objective=False):
    if not qrg_objective:
        return _V11_BASE_FINETUNE_WITH_VALIDATION(model, train_loader, val_loader, device, class_weight=class_weight, min_best_epoch=min_best_epoch, calibrate_qrg=calibrate_qrg, qrg_objective=False)
    del class_weight, calibrate_qrg
    return _v11_train_validation_stage(model, train_loader, val_loader, cfg.FINETUNE_EPOCHS, cfg.LR_QUANTUM, device, 'finetune', int(min_best_epoch))

def train_qrg_staged(model: QRGEEG, train_loader, val_loader, n_classes: int, device):
    pre_history, pre_epoch = pretrain_node_encoder(model, train_loader, val_loader, n_classes, device)
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    edge_only = getattr(model, 'ablation', None) == 'edge_only'
    model.node_encoder.freeze_temporal()
    if edge_only:
        for parameter in model.node_encoder.parameters():
            parameter.requires_grad = False
    model, stage2_history, stage2_epoch = train_plain_model(model, train_loader, val_loader, cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, node_stage='frozen', class_weight=None, min_best_epoch=cfg.QRG_MIN_STAGE2_BEST_EPOCH, calibrate_qrg=False, qrg_objective=True)
    if not edge_only:
        model.node_encoder.unfreeze_final()
    model, finetune_history, finetune_epoch = _run_finetune_with_validation(model, train_loader, val_loader, device, class_weight=None, min_best_epoch=cfg.QRG_MIN_FINETUNE_BEST_EPOCH, calibrate_qrg=False, qrg_objective=True)
    return (model, {'pretrain': pre_history, 'stage2': stage2_history, 'finetune': finetune_history}, {'pretrain': int(pre_epoch), 'stage2': int(stage2_epoch), 'finetune': int(finetune_epoch)})

def _v11_train_fixed_stage(model: nn.Module, train_loader, epochs: int, max_epochs: int, fallback_lr: float, device, stage: str):
    if int(epochs) <= 0:
        return model.to(device)
    model = model.to(device)
    criterion = _make_training_criterion(qrg_objective=True)
    optimizer = _v11_optimizer(model, stage, fallback_lr)
    scheduler = _v11_cosine_scheduler(optimizer, max_epochs)
    ema = _V11EMA(model, cfg.QRG_EMA_DECAY)
    _fixed_epoch_counter = 0
    for _ in range(int(epochs)):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, int(epochs), '_v11_train_fixed_stage')
        model.train()
        if stage == 'stage2':
            _set_node_stage_mode(model, 'frozen')
        elif stage == 'finetune':
            _set_node_stage_mode(model, 'finetune')
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
            ema.update(model)
        scheduler.step()
        if stage == 'finetune' and hasattr(model, 'fit_node_scaler'):
            _refresh_node_scaler(model, train_loader, device)
            _v11_sync_scaler(model, ema)
    model.load_state_dict(ema.snapshot())
    if hasattr(model, 'fit_node_scaler'):
        _refresh_node_scaler(model, train_loader, device)
    return model

def train_qrg_fixed(model: QRGEEG, train_loader, n_classes: int, device, plan: Mapping[str, int]):
    edge_only = getattr(model, 'ablation', None) == 'edge_only'
    pretrain_epochs = 0 if edge_only else max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, int(plan.get('pretrain', cfg.PRETRAIN_EPOCHS)))
    stage2_epochs = max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, int(plan.get('stage2', cfg.QUANTUM_EPOCHS)))
    finetune_epochs = max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, int(plan.get('finetune', cfg.FINETUNE_EPOCHS)))
    if not edge_only:
        wrapper = _v11_canonical_node_pretrainer(model, n_classes).to(device)
        wrapper = _v11_train_fixed_stage(wrapper, train_loader, pretrain_epochs, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device, 'pretrain')
        model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    else:
        for parameter in model.node_encoder.parameters():
            parameter.requires_grad = False
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    if edge_only:
        for parameter in model.node_encoder.parameters():
            parameter.requires_grad = False
    model = _v11_train_fixed_stage(model, train_loader, stage2_epochs, cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, 'stage2')
    if not edge_only:
        model.node_encoder.unfreeze_final()
    model = _v11_train_fixed_stage(model, train_loader, finetune_epochs, cfg.FINETUNE_EPOCHS, cfg.LR_QUANTUM, device, 'finetune')
    _refresh_node_scaler(model, train_loader, device)
    return model

def train_graph_control_staged(model, train_loader, val_loader, n_classes: int, device):
    wrapper = _v11_canonical_node_pretrainer(model, n_classes).to(device)
    wrapper, pre_history, pre_epoch = _v11_train_validation_stage(wrapper, train_loader, val_loader, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device, 'pretrain', cfg.QRG_MIN_PRETRAIN_BEST_EPOCH)
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model, stage2_history, stage2_epoch = _v11_train_validation_stage(model, train_loader, val_loader, cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, 'stage2', cfg.QRG_MIN_STAGE2_BEST_EPOCH)
    model.node_encoder.unfreeze_final()
    model, finetune_history, finetune_epoch = _v11_train_validation_stage(model, train_loader, val_loader, cfg.FINETUNE_EPOCHS, cfg.LR_QUANTUM, device, 'finetune', cfg.QRG_MIN_FINETUNE_BEST_EPOCH)
    return (model, {'pretrain': pre_history, 'stage2': stage2_history, 'finetune': finetune_history}, {'pretrain': int(pre_epoch), 'stage2': int(stage2_epoch), 'finetune': int(finetune_epoch)})

def train_graph_control_fixed(model, train_loader, n_classes: int, device, plan: Mapping[str, int]):
    wrapper = _v11_canonical_node_pretrainer(model, n_classes).to(device)
    wrapper = _v11_train_fixed_stage(wrapper, train_loader, int(plan.get('pretrain', cfg.PRETRAIN_EPOCHS)), cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, device, 'pretrain')
    model.node_encoder.load_state_dict(wrapper.encoder.state_dict())
    model = model.to(device)
    _refresh_node_scaler(model, train_loader, device)
    model.node_encoder.freeze_temporal()
    model = _v11_train_fixed_stage(model, train_loader, int(plan.get('stage2', cfg.QUANTUM_EPOCHS)), cfg.QUANTUM_EPOCHS, cfg.LR_QUANTUM, device, 'stage2')
    model.node_encoder.unfreeze_final()
    model = _v11_train_fixed_stage(model, train_loader, int(plan.get('finetune', cfg.FINETUNE_EPOCHS)), cfg.FINETUNE_EPOCHS, cfg.LR_QUANTUM, device, 'finetune')
    _refresh_node_scaler(model, train_loader, device)
    return model

def _v11_ring_edges(q: int) -> List[Tuple[int, int]]:
    edges = []
    for left in range(int(q)):
        right = (left + 1) % int(q)
        edge = tuple(sorted((left, right)))
        if edge not in edges:
            edges.append(edge)
    return edges

class ControlledQRGEEG(QRGEEG):

    def __init__(self, q: int, graph_edges: Sequence[Tuple[int, int]], readout_edges: Sequence[Tuple[int, int]], n_classes: int, layers: int, variant: str, ablation: str):
        mode = str(ablation)
        if mode == 'node_only':
            circuit_edges = []
            measured_edges = []
        elif mode == 'fixed_entanglement':
            circuit_edges = _v11_ring_edges(q)
            measured_edges = list(circuit_edges)
        else:
            circuit_edges = list(graph_edges)
            measured_edges = list(readout_edges)
        super().__init__(q, circuit_edges, measured_edges, n_classes, layers, variant)
        self.ablation = mode
        if self.ablation == 'shuffled_edge':
            count = len(circuit_edges)
            if count > 1:
                permutation = np.roll(np.arange(count, dtype=np.int64), 1)
            elif count == 1:
                raise RuntimeError('Shuffled-edge control requires at least two graph edges')
            else:
                permutation = np.empty(0, dtype=np.int64)
            self.register_buffer('edge_permutation', torch.as_tensor(permutation, dtype=torch.long))
        if self.ablation == 'edge_only':
            for parameter in self.node_encoder.parameters():
                parameter.requires_grad = False

    def fit_node_scaler(self, loader, device) -> None:
        if self.ablation == 'edge_only':
            with torch.no_grad():
                self.node_scaler.center.zero_()
                self.node_scaler.scale.fill_(1.0)
                self.node_scaler.fitted = True
            return
        super().fit_node_scaler(loader, device)

    def _controlled_inputs(self, batch: Dict[str, torch.Tensor]):
        batch_size = int(batch['edges'].shape[0])
        if self.ablation == 'edge_only':
            node_values = torch.ones(batch_size, self.circuit.q, dtype=batch['edges'].dtype, device=batch['edges'].device)
            deep_vectors = torch.zeros(batch_size, self.circuit.q, self.circuit.deep_dim, dtype=batch['edges'].dtype, device=batch['edges'].device)
        else:
            deep_vectors, node_values = self.node_encoder(batch['x_latent'])
            node_values = self.node_scaler(node_values)
        if self.ablation == 'node_only':
            edge_values = batch['edges'].new_zeros((batch_size, 0))
        elif self.ablation == 'fixed_entanglement':
            edge_values = batch['edges'].new_ones((batch_size, len(self.circuit.edges)))
        elif self.ablation == 'shuffled_edge':
            edge_values = batch['edges'].index_select(1, self.edge_permutation)
        else:
            edge_values = batch['edges']
        if self.ablation == 'edge_only':
            edge_values = edge_values.abs()
            edge_values = edge_values.mean(dim=1, keepdim=True).expand_as(edge_values)
            edge_values = F.layer_norm(edge_values, (edge_values.shape[1],))
        return (deep_vectors, node_values, edge_values)

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool=False):
        deep_vectors, node_values, edge_values = self._controlled_inputs(batch)
        measurements = self.circuit(node_values, edge_values, deep_vectors)
        logits = self.readout(measurements)
        return (logits, measurements) if return_features else logits

def build_qrg_model(fold: PreparedFold, n_classes: int, variant: str='A', ablation: Optional[str]=None, layers: Optional[int]=None) -> QRGEEG:
    variant = 'A' if ablation is not None else str(variant).upper()
    layers = int(layers or 2)
    if ablation is None:
        model = QRGEEG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, layers, variant)
        model.ablation = None
        return model
    return ControlledQRGEEG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, layers, 'A', str(ablation))

def _v11_average_metrics(metric_rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    names = ('macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'auroc', 'target_auprc')
    result = {}
    for name in names:
        values = [float(row[name]) for row in metric_rows if name in row and np.isfinite(row[name])]
        result[name] = float(np.mean(values)) if values else np.nan
    return result

def _v11_source_balanced_threshold(records: Sequence[Mapping[str, Any]]) -> float:
    all_scores = np.concatenate([np.asarray(record['prob'], dtype=float)[:, 1] for record in records])
    grid = np.linspace(0.05, 0.95, 361)
    quantiles = np.quantile(all_scores, np.linspace(0.01, 0.99, 99))
    candidates = np.unique(np.clip(np.r_[grid, quantiles, 0.5], 0.01, 0.99))
    evaluated = []
    for threshold in candidates:
        subject_metrics = []
        subject_min_class = []
        for record in records:
            y = np.asarray(record['y'], dtype=int)
            probability = np.asarray(record['prob'], dtype=float)
            subject_metrics.append(multiclass_metrics(y, probability, 'p300', decision_threshold=float(threshold)))
            scores = per_class_f1(y, probability, decision_threshold=float(threshold))
            subject_min_class.append(float(np.min(scores)))
        averaged = _v11_average_metrics(subject_metrics)
        evaluated.append({'threshold': float(threshold), 'metrics': averaged, 'min_class_f1': float(np.mean(subject_min_class)), 'joint_score': _v11_joint_score(averaged, 'p300')})
    base = min(evaluated, key=lambda item: abs(item['threshold'] - 0.5))
    best_macro = max((_finite_metric(item['metrics']['macro_f1']) for item in evaluated))
    eligible = [item for item in evaluated if _finite_metric(item['metrics']['macro_f1']) >= max(_finite_metric(base['metrics']['macro_f1']) - 0.002, best_macro - float(cfg.QRG_MACRO_TOLERANCE)) and _finite_metric(item['metrics']['accuracy']) >= _finite_metric(base['metrics']['accuracy']) - 0.005 and (_finite_metric(item['metrics']['balanced_accuracy']) >= _finite_metric(base['metrics']['balanced_accuracy']) - 0.01) and (_finite_metric(item['metrics']['kappa']) >= _finite_metric(base['metrics']['kappa']) - 0.01)]
    if not eligible:
        return 0.5
    best = max(eligible, key=lambda item: (item['joint_score'], item['min_class_f1'], _finite_metric(item['metrics']['macro_f1']), _finite_metric(item['metrics']['accuracy']), -abs(item['threshold'] - 0.5)))
    return float(best['threshold'])

def _v11_source_balanced_tau(records: Sequence[Mapping[str, Any]]) -> float:
    evaluated = []
    for tau in cfg.QRG_PRIOR_TAU_GRID:
        subject_metrics = []
        subject_min_class = []
        for record in records:
            y = np.asarray(record['y'], dtype=int)
            probability = apply_prior_correction(np.asarray(record['prob'], dtype=float), np.asarray(record['source_priors'], dtype=float), float(tau))
            subject_metrics.append(multiclass_metrics(y, probability, 'mi'))
            scores = per_class_f1(y, probability)
            subject_min_class.append(float(np.min(scores)))
        averaged = _v11_average_metrics(subject_metrics)
        evaluated.append({'tau': float(tau), 'metrics': averaged, 'min_class_f1': float(np.mean(subject_min_class)), 'joint_score': _v11_joint_score(averaged, 'mi')})
    base = min(evaluated, key=lambda item: abs(item['tau']))
    best_macro = max((_finite_metric(item['metrics']['macro_f1']) for item in evaluated))
    eligible = [item for item in evaluated if _finite_metric(item['metrics']['macro_f1']) >= max(_finite_metric(base['metrics']['macro_f1']) - 0.002, best_macro - float(cfg.QRG_MACRO_TOLERANCE)) and _finite_metric(item['metrics']['accuracy']) >= _finite_metric(base['metrics']['accuracy']) - 0.005 and (_finite_metric(item['metrics']['balanced_accuracy']) >= _finite_metric(base['metrics']['balanced_accuracy']) - 0.01) and (_finite_metric(item['metrics']['kappa']) >= _finite_metric(base['metrics']['kappa']) - 0.01)]
    if not eligible:
        return 0.0
    best = max(eligible, key=lambda item: (item['joint_score'], item['min_class_f1'], _finite_metric(item['metrics']['macro_f1']), _finite_metric(item['metrics']['accuracy']), -abs(item['tau'])))
    return float(best['tau'])

def _v11_representations(bundle: EEGDatasetBundle) -> List[Tuple[str, str]]:
    if bundle.task == 'mi':
        return [('csp', 'plain'), ('pca', 'plain')]
    if bundle.task == 'p300':
        return [('xdawn', 'plain'), ('xdawn', 'erp_residual')]
    raise ValueError(f'Unsupported task {bundle.task!r}')

def select_qrg_source_only(bundle: EEGDatasetBundle, test_subject: str):
    cache = RUN_DIR / f'source_selection_v11__{bundle.name}__test_{test_subject}.json'
    experiment_fingerprint = str(globals().get('EXPERIMENT_FINGERPRINT', cfg.QRG_SELECTION_VERSION))
    if cache.exists() and cfg.RESUME:
        payload = json.loads(cache.read_text())
        if payload.get('selection_version') != cfg.QRG_SELECTION_VERSION:
            raise RuntimeError(f'Incompatible V11 source-selection cache: {cache}. Use a new RUN_NAME rather than mixing protocols.')
        if payload.get('experiment_fingerprint') != experiment_fingerprint:
            raise RuntimeError(f'Stale source-selection fingerprint: {cache}')
        if not isinstance(payload.get('proxy_policy'), dict):
            raise RuntimeError(f'Source-selection cache lacks proxy policy: {cache}')
        selected = payload['selected']
        if selected.get('decision_threshold') is None:
            selected['decision_threshold'] = np.nan
        fixed = selected.get('variant') == 'A' and int(selected.get('layers', -1)) == 2 and (abs(float(selected.get('graph_multiplier', np.nan)) - 1.5) < 1e-12)
        if not fixed:
            raise RuntimeError(f'Invalid fixed-primary selection in {cache}')
        return (selected, {key: int(value) for key, value in payload['stage_plan'].items()})
    source_subjects = [subject for subject in sorted(np.unique(bundle.subjects.astype(str)).tolist()) if str(subject) != str(test_subject)]
    if bundle.name == 'EEGET_ALS_MI' and bool(getattr(cfg, 'QRG_EEGET_USE_ALL_SOURCE_VALIDATORS', False)):
        val_subjects = list(source_subjects)
    else:
        val_subjects = list(choose_inner_subjects(source_subjects, 'paper', cfg.PRIMARY_SEED))
    if not val_subjects:
        raise RuntimeError(f'No source validation subject for {bundle.name}/{test_subject}')
    all_candidates = [(int(q), str(projection_mode), str(edge_mode)) for projection_mode, edge_mode in _v11_representations(bundle) for q in (6, 8) if q <= int(bundle.X.shape[1])]
    proxy_subject = val_subjects[0]
    proxy_rows = []
    proxy_failures = []
    old_budgets = {'PRETRAIN_EPOCHS': cfg.PRETRAIN_EPOCHS, 'QUANTUM_EPOCHS': cfg.QUANTUM_EPOCHS, 'FINETUNE_EPOCHS': cfg.FINETUNE_EPOCHS, 'PATIENCE': cfg.PATIENCE}
    try:
        cfg.PRETRAIN_EPOCHS = max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, int(getattr(cfg, 'QRG_PROXY_PRETRAIN_EPOCHS', 12)))
        cfg.QUANTUM_EPOCHS = max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, int(getattr(cfg, 'QRG_PROXY_QUANTUM_EPOCHS', 12)))
        cfg.FINETUNE_EPOCHS = max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, int(getattr(cfg, 'QRG_PROXY_FINETUNE_EPOCHS', 6)))
        cfg.PATIENCE = int(getattr(cfg, 'QRG_PROXY_PATIENCE', 4))
        for q, projection_mode, edge_mode in all_candidates:
            try:
                train_index, val_index, test_index = source_validation_indices(bundle, test_subject, proxy_subject)
                proxy_fold = prepare_fold(bundle, train_index, val_index, test_index, q, 1.5, projection_mode, edge_mode)
                output = run_single_model('B8_QRG_EEG', proxy_fold, cfg.PRIMARY_SEED, 'A', 2, evaluate_test=False)
                y_val = np.asarray(output['val_y'], dtype=int)
                p_val = np.asarray(output['val_prob_raw'], dtype=float)
                metrics = multiclass_metrics(y_val, p_val, bundle.task)
                class_scores = per_class_f1(y_val, p_val)
                proxy_rows.append({'q': q, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'val_subject': str(proxy_subject), 'macro_f1': float(metrics['macro_f1']), 'min_class_f1': float(np.min(class_scores)), 'secondary_score': float(_v11_secondary_score(metrics, bundle.task)), 'joint_score': float(_v11_joint_score(metrics, bundle.task)), **{f'raw_{name}': value for name, value in metrics.items()}})
            except Exception as error:
                proxy_failures.append({'q': q, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'val_subject': str(proxy_subject), 'error': repr(error)})
    finally:
        for name, value in old_budgets.items():
            setattr(cfg, name, value)
    if not proxy_rows:
        raise RuntimeError(f'Every V11 proxy candidate failed for {bundle.name}/{test_subject}: {proxy_failures}')
    proxy_best_macro = max((_finite_metric(row['macro_f1']) for row in proxy_rows))
    proxy_rows = sorted(proxy_rows, key=lambda row: (_finite_metric(row['macro_f1']) >= proxy_best_macro - float(cfg.QRG_MACRO_TOLERANCE), _finite_metric(row['joint_score']), _finite_metric(row['min_class_f1']), _finite_metric(row['macro_f1']), _finite_metric(row['secondary_score']), -int(row['q'])), reverse=True)
    for rank, row in enumerate(proxy_rows, start=1):
        row['proxy_rank'] = rank
    shortlist_count = min(max(1, int(getattr(cfg, 'QRG_PROXY_SHORTLIST', 2))), len(proxy_rows))
    shortlisted = {(int(row['q']), str(row['projection_mode']), str(row['edge_mode'])) for row in proxy_rows[:shortlist_count]}
    pd.DataFrame(proxy_rows).to_csv(RUN_DIR / f'source_proxy_v11__{bundle.name}__test_{test_subject}.csv', index=False)
    if proxy_failures:
        _v11_atomic_text(RUN_DIR / f'source_proxy_v11_failures__{bundle.name}__test_{test_subject}.json', json.dumps(proxy_failures, indent=2, default=str))
        raise RuntimeError(f'V11 proxy grid was incomplete for {bundle.name}/{test_subject}; paper mode does not select from a silently reduced grid: {proxy_failures}')
    candidate_records = {}
    flat_rows = []
    failures = []
    for projection_mode, edge_mode in _v11_representations(bundle):
        for q in (6, 8):
            if q > int(bundle.X.shape[1]):
                continue
            key = (int(q), str(projection_mode), str(edge_mode))
            if key not in shortlisted:
                continue
            records = []
            for val_subject in val_subjects:
                try:
                    train_index, val_index, test_index = source_validation_indices(bundle, test_subject, val_subject)
                    fold = prepare_fold(bundle, train_index, val_index, test_index, int(q), 1.5, projection_mode, edge_mode)
                    output = run_single_model('B8_QRG_EEG', fold, cfg.PRIMARY_SEED, 'A', 2, evaluate_test=False)
                    y_val = np.asarray(output['val_y'], dtype=int)
                    p_val = np.asarray(output['val_prob_raw'], dtype=float)
                    raw_metrics = multiclass_metrics(y_val, p_val, bundle.task)
                    class_scores = per_class_f1(y_val, p_val)
                    record = {'val_subject': str(val_subject), 'y': y_val, 'prob': p_val, 'source_priors': np.asarray(output['source_priors'], dtype=float), 'metrics': raw_metrics, 'min_class_f1': float(np.min(class_scores)), 'stage_plan': {name: int(value) for name, value in output['stage_plan'].items()}}
                    records.append(record)
                    flat_rows.append({'q': int(q), 'variant': 'A', 'layers': 2, 'graph_multiplier': 1.5, 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'val_subject': str(val_subject), 'selection_probabilities': 'raw_uncalibrated', 'min_class_f1': record['min_class_f1'], **raw_metrics, 'stage_plan': json.dumps(record['stage_plan'])})
                except Exception as error:
                    failures.append({'q': int(q), 'projection_mode': projection_mode, 'edge_mode': edge_mode, 'val_subject': str(val_subject), 'error': repr(error)})
            if len(records) == len(val_subjects):
                candidate_records[key] = records
    tuning_frame = pd.DataFrame(flat_rows)
    tuning_frame.to_csv(RUN_DIR / f'source_tuning_v11__{bundle.name}__test_{test_subject}.csv', index=False)
    if failures:
        _v11_atomic_text(RUN_DIR / f'source_tuning_v11_failures__{bundle.name}__test_{test_subject}.json', json.dumps(failures, indent=2, default=str))
        raise RuntimeError(f'V11 full source-validation grid was incomplete for {bundle.name}/{test_subject}: {failures}')
    if not candidate_records:
        raise RuntimeError(f'Every V11 source-only candidate failed for {bundle.name}/{test_subject}: {failures}')
    summaries = []
    for key, records in candidate_records.items():
        metrics = [record['metrics'] for record in records]
        averaged = _v11_average_metrics(metrics)
        macro_values = np.asarray([record['metrics']['macro_f1'] for record in records], dtype=float)
        joint_values = np.asarray([_v11_joint_score(record['metrics'], bundle.task) for record in records], dtype=float)
        summary = {'q': key[0], 'projection_mode': key[1], 'edge_mode': key[2], 'macro_f1_mean': float(np.mean(macro_values)), 'macro_f1_worst': float(np.min(macro_values)), 'macro_f1_std': float(np.std(macro_values, ddof=0)), 'joint_score_mean': float(np.mean(joint_values)), 'joint_score_worst': float(np.min(joint_values)), 'min_class_f1_mean': float(np.mean([record['min_class_f1'] for record in records])), 'secondary_mean': float(_v11_secondary_score(averaged, bundle.task)), **{f'{name}_mean': value for name, value in averaged.items()}}
        summaries.append(summary)
    best_macro_mean = max((_finite_metric(row['macro_f1_mean']) for row in summaries))
    eligible_summaries = [row for row in summaries if _finite_metric(row['macro_f1_mean']) >= best_macro_mean - float(cfg.QRG_MACRO_TOLERANCE) - 1e-12]
    best = max(eligible_summaries, key=lambda row: (_finite_metric(row['joint_score_worst']), _finite_metric(row['joint_score_mean']), _finite_metric(row['macro_f1_worst']), _finite_metric(row['min_class_f1_mean']), _finite_metric(row['macro_f1_mean']), _finite_metric(row['secondary_mean']), -_finite_metric(row['macro_f1_std'], default=1000000000.0), -int(row['q'])))
    chosen_key = (int(best['q']), str(best['projection_mode']), str(best['edge_mode']))
    chosen_records = candidate_records[chosen_key]
    stage_plan = {}
    for stage in ('pretrain', 'stage2', 'finetune'):
        stage_plan[stage] = int(round(np.median([record['stage_plan'][stage] for record in chosen_records])))
    stage_plan['pretrain'] = max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, stage_plan['pretrain'])
    stage_plan['stage2'] = max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, stage_plan['stage2'])
    stage_plan['finetune'] = max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, stage_plan['finetune'])
    selected = {'variant': 'A', 'q': int(best['q']), 'layers': 2, 'graph_multiplier': 1.5, 'projection_mode': str(best['projection_mode']), 'edge_mode': str(best['edge_mode']), 'training_recipe': str(getattr(cfg, 'TRAINING_RECIPE', 'v11_hybrid_subject_risk_65_35_ovr_rank'))}
    if bundle.task == 'p300':
        selected['decision_threshold'] = float(_v11_source_balanced_threshold(chosen_records))
        selected['prior_tau'] = 0.0
    else:
        selected['decision_threshold'] = np.nan
        selected['prior_tau'] = float(_v11_source_balanced_tau(chosen_records))
    payload = {'selection_version': cfg.QRG_SELECTION_VERSION, 'experiment_fingerprint': experiment_fingerprint, 'selected': selected, 'stage_plan': stage_plan, 'candidate_summary': summaries, 'proxy_policy': {'outer_test_used': False, 'validator': str(proxy_subject), 'pretrain_epochs': int(getattr(cfg, 'QRG_PROXY_PRETRAIN_EPOCHS', 12)), 'quantum_epochs': int(getattr(cfg, 'QRG_PROXY_QUANTUM_EPOCHS', 12)), 'finetune_epochs': int(getattr(cfg, 'QRG_PROXY_FINETUNE_EPOCHS', 6)), 'patience': int(getattr(cfg, 'QRG_PROXY_PATIENCE', 4)), 'shortlist_count': int(shortlist_count), 'ranking': proxy_rows, 'shortlisted': [list(item) for item in sorted(shortlisted)]}, 'source_validation_subjects': [str(x) for x in val_subjects], 'checkpoint_selection': 'raw_macro_f1_primary_no_calibration', 'calibration': 'post_structure_equal_weight_per_source_subject', 'outer_test_labels_used_for_selection': False}
    strict_payload = json.dumps(_v11_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
    _v11_atomic_text(cache, strict_payload)
    _v11_atomic_text(RUN_DIR / f'selected_qrg__{bundle.name}__test_{test_subject}.json', strict_payload)
    return (selected, stage_plan)

def _v11_atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def _v11_reported_model(bundle: EEGDatasetBundle, model_name: str) -> str:
    if bundle.task == 'p300' and model_name == 'B5_GRAPH_CSPNET':
        return 'B5_XDAWN_BLDA'
    return str(model_name)

def _v11_structure_signature(selected: Mapping[str, Any]) -> Dict[str, Any]:
    return {'variant': str(selected['variant']), 'q': int(selected['q']), 'layers': int(selected['layers']), 'graph_multiplier': float(selected['graph_multiplier']), 'projection_mode': str(selected['projection_mode']), 'edge_mode': str(selected['edge_mode'])}

def _v11_fold_source_priors(arrays: FoldArrays) -> np.ndarray:
    y = np.asarray(arrays.y, dtype=np.int64)
    subjects = np.asarray(arrays.subject_codes, dtype=np.int64)
    n_classes = int(y.max()) + 1
    pieces = []
    for subject in sorted((int(x) for x in np.unique(subjects) if int(x) >= 0)):
        values = y[subjects == subject]
        if len(values):
            counts = np.bincount(values, minlength=n_classes).astype(np.float64)
            pieces.append(counts / counts.sum())
    if not pieces:
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        return counts / counts.sum()
    prior = np.maximum(np.mean(np.stack(pieces), axis=0), 1e-08)
    return prior / prior.sum()

def _v11_fixed_qrg_validation_output(model_name: str, fold: PreparedFold, selected: Mapping[str, Any], qrg_plan: Mapping[str, int]):
    seed_everything(cfg.PRIMARY_SEED)
    requirements = model_requirements(model_name, fold.task)
    batch_size = cfg.Q_BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.Q_BATCH_SIZE_CPU
    train_loader, val_loader, _ = make_loaders(fold, batch_size, requirements['filter_bank'], requirements['spd_bands'], requirements['plv'])
    model = build_torch_model(model_name, fold, len(fold.class_names), 'A', 2)
    if not isinstance(model, QRGEEG):
        raise TypeError(f'Expected a QRG control, got {type(model).__name__}')
    model = train_qrg_fixed(model, train_loader, len(fold.class_names), DEVICE, qrg_plan)
    y_val, p_val, _ = predict_torch(model, val_loader, DEVICE)
    return {'val_y': y_val, 'val_prob_raw': p_val, 'source_priors': _source_subject_balanced_priors(train_loader)}

def _v11_model_plan(bundle: EEGDatasetBundle, test_subject: str, model_name: str, selected: Mapping[str, Any], qrg_plan: Mapping[str, int]) -> Dict[str, Any]:
    qrg_controls = {'NODE_ONLY_VQC', 'FIXED_ENTANGLEMENT_VQC', 'SHUFFLED_EDGE_QRG', 'EDGE_ONLY_VQC'}
    if model_name == 'B8_QRG_EEG':
        return {**{name: int(value) for name, value in qrg_plan.items()}, 'decision_threshold': float(selected.get('decision_threshold', np.nan)), 'prior_tau': float(selected.get('prior_tau', 0.0)), 'calibration_subjects': 'nested_source_oof'}
    cache = RUN_DIR / 'model_plans' / f'{bundle.name}__test_{test_subject}__{model_name}.json'
    fingerprint = str(globals().get('EXPERIMENT_FINGERPRINT', cfg.QRG_SELECTION_VERSION))
    structure = _v11_structure_signature(selected)
    if cache.exists() and cfg.RESUME:
        payload = json.loads(cache.read_text())
        if payload.get('experiment_fingerprint') != fingerprint:
            raise RuntimeError(f'Stale model-plan fingerprint: {cache}')
        if payload.get('structure') != structure:
            raise RuntimeError(f'Stale model-plan structure: {cache}')
        plan = payload['plan']
        if plan.get('decision_threshold') is None:
            plan['decision_threshold'] = np.nan
        return plan
    source_subjects = [subject for subject in sorted(np.unique(bundle.subjects.astype(str)).tolist()) if str(subject) != str(test_subject)]
    candidate_validators = list(choose_inner_subjects(source_subjects, 'paper', cfg.PRIMARY_SEED))
    if bundle.name == 'EEGET_ALS_MI' and bool(getattr(cfg, 'QRG_EEGET_USE_ALL_SOURCE_VALIDATORS', False)):
        val_subjects = list(source_subjects)
    else:
        val_subjects = candidate_validators
    records = []
    epoch_plans = []
    for val_subject in val_subjects:
        train_index, val_index, test_index = source_validation_indices(bundle, test_subject, val_subject)
        fold = prepare_fold(bundle, train_index, val_index, test_index, selected['q'], selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
        if model_name in qrg_controls:
            output = _v11_fixed_qrg_validation_output(model_name, fold, selected, qrg_plan)
            epoch_plans.append({name: int(value) for name, value in qrg_plan.items()})
        else:
            output = run_single_model(model_name, fold, cfg.PRIMARY_SEED, 'A', 2, evaluate_test=False)
            epoch_plans.append({name: int(value) for name, value in (output.get('stage_plan') or {}).items()})
        records.append({'subject': str(val_subject), 'y': np.asarray(output['val_y'], dtype=int), 'prob': np.asarray(output['val_prob_raw'], dtype=float), 'source_priors': np.asarray(output['source_priors'], dtype=float)})
    if not records:
        raise RuntimeError(f'No source-only plan record for {bundle.name}/{test_subject}/{model_name}')
    if model_name in qrg_controls:
        plan = {name: int(value) for name, value in qrg_plan.items()}
        if model_name == 'EDGE_ONLY_VQC':
            plan['pretrain'] = 0
    else:
        stage_names = sorted(set().union(*(item.keys() for item in epoch_plans)))
        plan = {stage: max(1, int(round(np.median([item[stage] for item in epoch_plans if stage in item])))) for stage in stage_names}
    if bundle.task == 'p300':
        plan['decision_threshold'] = float(_v11_source_balanced_threshold(records))
        plan['prior_tau'] = 0.0
    else:
        plan['decision_threshold'] = np.nan
        plan['prior_tau'] = float(_v11_source_balanced_tau(records))
    plan['calibration_subjects'] = ';'.join((str(x) for x in val_subjects))
    payload = {'experiment_fingerprint': fingerprint, 'structure': structure, 'model': str(model_name), 'plan': plan, 'outer_test_labels_used': False, 'decision_policy': 'model_specific_source_only'}
    _v11_atomic_text(cache, json.dumps(_v11_json_ready(payload), indent=2, sort_keys=True, allow_nan=False))
    return plan

def run_final_model(model_name: str, fold: PreparedFold, seed: int, selected: Mapping[str, Any], plan: Mapping[str, Any]):
    local_selected = dict(selected)
    local_selected['decision_threshold'] = plan.get('decision_threshold', np.nan)
    local_selected['prior_tau'] = plan.get('prior_tau', 0.0)
    output = _V11_BASE_RUN_FINAL_MODEL(model_name, fold, seed, local_selected, plan)
    model_object = output.get('model_object')
    if isinstance(model_object, nn.Module):
        if isinstance(model_object, ControlledQRGEEG) and model_object.ablation == 'edge_only':
            output['trainable_params'] = int(sum((parameter.numel() for parameter in model_object.parameters() if parameter.requires_grad)))
        else:
            output['trainable_params'] = int(sum((parameter.numel() for parameter in model_object.parameters())))
    if isinstance(model_object, (QEEGNetStyle, RiemannianVQC)):
        output['two_qubit_gates'] = int(model_object.circuit.q * model_object.circuit.layers)
        output['circuit_depth'] = int(1 + model_object.circuit.layers * (model_object.circuit.q + 1))
    if isinstance(output.get('model_object'), QRGEEG):
        return output
    probability = np.asarray(output['prob'], dtype=float)
    threshold = None
    prior_tau = 0.0
    if fold.task == 'p300':
        candidate = plan.get('decision_threshold', np.nan)
        if candidate is not None and np.isfinite(candidate):
            threshold = float(candidate)
    elif fold.task == 'mi':
        prior_tau = float(plan.get('prior_tau', 0.0) or 0.0)
        probability = apply_prior_correction(probability, _v11_fold_source_priors(fold.train), prior_tau)
    output['prob'] = probability
    output['decision_threshold'] = float(threshold) if threshold is not None else np.nan
    output['prior_tau'] = float(prior_tau)
    output['metrics'] = multiclass_metrics(output['y'], probability, fold.task, decision_threshold=threshold)
    output['class_f1'] = per_class_f1(output['y'], probability, decision_threshold=threshold).tolist()
    return output

def _v11_effective_variant(model, selected: Mapping[str, Any]) -> str:
    if isinstance(model, QRGEEG):
        return str(model.circuit.variant)
    return str(selected['variant'])

def _v11_effective_edges(model, fold: PreparedFold) -> int:
    if isinstance(model, QRGEEG):
        return int(len(model.circuit.edges))
    if isinstance(model, (QEEGNetStyle, RiemannianVQC)):
        return int(model.circuit.q)
    if isinstance(model, SPDTimeFrequencyGraphNet):
        return int(model.n_bands)
    if isinstance(model, PLVGATBaseline):
        return int(len(model.complete_edges))
    if hasattr(model, 'edges'):
        try:
            return int(len(model.edges))
        except Exception:
            pass
    return 0

def _v11_checkpoint_atomic(model: nn.Module, path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as handle:
        torch.save({'state_dict': model.state_dict(), **dict(payload)}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path

def run_outer_fold_paper(bundle: EEGDatasetBundle, test_subject: str, models: Sequence[str], phase: str='main_B1_B8'):
    missing_models = []
    for model_name in models:
        reported = _v11_reported_model(bundle, model_name)
        if not all((store.completed({'dataset': bundle.name, 'model': reported, 'seed': seed, 'test_subject': str(test_subject)}) for seed in cfg.SEEDS)):
            missing_models.append(model_name)
    if not missing_models:
        print('Skipping completed phase/fold', phase, bundle.name, test_subject)
        return
    selected, qrg_plan = select_qrg_source_only(bundle, test_subject)
    source_index, empty_val_index, test_index = split_indices(bundle, test_subject, None)
    fold = prepare_fold(bundle, source_index, empty_val_index, test_index, selected['q'], selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
    fold_dir = RUN_DIR / 'fold_artifacts' / f'{bundle.name}__test_{test_subject}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    _v11_atomic_text(fold_dir / 'fit_audit_final_all_source.json', json.dumps(fold.fit_audit, indent=2, default=str))
    for model_name in missing_models:
        reported = _v11_reported_model(bundle, model_name)
        missing_seeds = [int(seed) for seed in cfg.SEEDS if not store.completed({'dataset': bundle.name, 'model': reported, 'seed': seed, 'test_subject': str(test_subject)})]
        if not missing_seeds:
            continue
        try:
            plan = _v11_model_plan(bundle, test_subject, model_name, selected, qrg_plan)
        except Exception as error:
            with open(RUN_DIR / 'errors.jsonl', 'a', encoding='utf-8') as handle:
                for seed in missing_seeds:
                    handle.write(json.dumps({'dataset': bundle.name, 'model': reported, 'seed': seed, 'test_subject': str(test_subject), 'phase': phase, 'stage': 'source_plan', 'error': repr(error)}) + '\n')
            print('FAILED SOURCE PLAN:', bundle.name, test_subject, reported, repr(error))
            continue
        for seed in missing_seeds:
            logical_key = {'dataset': bundle.name, 'model': reported, 'seed': int(seed), 'test_subject': str(test_subject)}
            try:
                print('FINAL OUTER:', phase, bundle.name, 'test', test_subject, 'seed', seed, 'model', reported)
                output = run_final_model(model_name, fold, int(seed), selected, plan)
                if model_name in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION'}:
                    gap = float(output.get('parameter_match_gap', np.nan))
                    if not np.isfinite(gap) or gap > float(cfg.PARAM_MATCH_TOLERANCE):
                        raise RuntimeError(f'Parameter matching failed for {model_name}: gap={gap}, tolerance={cfg.PARAM_MATCH_TOLERANCE}')
                model_object = output['model_object']
                variant = _v11_effective_variant(model_object, selected)
                n_edges = _v11_effective_edges(model_object, fold)
                epoch_total = sum((int(plan[name]) for name in ('pretrain', 'stage2', 'finetune', 'plain') if name in plan))
                training_recipe = str(selected['training_recipe']) if isinstance(model_object, (QRGEEG, SameGraphGAT, MatchedClassicalInteraction)) else 'n/a'
                row = {'dataset': bundle.name, 'model': reported, 'seed': int(seed), 'test_subject': str(test_subject), 'variant': variant, 'q': int(selected['q']), 'layers': int(selected['layers']), 'graph_multiplier': float(selected['graph_multiplier']), 'projection_mode': str(selected['projection_mode']), 'edge_mode': str(selected['edge_mode']), 'val_subject': 'source-only nested; final refit all source', 'training_recipe': training_recipe, 'n_edges': int(n_edges), 'best_epoch': int(epoch_total) if epoch_total else np.nan, 'decision_threshold': output.get('decision_threshold', np.nan), 'prior_tau': float(output.get('prior_tau', 0.0)), 'trainable_params': int(output['trainable_params']), 'quantum_params': int(output['quantum_params']), 'two_qubit_gates': int(output['two_qubit_gates']), 'circuit_depth': int(output['circuit_depth']), 'circuit_evaluations_per_epoch': int(output['circuit_evaluations_per_epoch']), 'parameter_match_target': output['parameter_match_target'], 'parameter_match_gap': output['parameter_match_gap'], 'wall_seconds': float(output['wall_seconds']), 'class_f1': json.dumps(output['class_f1']), 'macro_f1': float(output['metrics']['macro_f1']), 'accuracy': float(output['metrics']['accuracy']), 'balanced_accuracy': float(output['metrics']['balanced_accuracy']), 'kappa': float(output['metrics']['kappa']), 'auroc': float(output['metrics']['auroc']), 'target_auprc': float(output['metrics']['target_auprc'])}
                checkpoint_path = None
                if cfg.SAVE_CHECKPOINTS and isinstance(model_object, nn.Module):
                    checkpoint_path = _v11_checkpoint_atomic(model_object, fold_dir / f'{reported}__seed{seed}.pt', {'row': row, 'fit_audit': fold.fit_audit, 'training_plan': plan, 'source_selected_qrg': selected, 'experiment_fingerprint': globals().get('EXPERIMENT_FINGERPRINT')})
                store.add(row, {'y': np.asarray(output['y'], dtype=np.int64), 'prob': np.asarray(output['prob'], dtype=np.float64), 'features': np.asarray(output['features'], dtype=np.float32) if output['features'] is not None else np.empty((0,), dtype=np.float32)}, internal={'phase': phase, 'training_plan': plan, 'selected_qrg': selected, 'actual_variant': variant, 'actual_n_edges': n_edges}, checkpoint_path=checkpoint_path)
                print(row)
            except Exception as error:
                with open(RUN_DIR / 'errors.jsonl', 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps({**logical_key, 'phase': phase, 'stage': 'final_refit_or_commit', 'error': repr(error)}) + '\n')
                print('FAILED:', repr(error))
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def assert_v11_active() -> None:
    errors = []
    if str(cfg.RUN_MODE) != 'paper':
        errors.append("This experiment requires RUN_MODE='paper'")
    if str(cfg.RUN_MODE) == 'paper' and str(cfg.RUN_SCOPE) != 'full_submission':
        errors.append('RUN_SCOPE must be full_submission')
    if tuple((int(x) for x in cfg.SEEDS)) != (11, 22, 33):
        errors.append('SEEDS must be exactly (11,22,33)')
    if tuple(cfg.PAPER_VARIANT_CANDIDATES) != ('A',):
        errors.append("primary variant candidates must be ('A',)")
    if int(cfg.L_DEFAULT) != 2:
        errors.append('primary circuit depth must be L=2')
    if abs(float(cfg.GRAPH_MULTIPLIER) - 1.5) > 1e-12:
        errors.append('graph multiplier must be 1.5')
    if abs(float(cfg.QRG_NATURAL_SUBJECT_RISK) + float(cfg.QRG_SUBJECT_CLASS_RISK) - 1.0) > 1e-12:
        errors.append('QRG 65/35 classification-risk weights must sum to one')
    if not bool(getattr(cfg, 'RUN_SUPPLEMENTARY_CONTROLS', True)):
        errors.append('supplementary controls must be enabled')
    if 'EXPERIMENT_FINGERPRINT' not in globals():
        errors.append('EXPERIMENT_FINGERPRINT is undefined')
    if 'store' not in globals() or not all((hasattr(store, name) for name in ('completed', 'add', 'artifact_path'))):
        errors.append('transactional ResultStore is unavailable')
    if errors:
        raise RuntimeError('Protocol check failed: ' + '; '.join(errors))
    print('Protocol check PASS:', cfg.QRG_SELECTION_VERSION)

@torch.no_grad()
def qrg_angle_diagnostics(model: QRGEEG, loader, device) -> Dict[str, float]:
    model.eval()
    node_arguments = []
    edge_arguments = []
    measurements = []
    labels = []
    for batch_number, batch in enumerate(loader):
        batch = move_batch(batch, device)
        if isinstance(model, ControlledQRGEEG):
            deep, node, edge = model._controlled_inputs(batch)
        else:
            deep, node = model.node_encoder(batch['x_latent'])
            node = model.node_scaler(node)
            edge = batch['edges']
        node_arguments.append((node * model.circuit.theta_node).detach().cpu().numpy())
        if len(model.circuit.edges):
            edge_arguments.append((edge * model.circuit.theta_edge).detach().cpu().numpy())
        measured = model.circuit(node, edge, deep)
        measurements.append(measured.detach().cpu().numpy())
        labels.append(batch['y'].detach().cpu().numpy())
        if batch_number >= 15:
            break
    if not node_arguments:
        return {}
    node_array = np.concatenate(node_arguments, axis=0)
    edge_array = np.concatenate(edge_arguments, axis=0) if edge_arguments else np.empty((len(node_array), 0), dtype=float)
    feature_array = np.concatenate(measurements, axis=0)
    return {'node_arg_abs_mean': float(np.mean(np.abs(node_array))), 'node_sat_frac_gt2': float(np.mean(np.abs(node_array) > 2.0)), 'edge_arg_abs_mean': float(np.mean(np.abs(edge_array))) if edge_array.size else 0.0, 'edge_sat_frac_gt2': float(np.mean(np.abs(edge_array) > 2.0)) if edge_array.size else 0.0, 'quantum_feature_var_mean': float(np.mean(np.var(feature_array, axis=0))), 'effective_graph_edges': int(len(model.circuit.edges)), 'effective_readout_features': int(model.readout.in_features)}
import itertools
from contextlib import contextmanager
from sklearn.model_selection import GroupKFold
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
assert '_V11EMA' in globals(), 'Training core is not initialized.'

def _v12_calibrate(records: Sequence[Mapping[str, Any]], task: str) -> Dict[str, float]:
    rule = str(cfg.V12_CALIBRATION_RULE)
    if rule == 'v11_tolerance':
        if task == 'p300':
            return {'decision_threshold': float(_V11_SOURCE_BALANCED_THRESHOLD(records)), 'prior_tau': 0.0}
        return {'decision_threshold': float('nan'), 'prior_tau': float(_V11_SOURCE_BALANCED_TAU(records))}
    if rule != 'balanced_accuracy':
        raise ValueError(f'Unknown V12_CALIBRATION_RULE {rule!r}')
    if task == 'p300':
        pooled = np.concatenate([np.asarray(r['prob'], float)[:, 1] for r in records])
        candidates = np.unique(np.clip(np.r_[np.linspace(0.02, 0.98, 97), np.quantile(pooled, np.linspace(0.005, 0.995, 199)), 0.5], 0.0001, 1 - 0.0001))
        best = None
        for t in candidates:
            m = _v11_average_metrics([multiclass_metrics(np.asarray(r['y'], int), np.asarray(r['prob'], float), 'p300', decision_threshold=float(t)) for r in records])
            key = (_finite_metric(m['balanced_accuracy']), _finite_metric(m['macro_f1']), -abs(float(t) - 0.5))
            if best is None or key > best[0]:
                best = (key, float(t))
        return {'decision_threshold': best[1], 'prior_tau': 0.0}
    best = None
    for tau in cfg.QRG_PRIOR_TAU_GRID:
        m = _v11_average_metrics([multiclass_metrics(np.asarray(r['y'], int), apply_prior_correction(np.asarray(r['prob'], float), np.asarray(r['source_priors'], float), float(tau)), 'mi') for r in records])
        key = (_finite_metric(m['balanced_accuracy']), _finite_metric(m['macro_f1']), -abs(float(tau)))
        if best is None or key > best[0]:
            best = (key, float(tau))
    return {'decision_threshold': float('nan'), 'prior_tau': best[1]}

def _v12_apply_decision(prob_raw: np.ndarray, task: str, calib: Mapping[str, float], priors: np.ndarray):
    prob_raw = np.asarray(prob_raw, float)
    if task == 'p300':
        t = calib.get('decision_threshold', np.nan)
        return (prob_raw, float(t) if t is not None and np.isfinite(t) else None)
    return (apply_prior_correction(prob_raw, np.asarray(priors, float), float(calib.get('prior_tau', 0.0) or 0.0)), None)
if '_V11_SOURCE_BALANCED_THRESHOLD' not in globals():
    _V11_SOURCE_BALANCED_THRESHOLD = _v11_source_balanced_threshold
    _V11_SOURCE_BALANCED_TAU = _v11_source_balanced_tau
_v11_source_balanced_threshold = lambda records: _v12_calibrate(records, 'p300')['decision_threshold']
_v11_source_balanced_tau = lambda records: _v12_calibrate(records, 'mi')['prior_tau']
_V12_HP: Dict[str, Any] = {}

@contextmanager
def _v12_hp(hp: Optional[Mapping[str, Any]]):
    global _V12_HP
    previous = dict(_V12_HP)
    _V12_HP = dict(hp or {})
    try:
        yield
    finally:
        _V12_HP = previous

def _v12_apply_dropout(model: nn.Module) -> nn.Module:
    if 'dropout' in _V12_HP:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(_V12_HP['dropout'])
    return model

def _v12_train_baseline_validation_stage(model, train_loader, val_loader, epochs, lr, device, patience, node_stage, class_weight, min_best_epoch):
    lr = float(_V12_HP.get('lr', lr))
    if str(cfg.V12_BASELINE_TRAINING) == 'v11_plain':
        return _V11_BASELINE_VALIDATION_STAGE(model, train_loader, val_loader, epochs, lr, device, patience, node_stage, class_weight, min_best_epoch)
    model = model.to(device)
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    maximum = max(1, int(epochs))
    scheduler = _v11_cosine_scheduler(optimizer, maximum)
    ema = _V11EMA(model, cfg.QRG_EMA_DECAY)
    records, history, best_macro, wait = ([], [], -1000000000.0, 0)
    _fixed_epoch_counter = 0
    for epoch in range(maximum):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, maximum, '_v12_train_baseline_validation_stage')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
            ema.update(model)
            losses.append(float(loss.detach()))
        scheduler.step()
        live = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.snapshot())
        candidate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if len(val_loader.dataset):
            y_val, p_val, _ = predict_torch(model, val_loader, device)
            key = _v11_checkpoint_key(y_val, p_val, train_loader.dataset.task)
        else:
            key = (-float(np.mean(losses)) if losses else -1000000000.0, 0.0, 0.0)
        model.load_state_dict(live)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)) if losses else np.nan, 'val_macro_f1_raw': key[0], 'learning_rate': float(scheduler.get_last_lr()[0])})
        fixed_epoch_end()
        if epoch + 1 < max(1, int(min_best_epoch)):
            continue
        records.append({'state': candidate, 'epoch': epoch + 1, 'macro_f1': key[0], 'min_class_f1': key[1], 'joint_score': key[2]})
        if key[0] > best_macro + 1e-12:
            best_macro, wait = (key[0], 0)
        else:
            wait += 1
            if wait >= int(patience or cfg.PATIENCE):
                break
    if records:
        floor = best_macro - float(cfg.QRG_MACRO_TOLERANCE)
        chosen = max([r for r in records if r['macro_f1'] >= floor - 1e-12], key=lambda r: (r['joint_score'], r['min_class_f1'], r['macro_f1'], -r['epoch']))
        model.load_state_dict(chosen['state'])
        best_epoch = int(chosen['epoch'])
    else:
        model.load_state_dict(ema.snapshot())
        best_epoch = max(1, int(min_best_epoch))
    return (model, pd.DataFrame(history), best_epoch)
if '_V11_BASELINE_VALIDATION_STAGE' not in globals():
    _V11_BASELINE_VALIDATION_STAGE = _v11_train_baseline_validation_stage
    _V11_TRAIN_PLAIN_FIXED = train_plain_fixed
_v11_train_baseline_validation_stage = _v12_train_baseline_validation_stage

def train_plain_fixed(model, train_loader, epochs: int, lr: float, device, node_stage=None, class_weight=None, qrg_objective: bool=False):
    if qrg_objective or str(cfg.V12_BASELINE_TRAINING) == 'v11_plain':
        return _V11_TRAIN_PLAIN_FIXED(model, train_loader, epochs, float(_V12_HP.get('lr', lr)), device, node_stage=node_stage, class_weight=class_weight, qrg_objective=qrg_objective)
    lr = float(_V12_HP.get('lr', lr))
    model = model.to(device)
    criterion = _make_training_criterion(class_weight=class_weight, qrg_objective=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    is_quantum = isinstance(model, (QEEGNetStyle, RiemannianVQC))
    scheduler = _v11_cosine_scheduler(optimizer, int(cfg.QUANTUM_EPOCHS if is_quantum else cfg.PRETRAIN_EPOCHS))
    ema = _V11EMA(model, cfg.QRG_EMA_DECAY)
    _fixed_epoch_counter = 0
    for _ in range(max(1, int(epochs))):
        _fixed_epoch_counter += 1
        fixed_epoch_start(_fixed_epoch_counter, max(1, int(epochs)), 'train_plain_fixed')
        model.train()
        if node_stage:
            _set_node_stage_mode(model, node_stage)
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch['y'], batch['subject'], _batch_sbce_weight(batch, train_loader))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            fixed_training_step(model)
            ema.update(model)
        scheduler.step()
    model.load_state_dict(ema.snapshot())
    return model

def _v12_validators(bundle: EEGDatasetBundle, test_subject: str) -> List[str]:
    source = [s for s in sorted(np.unique(bundle.subjects.astype(str)).tolist()) if s != str(test_subject)]
    if bundle.name == 'EEGET_ALS_MI' and bool(getattr(cfg, 'QRG_EEGET_USE_ALL_SOURCE_VALIDATORS', False)):
        return list(source)
    return list(choose_inner_subjects(source, 'paper', cfg.PRIMARY_SEED))

def _v12_select_by_validation(summaries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    best_mean = max((_finite_metric(s['macro_f1_mean']) for s in summaries))
    eligible = [s for s in summaries if _finite_metric(s['macro_f1_mean']) >= best_mean - float(cfg.QRG_MACRO_TOLERANCE) - 1e-12]
    return max(eligible, key=lambda s: (_finite_metric(s['joint_worst']), _finite_metric(s['joint_mean']), _finite_metric(s['macro_f1_worst']), _finite_metric(s['macro_f1_mean'])))

def _v12_summarise(records: Sequence[Mapping[str, Any]], task: str) -> Dict[str, float]:
    metrics = [multiclass_metrics(r['y'], r['prob'], task) for r in records]
    macro = np.array([m['macro_f1'] for m in metrics], float)
    joint = np.array([_v11_joint_score(m, task) for m in metrics], float)
    return {'macro_f1_mean': float(np.mean(macro)), 'macro_f1_worst': float(np.min(macro)), 'joint_mean': float(np.mean(joint)), 'joint_worst': float(np.min(joint))}

def _v12_grid(model_name: str, task: str) -> List[Dict[str, Any]]:
    if model_name in _V12_NEURAL_GRID_MODELS:
        return [{'lr': lr, 'dropout': do} for lr in cfg.V12_BASELINE_LR_GRID for do in cfg.V12_BASELINE_DROPOUT_GRID]
    if model_name in ('B1_RIEMANNIAN_TS_LR',) or (model_name == 'B8_XDAWNCOV_FBCSP' and task == 'p300'):
        return [{'C': c} for c in cfg.V12_LINEAR_C_GRID]
    if model_name == 'RFF_E2E':
        return [{'sigma': s} for s in cfg.V12_RFF_SIGMA_GRID]
    return [{}]

class CovarianceMap(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        values = np.maximum(np.linalg.eigvalsh(0.5 * (X + X.swapaxes(-1, -2))), 1e-09)
        matrices = np.eye(X.shape[-1], dtype=X.dtype)[None, :, :] * values[:, None, :]
        scale = np.sqrt(np.diagonal(matrices, axis1=-2, axis2=-1))
        return (matrices / (scale[..., :, None] * scale[..., None, :])).astype(np.float32)

class LatentLinearClassifier(NodeBackbonePretrainer):

    def forward(self, batch):
        _, values = self.encoder(batch['x_latent'])
        values = torch.sort(values, dim=1).values
        values = values.mean(dim=1, keepdim=True).expand_as(values)
        values = F.layer_norm(values, (values.shape[1],))
        return self.classifier(values)

def _v12_erp_features(arrays):
    values = np.sort(arrays.x_latent_riem, axis=-1)
    values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    values = values.mean(axis=1, keepdims=True).repeat(arrays.x_latent_riem.shape[1], axis=1)
    return values.reshape(len(arrays.y), -1)

class EEGConformerStyle(nn.Module):

    def __init__(self, n_channels: int, n_times: int, n_classes: int, emb: int=40, depth: int=2, heads: int=4, dropout: float=0.5):
        super().__init__()
        self.patch = nn.Sequential(nn.Conv2d(1, emb, (1, 25), padding=(0, 12)), nn.Conv2d(emb, emb, (n_channels, 1)), nn.BatchNorm2d(emb), nn.ELU(), nn.AvgPool2d((1, 16), stride=(1, 8)), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=2 * emb, dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(emb), nn.Dropout(dropout), nn.Linear(emb, n_classes))

    def forward(self, batch):
        tokens = self.patch(batch['x_deep'].unsqueeze(1)).squeeze(2).transpose(1, 2)
        tokens = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        tokens = tokens - tokens.mean(dim=1, keepdim=True)
        return self.head(self.encoder(tokens).mean(1))

class EEGNetOnLatent(nn.Module):

    def __init__(self, q: int, n_times: int, n_classes: int):
        super().__init__()
        self.net = EEGNetClassifier(q, n_times, n_classes)

    def forward(self, batch):
        return self.net.classifier(self.net.features(batch['x_latent']))

class TrigSurrogate(MatchedClassicalInteraction):

    def __init__(self, node_encoder, edges, n_classes: int, max_order: int=1):
        nn.Module.__init__(self)
        self.node_encoder = node_encoder
        self.edges = [tuple(sorted(e)) for e in edges]
        q = node_encoder.q
        self.node_scaler = RobustTorchScaler(q)
        self.theta_node = nn.Parameter(cfg.THETA_INIT + cfg.THETA_INIT_STD * torch.randn(q))
        self.theta_edge = nn.Parameter(cfg.THETA_INIT + cfg.THETA_INIT_STD * torch.randn(len(self.edges)))
        self.max_order = int(max_order)
        self.terms = []
        for i in range(q):
            incident = [(k, b if a == i else a) for k, (a, b) in enumerate(self.edges) if i in (a, b)]
            for order in range(0, min(self.max_order, len(incident)) + 1):
                for subset in itertools.combinations(incident, order):
                    used = {k for k, _ in subset}
                    self.terms.append((i, list(subset), [k for k, _ in incident if k not in used]))
        coord = list(range(q)) + [i for i, _, _ in self.terms] + [q + k for k in range(len(self.edges)) for _ in range(6)]
        self.register_buffer('coordinate', torch.as_tensor(coord, dtype=torch.long))
        lam_z = math.cos(cfg.MIXER_X_INIT) * math.cos(cfg.MIXER_Y_INIT)
        init = torch.cat([torch.full((q,), lam_z), 0.3 * torch.randn(len(coord) - q)])
        self.term_weight = nn.Parameter(init)
        self.n_coordinates = q + len(self.edges)
        self.readout = nn.Linear(self.n_coordinates, n_classes)

    def features(self, v, e):
        alpha = cfg.ALPHA_MAX * torch.tanh(v * self.theta_node)
        beta = cfg.BETA_MAX * torch.tanh(e * self.theta_edge)
        ca, sa, cb, sb = (torch.cos(alpha), torch.sin(alpha).abs(), torch.cos(beta), torch.sin(beta).abs())
        cols = [ca]
        light = []
        for i, subset, rest in self.terms:
            term = sa[:, i]
            for k, j in subset:
                term = term * sb[:, k] * ca[:, j]
            for k in rest:
                term = term * cb[:, k]
            light.append(term)
        cols.append(torch.stack(light, 1))
        pair = []
        for k, (i, j) in enumerate(self.edges):
            pair += [ca[:, i] * ca[:, j], sa[:, i] * sa[:, j], sa[:, j] * sb[:, k], sa[:, i] * sb[:, k], ca[:, i] * sa[:, j] * cb[:, k], ca[:, j] * sa[:, i] * cb[:, k]]
        if pair:
            cols.append(torch.stack(pair, 1))
        return torch.cat(cols, 1)

    def measurements(self, v, e):
        phi = self.features(v, e) * self.term_weight
        return phi.new_zeros((phi.shape[0], self.n_coordinates)).index_add_(1, self.coordinate, phi)

    def forward(self, batch):
        _, v = self.node_encoder(batch['x_latent'])
        v = self.node_scaler(v)
        v = v.mean(dim=1, keepdim=True).expand_as(v)
        v = F.layer_norm(v, (v.shape[1],))
        return self.readout(self.measurements(v, batch['edges']))

class RFFEndToEnd(MatchedClassicalInteraction):

    def __init__(self, node_encoder, edges, n_classes: int, n_features: int=1024, sigma: float=math.pi / 2):
        nn.Module.__init__(self)
        self.node_encoder = node_encoder
        self.edges = list(edges)
        self.node_scaler = RobustTorchScaler(node_encoder.q)
        dim = node_encoder.q + len(self.edges)
        generator = torch.Generator().manual_seed(int(cfg.PRIMARY_SEED))
        W = torch.randn(dim, int(n_features), generator=generator) / float(sigma)
        self.register_buffer('W', W.mean(dim=0, keepdim=True).expand_as(W).clone())
        self.register_buffer('b', 2 * math.pi * torch.rand(int(n_features), generator=generator))
        self.head = nn.Linear(int(n_features), n_classes)

    def forward(self, batch):
        _, v = self.node_encoder(batch['x_latent'])
        z = torch.cat([self.node_scaler(v), batch['edges']], 1)
        z = z - z.mean(dim=1, keepdim=True)
        return self.head(math.sqrt(2.0 / self.W.shape[1]) * torch.cos(z @ self.W + self.b))

class V12ControlledQRG(ControlledQRGEEG):

    def __init__(self, q, graph_edges, readout_edges, n_classes, layers, ablation, permutation=None):
        mode = str(ablation)
        base_mode = {'node_only_eq': 'full', 'fixed_same_e': 'full', 'shuffled_perm': 'full', 'mixer_none': 'full', 'mixer_ry_only': 'full', 'graph_full': 'full'}.get(mode, mode)
        ControlledQRGEEG.__init__(self, q, graph_edges, readout_edges, n_classes, layers, 'A', base_mode if base_mode != 'full' else 'full')
        self.ablation = mode
        if mode == 'node_only_eq':
            self.circuit = GraphConditionedCircuit(q, [], list(readout_edges), layers, 'A', cfg.NODE_FEATURE_DIM)
            self.readout = nn.Linear(q + len(readout_edges), n_classes)
        if mode == 'shuffled_perm':
            self.register_buffer('edge_permutation', torch.as_tensor(np.asarray(permutation), dtype=torch.long))
        if mode in ('mixer_none', 'mixer_ry_only'):
            with torch.no_grad():
                self.circuit.mixer_x.zero_()
                if mode == 'mixer_none':
                    self.circuit.mixer_y.zero_()
            self.circuit.mixer_x.requires_grad = False
            if mode == 'mixer_none':
                self.circuit.mixer_y.requires_grad = False

    def _controlled_inputs(self, batch):
        if self.ablation not in ('node_only_eq', 'fixed_same_e', 'shuffled_perm', 'mixer_none', 'mixer_ry_only', 'graph_full'):
            return ControlledQRGEEG._controlled_inputs(self, batch)
        deep_vectors, node_values = self.node_encoder(batch['x_latent'])
        node_values = self.node_scaler(node_values)
        e = batch['edges']
        if self.ablation == 'node_only_eq':
            e = e.new_zeros((e.shape[0], 0))
        elif self.ablation == 'fixed_same_e':
            e = torch.ones_like(e)
        elif self.ablation == 'shuffled_perm':
            e = e.index_select(1, self.edge_permutation)
        node_values = node_values.mean(dim=1, keepdim=True).expand_as(node_values)
        node_values = F.layer_norm(node_values, (node_values.shape[1],))
        if self.ablation in ('shuffled_perm', 'graph_full', 'mixer_none', 'mixer_ry_only') and e.shape[1]:
            e = e.square()
            e = e.mean(dim=1, keepdim=True).expand_as(e)
            e = F.layer_norm(e, (e.shape[1],))
        return (deep_vectors, node_values, e)

def _v12_edge_permutations(edges: Sequence[Tuple[int, int]], q: int, n: int, seed_key: str):
    edges = [tuple(sorted(e)) for e in edges]
    graph = nx.Graph()
    graph.add_nodes_from(range(q))
    graph.add_edges_from(edges)
    index = {e: k for k, e in enumerate(edges)}
    induced = set()
    for count, sigma in enumerate(nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter()):
        induced.add(tuple((index[tuple(sorted((sigma[a], sigma[b])))] for a, b in edges)))
        if count > 20000:
            break
    rng = np.random.default_rng(int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16))
    chosen, excluded, tries = ([], 0, 0)
    while len(chosen) < int(n) and tries < 100000:
        tries += 1
        perm = tuple((int(x) for x in rng.permutation(len(edges))))
        if perm in induced:
            excluded += 1
            continue
        if perm not in chosen:
            chosen.append(perm)
    return ([list(p) for p in chosen], int(excluded), int(len(induced)))

def _v12_random_topology_fold(fold: PreparedFold, graph_index: int, seed_key: str) -> PreparedFold:
    q, n_edges = (fold.q, len(fold.graph_edges))
    rng = np.random.default_rng(int(hashlib.sha256(f'{seed_key}|{graph_index}'.encode()).hexdigest()[:8], 16))
    pairs = [(i, j) for i in range(q) for j in range(i + 1, q)]
    for _ in range(10000):
        pick = [pairs[k] for k in rng.choice(len(pairs), size=n_edges, replace=False)]
        g = nx.Graph()
        g.add_nodes_from(range(q))
        g.add_edges_from(pick)
        if nx.is_connected(g) and sorted(pick) != sorted(map(tuple, fold.graph_edges)):
            break
    edges = sorted((tuple(sorted(e)) for e in pick))
    H = fold.v12_H
    raw = {part: np.stack([[h[i, j] for i, j in edges] for h in H[part]]) if len(H[part]) else np.empty((0, len(edges)), np.float32) for part in ('train', 'val', 'test')}
    scaler = RobustFeatureScaler().fit(raw['train'])
    new = copy.copy(fold)
    new.graph_edges = edges
    new.readout_edges = edges
    for part in ('train', 'val', 'test'):
        arrays = copy.copy(getattr(fold, part))
        arrays.edges = scaler.transform(raw[part]).astype(np.float32) if len(raw[part]) else raw[part].astype(np.float32)
        setattr(new, part, arrays)
    new.fit_audit = dict(fold.fit_audit, v12_random_topology=edges, v12_graph_index=int(graph_index))
    return new

def _v12_euclidean_alignment_fold(fold: PreparedFold) -> PreparedFold:
    new = copy.copy(fold)
    for part in ('train', 'val', 'test'):
        arrays = copy.copy(getattr(fold, part))
        X = np.array(arrays.x_deep, dtype=np.float64, copy=True)
        for subject in np.unique(arrays.subject_names.astype(str)):
            mask = arrays.subject_names.astype(str) == subject
            R = np.mean([x @ x.T / x.shape[1] for x in X[mask]], axis=0)
            R = np.diag(np.diag(R))
            _, inv_sqrt = spd_sqrt_and_inv(R + 1e-06 * np.trace(R) / len(R) * np.eye(len(R)))
            X[mask] = np.einsum('cd,ndt->nct', inv_sqrt, X[mask])
        arrays.x_deep = X.astype(np.float32)
        setattr(new, part, arrays)
    return new

class _SklearnProba:

    def __init__(self, estimator, n_params):
        self.estimator, self.n_params = (estimator, int(n_params))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

def _v12_fit_standard_pipeline(model_name: str, fold: PreparedFold, hp: Mapping[str, Any]):
    from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    from pyriemann.spatialfilters import CSP as RCSP
    train = fold.train
    if fold.task == 'p300' and model_name == 'B8_XDAWNCOV_FBCSP':
        est = make_pipeline(XdawnCovariances(nfilter=int(cfg.V12_B8_NFILTER), estimator='oas'), CovarianceMap(), TangentSpace(metric='riemann'), LogisticRegression(C=float(hp.get('C', 1.0)), class_weight='balanced', max_iter=3000))
        est.fit(train.x_riem.astype(np.float64), train.y)
        lr = est[-1]
        return (lambda a: est.predict_proba(a.x_riem.astype(np.float64)), int(lr.coef_.size + lr.intercept_.size))
    if fold.task == 'p300' and model_name == 'B9_ERPCOV_MDM':
        est = make_pipeline(ERPCovariances(estimator='oas'), CovarianceMap(), MDM(metric='riemann'))
        est.fit(train.x_riem.astype(np.float64), train.y)
        d = est[0].transform(train.x_riem[:1].astype(np.float64)).shape[-1]
        return (lambda a: est.predict_proba(a.x_riem.astype(np.float64)), int(len(np.unique(train.y)) * d * (d + 1) // 2))
    if fold.task == 'mi' and model_name == 'B8_XDAWNCOV_FBCSP':
        bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 28), (28, 40)]

        def band_features(arrays, fitted=None):
            fb = filter_bank_numpy(arrays.x_raw, fold.sfreq, bands)
            feats, out = ([], fitted or [])
            for k in range(len(bands)):
                covs = Covariances(estimator='oas').transform(fb[:, k].astype(np.float64))
                if fitted is None:
                    csp = RCSP(nfilter=int(cfg.V12_B8_CSP_FILTERS), log=True).fit(covs, arrays.y)
                    out.append(csp)
                values = np.sort(out[k].transform(covs), axis=1)
            values = values.mean(axis=1, keepdims=True).repeat(values.shape[1], axis=1)
            values = values - values.mean(axis=1, keepdims=True)
            feats.append(values)
            return (np.nan_to_num(np.concatenate(feats, 1), nan=0.0, posinf=0.0, neginf=0.0), out)
        Ftr, csps = band_features(train)
        k_best = int(min(cfg.V12_B8_MI_SELECT, Ftr.shape[1]))
        sel = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=cfg.PRIMARY_SEED), k=k_best).fit(Ftr, train.y)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(sel.transform(Ftr), train.y)
        predict = lambda a: lda.predict_proba(sel.transform(band_features(a, csps)[0]))
        return (predict, int(np.size(lda.coef_) + np.size(lda.intercept_) + sum((np.size(c.filters_) for c in csps))))
    raise KeyError(f'{model_name} is not defined for task {fold.task}')

def _v12_fit_b1(fold, hp):
    rows, columns = np.triu_indices(fold.q)
    keep = rows == columns

    def design(arrays):
        values = arrays.tangent_vec[:, keep]
        values = values.mean(axis=1, keepdims=True).repeat(values.shape[1], axis=1)
        return values - values.mean(axis=1, keepdims=True)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=float(hp.get('C', 1.0)), random_state=cfg.PRIMARY_SEED).fit(design(fold.train), fold.train.y)
    return (lambda arrays: lr.predict_proba(design(arrays)), int(lr.coef_.size + lr.intercept_.size))
_V12_NEURAL_GRID_MODELS = {'B2_EEGNET', 'B4_QEEGNET', 'B3_TASK_STRONG', 'B5_GRAPH_CSPNET', 'CRGNET', 'B10_EEG_CONFORMER', 'C2_EEGNET_LATENT', 'EA_EEGNET', 'RIEMANNIAN_VQC', 'PLV_GAT'}
_V12_QUANTUM_PLAIN = {'B4_QEEGNET', 'RIEMANNIAN_VQC'}

def _v12_family(model_name: str, task: str) -> str:
    if model_name == 'B8_QRG_EEG':
        return 'qrg'
    if model_name in ('B1_RIEMANNIAN_TS_LR',):
        return 'b1'
    if model_name in ('B8_XDAWNCOV_FBCSP', 'B9_ERPCOV_MDM'):
        return 'std'
    if task == 'p300' and model_name in ('B3_TASK_STRONG', 'B5_GRAPH_CSPNET'):
        return 'lda'
    if model_name in ('B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE', 'RFF_E2E'):
        return 'graph_control'
    if model_name == 'C1_STAGE1_HEAD':
        return 'stage1'
    if model_name in ('EDGE_ONLY_VQC', 'C3_NODE_ONLY_EQ', 'C5_FIXED_SAME_E') or model_name.startswith(('C6_RANDOM_TOPOLOGY_G', 'C7_SHUFFLED_TT_P', 'QRG_DIAG_')):
        return 'qrg_control'
    if model_name in ('B7_FROZEN', 'RFF_FROZEN', 'SVM_FROZEN'):
        return 'frozen'
    if model_name.startswith('C8_SHUFFLED_TEST_P'):
        return 'test_only'
    return 'plain'

def _v12_index(model_name):
    return int(re.search('(\\d+)$', model_name).group(1))

def _v12_build(model_name: str, fold: PreparedFold, n_classes: int, hp: Mapping[str, Any], extra: Mapping[str, Any]) -> nn.Module:
    C, T = fold.train.x_deep.shape[1:]
    q = fold.q
    new_encoder = lambda: SharedTemporalNodeEncoder(q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
    if model_name == 'B10_EEG_CONFORMER':
        model = EEGConformerStyle(C, T, n_classes)
    elif model_name == 'C2_EEGNET_LATENT':
        model = EEGNetOnLatent(q, fold.train.x_latent_deep.shape[-1], n_classes)
    elif model_name == 'EA_EEGNET':
        model = EEGNetClassifier(C, T, n_classes)
    elif model_name == 'B12_TRIG_SURROGATE':
        target = _v11_total_architectural_parameters(build_qrg_model(fold, n_classes, 'A', layers=2))
        options = []
        for order in (2, 1, 0):
            candidate = TrigSurrogate(new_encoder(), fold.graph_edges, n_classes, max_order=order)
            actual = _v11_total_architectural_parameters(candidate)
            options.append((abs(actual - target) / max(1, target), order, actual, candidate))
        within = [o for o in options if o[0] <= float(cfg.PARAM_MATCH_TOLERANCE)]
        gap, order, actual, model = within[0] if within else min(options, key=lambda o: o[0])
        model.parameter_match_target, model.parameter_match_actual, model.parameter_match_gap = (int(target), int(actual), float(gap))
    elif model_name == 'C1_STAGE1_HEAD':
        model = build_qrg_model(fold, n_classes, 'A', layers=2)
    elif model_name == 'RFF_E2E':
        model = RFFEndToEnd(new_encoder(), fold.graph_edges, n_classes, int(cfg.V12_RFF_E2E_FEATURES), float(hp.get('sigma', math.pi / 2)))
    elif model_name == 'C3_NODE_ONLY_EQ':
        model = V12ControlledQRG(q, fold.graph_edges, fold.readout_edges, n_classes, 2, 'node_only_eq')
    elif model_name == 'C5_FIXED_SAME_E':
        model = V12ControlledQRG(q, fold.graph_edges, fold.readout_edges, n_classes, 2, 'fixed_same_e')
    elif model_name.startswith('C7_SHUFFLED_TT_P'):
        model = V12ControlledQRG(q, fold.graph_edges, fold.readout_edges, n_classes, 2, 'shuffled_perm', permutation=extra['permutation'])
    elif model_name.startswith('C6_RANDOM_TOPOLOGY_G') or model_name in ('QRG_DIAG_Q6', 'QRG_DIAG_Q8'):
        model = build_qrg_model(fold, n_classes, 'A', layers=2)
    elif model_name == 'QRG_DIAG_L4':
        model = build_qrg_model(fold, n_classes, 'A', layers=4)
    elif model_name in ('QRG_DIAG_NOMIX', 'QRG_DIAG_RYONLY'):
        model = V12ControlledQRG(q, fold.graph_edges, fold.readout_edges, n_classes, 2, 'mixer_none' if model_name.endswith('NOMIX') else 'mixer_ry_only')
    else:
        model = build_torch_model(model_name, fold, n_classes, 'A', 2)
    with _v12_hp(hp):
        return _v12_apply_dropout(model)

def _v12_requirements(model_name: str, task: str) -> Dict[str, bool]:
    return model_requirements(model_name, task)

def _v12_loaders(model_name: str, fold: PreparedFold):
    family = _v12_family(model_name, fold.task)
    quantum_batch = family in ('qrg', 'graph_control', 'qrg_control', 'stage1') or model_name in _V12_QUANTUM_PLAIN
    batch = (cfg.Q_BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.Q_BATCH_SIZE_CPU) if quantum_batch else cfg.BATCH_SIZE_GPU if DEVICE.type == 'cuda' else cfg.BATCH_SIZE_CPU
    req = _v12_requirements(model_name, fold.task)
    return make_loaders(fold, batch, req['filter_bank'], req['spd_bands'], req['plv'])

def _v12_transform_fold(model_name: str, fold: PreparedFold, key: str) -> PreparedFold:
    if model_name.startswith('C6_RANDOM_TOPOLOGY_G'):
        return _v12_random_topology_fold(fold, _v12_index(model_name), key)
    if model_name == 'EA_EEGNET':
        return _v12_euclidean_alignment_fold(fold)
    return fold

def _v12_torch_params(model: nn.Module) -> int:
    return int(sum((p.numel() for p in model.parameters())))

def _v12_fit(model_name: str, fold: PreparedFold, seed: int, hp: Mapping[str, Any], mode: str, epochs: Optional[Mapping[str, int]]=None, qrg_plan: Optional[Mapping[str, int]]=None, extra: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]:
    seed_everything(int(seed))
    extra = dict(extra or {})
    K = len(fold.class_names)
    family = _v12_family(model_name, fold.task)
    evaluate = fold.val if mode == 'validate' else fold.test
    t0 = time.perf_counter()
    out: Dict[str, Any] = {'epochs': {}}
    if family in ('b1', 'std', 'lda'):
        if family == 'b1':
            predict, n_params = _v12_fit_b1(fold, hp)
        elif family == 'std':
            predict, n_params = _v12_fit_standard_pipeline(model_name, fold, hp)
        else:
            Xtr = _v12_erp_features(fold.train)
            est = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, fold.train.y) if model_name == 'B3_TASK_STRONG' else BinaryBLDA().fit(Xtr, fold.train.y)
            coef = est.coef_ if hasattr(est, 'coef_') else est.model.coef_
            n_params = int(np.size(coef) + (np.size(est.intercept_) if hasattr(est, 'intercept_') else 1))
            predict = lambda a: est.predict_proba(_v12_erp_features(a))
        t1 = time.perf_counter()
        prob = predict(evaluate) if len(evaluate.y) else np.empty((0, K))
        out.update(y=evaluate.y, prob_raw=np.asarray(prob, float), n_params=n_params, model=None, infer_s=time.perf_counter() - t1)
    else:
        train_loader, val_loader, test_loader = _v12_loaders(model_name, fold)
        eval_loader = val_loader if mode == 'validate' else test_loader
        model = _v12_build(model_name, fold, K, hp, extra)
        with _v12_hp(hp):
            if family == 'qrg' or family == 'qrg_control':
                if mode == 'validate' and family == 'qrg':
                    model, _, stages = train_qrg_staged(model, train_loader, val_loader, K, DEVICE)
                    out['epochs'] = dict(stages)
                else:
                    model = train_qrg_fixed(model, train_loader, K, DEVICE, dict(epochs or qrg_plan))
                    out['epochs'] = dict(epochs or qrg_plan)
            elif family == 'graph_control':
                if mode == 'validate':
                    model, _, stages = train_graph_control_staged(model, train_loader, val_loader, K, DEVICE)
                    out['epochs'] = dict(stages)
                else:
                    model = train_graph_control_fixed(model, train_loader, K, DEVICE, dict(epochs))
                    out['epochs'] = dict(epochs)
            elif family == 'stage1':
                pretrainer = _v11_canonical_node_pretrainer(model, K)
                model = LatentLinearClassifier(pretrainer.encoder, K)
                model.classifier = pretrainer.classifier
                model = model.to(DEVICE)
                if mode == 'validate':
                    model, _, best = _v11_train_validation_stage(model, train_loader, val_loader, cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, DEVICE, 'pretrain', cfg.QRG_MIN_PRETRAIN_BEST_EPOCH)
                    out['epochs'] = {'pretrain': int(best)}
                else:
                    model = _v11_train_fixed_stage(model, train_loader, int(epochs['pretrain']), cfg.PRETRAIN_EPOCHS, cfg.LR_PRETRAIN, DEVICE, 'pretrain')
                    out['epochs'] = dict(epochs)
            else:
                quantum = model_name in _V12_QUANTUM_PLAIN
                max_epochs = cfg.QUANTUM_EPOCHS if quantum else cfg.PRETRAIN_EPOCHS
                lr = cfg.LR_QUANTUM if quantum else cfg.LR_PRETRAIN
                if mode == 'validate':
                    model, _, best = train_plain_model(model, train_loader, val_loader, max_epochs, lr, DEVICE)
                    out['epochs'] = {'plain': max(1, int(best))}
                else:
                    model = train_plain_fixed(model, train_loader, int(epochs['plain']), lr, DEVICE)
                    out['epochs'] = dict(epochs)
        t1 = time.perf_counter()
        y, prob, _ = predict_torch(model, eval_loader, DEVICE)
        out.update(y=y, prob_raw=np.asarray(prob, float), n_params=_v12_torch_params(model), model=model, infer_s=time.perf_counter() - t1)
        for name in ('parameter_match_target', 'parameter_match_gap'):
            out[name] = getattr(model, name, np.nan)
    out['source_priors'] = _v11_fold_source_priors(fold.train)
    out['wall_seconds'] = time.perf_counter() - t0
    return out

def _v12_plan(bundle: EEGDatasetBundle, test_subject: str, model_name: str, selected: Mapping[str, Any], qrg_plan: Mapping[str, int], extra: Mapping[str, Any]) -> Dict[str, Any]:
    cache_dir = RUN_DIR / 'model_plans_v12'
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f'{bundle.name}__test_{test_subject}__{model_name}.json'
    structure = _v11_structure_signature(selected)
    if cache.exists() and cfg.RESUME:
        payload = json.loads(cache.read_text())
        if payload.get('fingerprint') == EXPERIMENT_FINGERPRINT and payload.get('structure') == structure:
            plan = payload['plan']
            plan['calibration']['decision_threshold'] = float(plan['calibration'].get('decision_threshold') or np.nan) if plan['calibration'].get('decision_threshold') is not None else float('nan')
            return plan
    family = _v12_family(model_name, bundle.task)
    if model_name == 'B8_QRG_EEG':
        plan = {'hp': {}, 'epochs': dict(qrg_plan), 'validators': 'nested QRG selection', 'calibration': {'decision_threshold': float(selected.get('decision_threshold', np.nan)), 'prior_tau': float(selected.get('prior_tau', 0.0) or 0.0)}, 'grid': []}
    else:
        validators = _v12_validators(bundle, test_subject)
        q_override = {'QRG_DIAG_Q6': 6, 'QRG_DIAG_Q8': 8}.get(model_name)
        folds = {}
        for v in validators:
            tr, va, te = source_validation_indices(bundle, test_subject, v)
            f = prepare_fold(bundle, tr, va, te, int(q_override or selected['q']), selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
            folds[v] = _v12_transform_fold(model_name, f, f'{bundle.name}|{test_subject}')
        summaries = []
        for hp in _v12_grid(model_name, bundle.task):
            records, epoch_rows = ([], [])
            for v, f in folds.items():
                ex = dict(extra)
                if model_name.startswith('C7_SHUFFLED_TT_P'):
                    perms, _, _ = _v12_edge_permutations(f.graph_edges, f.q, cfg.V12_N_PERMUTATIONS, f'{bundle.name}|{test_subject}|{v}')
                    ex['permutation'] = perms[_v12_index(model_name) % len(perms)]
                fit = _v12_fit(model_name, f, cfg.PRIMARY_SEED, hp, 'validate', qrg_plan=qrg_plan, extra=ex)
                records.append({'subject': v, 'y': np.asarray(fit['y'], int), 'prob': fit['prob_raw'], 'source_priors': fit['source_priors']})
                epoch_rows.append(fit['epochs'])
            summaries.append({'hp': hp, 'records': records, 'epochs': epoch_rows, **_v12_summarise(records, bundle.task)})
        chosen = _v12_select_by_validation(summaries)
        if family == 'qrg_control':
            epochs = {k: int(v) for k, v in qrg_plan.items()}
            if model_name == 'EDGE_ONLY_VQC':
                epochs['pretrain'] = 0
        elif chosen['epochs'] and chosen['epochs'][0]:
            names = sorted(set().union(*[e.keys() for e in chosen['epochs']]))
            epochs = {n: max(1, int(round(np.median([e[n] for e in chosen['epochs'] if n in e])))) for n in names}
        else:
            epochs = {}
        plan = {'hp': chosen['hp'], 'epochs': epochs, 'validators': validators, 'calibration': _v12_calibrate(chosen['records'], bundle.task), 'grid': [{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst', 'joint_mean', 'joint_worst')} for s in summaries]}
    _v11_atomic_text(cache, json.dumps(_v11_json_ready({'fingerprint': EXPERIMENT_FINGERPRINT, 'structure': structure, 'model': model_name, 'plan': plan, 'outer_test_labels_used': False}), indent=2, allow_nan=False))
    return plan

def _v12_sklearn_params(est) -> int:
    final = est.steps[-1][1] if hasattr(est, 'steps') else est
    if hasattr(final, 'coefs_'):
        return int(sum((np.size(w) for w in final.coefs_)) + sum((np.size(b) for b in final.intercepts_)))
    if hasattr(final, 'dual_coef_'):
        return int(np.size(final.dual_coef_) + np.size(final.intercept_))
    return int(np.size(getattr(final, 'coef_', np.zeros(0))) + np.size(getattr(final, 'intercept_', np.zeros(0))))

def _v12_qrg_checkpoint(bundle_name: str, test_subject: str, seed: int) -> Path:
    return RUN_DIR / 'fold_artifacts' / f'{bundle_name}__test_{test_subject}' / f'B8_QRG_EEG__seed{seed}.pt'

def _v12_load_qrg(fold: PreparedFold, bundle_name: str, test_subject: str, seed: int) -> QRGEEG:
    path = _v12_qrg_checkpoint(bundle_name, test_subject, seed)
    if not path.exists():
        raise RuntimeError(f'QRG checkpoint missing (run the main phase first): {path}')
    model = build_qrg_model(fold, len(fold.class_names), 'A', layers=2)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])
    return model.to(DEVICE).eval()

@torch.no_grad()
def _v12_frozen_features(model: QRGEEG, arrays: FoldArrays) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    for start in range(0, len(arrays.y), 256):
        x = torch.as_tensor(arrays.x_latent_deep[start:start + 256], device=DEVICE)
        values.append(model.node_scaler(model.node_encoder(x)[1]).cpu().numpy())
    v = np.concatenate(values) if values else np.empty((0, model.circuit.q))
    return (v.astype(np.float64), np.asarray(arrays.edges, np.float64))

def _v12_frozen_estimators(model_name: str) -> List[Tuple[Dict[str, Any], Any]]:
    if model_name == 'RFF_FROZEN':
        return [({'D': d, 'sigma': s}, lambda d=d, s=s: make_pipeline(RBFSampler(gamma=1.0 / (2 * s * s), n_components=int(d), random_state=cfg.PRIMARY_SEED), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000))) for d in cfg.V12_RFF_D_GRID for s in cfg.V12_RFF_SIGMA_GRID]
    if model_name == 'SVM_FROZEN':
        return [({'C': c, 'gamma': g}, lambda c=c, g=g: SVC(C=float(c), gamma=g, kernel='rbf', probability=True, class_weight='balanced', random_state=cfg.PRIMARY_SEED)) for c in cfg.V12_SVM_C_GRID for g in cfg.V12_SVM_GAMMA_GRID]
    return [({'hidden': h}, lambda h=h: MLPClassifier(hidden_layer_sizes=(int(h),), activation='tanh', alpha=0.0001, max_iter=600, early_stopping=False, random_state=cfg.PRIMARY_SEED)) for h in cfg.V12_B7_FROZEN_HIDDEN_GRID]

def _v12_frozen_design(model_name, v, e, edges):
    if model_name == 'B7_FROZEN':
        values = np.column_stack([v] + [np.square(e[:, k]) * np.square(v[:, i] - v[:, j]) for k, (i, j) in enumerate(edges)])
    else:
        values = np.column_stack([v, e])
    values = np.sort(values, axis=1)
    values = values.mean(axis=1, keepdims=True).repeat(values.shape[1], axis=1)
    return values - values.mean(axis=1, keepdims=True)

def _v12_run_frozen(model_name: str, fold: PreparedFold, bundle_name: str, test_subject: str, seed: int):
    t0 = time.perf_counter()
    qrg = _v12_load_qrg(fold, bundle_name, test_subject, seed)
    vtr, etr = _v12_frozen_features(qrg, fold.train)
    vte, ete = _v12_frozen_features(qrg, fold.test)
    Xtr = _v12_frozen_design(model_name, vtr, etr, fold.graph_edges)
    Xte = _v12_frozen_design(model_name, vte, ete, fold.graph_edges)
    groups = fold.train.subject_names.astype(str)
    splitter = GroupKFold(n_splits=int(min(3, len(np.unique(groups)))))
    task = fold.task
    summaries = []
    for hp, make in _v12_frozen_estimators(model_name):
        records = []
        for tr, va in splitter.split(Xtr, fold.train.y, groups):
            est = make().fit(Xtr[tr], fold.train.y[tr])
            prob = est.predict_proba(Xtr[va])
            for s in np.unique(groups[va]):
                m = groups[va] == s
                records.append({'subject': s, 'y': fold.train.y[va][m], 'prob': prob[m], 'source_priors': _v11_fold_source_priors(fold.train)})
        summaries.append({'hp': hp, 'records': records, **_v12_summarise(records, task)})
    chosen = _v12_select_by_validation(summaries)
    make = dict(((json.dumps(h, sort_keys=True), m) for h, m in _v12_frozen_estimators(model_name)))[json.dumps(chosen['hp'], sort_keys=True)]
    est = make().fit(Xtr, fold.train.y)
    t1 = time.perf_counter()
    prob = est.predict_proba(Xte)
    n_params = _v12_sklearn_params(est)
    return {'y': fold.test.y, 'prob_raw': np.asarray(prob, float), 'n_params': n_params, 'model': None, 'infer_s': time.perf_counter() - t1, 'epochs': {}, 'source_priors': _v11_fold_source_priors(fold.train), 'wall_seconds': time.perf_counter() - t0, 'plan': {'hp': chosen['hp'], 'epochs': {}, 'validators': 'GroupKFold(source)', 'calibration': _v12_calibrate(chosen['records'], task), 'grid': [{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst')} for s in summaries]}}

@torch.no_grad()
def _v12_run_test_only(model_name: str, fold: PreparedFold, bundle_name: str, test_subject: str, seed: int, qrg_calibration: Mapping[str, float]):
    t0 = time.perf_counter()
    qrg = _v12_load_qrg(fold, bundle_name, test_subject, seed)
    perms, excluded, n_auto = _v12_edge_permutations(fold.graph_edges, fold.q, cfg.V12_N_PERMUTATIONS, f'{bundle_name}|{test_subject}|final')
    perm = torch.as_tensor(perms[_v12_index(model_name) % len(perms)], dtype=torch.long, device=DEVICE)
    _, _, test_loader = make_loaders(fold, 256)
    ys, probs = ([], [])
    for batch in test_loader:
        batch = move_batch(batch, DEVICE)
        x = batch['x_latent']
        x = x.mean(dim=-1, keepdim=True).expand_as(x)
        batch['x_latent'] = x - x.mean(dim=-1, keepdim=True)
        e = batch['edges']
        e = e.mean(dim=1, keepdim=True).expand_as(e)
        batch['edges'] = F.layer_norm(e, (e.shape[1],))
        d, v = qrg.node_encoder(batch['x_latent'])
        m = qrg.circuit(qrg.node_scaler(v), batch['edges'].index_select(1, perm), d)
        probs.append(torch.softmax(qrg.readout(m), 1).cpu().numpy())
        ys.append(batch['y'].cpu().numpy())
    return {'y': np.concatenate(ys), 'prob_raw': np.concatenate(probs), 'n_params': _v12_torch_params(qrg), 'model': None, 'infer_s': time.perf_counter() - t0, 'epochs': {}, 'source_priors': _v11_fold_source_priors(fold.train), 'wall_seconds': time.perf_counter() - t0, 'automorphisms_excluded': excluded, 'n_automorphisms': n_auto, 'plan': {'hp': {}, 'epochs': {}, 'validators': 'QRG-EEG decision rule', 'calibration': dict(qrg_calibration), 'grid': []}}

def _v12_row(bundle, model_name, seed, test_subject, selected, fold_used, fit, plan, metrics, class_f1, model_obj):
    epochs_total = int(sum((int(v) for v in (plan.get('epochs') or {}).values()))) if plan.get('epochs') else np.nan
    is_q = isinstance(model_obj, QRGEEG)
    return {'dataset': bundle.name, 'model': _v11_reported_model(bundle, model_name), 'seed': int(seed), 'test_subject': str(test_subject), 'variant': 'A' if is_q else 'n/a', 'q': int(fold_used.q), 'layers': int(model_obj.circuit.layers) if is_q else int(selected['layers']), 'graph_multiplier': float(selected['graph_multiplier']), 'projection_mode': str(selected['projection_mode']), 'edge_mode': str(selected['edge_mode']), 'val_subject': ';'.join(map(str, plan['validators'])) if isinstance(plan['validators'], list) else str(plan['validators']), 'training_recipe': 'qrg_objective' if isinstance(model_obj, (QRGEEG, SameGraphGAT, MatchedClassicalInteraction)) or _v12_family(model_name, bundle.task) == 'stage1' else 'subject_balanced_ce', 'n_edges': int(len(model_obj.circuit.edges)) if is_q else int(len(fold_used.graph_edges)), 'best_epoch': epochs_total, 'decision_threshold': plan['calibration'].get('decision_threshold', np.nan), 'prior_tau': float(plan['calibration'].get('prior_tau', 0.0) or 0.0), 'trainable_params': int(fit['n_params']), 'quantum_params': int(sum((p.numel() for p in model_obj.circuit.parameters()))) if is_q else 0, 'two_qubit_gates': int(model_obj.two_qubit_gates_per_trial()) if is_q else 0, 'circuit_depth': int(model_obj.circuit.layers * (1 + len(model_obj.circuit.edges) + 2)) if is_q else 0, 'circuit_evaluations_per_epoch': int(len(fold_used.train.y)) if is_q else 0, 'parameter_match_target': fit.get('parameter_match_target', np.nan), 'parameter_match_gap': fit.get('parameter_match_gap', np.nan), 'wall_seconds': float(fit['wall_seconds']), 'class_f1': json.dumps([float(x) for x in class_f1]), 'macro_f1': float(metrics['macro_f1']), 'accuracy': float(metrics['accuracy']), 'balanced_accuracy': float(metrics['balanced_accuracy']), 'kappa': float(metrics['kappa']), 'auroc': float(metrics['auroc']), 'target_auprc': float(metrics['target_auprc'])}

def _v12_commit(bundle, model_name, seed, test_subject, selected, fold_used, fit, plan, phase):
    calib = plan['calibration']
    prob_dec, threshold = _v12_apply_decision(fit['prob_raw'], fold_used.task, calib, fit['source_priors'])
    metrics = multiclass_metrics(fit['y'], prob_dec, fold_used.task, decision_threshold=threshold)
    class_f1 = per_class_f1(fit['y'], prob_dec, decision_threshold=threshold)
    uncal = multiclass_metrics(fit['y'], fit['prob_raw'], fold_used.task)
    model_obj = fit.get('model') if fit.get('model') is not None else fit.get('model_for_row')
    row = _v12_row(bundle, model_name, seed, test_subject, selected, fold_used, fit, plan, metrics, class_f1, model_obj)
    fold_dir = RUN_DIR / 'fold_artifacts' / f'{bundle.name}__test_{test_subject}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if cfg.SAVE_CHECKPOINTS and isinstance(fit.get('model'), nn.Module):
        checkpoint = _v11_checkpoint_atomic(fit['model'], fold_dir / f"{row['model']}__seed{seed}.pt", {'row': row, 'plan': plan, 'selected': selected, 'graph_edges': list(map(list, fold_used.graph_edges)), 'experiment_fingerprint': EXPERIMENT_FINGERPRINT})
    store.add(row, {'y': np.asarray(fit['y'], np.int64), 'prob': np.asarray(prob_dec, np.float64), 'prob_raw': np.asarray(fit['prob_raw'], np.float64), 'source_priors': np.asarray(fit['source_priors']), 'decision_threshold': np.asarray([np.nan if threshold is None else threshold]), 'prior_tau': np.asarray([float(calib.get('prior_tau', 0.0) or 0.0)]), 'features': np.empty((0,), np.float32)}, internal=_v11_json_ready({'phase': phase, 'plan': plan, 'selected_qrg': selected, 'uncalibrated_metrics': uncal, 'infer_seconds': fit.get('infer_s'), 'n_test': int(len(fit['y'])), 'automorphisms_excluded': fit.get('automorphisms_excluded'), 'graph_edges': [list(map(int, e)) for e in fold_used.graph_edges]}), checkpoint_path=checkpoint)
    return row

def run_outer_fold_v12(bundle: EEGDatasetBundle, test_subject: str, models: Sequence[str], phase: str) -> None:
    todo = [m for m in models if not all((store.completed({'dataset': bundle.name, 'model': _v11_reported_model(bundle, m), 'seed': s, 'test_subject': str(test_subject)}) for s in cfg.SEEDS))]
    if not todo:
        return
    selected, qrg_plan = select_qrg_source_only(bundle, test_subject)
    todo = [m for m in todo if m != f"QRG_DIAG_Q{int(selected['q'])}"]
    src, empty_val, test_idx = split_indices(bundle, test_subject, None)
    base_fold = prepare_fold(bundle, src, empty_val, test_idx, selected['q'], selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
    folds_q = {}
    qrg_calibration = None
    for model_name in todo:
        family = _v12_family(model_name, bundle.task)
        try:
            if model_name in ('QRG_DIAG_Q6', 'QRG_DIAG_Q8'):
                qq = 6 if model_name.endswith('Q6') else 8
                if qq not in folds_q:
                    folds_q[qq] = prepare_fold(bundle, src, empty_val, test_idx, qq, selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
                fold = folds_q[qq]
            else:
                fold = _v12_transform_fold(model_name, base_fold, f'{bundle.name}|{test_subject}')
            if family in ('frozen', 'test_only'):
                plan = None
            else:
                extra = {}
                plan = _v12_plan(bundle, test_subject, model_name, selected, qrg_plan, extra)
            for seed in cfg.SEEDS:
                key = {'dataset': bundle.name, 'model': _v11_reported_model(bundle, model_name), 'seed': seed, 'test_subject': str(test_subject)}
                if store.completed(key):
                    continue
                print('V12 OUTER:', phase, bundle.name, test_subject, model_name, 'seed', seed)
                if family == 'frozen':
                    fit = _v12_run_frozen(model_name, fold, bundle.name, test_subject, seed)
                    plan_used = fit['plan']
                    fit['model_for_row'] = nn.Identity()
                elif family == 'test_only':
                    if qrg_calibration is None:
                        qrg_calibration = _v12_plan(bundle, test_subject, 'B8_QRG_EEG', selected, qrg_plan, {})['calibration']
                    fit = _v12_run_test_only(model_name, fold, bundle.name, test_subject, seed, qrg_calibration)
                    plan_used = fit['plan']
                    fit['model_for_row'] = _v12_load_qrg(fold, bundle.name, test_subject, seed)
                else:
                    extra = {}
                    if model_name.startswith('C7_SHUFFLED_TT_P'):
                        perms, excluded, _ = _v12_edge_permutations(fold.graph_edges, fold.q, cfg.V12_N_PERMUTATIONS, f'{bundle.name}|{test_subject}|final')
                        extra['permutation'] = perms[_v12_index(model_name) % len(perms)]
                    fit = _v12_fit(model_name, fold, seed, plan['hp'], 'final', epochs=plan['epochs'], qrg_plan=qrg_plan, extra=extra)
                    if model_name.startswith('C7_SHUFFLED_TT_P'):
                        fit['automorphisms_excluded'] = excluded
                    plan_used = plan
                    if model_name in ('B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE'):
                        gap = float(fit.get('parameter_match_gap', np.nan))
                        if not np.isfinite(gap) or gap > float(cfg.PARAM_MATCH_TOLERANCE):
                            raise RuntimeError(f'Parameter matching failed for {model_name}: gap={gap}')
                    if fit.get('model') is None:
                        fit['model_for_row'] = nn.Identity()
                row = _v12_commit(bundle, model_name, seed, test_subject, selected, fold, fit, plan_used, phase)
                print({k: row[k] for k in ('dataset', 'model', 'seed', 'test_subject', 'macro_f1', 'auroc')})
                gc.collect()
        except Exception as error:
            with open(RUN_DIR / 'errors.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps({'dataset': bundle.name, 'model': model_name, 'test_subject': str(test_subject), 'phase': phase, 'error': repr(error)}) + '\n')
            print('FAILED:', bundle.name, test_subject, model_name, repr(error))

def _v12_phase_models() -> Dict[str, List[str]]:
    return {'main': list(cfg.V12_MAIN_MODELS), 'rq2': list(cfg.V12_RQ2_MODELS) + [f'C6_RANDOM_TOPOLOGY_G{k}' for k in range(int(cfg.V12_N_RANDOM_TOPOLOGIES))] + [f'C7_SHUFFLED_TT_P{k}' for k in range(int(cfg.V12_N_PERMUTATIONS))] + [f'C8_SHUFFLED_TEST_P{k}' for k in range(int(cfg.V12_N_PERMUTATIONS))], 'surrogates': list(cfg.V12_SURROGATE_MODELS), 'transductive': ['EA_EEGNET'], 'diagnostics': ['QRG_DIAG_NOMIX', 'QRG_DIAG_RYONLY', 'QRG_DIAG_Q6', 'QRG_DIAG_Q8', 'QRG_DIAG_L4']}

def _v12_compatible(bundle: EEGDatasetBundle, model_name: str) -> bool:
    if model_name == 'PLV_GAT':
        return bundle.name == 'UCL_ALS_MI'
    if model_name == 'B9_ERPCOV_MDM':
        return bundle.task == 'p300'
    return True

def v12_diagnostic_subjects(bundle: EEGDatasetBundle) -> List[str]:
    given = dict(cfg.V12_DIAG_SUBJECTS).get(bundle.name)
    subjects = outer_subjects(bundle, cfg.MAX_OUTER_FOLDS, cfg.PRIMARY_SEED)
    return [s for s in subjects if s in given] if given else list(subjects[:dict(cfg.DIAG_SUBJECTS_PER_DATASET).get(bundle.name, 2)])

def run_full_submission_v12(phases: Sequence[str]=('main', 'rq2', 'surrogates', 'transductive', 'diagnostics')) -> None:
    models = _v12_phase_models()
    for phase in phases:
        print(f'\n===== V12 phase: {phase} =====')
        for bundle in loaded_datasets.values():
            subjects = v12_diagnostic_subjects(bundle) if phase == 'diagnostics' else outer_subjects(bundle, cfg.MAX_OUTER_FOLDS, cfg.PRIMARY_SEED)
            for subject in subjects:
                run_outer_fold_v12(bundle, subject, [m for m in models[phase] if _v12_compatible(bundle, m)], phase)

class DiagnosticQRG(QRGEEG):

    def forward(self, batch, return_features=False):
        values = dict(batch)
        x = values['x_latent']
        x = x.mean(dim=-1, keepdim=True).expand_as(x)
        values['x_latent'] = x - x.mean(dim=-1, keepdim=True)
        e = values['edges']
        e = e.mean(dim=1, keepdim=True).expand_as(e)
        values['edges'] = F.layer_norm(e, (e.shape[1],))
        return super().forward(values, return_features=return_features)
assert 'run_outer_fold_v12' in globals(), 'Source-selection core is not initialized.'

def _v13_set_determinism() -> None:
    if not bool(cfg.V13_DETERMINISTIC):
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
_v13_set_determinism()
_V13_STRUCT: Dict[str, Any] = {'layers': 2, 'pool': 'avg'}

def _v13_activate(selected: Mapping[str, Any]) -> None:
    _V13_STRUCT['layers'] = int(selected.get('layers', 2))
    _V13_STRUCT['pool'] = str(selected.get('pool', 'avg'))

def _v13_layers() -> int:
    return int(_V13_STRUCT['layers'])

class V13GlobalMaxPool1d(nn.Module):

    def forward(self, x):
        return x.amax(dim=-1, keepdim=True)
if '_V13_ENCODER_INIT' not in globals():
    _V13_ENCODER_INIT = SharedTemporalNodeEncoder.__init__

def _v13_encoder_init(self, *args, **kwargs):
    _V13_ENCODER_INIT(self, *args, **kwargs)
    self.pool_mode = str(_V13_STRUCT['pool'])
    if self.pool_mode == 'max':
        for parent in self.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.AdaptiveAvgPool1d) and child.output_size in (1, (1,)):
                    setattr(parent, name, V13GlobalMaxPool1d())
SharedTemporalNodeEncoder.__init__ = _v13_encoder_init
if '_V13_STRUCTURE_SIGNATURE' not in globals():
    _V13_STRUCTURE_SIGNATURE = _v11_structure_signature

def _v11_structure_signature(selected: Mapping[str, Any]) -> Dict[str, Any]:
    out = _V13_STRUCTURE_SIGNATURE(selected)
    out['pool'] = str(selected.get('pool', 'avg'))
    return out

@contextmanager
def _v13_proxy_budget():
    names = ('PRETRAIN_EPOCHS', 'QUANTUM_EPOCHS', 'FINETUNE_EPOCHS', 'PATIENCE')
    old = {n: getattr(cfg, n) for n in names}
    try:
        cfg.PRETRAIN_EPOCHS = max(cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, int(cfg.QRG_PROXY_PRETRAIN_EPOCHS))
        cfg.QUANTUM_EPOCHS = max(cfg.QRG_MIN_STAGE2_BEST_EPOCH, int(cfg.QRG_PROXY_QUANTUM_EPOCHS))
        cfg.FINETUNE_EPOCHS = max(cfg.QRG_MIN_FINETUNE_BEST_EPOCH, int(cfg.QRG_PROXY_FINETUNE_EPOCHS))
        cfg.PATIENCE = int(cfg.QRG_PROXY_PATIENCE)
        yield
    finally:
        for n, v in old.items():
            setattr(cfg, n, v)

def _v13_proxy_key(m: Mapping[str, float], min_class: float, best_macro: float, task: str, tie: Tuple) -> Tuple:
    return (_finite_metric(m['macro_f1']) >= best_macro - float(cfg.QRG_MACRO_TOLERANCE), _finite_metric(_v11_joint_score(m, task)), _finite_metric(min_class), _finite_metric(m['macro_f1']), _finite_metric(_v11_secondary_score(m, task))) + tuple(tie)

def _v13_full_summary(records: Sequence[Mapping[str, Any]], task: str) -> Dict[str, float]:
    metrics = [multiclass_metrics(np.asarray(r['y'], int), np.asarray(r['prob'], float), task) for r in records]
    macro = np.array([m['macro_f1'] for m in metrics], float)
    joint = np.array([_v11_joint_score(m, task) for m in metrics], float)
    mins = [float(np.min(per_class_f1(np.asarray(r['y'], int), np.asarray(r['prob'], float)))) for r in records]
    return {'macro_f1_mean': float(np.mean(macro)), 'macro_f1_worst': float(np.min(macro)), 'macro_f1_std': float(np.std(macro)), 'joint_mean': float(np.mean(joint)), 'joint_worst': float(np.min(joint)), 'min_class_f1_mean': float(np.mean(mins)), 'secondary_mean': float(_v11_secondary_score(_v11_average_metrics(metrics), task))}

def _v13_full_key(s: Mapping[str, float], tie: Tuple) -> Tuple:
    return (_finite_metric(s['joint_worst']), _finite_metric(s['joint_mean']), _finite_metric(s['macro_f1_worst']), _finite_metric(s['min_class_f1_mean']), _finite_metric(s['macro_f1_mean']), _finite_metric(s['secondary_mean']), -_finite_metric(s['macro_f1_std'], default=1000000000.0)) + tuple(tie)

def _v13_choose(summaries: Sequence[Dict[str, Any]], tie_of) -> Dict[str, Any]:
    best = max((_finite_metric(s['macro_f1_mean']) for s in summaries))
    eligible = [s for s in summaries if _finite_metric(s['macro_f1_mean']) >= best - float(cfg.QRG_MACRO_TOLERANCE) - 1e-12]
    return max(eligible, key=lambda s: _v13_full_key(s, tie_of(s)))

def _v13_qrg_candidates(bundle: EEGDatasetBundle) -> List[Tuple[int, str, str, int, float, str]]:
    pools = tuple(cfg.V13_P300_POOL_GRID) if bundle.task == 'p300' else ('avg',)
    return [(int(q), str(p), str(e), int(L), float(g), str(pool)) for p, e in _v11_representations(bundle) for q in (6, 8) if q <= int(bundle.X.shape[1]) for L in cfg.V13_L_GRID for g in cfg.V13_G_GRID for pool in pools]

def _v13_qrg_tie(key) -> Tuple:
    q, _, _, L, g, pool = key
    return (-int(q), -int(L), -abs(float(g) - 1.5), pool == 'avg')

def _v13_qrg_validation(bundle, test_subject, val_subject, key, seed):
    q, proj, edge, L, g, pool = key
    tr, va, te = source_validation_indices(bundle, test_subject, val_subject)
    _v13_activate({'layers': L, 'pool': pool})
    fold = prepare_fold(bundle, tr, va, te, q, g, proj, edge)
    out = run_single_model('B8_QRG_EEG', fold, seed, 'A', L, evaluate_test=False)
    return {'y': np.asarray(out['val_y'], int), 'prob': np.asarray(out['val_prob_raw'], float), 'source_priors': np.asarray(out['source_priors'], float), 'stage_plan': {k: int(v) for k, v in out['stage_plan'].items()}}

def select_qrg_source_only_v13(bundle: EEGDatasetBundle, test_subject: str):
    cache = RUN_DIR / f'source_selection_v13__{bundle.name}__test_{test_subject}.json'
    if cache.exists() and cfg.RESUME:
        payload = json.loads(cache.read_text())
        if payload.get('experiment_fingerprint') != EXPERIMENT_FINGERPRINT:
            raise RuntimeError(f'Stale V13 selection cache (configuration changed): {cache}')
        selected = payload['selected']
        selected['decision_threshold'] = np.nan if selected.get('decision_threshold') is None else float(selected['decision_threshold'])
        _v13_activate(selected)
        return (selected, {k: int(v) for k, v in payload['stage_plan'].items()})
    validators = _v12_validators(bundle, test_subject)
    candidates = _v13_qrg_candidates(bundle)
    proxy = []
    with _v13_proxy_budget():
        for key in candidates:
            rec = _v13_qrg_validation(bundle, test_subject, validators[0], key, cfg.PRIMARY_SEED)
            m = multiclass_metrics(rec['y'], rec['prob'], bundle.task)
            proxy.append({'key': key, 'metrics': m, 'min_class_f1': float(np.min(per_class_f1(rec['y'], rec['prob'])))})
    best_macro = max((_finite_metric(r['metrics']['macro_f1']) for r in proxy))
    proxy.sort(key=lambda r: _v13_proxy_key(r['metrics'], r['min_class_f1'], best_macro, bundle.task, _v13_qrg_tie(r['key'])), reverse=True)
    shortlist = [r['key'] for r in proxy[:max(1, int(cfg.QRG_PROXY_SHORTLIST))]]
    summaries = []
    for key in shortlist:
        records = [_v13_qrg_validation(bundle, test_subject, v, key, cfg.PRIMARY_SEED) for v in validators]
        summaries.append({'key': key, 'records': records, **_v13_full_summary(records, bundle.task)})
    chosen = _v13_choose(summaries, lambda s: _v13_qrg_tie(s['key']))
    q, proj, edge, L, g, pool = chosen['key']
    stage_plan = {}
    floors = {'pretrain': cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, 'stage2': cfg.QRG_MIN_STAGE2_BEST_EPOCH, 'finetune': cfg.QRG_MIN_FINETUNE_BEST_EPOCH}
    for stage, floor in floors.items():
        stage_plan[stage] = max(int(floor), int(round(np.median([r['stage_plan'][stage] for r in chosen['records']]))))
    selected = {'variant': 'A', 'q': int(q), 'layers': int(L), 'graph_multiplier': float(g), 'pool': str(pool), 'projection_mode': str(proj), 'edge_mode': str(edge), 'training_recipe': str(getattr(cfg, 'TRAINING_RECIPE', ''))}
    calib = _v12_calibrate(chosen['records'], bundle.task)
    selected['decision_threshold'] = float(calib['decision_threshold']) if bundle.task == 'p300' else np.nan
    selected['prior_tau'] = float(calib['prior_tau'])
    payload = {'experiment_fingerprint': EXPERIMENT_FINGERPRINT, 'selected': selected, 'stage_plan': stage_plan, 'validators': [str(v) for v in validators], 'outer_test_labels_used_for_selection': False, 'proxy_ranking': [{'key': list(r['key']), **{k: r['metrics'][k] for k in ('macro_f1', 'balanced_accuracy', 'auroc')}, 'min_class_f1': r['min_class_f1']} for r in proxy], 'full_validation': [{'key': list(s['key']), **{k: v for k, v in s.items() if k not in ('key', 'records')}} for s in summaries]}
    _v11_atomic_text(cache, json.dumps(_v11_json_ready(payload), indent=2, allow_nan=False))
    _v13_activate(selected)
    return (selected, stage_plan)
if bool(cfg.V13_QRG_SEARCH):
    if '_V13_V11_SELECT' not in globals():
        _V13_V11_SELECT = select_qrg_source_only
    select_qrg_source_only = select_qrg_source_only_v13

def _v13_search_k(bundle: EEGDatasetBundle) -> int:
    return int(cfg.V13_SEARCH_K) if cfg.V13_SEARCH_K else len(_v13_qrg_candidates(bundle))
_V13_GRAPH_CONTROLS = {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE', 'RFF_E2E'}
_V13_LINEAR = {'B1_RIEMANNIAN_TS_LR'}

def _v13_space(model_name: str, task: str):
    loguni = lambda r, a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    if model_name in _V12_NEURAL_GRID_MODELS:
        quantum = model_name in _V12_QUANTUM_PLAIN
        default = {'lr': float(cfg.LR_QUANTUM if quantum else cfg.LR_PRETRAIN), 'dropout': float(cfg.DROPOUT), 'weight_decay': float(cfg.WEIGHT_DECAY)}
        return (default, lambda r: {'lr': loguni(r, 0.0001, 0.005), 'dropout': float(r.uniform(0.1, 0.6)), 'weight_decay': loguni(r, 1e-05, 0.01)})
    if model_name in _V13_GRAPH_CONTROLS:
        default = {'lr_scale': 1.0, 'weight_decay': float(cfg.WEIGHT_DECAY)}
        if model_name == 'RFF_E2E':
            default['sigma'] = math.pi / 2
            return (default, lambda r: {'lr_scale': loguni(r, 0.25, 4.0), 'weight_decay': loguni(r, 1e-05, 0.01), 'sigma': loguni(r, math.pi / 8, 2 * math.pi)})
        return (default, lambda r: {'lr_scale': loguni(r, 0.25, 4.0), 'weight_decay': loguni(r, 1e-05, 0.01)})
    if model_name in _V13_LINEAR or (model_name == 'B8_XDAWNCOV_FBCSP' and task == 'p300'):
        return ({'C': 1.0}, lambda r: {'C': loguni(r, 0.001, 100.0)})
    if model_name == 'B8_XDAWNCOV_FBCSP' and task == 'mi':
        default = {'csp': int(cfg.V12_B8_CSP_FILTERS), 'select': int(cfg.V12_B8_MI_SELECT)}
        return (default, lambda r: {'csp': int(r.choice([2, 4, 6])), 'select': int(r.choice([4, 6, 8, 12, 16]))})
    if model_name == 'B8_XDAWNCOV_FBCSP' and task == 'p300':
        return ({'C': 1.0}, lambda r: {'C': loguni(r, 0.001, 100.0)})
    return None

def _v13_configs(model_name: str, bundle: EEGDatasetBundle, test_subject: str) -> List[Dict[str, Any]]:
    space = _v13_space(model_name, bundle.task)
    if space is None or not bool(cfg.V13_EQUAL_BUDGET):
        return _v12_grid(model_name, bundle.task)
    default, sample = space
    rng = np.random.default_rng(int(hashlib.sha256(f'{model_name}|{bundle.name}|{test_subject}'.encode()).hexdigest()[:8], 16))
    configs, seen = ([dict(default)], {json.dumps(default, sort_keys=True)})
    if model_name == 'B8_XDAWNCOV_FBCSP' and bundle.task == 'mi':
        candidates = [{'csp': c, 'select': k} for c in (2, 4, 6) for k in (4, 6, 8, 12, 16)]
        rng.shuffle(candidates)
        for candidate in candidates:
            token = json.dumps(candidate, sort_keys=True)
            if token not in seen:
                seen.add(token)
                configs.append(candidate)
        return configs[:_v13_search_k(bundle)]
    _fixed_sampling_attempts = 0
    while len(configs) < _v13_search_k(bundle):
        _fixed_sampling_attempts += 1
        if _fixed_sampling_attempts > max(100000, 10000 * _v13_search_k(bundle)):
            raise RuntimeError('Hyperparameter rejection sampling exhausted its safety bound; no search candidates were removed or accepted with changed rules.')
        c = sample(rng)
        k = json.dumps(c, sort_keys=True)
        if k not in seen:
            seen.add(k)
            configs.append(c)
    return configs

@contextmanager
def _v13_cfg_overrides(hp: Mapping[str, Any]):
    names = ['WEIGHT_DECAY', 'QRG_LR_STAGE2_CIRCUIT', 'QRG_LR_STAGE2_READOUT', 'QRG_LR_FINETUNE_BACKBONE', 'QRG_LR_FINETUNE_CIRCUIT', 'QRG_LR_FINETUNE_READOUT', 'V12_B8_CSP_FILTERS', 'V12_B8_MI_SELECT', 'V12_RFF_E2E_FEATURES']
    old = {n: getattr(cfg, n) for n in names if hasattr(cfg, n)}
    try:
        if 'weight_decay' in hp:
            cfg.WEIGHT_DECAY = float(hp['weight_decay'])
        if 'lr_scale' in hp:
            for n in names[1:6]:
                setattr(cfg, n, float(old[n]) * float(hp['lr_scale']))
        if 'csp' in hp:
            cfg.V12_B8_CSP_FILTERS = int(hp['csp'])
        if 'select' in hp:
            cfg.V12_B8_MI_SELECT = int(hp['select'])
        yield
    finally:
        for n, v in old.items():
            setattr(cfg, n, v)
if '_V13_V12_FIT' not in globals():
    _V13_V12_FIT = _v12_fit

def _v12_fit(model_name, fold, seed, hp, mode, epochs=None, qrg_plan=None, extra=None):
    with _v13_cfg_overrides(hp or {}):
        return _V13_V12_FIT(model_name, fold, seed, hp, mode, epochs=epochs, qrg_plan=qrg_plan, extra=extra)
if '_V13_V12_PLAN' not in globals():
    _V13_V12_PLAN = _v12_plan

def _v12_plan(bundle, test_subject, model_name, selected, qrg_plan, extra):
    configs = _v13_configs(model_name, bundle, test_subject)
    family = _v12_family(model_name, bundle.task)
    if model_name == 'B8_QRG_EEG' or family == 'qrg_control' or len(configs) <= 1 or (not bool(cfg.V13_EQUAL_BUDGET)):
        return _V13_V12_PLAN(bundle, test_subject, model_name, selected, qrg_plan, extra)
    cache = RUN_DIR / 'model_plans_v13' / f'{bundle.name}__test_{test_subject}__{model_name}.json'
    cache.parent.mkdir(exist_ok=True)
    structure = _v11_structure_signature(selected)
    if cache.exists() and cfg.RESUME:
        payload = json.loads(cache.read_text())
        if payload.get('fingerprint') == EXPERIMENT_FINGERPRINT and payload.get('structure') == structure:
            plan = payload['plan']
            t = plan['calibration'].get('decision_threshold')
            plan['calibration']['decision_threshold'] = float('nan') if t is None else float(t)
            return plan
    validators = _v12_validators(bundle, test_subject)
    folds = {}
    for v in validators:
        tr, va, te = source_validation_indices(bundle, test_subject, v)
        f = prepare_fold(bundle, tr, va, te, int(selected['q']), selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
        folds[v] = _v12_transform_fold(model_name, f, f'{bundle.name}|{test_subject}')
    proxy = []
    with _v13_proxy_budget():
        for i, hp in enumerate(configs):
            fit = _v12_fit(model_name, folds[validators[0]], cfg.PRIMARY_SEED, hp, 'validate', qrg_plan=qrg_plan, extra=extra)
            m = multiclass_metrics(np.asarray(fit['y'], int), fit['prob_raw'], bundle.task)
            proxy.append({'i': i, 'hp': hp, 'metrics': m, 'min_class_f1': float(np.min(per_class_f1(np.asarray(fit['y'], int), fit['prob_raw'])))})
    best_macro = max((_finite_metric(r['metrics']['macro_f1']) for r in proxy))
    proxy.sort(key=lambda r: _v13_proxy_key(r['metrics'], r['min_class_f1'], best_macro, bundle.task, (-r['i'],)), reverse=True)
    summaries = []
    for r in proxy[:max(1, int(cfg.QRG_PROXY_SHORTLIST))]:
        records, epoch_rows = ([], [])
        for v in validators:
            fit = _v12_fit(model_name, folds[v], cfg.PRIMARY_SEED, r['hp'], 'validate', qrg_plan=qrg_plan, extra=extra)
            records.append({'subject': v, 'y': np.asarray(fit['y'], int), 'prob': fit['prob_raw'], 'source_priors': fit['source_priors']})
            epoch_rows.append(fit['epochs'])
        summaries.append({'i': r['i'], 'hp': r['hp'], 'records': records, 'epochs': epoch_rows, **_v13_full_summary(records, bundle.task)})
    chosen = _v13_choose(summaries, lambda s: (-s['i'],))
    if chosen['epochs'] and chosen['epochs'][0]:
        names = sorted(set().union(*[e.keys() for e in chosen['epochs']]))
        epochs = {n: max(1, int(round(np.median([e[n] for e in chosen['epochs'] if n in e])))) for n in names}
    else:
        epochs = {}
    plan = {'hp': chosen['hp'], 'epochs': epochs, 'validators': validators, 'calibration': _v12_calibrate(chosen['records'], bundle.task), 'search_k': len(configs), 'proxy': [{'hp': r['hp'], 'macro_f1': r['metrics']['macro_f1'], 'auroc': r['metrics']['auroc']} for r in proxy], 'grid': [{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst', 'joint_mean', 'joint_worst')} for s in summaries]}
    _v11_atomic_text(cache, json.dumps(_v11_json_ready({'fingerprint': EXPERIMENT_FINGERPRINT, 'structure': structure, 'model': model_name, 'plan': plan, 'outer_test_labels_used': False}), indent=2, allow_nan=False))
    return plan
if '_V13_V12_BUILD' not in globals():
    _V13_V12_BUILD = _v12_build

def _v12_build(model_name, fold, n_classes, hp, extra):
    L = _v13_layers()
    if model_name in ('B8_QRG_EEG', 'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'EDGE_ONLY_VQC'):
        with _v12_hp(hp):
            return _v12_apply_dropout(build_torch_model(model_name, fold, n_classes, 'A', L))
    if model_name == 'B12_TRIG_SURROGATE':
        target = _v11_total_architectural_parameters(build_qrg_model(fold, n_classes, 'A', layers=L))
        new_encoder = lambda: SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
        options = []
        for order in (2, 1, 0):
            cand = TrigSurrogate(new_encoder(), fold.graph_edges, n_classes, max_order=order)
            actual = _v11_total_architectural_parameters(cand)
            options.append((abs(actual - target) / max(1, target), order, actual, cand))
        within = [o for o in options if o[0] <= float(cfg.PARAM_MATCH_TOLERANCE)]
        gap, order, actual, model = within[0] if within else min(options, key=lambda o: o[0])
        model.parameter_match_target, model.parameter_match_actual, model.parameter_match_gap = (int(target), int(actual), float(gap))
        return model
    if model_name in ('C3_NODE_ONLY_EQ', 'C5_FIXED_SAME_E', 'QRG_DIAG_NOMIX', 'QRG_DIAG_RYONLY') or model_name.startswith('C7_SHUFFLED_TT_P'):
        ablation = {'C3_NODE_ONLY_EQ': 'node_only_eq', 'C5_FIXED_SAME_E': 'fixed_same_e', 'QRG_DIAG_NOMIX': 'mixer_none', 'QRG_DIAG_RYONLY': 'mixer_ry_only'}.get(model_name, 'shuffled_perm')
        return V12ControlledQRG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, L, ablation, permutation=(extra or {}).get('permutation'))
    if model_name.startswith('C6_RANDOM_TOPOLOGY_G'):
        return V12ControlledQRG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, L, 'graph_full')
    if model_name in ('QRG_DIAG_Q6', 'QRG_DIAG_Q8'):
        return DiagnosticQRG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, L, 'A')
    if model_name in ('C1_STAGE1_HEAD',):
        return build_qrg_model(fold, n_classes, 'A', layers=L)
    if model_name in ('QRG_DIAG_L2', 'QRG_DIAG_L4'):
        return DiagnosticQRG(fold.q, fold.graph_edges, fold.readout_edges, n_classes, 2 if model_name.endswith('L2') else 4, 'A')
    return _V13_V12_BUILD(model_name, fold, n_classes, hp, extra)

def _v12_family(model_name: str, task: str, _base=_v12_family) -> str:
    return 'qrg_control' if model_name == 'QRG_DIAG_L2' else _base(model_name, task)
if '_V13_LOAD_QRG' not in globals():
    _V13_LOAD_QRG = _v12_load_qrg

def _v12_load_qrg(fold, bundle_name, test_subject, seed):
    payload = torch.load(_v12_qrg_checkpoint(bundle_name, test_subject, seed), map_location='cpu', weights_only=False)
    _v13_activate(payload.get('selected', {}))
    model = build_qrg_model(fold, len(fold.class_names), 'A', layers=_v13_layers())
    model.load_state_dict(payload['state_dict'])
    return model.to(DEVICE).eval()
if '_V13_RUN_OUTER' not in globals():
    _V13_RUN_OUTER = run_outer_fold_v12

def run_outer_fold_v12(bundle, test_subject, models, phase):
    selected, _ = select_qrg_source_only(bundle, test_subject)
    same = {f"QRG_DIAG_Q{int(selected['q'])}", f"QRG_DIAG_L{int(selected.get('layers', 2))}"}
    _v13_activate(selected)
    return _V13_RUN_OUTER(bundle, test_subject, [m for m in models if m not in same], phase)
if '_V13_PHASES_BASE' not in globals():
    _V13_PHASES_BASE = _v12_phase_models

def _v12_phase_models():
    out = _V13_PHASES_BASE()
    out['diagnostics'] = ['QRG_DIAG_NOMIX', 'QRG_DIAG_RYONLY', 'QRG_DIAG_Q6', 'QRG_DIAG_Q8', 'QRG_DIAG_L2', 'QRG_DIAG_L4']
    return out
_FIXED_DATA_IDENTITIES = {}
_FIXED_GUARDS = []
_FIXED_FIT_EVIDENCE = {}
_FIXED_LAST_BATCH_LOG = 0.0
_FIXED_CANDIDATE_COMPUTATIONS = 0
_FIXED_PROXY_DEPTH = 0

def fixed_array_hash(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        a = a.astype(str)
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(fixed_canonical(a.shape).encode())
    view = memoryview(a.reshape(-1)).cast('B')
    for start in range(0, len(view), 1 << 24):
        h.update(view[start:start + (1 << 24)])
    return h.hexdigest()

def fixed_bundle_identity(bundle):
    key = id(bundle)
    cached = _FIXED_DATA_IDENTITIES.get(key)
    if cached is not None and cached[0] is bundle:
        return cached[1]
    with fixed_operation('input_identity', dataset=bundle.name, device='cpu'):
        identity = fixed_digest(dict(name=bundle.name, task=bundle.task, X=fixed_array_hash(bundle.X), y=fixed_array_hash(bundle.y), subjects=fixed_array_hash(bundle.subjects), sessions=fixed_array_hash(bundle.sessions), sfreq=bundle.sfreq, channels=bundle.ch_names, classes=bundle.class_names))
    _FIXED_DATA_IDENTITIES[key] = (bundle, identity)
    return identity

def fixed_epoch_start(epoch, maximum, trainer):
    if not _FIXED_CONTEXT.get('model'):
        return
    _FIXED_CONTEXT.update(epoch=int(epoch), max_epochs=int(maximum), batch=0, trainer=trainer)
    if _FIXED_CONTEXT.get('model') == 'B10_EEG_CONFORMER':
        fixed_emit('EPOCH', device=str(DEVICE))

def fixed_epoch_end():
    if _FIXED_CONTEXT.get('model') == 'B10_EEG_CONFORMER':
        fixed_emit('EPOCH_DONE', device=str(DEVICE))

def fixed_b10_fold(bundle, test_subject, validator, selected):
    if validator is None:
        tr, va, te = split_indices(bundle, test_subject, None)
    else:
        tr, va, te = source_validation_indices(bundle, test_subject, validator)
    train_subjects = sorted(np.unique(bundle.subjects[tr].astype(str)).tolist())
    assert_no_subject_overlap(bundle.subjects[tr], bundle.subjects[va], bundle.subjects[te])
    with fixed_operation('b10_source_normalization', dataset=bundle.name, test_subject=str(test_subject), model='B10_EEG_CONFORMER', validator=str(validator), device='cpu', target_device=str(DEVICE)):
        raw_train = bundle.X[tr].astype(np.float32)
        scaler = RobustChannelScaler().fit(raw_train)

        def part(ids):
            ids = np.asarray(ids, dtype=int)
            raw = bundle.X[ids].astype(np.float32)
            n = len(ids)
            T = raw.shape[-1]
            empty_signal = np.empty((n, 0, T), np.float32)
            empty_features = np.empty((n, 0), np.float32)
            return FoldArrays(raw, scaler.transform(raw), empty_signal, empty_signal, empty_signal, empty_features, empty_features, bundle.y[ids].astype(np.int64), make_subject_codes(bundle.subjects[ids], train_subjects), bundle.subjects[ids].astype(object), bundle.sessions[ids].astype(object))
        return PreparedFold(bundle.name, bundle.task, bundle.class_names, bundle.sfreq, int(selected['q']), [], [], np.empty((bundle.X.shape[1], 0), np.float32), part(tr), part(va), part(te), dict(fit_subjects=train_subjects, validation_subjects=sorted(np.unique(bundle.subjects[va]).astype(str).tolist()), test_subjects=[str(test_subject)], runtime_role='B10 validation inputs only; original source-fitted raw-channel normalization'))

def fixed_prepare_validation_fold(bundle, test_subject, validator, selected, model):
    if model == 'B10_EEG_CONFORMER':
        return fixed_b10_fold(bundle, test_subject, validator, selected)
    with fixed_operation('prepare_fold', dataset=bundle.name, test_subject=str(test_subject), model=model, validator=str(validator), device='cpu', target_device=str(DEVICE)):
        tr, va, te = source_validation_indices(bundle, test_subject, validator)
        fold = prepare_fold(bundle, tr, va, te, int(selected['q']), selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
        return _v12_transform_fold(model, fold, f'{bundle.name}|{test_subject}')

def fixed_candidate_contract(bundle, test_subject, model, selected, hp, stage, validator, index, total):
    return dict(schema=1, scientific_fingerprint=EXPERIMENT_FINGERPRINT, scientific_source_sha256=SOURCE_IDENTITY_SHA256, runtime_version=FIXED_RUNTIME_VERSION, data_identity=fixed_bundle_identity(bundle), dataset=bundle.name, test_subject=str(test_subject), model=model, structure=_v11_structure_signature(selected), hp=dict(hp), stage=stage, validator=str(validator), candidate_index=int(index), candidate_total=int(total), seed=int(cfg.PRIMARY_SEED), budgets={n: getattr(cfg, n) for n in ('PRETRAIN_EPOCHS', 'QUANTUM_EPOCHS', 'FINETUNE_EPOCHS', 'PATIENCE', 'QRG_MIN_PRETRAIN_BEST_EPOCH', 'QRG_MIN_STAGE2_BEST_EPOCH', 'QRG_MIN_FINETUNE_BEST_EPOCH')}, execution_device='cpu' if _v12_family(model, bundle.task) in ('b1', 'std', 'lda', 'frozen') else str(DEVICE), environment=_FIXED_RUNTIME_MANIFEST)

def fixed_candidate_paths(root, contract):
    token = fixed_digest(contract)
    folder = Path(root) / contract['dataset'] / ('test_' + contract['test_subject']) / contract['model']
    return (folder / (token + '.json'), folder / (token + '.npz'))

def fixed_atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def fixed_candidate(root, contract, compute):
    global _FIXED_CANDIDATE_COMPUTATIONS
    metadata_path, prediction_path = fixed_candidate_paths(root, contract)
    fields = dict(dataset=contract['dataset'], test_subject=contract['test_subject'], model=contract['model'], config=contract['candidate_index'] + 1, total_configs=contract['candidate_total'], validator=contract['validator'], stage=contract['stage'], device=contract.get('execution_device', str(DEVICE)))
    with fixed_operation(contract['stage'], **{k: v for k, v in fields.items() if k != 'stage'}):
        if metadata_path.exists() and cfg.RESUME:
            meta = json.loads(metadata_path.read_text())
            if meta.get('status') != 'complete' or meta.get('contract') != json.loads(fixed_canonical(contract)):
                raise RuntimeError('Candidate cache contract mismatch: ' + str(metadata_path))
            if not prediction_path.is_file() or fixed_sha256(prediction_path) != meta['prediction_sha256']:
                raise RuntimeError('Candidate cache artifact missing or corrupt: ' + str(prediction_path))
            with np.load(prediction_path, allow_pickle=False) as z:
                arrays = {n: z[n].copy() for n in z.files}
            fixed_check_candidate_arrays(arrays)
            fixed_emit('CANDIDATE_RESUME', detail=str(metadata_path))
            return dict(arrays, epochs=meta['epochs'], gpu_evidence=meta.get('gpu_evidence', {}), cached=True)
        _FIXED_CANDIDATE_COMPUTATIONS += 1
        fixed_emit('CANDIDATE_FIT')
        try:
            result = compute()
            y = np.asarray(result['y'], np.int64)
            prob = np.asarray(result['prob_raw'], np.float64)
            priors = np.asarray(result['source_priors'], np.float64)
            epochs = {k: int(v) for k, v in result['epochs'].items()}
            if contract['model'] == 'B10_EEG_CONFORMER' and DEVICE.type == 'cuda' and (not all((_FIXED_FIT_EVIDENCE.get(k) for k in ('forward_cuda', 'backward_cuda')))):
                raise RuntimeError('B10 candidate has no measured CUDA forward/backward evidence')
            arrays = dict(y=y, prob=prob, source_priors=priors)
            if 'validation_subjects' in result:
                arrays['validation_subjects'] = np.asarray(result['validation_subjects']).astype(str)
            fixed_check_candidate_arrays(arrays)
            fixed_atomic_npz(prediction_path, arrays)
            meta = dict(status='complete', contract=contract, epochs=epochs, prediction_sha256=fixed_sha256(prediction_path), gpu_evidence=dict(_FIXED_FIT_EVIDENCE), created_in_action=FIXED_ACTION)
            fixed_atomic_json(metadata_path, meta)
            fixed_emit('CANDIDATE_SAVED', detail=str(metadata_path))
            return dict(arrays, epochs=epochs, gpu_evidence=meta['gpu_evidence'], cached=False)
        except BaseException as error:
            fixed_atomic_json(metadata_path.with_suffix('.interrupted.json'), dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', contract=contract, error=repr(error)))
            raise

def fixed_check_candidate_arrays(arrays):
    if not {'y', 'prob', 'source_priors'} <= set(arrays):
        raise ValueError('Candidate arrays are incomplete')
    y, p, priors = (arrays['y'], arrays['prob'], arrays['source_priors'])
    if y.dtype.kind not in 'iu' or y.ndim != 1 or p.ndim != 2 or (len(y) != len(p)) or (not len(y)) or (not np.isfinite(p).all()) or (p < 0).any() or (not np.allclose(p.sum(1), 1, rtol=1e-05, atol=1e-07)):
        raise ValueError('Candidate probabilities are invalid')
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Candidate label indices are invalid')
    if priors.shape != (p.shape[1],) or not np.isfinite(priors).all() or (priors < 0).any() or (not np.isclose(priors.sum(), 1, rtol=1e-05, atol=1e-07)):
        raise ValueError('Candidate source priors are invalid')
    if 'validation_subjects' in arrays and len(arrays['validation_subjects']) != len(y):
        raise ValueError('Candidate validator identities are incomplete')

def fixed_v13_plan(bundle, test_subject, model_name, selected, qrg_plan, extra, cache_root=None, write_plan=True):
    with fixed_operation('configuration_generation', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device='cpu', target_device=str(DEVICE)):
        configs = _v13_configs(model_name, bundle, test_subject)
    family = _v12_family(model_name, bundle.task)
    if model_name == 'B8_QRG_EEG' or family == 'qrg_control' or len(configs) <= 1 or (not bool(cfg.V13_EQUAL_BUDGET)):
        return fixed_v12_plan(bundle, test_subject, model_name, selected, qrg_plan, extra, cache_root=cache_root, write_plan=write_plan)
    path = RUN_DIR / 'model_plans_v13' / f'{bundle.name}__test_{test_subject}__{model_name}.json'
    structure = _v11_structure_signature(selected)
    if path.exists() and cfg.RESUME and write_plan:
        payload = json.loads(path.read_text())
        if payload.get('fingerprint') != EXPERIMENT_FINGERPRINT or payload.get('structure') != structure:
            raise RuntimeError('Existing model plan is incompatible: ' + str(path))
        plan = payload['plan']
        t = plan['calibration'].get('decision_threshold')
        plan['calibration']['decision_threshold'] = float('nan') if t is None else float(t)
        fixed_emit('PLAN_RESUME', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device='cpu', detail=str(path))
        return plan
    validators = _v12_validators(bundle, test_subject)
    if not validators:
        raise RuntimeError('No source-only validator was selected')
    root = cache_root or _FIXED_ROOT / 'candidates'
    folds = {}

    def fold_for(v):
        if v not in folds:
            folds[v] = fixed_prepare_validation_fold(bundle, test_subject, v, selected, model_name)
        return folds[v]
    proxy = []
    with _v13_proxy_budget():
        fixed_emit('PROXY_BUDGET', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device=str(DEVICE), detail=f'{len(configs)} configurations; original proxy epochs={cfg.PRETRAIN_EPOCHS}, patience={cfg.PATIENCE}')
        for i, hp in enumerate(configs):
            v = validators[0]
            contract = fixed_candidate_contract(bundle, test_subject, model_name, selected, hp, 'proxy', v, i, len(configs))
            result = fixed_candidate(root, contract, lambda hp=hp, v=v: _v12_fit(model_name, fold_for(v), cfg.PRIMARY_SEED, hp, 'validate', qrg_plan=qrg_plan, extra=extra))
            m = multiclass_metrics(result['y'], result['prob'], bundle.task)
            proxy.append(dict(i=i, hp=hp, metrics=m, min_class_f1=float(np.min(per_class_f1(result['y'], result['prob'])))))
    best_macro = max((_finite_metric(r['metrics']['macro_f1']) for r in proxy))
    proxy.sort(key=lambda r: _v13_proxy_key(r['metrics'], r['min_class_f1'], best_macro, bundle.task, (-r['i'],)), reverse=True)
    summaries = []
    for r in proxy[:max(1, int(cfg.QRG_PROXY_SHORTLIST))]:
        records = []
        epoch_rows = []
        for v in validators:
            contract = fixed_candidate_contract(bundle, test_subject, model_name, selected, r['hp'], 'full_validation', v, r['i'], len(configs))
            result = fixed_candidate(root, contract, lambda r=r, v=v: _v12_fit(model_name, fold_for(v), cfg.PRIMARY_SEED, r['hp'], 'validate', qrg_plan=qrg_plan, extra=extra))
            records.append(dict(subject=v, y=result['y'], prob=result['prob'], source_priors=result['source_priors']))
            epoch_rows.append(result['epochs'])
        summaries.append(dict(i=r['i'], hp=r['hp'], records=records, epochs=epoch_rows, **_v13_full_summary(records, bundle.task)))
    chosen = _v13_choose(summaries, lambda s: (-s['i'],))
    if chosen['epochs'] and chosen['epochs'][0]:
        names = sorted(set().union(*[e.keys() for e in chosen['epochs']]))
        epochs = {n: max(1, int(round(np.median([e[n] for e in chosen['epochs'] if n in e])))) for n in names}
    else:
        epochs = {}
    plan = dict(hp=chosen['hp'], epochs=epochs, validators=validators, calibration=_v12_calibrate(chosen['records'], bundle.task), search_k=len(configs), proxy=[dict(hp=r['hp'], macro_f1=r['metrics']['macro_f1'], auroc=r['metrics']['auroc']) for r in proxy], grid=[{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst', 'joint_mean', 'joint_worst')} for s in summaries])
    if write_plan:
        fixed_atomic_json(path, _v11_json_ready(dict(fingerprint=EXPERIMENT_FINGERPRINT, structure=structure, model=model_name, plan=plan, outer_test_labels_used=False)))
        fixed_emit('PLAN_SAVED', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device='cpu', detail=str(path))
    return plan

def fixed_v12_plan(bundle, test_subject, model_name, selected, qrg_plan, extra, cache_root=None, write_plan=True):
    path = RUN_DIR / 'model_plans_v12' / f'{bundle.name}__test_{test_subject}__{model_name}.json'
    structure = _v11_structure_signature(selected)
    if path.exists() and cfg.RESUME and write_plan:
        payload = json.loads(path.read_text())
        if payload.get('fingerprint') != EXPERIMENT_FINGERPRINT or payload.get('structure') != structure:
            raise RuntimeError('Existing V12 plan is incompatible: ' + str(path))
        plan = payload['plan']
        t = plan['calibration'].get('decision_threshold')
        plan['calibration']['decision_threshold'] = float('nan') if t is None else float(t)
        fixed_emit('PLAN_RESUME', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device='cpu', detail=str(path))
        return plan
    if model_name == 'B8_QRG_EEG' or model_name in {f"QRG_DIAG_Q{int(selected['q'])}", f"QRG_DIAG_L{int(selected.get('layers', 2))}"}:
        plan = dict(hp={}, epochs=dict(qrg_plan), validators='nested QRG selection', calibration=dict(decision_threshold=float(selected.get('decision_threshold', np.nan)), prior_tau=float(selected.get('prior_tau', 0.0))), grid=[])
    else:
        validators = _v12_validators(bundle, test_subject)
        q_override = {'QRG_DIAG_Q6': 6, 'QRG_DIAG_Q8': 8}.get(model_name)
        folds = {}
        configs = _v12_grid(model_name, bundle.task)
        summaries = []
        root = cache_root or _FIXED_ROOT / 'candidates'

        def fold_for(v):
            if v not in folds:
                if q_override:
                    tr, va, te = source_validation_indices(bundle, test_subject, v)
                    with fixed_operation('prepare_fold', dataset=bundle.name, test_subject=str(test_subject), model=model_name, validator=str(v), device='cpu'):
                        f = prepare_fold(bundle, tr, va, te, int(q_override), selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
                    folds[v] = _v12_transform_fold(model_name, f, f'{bundle.name}|{test_subject}')
                else:
                    folds[v] = fixed_prepare_validation_fold(bundle, test_subject, v, selected, model_name)
            return folds[v]
        for i, hp in enumerate(configs):
            records = []
            epoch_rows = []
            for v in validators:
                ex = dict(extra)
                if model_name.startswith('C7_SHUFFLED_TT_P'):
                    f = fold_for(v)
                    perms, _, _ = _v12_edge_permutations(f.graph_edges, f.q, cfg.V12_N_PERMUTATIONS, f'{bundle.name}|{test_subject}|{v}')
                    ex['permutation'] = perms[_v12_index(model_name) % len(perms)]
                contract = fixed_candidate_contract(bundle, test_subject, model_name, selected, hp, 'v12_validation', v, i, len(configs))
                contract['extra'] = _v11_json_ready(ex)
                result = fixed_candidate(root, contract, lambda hp=hp, v=v, ex=ex: _v12_fit(model_name, fold_for(v), cfg.PRIMARY_SEED, hp, 'validate', qrg_plan=qrg_plan, extra=ex))
                records.append(dict(subject=v, y=result['y'], prob=result['prob'], source_priors=result['source_priors']))
                epoch_rows.append(result['epochs'])
            summaries.append(dict(hp=hp, records=records, epochs=epoch_rows, **_v12_summarise(records, bundle.task)))
        chosen = _v12_select_by_validation(summaries)
        family = _v12_family(model_name, bundle.task)
        if family == 'qrg_control':
            epochs = {k: int(v) for k, v in qrg_plan.items()}
            if model_name == 'EDGE_ONLY_VQC':
                epochs['pretrain'] = 0
        elif chosen['epochs'] and chosen['epochs'][0]:
            names = sorted(set().union(*[e.keys() for e in chosen['epochs']]))
            epochs = {n: max(1, int(round(np.median([e[n] for e in chosen['epochs'] if n in e])))) for n in names}
        else:
            epochs = {}
        plan = dict(hp=chosen['hp'], epochs=epochs, validators=validators, calibration=_v12_calibrate(chosen['records'], bundle.task), grid=[{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst', 'joint_mean', 'joint_worst')} for s in summaries])
    if write_plan:
        fixed_atomic_json(path, _v11_json_ready(dict(fingerprint=EXPERIMENT_FINGERPRINT, structure=structure, model=model_name, plan=plan, outer_test_labels_used=False)))
    return plan

def fixed_frozen(model_name, fold, bundle_name, test_subject, seed):
    t0 = time.perf_counter()
    bundle = loaded_datasets[bundle_name]
    with fixed_operation('frozen_cuda_features', model=model_name, seed=int(seed), device=str(DEVICE)):
        qrg = _v12_load_qrg(fold, bundle_name, test_subject, seed)
        fixed_assert_model(qrg, DEVICE)
        vtr, etr = _v12_frozen_features(qrg, fold.train)
        vte, ete = _v12_frozen_features(qrg, fold.test)
    Xtr = _v12_frozen_design(model_name, vtr, etr, fold.graph_edges)
    Xte = _v12_frozen_design(model_name, vte, ete, fold.graph_edges)
    groups = fold.train.subject_names.astype(str)
    splitter = GroupKFold(n_splits=int(min(3, len(np.unique(groups)))))
    choices = _v12_frozen_estimators(model_name)
    summaries = []
    selected, _ = select_qrg_source_only(bundle, test_subject)
    feature_identity = fixed_array_hash(Xtr)
    for index, (hp, make) in enumerate(choices):
        records = []
        for cv_index, (tr, va) in enumerate(splitter.split(Xtr, fold.train.y, groups)):
            contract = fixed_candidate_contract(bundle, test_subject, model_name, selected, hp, 'frozen_cv', 'GroupKFold_' + str(cv_index + 1), index, len(choices))
            contract.update(seed=int(seed), feature_identity=feature_identity, train_indices=fixed_array_hash(tr), validation_indices=fixed_array_hash(va), execution_device='cpu')

            def compute(make=make, tr=tr, va=va):
                est = make().fit(Xtr[tr], fold.train.y[tr])
                prob = est.predict_proba(Xtr[va])
                return dict(y=fold.train.y[va], prob_raw=prob, source_priors=_v11_fold_source_priors(fold.train), epochs={}, validation_subjects=groups[va])
            result = fixed_candidate(_FIXED_ROOT / 'candidates', contract, compute)
            for subject in np.unique(result['validation_subjects']):
                mask = result['validation_subjects'] == subject
                records.append(dict(subject=subject, y=result['y'][mask], prob=result['prob'][mask], source_priors=result['source_priors']))
        summaries.append(dict(hp=hp, records=records, **_v12_summarise(records, fold.task)))
    chosen = _v12_select_by_validation(summaries)
    make = dict(((json.dumps(h, sort_keys=True), m) for h, m in choices))[json.dumps(chosen['hp'], sort_keys=True)]
    with fixed_operation('frozen_final_fit', device='cpu'):
        est = make().fit(Xtr, fold.train.y)
    t1 = time.perf_counter()
    prob = est.predict_proba(Xte)
    n_params = _v12_sklearn_params(est)
    return dict(y=fold.test.y, prob_raw=np.asarray(prob, float), n_params=n_params, model=None, infer_s=time.perf_counter() - t1, epochs={}, source_priors=_v11_fold_source_priors(fold.train), wall_seconds=time.perf_counter() - t0, plan=dict(hp=chosen['hp'], epochs={}, validators='GroupKFold(source)', calibration=_v12_calibrate(chosen['records'], fold.task), grid=[{k: s[k] for k in ('hp', 'macro_f1_mean', 'macro_f1_worst')} for s in summaries]))

def fixed_install_wrappers():
    global _v12_build, _v12_fit, _v12_plan, _v12_calibrate, predict_torch, run_outer_fold_v12, _v13_proxy_budget, _v13_qrg_validation, _v12_run_frozen, prepare_fold
    original_prepare = prepare_fold

    def prepared(bundle, tr, va, te, q, graph_multiplier, projection_mode=None, edge_mode=None):
        return fixed_cached_prepare(original_prepare, bundle, tr, va, te, q, graph_multiplier, projection_mode, edge_mode)
    prepare_fold = prepared
    original_build = _v12_build
    original_fit = _v12_fit
    original_calibrate = _v12_calibrate
    original_predict = predict_torch
    original_proxy = _v13_proxy_budget
    original_qrg_validation = _v13_qrg_validation

    def build(model_name, fold, n_classes, hp, extra):
        model = original_build(model_name, fold, n_classes, hp, extra)
        model = model.to(DEVICE)
        device = fixed_assert_model(model, DEVICE)
        if _FIXED_CONTEXT.get('model'):
            handle = model.register_forward_pre_hook(fixed_model_forward_guard)
            _FIXED_GUARDS.append(handle)
            fixed_emit('MODEL_CUDA', device=device, detail='Parameters and registered buffers checked')
        return model

    def fit(model_name, fold, seed, hp, mode, epochs=None, qrg_plan=None, extra=None):
        _FIXED_FIT_EVIDENCE.clear()
        start_handles = len(_FIXED_GUARDS)
        execution_device = 'cpu' if _v12_family(model_name, fold.task) in ('b1', 'std', 'lda', 'frozen') else str(DEVICE)
        with fixed_operation('fit_' + mode, model=model_name, seed=int(seed), device=execution_device):
            try:
                return original_fit(model_name, fold, seed, hp, mode, epochs=epochs, qrg_plan=qrg_plan, extra=extra)
            finally:
                for handle in _FIXED_GUARDS[start_handles:]:
                    handle.remove()
                del _FIXED_GUARDS[start_handles:]

    def calibrate(records, task):
        with fixed_operation('source_only_calibration', device='cpu'):
            return original_calibrate(records, task)

    def predict(model, loader, device, *args, **kwargs):
        with fixed_operation('prediction', device=str(device)):
            return original_predict(model, loader, device, *args, **kwargs)

    @contextlib.contextmanager
    def proxy_budget():
        global _FIXED_PROXY_DEPTH
        with original_proxy():
            _FIXED_PROXY_DEPTH += 1
            try:
                yield
            finally:
                _FIXED_PROXY_DEPTH -= 1

    def qrg_validation(bundle, test_subject, val_subject, key, seed):
        q, proj, edge, L, g, pool = key
        _v13_activate({'layers': L, 'pool': pool})
        selected = dict(variant='A', q=int(q), layers=int(L), graph_multiplier=float(g), projection_mode=str(proj), edge_mode=str(edge), pool=str(pool))
        candidates = _v13_qrg_candidates(bundle)
        index = candidates.index(key)
        stage = 'qrg_proxy' if _FIXED_PROXY_DEPTH else 'qrg_full_validation'
        contract = fixed_candidate_contract(bundle, test_subject, 'B8_QRG_EEG', selected, dict(qrg_key=list(key)), stage, val_subject, index, len(candidates))
        contract['seed'] = int(seed)

        def compute():
            _FIXED_FIT_EVIDENCE.clear()
            with fixed_operation('qrg_validation_inputs_and_training', device=str(DEVICE)):
                r = original_qrg_validation(bundle, test_subject, val_subject, key, seed)
            return dict(y=r['y'], prob_raw=r['prob'], source_priors=r['source_priors'], epochs=r['stage_plan'])
        r = fixed_candidate(_FIXED_ROOT / 'candidates', contract, compute)
        return dict(y=r['y'], prob=r['prob'], source_priors=r['source_priors'], stage_plan=r['epochs'])
    _v12_build = build
    _v12_fit = fit
    _v12_plan = fixed_v13_plan
    _v12_calibrate = calibrate
    predict_torch = predict
    run_outer_fold_v12 = fixed_outer
    _v13_proxy_budget = proxy_budget
    _v13_qrg_validation = qrg_validation
    _v12_run_frozen = fixed_frozen
_FIXED_FOLD_CACHE = collections.OrderedDict()
_FIXED_FOLD_CACHE_BYTES = 0
_FIXED_SHARED_FOLD_KEYS = {}

def fixed_assert_model(model, device):
    target = torch.device(device)
    if target.type != 'cuda' and bool(cfg.REQUIRE_CUDA_PAPER):
        raise RuntimeError('This full run requires CUDA')
    expected = target.index if target.index is not None else torch.cuda.current_device() if target.type == 'cuda' else None
    bad = [str(t.device) for t in list(model.parameters()) + list(model.buffers()) if t.device.type != target.type or (target.type == 'cuda' and t.device.index != expected)]
    if bad:
        raise RuntimeError('Model parameters/buffers are on a different device: ' + str(bad[:5]))
    return str(torch.device(target.type, expected)) if expected is not None else str(target)

def fixed_check_operation_time():
    return None

def fixed_model_forward_guard(model, args):
    batch = args[0] if args else None
    if not isinstance(batch, dict):
        return
    device = fixed_assert_model(model, DEVICE)
    expected = torch.cuda.current_device() if DEVICE.type == 'cuda' else None
    for name, value in batch.items():
        if torch.is_tensor(value) and (value.device.type != DEVICE.type or (DEVICE.type == 'cuda' and value.device.index != expected)):
            raise RuntimeError('Batch is on the wrong device: ' + name)
    if not _FIXED_FIT_EVIDENCE.get('forward_checked'):
        _FIXED_FIT_EVIDENCE.update(forward_checked=True, forward_cuda=DEVICE.type == 'cuda', model_device=device, batch_device=str(batch['x_deep'].device) if 'x_deep' in batch else device)
        fixed_emit('FORWARD_DEVICE', device=device)

def fixed_training_step(model):
    global _FIXED_LAST_BATCH_LOG
    if not _FIXED_CONTEXT.get('model'):
        return
    _FIXED_CONTEXT['batch'] = int(_FIXED_CONTEXT.get('batch', 0)) + 1
    if not _FIXED_FIT_EVIDENCE.get('backward_checked'):
        device = fixed_assert_model(model, DEVICE)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all((bool(torch.isfinite(g).all()) and g.device.type == DEVICE.type for g in gradients)):
            raise FloatingPointError('Missing, non-finite or incorrectly placed gradients')
        _FIXED_FIT_EVIDENCE.update(backward_checked=True, backward_cuda=DEVICE.type == 'cuda', gradient_device=str(gradients[0].device))
        fixed_emit('BACKWARD_DEVICE', device=device)
    if time.monotonic() - _FIXED_LAST_BATCH_LOG >= 30:
        fixed_emit('BATCH_PROGRESS', device=str(DEVICE))
        _FIXED_LAST_BATCH_LOG = time.monotonic()

def fixed_fold_bytes(fold):
    seen = set()
    total = 0
    for part in (fold.train, fold.val, fold.test):
        for value in vars(part).values():
            if isinstance(value, np.ndarray) and id(value) not in seen:
                seen.add(id(value))
                total += value.nbytes
    for value in getattr(fold, 'v12_H', {}).values():
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += value.nbytes
    return total

def fixed_fold_cache_limit():
    available = 8 * (1 << 30)
    try:
        line = next((s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
        available = int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        pass
    return int(min(2 * (1 << 30), max(64 * (1 << 20), available // 8)))

def fixed_cached_prepare(original, bundle, tr, va, te, q, graph_multiplier, projection_mode=None, edge_mode=None):
    global _FIXED_FOLD_CACHE_BYTES
    key = (id(bundle), fixed_array_hash(np.asarray(tr, int)), fixed_array_hash(np.asarray(va, int)), fixed_array_hash(np.asarray(te, int)), int(q), float(graph_multiplier), projection_mode, edge_mode)
    if key in _FIXED_FOLD_CACHE:
        fold, nbytes = _FIXED_FOLD_CACHE.pop(key)
        _FIXED_FOLD_CACHE[key] = (fold, nbytes)
        fixed_emit('FOLD_INPUT_CACHE_HIT', device='cpu')
        return fold
    with fixed_operation('prepare_fold_inputs', dataset=bundle.name, device='cpu', target_device=str(DEVICE)):
        fold = original(bundle, tr, va, te, q, graph_multiplier, projection_mode, edge_mode)
    nbytes = fixed_fold_bytes(fold)
    limit = fixed_fold_cache_limit()
    while _FIXED_FOLD_CACHE and (len(_FIXED_FOLD_CACHE) >= 4 or _FIXED_FOLD_CACHE_BYTES + nbytes > limit):
        _, (_, old_bytes) = _FIXED_FOLD_CACHE.popitem(last=False)
        _FIXED_FOLD_CACHE_BYTES -= old_bytes
    if nbytes <= limit:
        _FIXED_FOLD_CACHE[key] = (fold, nbytes)
        _FIXED_FOLD_CACHE_BYTES += nbytes
    return fold

def fixed_sample_rows(bundle, test_subject):
    first = outer_subjects(bundle, cfg.MAX_OUTER_FOLDS, cfg.PRIMARY_SEED)[0]
    if str(test_subject) != str(first):
        return
    rows = [p['row'] for k, p in store.records.items() if k[0] == bundle.name and k[3] == str(first)]
    _atomic_csv(RUN_DIR / (bundle.name + '__sample_results.csv'), pd.DataFrame(rows, columns=V8_PUBLIC_COLUMNS))

def fixed_outer(bundle, test_subject, models, phase):
    global _FIXED_LAST_RESULT
    todo = [m for m in models if not all((store.completed(dict(dataset=bundle.name, model=_v11_reported_model(bundle, m), seed=s, test_subject=str(test_subject))) for s in cfg.SEEDS))]
    if not todo:
        return
    first_model = todo[0]
    first_seed = next((s for s in cfg.SEEDS if not store.completed(dict(dataset=bundle.name, model=_v11_reported_model(bundle, first_model), seed=s, test_subject=str(test_subject)))))
    print('V12 OUTER:', phase, bundle.name, test_subject, first_model, 'seed', first_seed, flush=True)
    pending_header = (first_model, int(first_seed))
    with fixed_operation('source_selection', dataset=bundle.name, test_subject=str(test_subject), model='B8_QRG_EEG', device=str(DEVICE)), fixed_quiet():
        selected, qrg_plan = select_qrg_source_only(bundle, test_subject)
    _v13_activate(selected)
    src, empty, test = split_indices(bundle, test_subject, None)
    base_fold = None
    folds_q = {}
    qrg_calibration = None
    for model_name in todo:
        family = _v12_family(model_name, bundle.task)
        with fixed_operation('outer_model', dataset=bundle.name, test_subject=str(test_subject), model=model_name, device='cpu' if family in ('b1', 'std', 'lda', 'frozen') else str(DEVICE)):
            try:
                seeds = [s for s in cfg.SEEDS if not store.completed(dict(dataset=bundle.name, model=_v11_reported_model(bundle, model_name), seed=s, test_subject=str(test_subject)))]
                if not seeds:
                    continue
                if pending_header != (model_name, int(seeds[0])):
                    print('V12 OUTER:', phase, bundle.name, test_subject, model_name, 'seed', seeds[0], flush=True)
                if base_fold is None:
                    with fixed_quiet():
                        base_fold = prepare_fold(bundle, src, empty, test, selected['q'], selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
                if model_name in ('QRG_DIAG_Q6', 'QRG_DIAG_Q8'):
                    qq = 6 if model_name.endswith('Q6') else 8
                    if qq not in folds_q:
                        with fixed_quiet():
                            folds_q[qq] = prepare_fold(bundle, src, empty, test, qq, selected['graph_multiplier'], selected['projection_mode'], selected['edge_mode'])
                    fold = folds_q[qq]
                else:
                    with fixed_quiet():
                        fold = _v12_transform_fold(model_name, base_fold, f'{bundle.name}|{test_subject}')
                with fixed_quiet():
                    plan = None if family in ('frozen', 'test_only') else _v12_plan(bundle, test_subject, model_name, selected, qrg_plan, {})
                for seed_index, seed in enumerate(seeds):
                    if seed_index:
                        print('V12 OUTER:', phase, bundle.name, test_subject, model_name, 'seed', seed, flush=True)
                    _FIXED_CONTEXT['seed'] = int(seed)
                    with fixed_quiet():
                        if family == 'frozen':
                            fit = _v12_run_frozen(model_name, fold, bundle.name, test_subject, seed)
                            plan_used = fit['plan']
                            fit['model_for_row'] = nn.Identity()
                        elif family == 'test_only':
                            if qrg_calibration is None:
                                qrg_calibration = _v12_plan(bundle, test_subject, 'B8_QRG_EEG', selected, qrg_plan, {})['calibration']
                            fit = _v12_run_test_only(model_name, fold, bundle.name, test_subject, seed, qrg_calibration)
                            plan_used = fit['plan']
                            fit['model_for_row'] = _v12_load_qrg(fold, bundle.name, test_subject, seed)
                        else:
                            extra = {}
                            if model_name.startswith('C7_SHUFFLED_TT_P'):
                                perms, excluded, _ = _v12_edge_permutations(fold.graph_edges, fold.q, cfg.V12_N_PERMUTATIONS, f'{bundle.name}|{test_subject}|final')
                                extra['permutation'] = perms[_v12_index(model_name) % len(perms)]
                            fit = _v12_fit(model_name, fold, seed, plan['hp'], 'final', epochs=plan['epochs'], qrg_plan=qrg_plan, extra=extra)
                            if model_name.startswith('C7_SHUFFLED_TT_P'):
                                fit['automorphisms_excluded'] = excluded
                            plan_used = plan
                            if model_name in ('B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE'):
                                gap = float(fit.get('parameter_match_gap', np.nan))
                                if not np.isfinite(gap) or gap > float(cfg.PARAM_MATCH_TOLERANCE):
                                    raise RuntimeError('Parameter matching failed for ' + model_name + ': ' + str(gap))
                            if fit.get('model') is None:
                                fit['model_for_row'] = nn.Identity()
                        with fixed_operation('commit_record', device='cpu'):
                            row = _v12_commit(bundle, model_name, seed, test_subject, selected, fold, fit, plan_used, phase)
                    print({k: row[k] for k in ('dataset', 'model', 'seed', 'test_subject', 'macro_f1', 'auroc')}, flush=True)
                    _FIXED_LAST_RESULT = time.monotonic()
                    fixed_sample_rows(bundle, test_subject)
                    del fit
                    with fixed_operation('garbage_collection', device='cpu'):
                        gc.collect()
            except Exception as error:
                with (RUN_DIR / 'errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(dataset=bundle.name, model=model_name, test_subject=str(test_subject), phase=phase, error=repr(error))) + '\n')
                with (_FIXED_ROOT / 'tracebacks.log').open('a') as stream:
                    traceback.print_exc(file=stream)
                fixed_emit('MODEL_FAILED', detail=repr(error))
    fixed_sample_rows(bundle, test_subject)
fixed_install_wrappers()

def validate_final_bundle(bundle: EEGDatasetBundle) -> None:
    n = len(bundle.y)
    if n == 0 or bundle.X.shape[0] == 0:
        raise DatasetSchemaError(f'{bundle.name}: zero trials after preprocessing.')
    if not len(bundle.X) == len(bundle.y) == len(bundle.subjects) == len(bundle.sessions):
        raise DatasetSchemaError(f'{bundle.name}: X/y/subject/session lengths are inconsistent.')
    if not np.isfinite(bundle.X).all():
        raise DatasetSchemaError(f'{bundle.name}: non-finite EEG values remain after preprocessing.')
    classes = set(np.unique(bundle.y).tolist())
    subjects = sorted(np.unique(bundle.subjects).tolist(), key=str)
    expected_classes = set(range(len(bundle.class_names)))
    if classes != expected_classes:
        raise DatasetSchemaError(f'{bundle.name}: expected class indices {sorted(expected_classes)}, got {sorted(classes)}.')
    if len(subjects) < 2:
        raise DatasetSchemaError(f'{bundle.name}: fewer than two subjects remain ({subjects}).')
    for subject in subjects:
        idx = np.flatnonzero(bundle.subjects.astype(str) == str(subject))
        present = set(np.unique(bundle.y[idx]).tolist())
        missing = expected_classes - present
        if missing:
            if cfg.RUN_MODE == 'paper':
                raise DatasetSchemaError(f'{bundle.name} subject {subject}: missing classes {sorted(missing)}; AUROC/paired paper metrics would be undefined.')
            print(f'WARNING {bundle.name} subject {subject}: missing classes {sorted(missing)} in recorded data; patient retained without fabrication.')
    if bundle.name == 'BNCI2014_008':
        expected_n = min(8, cfg.MAX_SUBJECTS) if cfg.MAX_SUBJECTS is not None else 8
        if len(subjects) != expected_n:
            raise DatasetSchemaError(f'BNCI2014_008: expected {expected_n} ALS subjects in this mode, got {len(subjects)}.')
    elif bundle.name == 'UCL_ALS_MI' and len(subjects) != 8:
        raise DatasetSchemaError(f'UCL_ALS_MI: expected 8 ALS patients, got {len(subjects)}.')
    elif bundle.name == 'BIGP3BCI_ALS' and len(subjects) != 8:
        raise DatasetSchemaError(f'BIGP3BCI_ALS Study{cfg.BIGP3_STUDY}: expected 8 ALS participants, got {len(subjects)}.')
    elif bundle.name == 'EEGET_ALS_MI' and len(subjects) != 6:
        raise DatasetSchemaError(f'EEGET_ALS_MI: final execution protocol expects 6 ALS participants, got {len(subjects)}.')
loaded_datasets = {}

def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _atomic_text(path: Path, text_value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        handle.write(text_value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(Path(path), json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + '\n')

def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
import os
import re
import json
import time
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
RB_SUBJECT_SCOPE = 'first' if ARGS.scope == 'first' else 'all'
RB_SEARCH = 'equal_budget'
RB_DATASETS = tuple(ARGS.datasets)
RB_SEEDS = tuple(cfg.SEEDS)
RB_SUBJECTS = {}
RB_CONFIG = {'p300': dict(projection_mode='xdawn', edge_mode='erp_residual'), 'mi': dict(projection_mode='csp', edge_mode='plain'), 'common': dict(variant='A', q=8, layers=2, graph_multiplier=1.5)}
RB_P300_POOL = 'avg'
RB_MI_POOL = 'avg'
RB_ORIGINAL_RUN_DIR = Path(ARGS.reference_selections) if ARGS.reference_selections else Path()
RB_REPORTED_DIR = Path(ARGS.reference_checkpoints) if ARGS.reference_checkpoints else Path()
RB_REPORTED_B7 = 'B7_MATCHED_CLASSICAL_INTERACTION'
RB_RUN_BASELINES = True
RB_RUN_FORCED = True
RB_RUN_SHOTS = not ARGS.skip_shots
RB_RUN_EFFICIENCY = not ARGS.skip_efficiency
RB_REUSE_PREVIOUS = True
RB_ONLY_MODELS = ()
RB_FORCED_PARTS = ('representation', 'qubits_q6')
RB_SHOTS = (64, 256, 1024, 4096)
RB_SHOT_REPEATS = 5
RB_BOOTSTRAP = 100000
RB_ROOT = cfg.RESULTS_ROOT / 'experiments'
RB_QRG = 'B8_QRG_EEG'
RB_B7 = 'X0_B7_MATCHED'
RB_COHORT_ORDER = ('BIGP3BCI_ALS', 'BNCI2014_008', 'EEGET_ALS_MI', 'UCL_ALS_MI')
RB_SHORT = {'BIGP3BCI_ALS': 'bigP3BCI', 'BNCI2014_008': 'BNCI', 'EEGET_ALS_MI': 'EEGET', 'UCL_ALS_MI': 'UCL'}
if RB_SUBJECT_SCOPE not in ('first', 'all'):
    raise ValueError("RB_SUBJECT_SCOPE must be 'first' or 'all'")
if RB_SEARCH not in ('equal_budget', 'default'):
    raise ValueError("RB_SEARCH must be 'equal_budget' or 'default'")

def _rb_key(subject):
    return re.sub('[^0-9A-Za-z]', '', str(subject))

def _rb_bundle(name):
    if name not in loaded_datasets:
        loaded_datasets[name] = fixed_load_dataset(name)
    return loaded_datasets[name]

def _rb_subjects(bundle):
    if bundle.name in RB_SUBJECTS:
        return [str(s) for s in RB_SUBJECTS[bundle.name]]
    subjects = outer_subjects(bundle, cfg.MAX_OUTER_FOLDS, cfg.PRIMARY_SEED)
    return [str(s) for s in (subjects[:1] if RB_SUBJECT_SCOPE == 'first' else subjects)]

def _rb_cohorts():
    return [d for d in RB_COHORT_ORDER if d in RB_DATASETS]

def _rb_selected(bundle, subject):
    path = RB_ORIGINAL_RUN_DIR / f'source_selection_v13__{bundle.name}__test_{subject}.json'
    if str(RB_ORIGINAL_RUN_DIR) not in ('', '.') and path.is_file():
        selected = dict(json.loads(path.read_text())['selected'])
        selected.setdefault('variant', 'A')
        selected.setdefault('pool', 'avg')
        return (selected, 'reference selection file')
    selected = dict(RB_CONFIG['common'], **RB_CONFIG[bundle.task])
    selected['pool'] = RB_P300_POOL if bundle.task == 'p300' else RB_MI_POOL
    return (selected, 'reference configuration')

def _rb_outer_fold(bundle, subject, selected):
    _v13_activate(selected)
    src, empty, test = split_indices(bundle, subject, None)
    return prepare_fold(bundle, src, empty, test, int(selected['q']), float(selected['graph_multiplier']), selected['projection_mode'], selected['edge_mode'])

@contextlib.contextmanager
def _rb_workspace(name):
    global RUN_DIR, _FIXED_ROOT, EXPERIMENT_FINGERPRINT
    old = (globals().get('RUN_DIR'), globals().get('_FIXED_ROOT'), globals().get('EXPERIMENT_FINGERPRINT'))
    out = RB_ROOT / name
    (out / 'runtime').mkdir(parents=True, exist_ok=True)
    RUN_DIR, _FIXED_ROOT = (out, out / 'runtime')
    EXPERIMENT_FINGERPRINT = f'graph_experiments|{name}|{cfg.CODE_VERSION}|{RB_SEARCH}'
    try:
        yield out
    finally:
        RUN_DIR, _FIXED_ROOT, EXPERIMENT_FINGERPRINT = old

@contextlib.contextmanager
def _rb_search_mode():
    old = cfg.V13_SEARCH_K
    try:
        if RB_SEARCH == 'default':
            cfg.V13_SEARCH_K = 1
        yield
    finally:
        cfg.V13_SEARCH_K = old

@torch.no_grad()
def _rb_input_check(model, fold, label):
    model = model.to(DEVICE).eval()
    batch = move_batch(next(iter(make_loaders(fold, 32)[0])), DEVICE)
    base = model(batch)
    d_edge = float((model(dict(batch, edges=batch['edges'] + 0.5)) - base).abs().max())
    d_node = float((model(dict(batch, x_latent=batch['x_latent'] + 0.5 * torch.randn_like(batch['x_latent']))) - base).abs().max())
    if not (bool(torch.isfinite(base).all()) and d_edge > 1e-07 and (d_node > 1e-07)):
        raise RuntimeError(f'{label}: output ignores its inputs (edge change {d_edge:.2e}, node change {d_node:.2e})')

def _rb_metrics(y, prob_raw, task, calibration, priors):
    y = np.asarray(y, int)
    prob_raw = np.asarray(prob_raw, float)
    prob_dec, threshold = _v12_apply_decision(prob_raw, task, calibration, priors)
    cal = multiclass_metrics(y, prob_dec, task, decision_threshold=threshold)
    raw = multiclass_metrics(y, prob_raw, task)
    pred = _prediction_from_probability(prob_dec, threshold)
    out = {k: cal[k] for k in ('macro_f1', 'balanced_accuracy', 'auroc', 'accuracy', 'kappa', 'target_auprc')}
    out.update(macro_f1_uncal=raw['macro_f1'], balanced_accuracy_uncal=raw['balanced_accuracy'])
    out.update({f'recall_{k}': float(np.mean(pred[y == k] == k)) if np.any(y == k) else np.nan for k in range(prob_dec.shape[1])})
    return out

def _rb_fit(model_name, fold, seed, plan, qrg_plan):
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fit = _v12_fit(model_name, fold, seed, plan['hp'], 'final', epochs=plan['epochs'], qrg_plan=qrg_plan)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    fit['rb_wall_seconds'] = time.perf_counter() - t0
    fit['rb_peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == 'cuda' else np.nan
    return fit

def _rb_row(part, name, subject, model_name, seed, fold, selected, plan, fit):
    row = dict(part=part, dataset=name, task=fold.task, test_subject=str(subject), model=model_name, seed=int(seed), q=int(selected['q']), representation=f"{selected['projection_mode']}/{selected['edge_mode']}", n_params=int(fit.get('n_params', -1)), param_gap=float(fit.get('parameter_match_gap', np.nan)), hp=json.dumps(plan['hp'], sort_keys=True), epochs=json.dumps(plan['epochs'], sort_keys=True), wall_seconds=float(fit.get('rb_wall_seconds', np.nan)), peak_mem_mb=float(fit.get('rb_peak_mem_mb', np.nan)))
    row.update(_rb_metrics(fit['y'], fit['prob_raw'], fold.task, plan['calibration'], fit['source_priors']))
    save_experiment_artifacts(row, fold, selected, plan, fit)
    return row

def _rb_qrg_source_plan(bundle, subject, selected):
    key = (int(selected['q']), str(selected['projection_mode']), str(selected['edge_mode']), int(selected['layers']), float(selected['graph_multiplier']), str(selected.get('pool', 'avg')))
    if key not in _v13_qrg_candidates(bundle):
        raise RuntimeError(f'Configuration {key} is not in the declared QRG candidate set')
    records = [_v13_qrg_validation(bundle, subject, v, key, cfg.PRIMARY_SEED) for v in _v12_validators(bundle, subject)]
    floors = {'pretrain': cfg.QRG_MIN_PRETRAIN_BEST_EPOCH, 'stage2': cfg.QRG_MIN_STAGE2_BEST_EPOCH, 'finetune': cfg.QRG_MIN_FINETUNE_BEST_EPOCH}
    stage_plan = {s: max(int(f), int(round(np.median([r['stage_plan'][s] for r in records])))) for s, f in floors.items()}
    calibration = _v12_calibrate([dict(y=r['y'], prob=r['prob'], source_priors=r['source_priors']) for r in records], bundle.task)
    return dict(hp={}, epochs=stage_plan, calibration=calibration)

def _rb_expected(models, seeds=None):
    return {(d, s, m, int(seed)) for d in RB_DATASETS for s in _rb_subjects(_rb_bundle(d)) for m in models for seed in seeds or RB_SEEDS}

def _rb_reuse(paths, models, part=None, part_alias=None, seeds=None):
    if not RB_REUSE_PREVIOUS:
        return None
    expected = _rb_expected(models, seeds)
    for path in paths:
        if not Path(path).is_file():
            continue
        frame = pd.read_csv(path)
        frame['test_subject'] = frame['test_subject'].astype(str)
        if part is not None:
            if 'part' not in frame:
                frame.insert(0, 'part', part_alias or part)
            frame = frame[frame['part'] == part]
        keys = list(zip(frame['dataset'], frame['test_subject'], frame['model'], frame['seed'].astype(int)))
        if expected <= set(keys):
            print(f'[reuse] {len(expected)} completed rows from {path}', flush=True)
            return frame[[k in expected for k in keys]].reset_index(drop=True)
    return None

def _rb_patient_means(frame, model, metric):
    sel = frame[frame['model'] == model]
    return sel.groupby(['dataset', 'test_subject'])[metric].mean()

def _rb_boot_ci(values, n=None, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(values) < 2:
        return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(int(n or RB_BOOTSTRAP), len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def _rb_fmt(x, signed=True):
    if x is None or not np.isfinite(x):
        return 'n/a'
    s = f'{x:+.3f}' if signed else f'{x:.3f}'
    return s.replace('+0.', '+.').replace('-0.', '−.').replace('0.', '.', 1) if not signed else s.replace('+0.', '+.').replace('-0.', '−.')
GRAPH_MODELS = ('X1_NNCONV_ECC', 'X2_GINE', 'X3_PAIRWISE_MLP')
RB_NEW_GRAPH_MODELS = ('X0_B7_MATCHED', 'X1_NNCONV_ECC', 'X2_GINE', 'X3_PAIRWISE_MLP')
MODEL_LABELS = {'X0_B7_MATCHED': 'B7', 'X1_NNCONV_ECC': 'NNConv/ECC', 'X2_GINE': 'GINE', 'X3_PAIRWISE_MLP': 'Pairwise MLP', 'B8_QRG_EEG': 'QRG-EEG'}

def _graph_directed(edges):
    src, dst, eid = ([], [], [])
    for k, (i, j) in enumerate(edges):
        src += [i, j]
        dst += [j, i]
        eid += [k, k]
    return (torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long), torch.as_tensor(eid, dtype=torch.long))

class _GraphControlBase(MatchedClassicalInteraction):

    def _init_common(self, node_encoder, edges):
        nn.Module.__init__(self)
        self.node_encoder = node_encoder
        self.edges = [(int(a), int(b)) for a, b in edges]
        self.q = int(node_encoder.q)
        self.node_scaler = RobustTorchScaler(self.q)
        src, dst, eid = _graph_directed(self.edges)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
        self.register_buffer('eid', eid)
        self.register_buffer('deg', torch.bincount(dst, minlength=self.q).clamp_min(1).to(torch.float32))
        self.register_buffer('edge_i', torch.as_tensor([a for a, _ in self.edges], dtype=torch.long))
        self.register_buffer('edge_j', torch.as_tensor([b for _, b in self.edges], dtype=torch.long))

    def node_edge_inputs(self, batch):
        _, v = self.node_encoder(batch['x_latent'])
        return (self.node_scaler(v), batch['edges'])

    def aggregate(self, msg, mean):
        out = msg.new_zeros(msg.shape[0], self.q, msg.shape[2]).index_add_(1, self.dst, msg)
        return out / self.deg.view(1, -1, 1) if mean else out

class _ECCLayer(nn.Module):

    def __init__(self, din, dout, edge_hidden):
        super().__init__()
        self.din, self.dout = (int(din), int(dout))
        self.edge_net = nn.Sequential(nn.Linear(1, edge_hidden), nn.Tanh(), nn.Linear(edge_hidden, self.din * self.dout))
        self.root = nn.Linear(self.din, self.dout)

class NNConvECCBaseline(_GraphControlBase):

    def __init__(self, node_encoder, edges, n_classes, hidden=3, edge_hidden=2):
        self._init_common(node_encoder, edges)
        self.conv1 = _ECCLayer(1, hidden, edge_hidden)
        self.conv2 = _ECCLayer(hidden, hidden, edge_hidden)
        self.readout = nn.Linear(self.q * hidden, n_classes)
        self.config = dict(hidden=int(hidden), edge_hidden=int(edge_hidden), layers=2, aggregation='mean')

    def _conv(self, layer, h, e_dir):
        B, M = e_dir.shape
        theta = layer.edge_net(e_dir.unsqueeze(-1)).view(B, M, layer.din, layer.dout)
        msg = torch.einsum('bmi,bmio->bmo', h[:, self.src], theta)
        return layer.root(h) + self.aggregate(msg, mean=True)

    def forward(self, batch):
        v, e = self.node_edge_inputs(batch)
        e_dir = e[:, self.eid]
        h = F.elu(self._conv(self.conv1, v.unsqueeze(-1), e_dir))
        h = F.elu(self._conv(self.conv2, h, e_dir))
        return self.readout(h.flatten(1))

class _GINELayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.edge_lin = nn.Linear(1, dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, dim))

class GINEBaseline(_GraphControlBase):

    def __init__(self, node_encoder, edges, n_classes, hidden=2, n_layers=2):
        self._init_common(node_encoder, edges)
        self.lift = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([_GINELayer(hidden) for _ in range(int(n_layers))])
        self.readout = nn.Linear(self.q * hidden, n_classes)
        self.config = dict(hidden=int(hidden), layers=int(n_layers), aggregation='sum')

    def forward(self, batch):
        v, e = self.node_edge_inputs(batch)
        e_dir = e[:, self.eid].unsqueeze(-1)
        h = self.lift(v.unsqueeze(-1))
        for index, layer in enumerate(self.layers):
            msg = F.elu(h[:, self.src] + layer.edge_lin(e_dir))
            h = layer.mlp((1.0 + layer.eps) * h + self.aggregate(msg, mean=False))
            if index < len(self.layers) - 1:
                h = F.elu(h)
        return self.readout(h.flatten(1))

class PairwiseMLPBaseline(_GraphControlBase):

    def __init__(self, node_encoder, edges, n_classes, hidden=10, rank=1):
        self._init_common(node_encoder, edges)
        self.phi = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, rank))
        self.readout = nn.Linear(self.q + len(self.edges) * int(rank), n_classes)
        self.config = dict(hidden=int(hidden), rank=int(rank))

    def forward(self, batch):
        v, e = self.node_edge_inputs(batch)
        pair = torch.stack([v[:, self.edge_i], v[:, self.edge_j], e], dim=-1)
        return self.readout(torch.cat([v, self.phi(pair).flatten(1)], dim=1))

class MatchedInteractionBaseline(_GraphControlBase):

    def __init__(self, node_encoder, edges, n_classes, hidden=4):
        self._init_common(node_encoder, edges)
        self.net = nn.Sequential(nn.Linear(self.q + len(self.edges), hidden), nn.Tanh(), nn.Linear(hidden, n_classes))
        self.config = dict(hidden=int(hidden), interaction='e_ij * v_i * v_j')

    def forward(self, batch):
        v, e = self.node_edge_inputs(batch)
        return self.net(torch.cat([v, e * v[:, self.edge_i] * v[:, self.edge_j]], dim=1))
_GRAPH_MODEL_CLASSES = {'X0_B7_MATCHED': MatchedInteractionBaseline, 'X1_NNCONV_ECC': NNConvECCBaseline, 'X2_GINE': GINEBaseline, 'X3_PAIRWISE_MLP': PairwiseMLPBaseline}
_GRAPH_ARCHITECTURES = {'X0_B7_MATCHED': [dict(hidden=h) for h in range(1, int(cfg.PARAM_MATCH_MAX_HIDDEN) + 1)], 'X1_NNCONV_ECC': [dict(hidden=h, edge_hidden=m) for h in range(1, 17) for m in (1, 2, 4, 8)], 'X2_GINE': [dict(hidden=h, n_layers=n) for n in (2, 1) for h in range(1, 17)], 'X3_PAIRWISE_MLP': [dict(hidden=h, rank=r) for r in (1, 2, 3) for h in range(1, 65)]}

def _graph_total_params(model):
    return int(sum((p.numel() for p in model.parameters())))

def build_graph_control(model_name, fold, n_classes):
    target = _v11_total_architectural_parameters(build_qrg_model(fold, n_classes, 'A', layers=_v13_layers()))
    best = None
    for kw in _GRAPH_ARCHITECTURES[model_name]:
        encoder = SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
        candidate = _GRAPH_MODEL_CLASSES[model_name](encoder, fold.graph_edges, n_classes, **kw)
        actual = _graph_total_params(candidate)
        gap = abs(actual - target) / max(1, target)
        if best is None or gap < best[0] - 1e-12:
            best = (gap, actual, candidate)
    gap, actual, model = best
    if getattr(model, 'config', {}).get('layers', 2) != 2:
        two = []
        for kw in _GRAPH_ARCHITECTURES[model_name]:
            if kw.get('n_layers', 2) != 2:
                continue
            encoder = SharedTemporalNodeEncoder(fold.q, cfg.NODE_FEATURE_DIM, cfg.TEMPORAL_FILTERS, cfg.DROPOUT)
            cand = _GRAPH_MODEL_CLASSES[model_name](encoder, fold.graph_edges, n_classes, **kw)
            g = abs(_graph_total_params(cand) - target) / max(1, target)
            if g <= float(cfg.PARAM_MATCH_TOLERANCE) and (not two or g < two[0][0]):
                two = [(g, _graph_total_params(cand), cand)]
        if two:
            gap, actual, model = two[0]
    if gap > float(cfg.PARAM_MATCH_TOLERANCE):
        raise RuntimeError(f'Parameter matching failed for {model_name}: gap={gap:.3f}')
    model.parameter_match_target, model.parameter_match_actual, model.parameter_match_gap = (int(target), int(actual), float(gap))
    return model
if '_BASE_GRAPH_BUILD' not in globals():
    _BASE_GRAPH_BUILD = _v12_build
    _BASE_GRAPH_FAMILY = _v12_family

def _v12_build(model_name, fold, n_classes, hp, extra):
    if model_name in RB_NEW_GRAPH_MODELS:
        model = build_graph_control(model_name, fold, n_classes).to(DEVICE)
        fixed_assert_model(model, DEVICE)
        return model
    return _BASE_GRAPH_BUILD(model_name, fold, n_classes, hp, extra)

def _v12_family(model_name, task):
    return 'graph_control' if model_name in RB_NEW_GRAPH_MODELS else _BASE_GRAPH_FAMILY(model_name, task)
_V13_GRAPH_CONTROLS.update(RB_NEW_GRAPH_MODELS)

def _rb_reported_path(name, subject, model, seed):
    if str(RB_REPORTED_DIR) in ('', '.') or not RB_REPORTED_DIR.is_dir():
        return None
    for folder in [RB_REPORTED_DIR / f'{name}__test_{subject}'] + sorted(RB_REPORTED_DIR.rglob(f'{name}__test_{subject}')):
        path = folder / f'{model}__seed{seed}.pt'
        if path.is_file():
            return path
    return None

def rb_read_reported():
    rows, missing = ([], [])
    for name in RB_DATASETS:
        for subject in _rb_subjects(_rb_bundle(name)):
            for model, label in ((RB_QRG, RB_QRG), (RB_REPORTED_B7, RB_B7)):
                for seed in RB_SEEDS:
                    path = _rb_reported_path(name, subject, model, seed)
                    if path is None:
                        missing.append(f'{name}/{subject}/{model}/seed{seed}')
                        continue
                    row = torch.load(path, map_location='cpu', weights_only=False).get('row', {})
                    rows.append(dict(part='reported', dataset=name, test_subject=str(subject), model=label, seed=int(seed), macro_f1=float(row.get('macro_f1', np.nan)), auroc=float(row.get('auroc', np.nan)), checkpoint=str(path)))
    frame = pd.DataFrame(rows)
    if str(RB_REPORTED_DIR) in ('', '.'):
        print('[reported] No reference-checkpoint directory provided; reference rows and finite-shot inference are unavailable.')
    else:
        print(f'[reported] read {len(rows)} committed rows; {len(missing)} not found' + (f' (first: {missing[:3]})' if missing else ''))
    if not frame.empty:
        frame.to_csv(RB_ROOT / 'reported_rows_read.csv', index=False)
    return frame
RB_BASELINES_CSV = RB_ROOT / 'graph_baselines' / 'per_seed_results.csv'

def rb_run_baselines():
    reused = _rb_reuse([RB_BASELINES_CSV], GRAPH_MODELS)
    if reused is not None:
        if 'part' not in reused:
            reused.insert(0, 'part', 'baselines')
        return reused
    rows, done = ([], set())
    if RB_REUSE_PREVIOUS and RB_BASELINES_CSV.is_file():
        old = pd.read_csv(RB_BASELINES_CSV)
        if not old.empty:
            old['test_subject'] = old['test_subject'].astype(str)
            if 'part' not in old:
                old.insert(0, 'part', 'baselines')
            rows = old.to_dict('records')
            done = {(r['dataset'], str(r['test_subject']), r['model'], int(r['seed'])) for r in rows}
            print(f'[resume] keeping {len(rows)} finished rows from {RB_BASELINES_CSV}', flush=True)
    with _rb_workspace('graph_baselines') as out:
        for name in RB_DATASETS:
            bundle = _rb_bundle(name)
            for subject in _rb_subjects(bundle):
                selected, source = _rb_selected(bundle, subject)
                fold = _rb_outer_fold(bundle, subject, selected)
                print(f'\n[baselines] {name} | patient {subject} | {source}', flush=True)
                qrg_epochs = {'pretrain': cfg.PRETRAIN_EPOCHS, 'stage2': cfg.QUANTUM_EPOCHS, 'finetune': cfg.FINETUNE_EPOCHS}
                for model_name in GRAPH_MODELS:
                    todo = [s for s in RB_SEEDS if (name, str(subject), model_name, int(s)) not in done]
                    if not todo:
                        continue
                    _rb_input_check(build_graph_control(model_name, fold, len(fold.class_names)), fold, model_name)
                    with _rb_search_mode():
                        plan = _v12_plan(bundle, subject, model_name, selected, qrg_epochs, {})
                    for seed in todo:
                        _v13_activate(selected)
                        fit = _rb_fit(model_name, fold, seed, plan, qrg_epochs)
                        row = _rb_row('baselines', name, subject, model_name, seed, fold, selected, plan, fit)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(out / 'per_seed_results.csv', index=False)
                        print(f"  {MODEL_LABELS[model_name]:<13} seed {seed}: Macro-F1 {row['macro_f1']:.3f} | BAcc {row['balanced_accuracy']:.3f} | AUROC {row['auroc']:.3f} | params {row['n_params']} (gap {row['param_gap']:.1%})", flush=True)
                        del fit
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
    return pd.DataFrame(rows)
RB_FORCED_CSV = RB_ROOT / 'forced_per_seed.csv'
RB_FORCED_VARIANTS = {'representation': dict(models=(RB_QRG, RB_B7), override=lambda b: dict(zip(('projection_mode', 'edge_mode'), {'p300': ('xdawn', 'plain'), 'mi': ('pca', 'plain')}[b.task]))), 'qubits_q6': dict(models=(RB_QRG,), override=lambda b: dict(q=6))}
RB_FORCED_VARIANTS = {k: dict(v, models=tuple((m for m in v['models'] if not RB_ONLY_MODELS or m in RB_ONLY_MODELS))) for k, v in RB_FORCED_VARIANTS.items() if k in RB_FORCED_PARTS}
RB_FORCED_VARIANTS = {k: v for k, v in RB_FORCED_VARIANTS.items() if v['models']}

def rb_run_forced():
    rows = []
    for variant, spec in RB_FORCED_VARIANTS.items():
        reused = _rb_reuse([RB_FORCED_CSV], spec['models'], part=variant)
        if reused is not None:
            rows.extend(reused.to_dict('records'))
            continue
        with _rb_workspace(f'forced_{variant}'):
            for name in RB_DATASETS:
                bundle = _rb_bundle(name)
                for subject in _rb_subjects(bundle):
                    selected, source = _rb_selected(bundle, subject)
                    forced = dict(selected, **spec['override'](bundle))
                    if int(forced['q']) > int(bundle.X.shape[1]):
                        print(f"[{variant}] {name}: q={forced['q']} exceeds the channel count; skipped")
                        continue
                    fold = _rb_outer_fold(bundle, subject, forced)
                    K = len(fold.class_names)
                    print(f"\n[{variant}] {name} | patient {subject} | q={forced['q']} {forced['projection_mode']}/{forced['edge_mode']} (reference q={selected['q']} {selected['projection_mode']}/{selected['edge_mode']})", flush=True)
                    for model_name in spec['models']:
                        _rb_input_check(_v12_build(model_name, fold, K, {}, {}), fold, model_name)
                    qrg_plan = _rb_qrg_source_plan(bundle, subject, forced)
                    plans = {RB_QRG: qrg_plan}
                    if RB_B7 in spec['models']:
                        with _rb_search_mode():
                            plans[RB_B7] = _v12_plan(bundle, subject, RB_B7, forced, qrg_plan['epochs'], {})
                    for model_name in spec['models']:
                        for seed in RB_SEEDS:
                            _v13_activate(forced)
                            fit = _rb_fit(model_name, fold, seed, plans[model_name], qrg_plan['epochs'])
                            rows.append(_rb_row(variant, name, subject, model_name, seed, fold, forced, plans[model_name], fit))
                            pd.DataFrame(rows).to_csv(RB_FORCED_CSV, index=False)
                            r = rows[-1]
                            print(f"  {MODEL_LABELS[model_name]:<8} seed {seed}: Macro-F1 {r['macro_f1']:.3f} | BAcc {r['balanced_accuracy']:.3f} | AUROC {r['auroc']:.3f}", flush=True)
                            del fit
                        if DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
RB_SHOTS_CSV = RB_ROOT / 'finite_shot_per_repeat.csv'

@torch.no_grad()
def _rb_predict_shots(model, fold, shots, generator):
    backend = model.circuit.backend
    exact_measure = backend.measure

    def sampled_measure(state):
        prob = state.abs().square().real.clamp_min(0)
        prob = prob / prob.sum(dim=1, keepdim=True)
        idx = torch.multinomial(prob, int(shots), replacement=True, generator=generator)
        freq = torch.zeros_like(prob).scatter_add_(1, idx, torch.ones_like(idx, dtype=prob.dtype)) / float(shots)
        z = freq @ backend.z_signs.T
        return torch.cat([z, freq @ backend.zz_signs.T], dim=1) if len(backend.readout_edges) else z
    if shots is not None:
        backend.measure = sampled_measure
    try:
        ys, probs = ([], [])
        for batch in make_loaders(fold, 256)[2]:
            batch = move_batch(batch, DEVICE)
            probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
            ys.append(batch['y'].cpu().numpy())
    finally:
        backend.measure = exact_measure
    return (np.concatenate(ys), np.concatenate(probs))

def _rb_load_reported_qrg(bundle, subject, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    selected = dict(payload.get('selected') or _rb_selected(bundle, subject)[0])
    selected.setdefault('variant', 'A')
    selected.setdefault('pool', 'avg')
    fold = _rb_outer_fold(bundle, subject, selected)
    saved = [tuple(map(int, e)) for e in payload.get('graph_edges', [])]
    if saved and saved != [tuple(map(int, e)) for e in fold.graph_edges]:
        raise RuntimeError(f'{bundle.name}/{subject}: rebuilt graph differs from the checkpoint graph')
    model = build_qrg_model(fold, len(fold.class_names), 'A', layers=int(selected.get('layers', _v13_layers())))
    model.load_state_dict(payload['state_dict'])
    row, plan = (payload.get('row', {}), payload.get('plan') or {})
    cal = dict(plan.get('calibration') or {})
    for key in ('decision_threshold', 'prior_tau'):
        if key in row and row[key] is not None and np.isfinite(float(row[key])):
            cal[key] = float(row[key])
    cal = {'decision_threshold': float(cal.get('decision_threshold', np.nan) if cal.get('decision_threshold') is not None else np.nan), 'prior_tau': float(cal.get('prior_tau', 0.0) or 0.0)}
    return (model.to(DEVICE).eval(), fold, cal, float(row.get('macro_f1', np.nan)))

def rb_run_shots():
    if RB_REUSE_PREVIOUS and RB_SHOTS_CSV.is_file():
        frame = pd.read_csv(RB_SHOTS_CSV)
        frame['test_subject'] = frame['test_subject'].astype(str)
        if set(frame['shots']) >= set(RB_SHOTS) | {-1}:
            print(f'[reuse] finite-shot results from {RB_SHOTS_CSV}', flush=True)
            return frame
    rows, gaps = ([], [])
    for name in RB_DATASETS:
        bundle = _rb_bundle(name)
        for subject in _rb_subjects(bundle):
            for seed in RB_SEEDS:
                path = _rb_reported_path(name, subject, RB_QRG, seed)
                if path is None:
                    continue
                model, fold, cal, committed = _rb_load_reported_qrg(bundle, subject, path)
                priors = _v11_fold_source_priors(fold.train)
                for shots in (None,) + tuple(RB_SHOTS):
                    for r in range(1 if shots is None else int(RB_SHOT_REPEATS)):
                        gen = torch.Generator(device=DEVICE).manual_seed(int(seed) * 1000 + r)
                        y, prob = _rb_predict_shots(model, fold, shots, gen)
                        row = dict(dataset=name, test_subject=str(subject), seed=int(seed), shots=-1 if shots is None else int(shots), repeat=r)
                        row.update(_rb_metrics(y, prob, fold.task, cal, priors))
                        rows.append(row)
                        if shots is None and np.isfinite(committed):
                            gaps.append(abs(row['macro_f1'] - committed))
                del model
        print(f'[shots] {name}: done', flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        print('[shots] Not run: no reference checkpoint found.')
        return frame
    frame.to_csv(RB_SHOTS_CSV, index=False)
    if gaps:
        print(f'[shots] exact readout vs committed rows: max |ΔMacro-F1| = {max(gaps):.4f} (should be ~0)')
    return frame
RB_BENCH_Q = (4, 6, 8, 10, 12)
RB_BENCH_C = (8, 16, 32, 64, 128, 256)
RB_BENCH_BATCH = 64
RB_BENCH_REPEATS = 20

def _rb_sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def _rb_bench_graph(q):
    edges = [tuple(sorted((i, (i + 1) % q))) for i in range(q)]
    target, k = (int(round(1.5 * q)), 2)
    while len(edges) < target:
        for i in range(q):
            e = tuple(sorted((i, (i + k) % q)))
            if e[0] != e[1] and e not in edges and (len(edges) < target):
                edges.append(e)
        k += 1
    return edges[:target]

def _rb_time_call(fn):
    for _ in range(3):
        fn()
    _rb_sync()
    t0 = time.perf_counter()
    for _ in range(RB_BENCH_REPEATS):
        fn()
    _rb_sync()
    return (time.perf_counter() - t0) / RB_BENCH_REPEATS

def rb_run_efficiency():
    out = {}
    frames = [f for f in (RB_BASELINES, RB_FORCED) if not f.empty and 'wall_seconds' in f]
    trained = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not trained.empty:
        trained = trained[np.isfinite(trained['wall_seconds'])]
        out['runtime'] = trained.groupby(['part', 'model']).agg(fits=('wall_seconds', 'size'), median_seconds=('wall_seconds', 'median'), peak_mem_mb=('peak_mem_mb', 'median'), params=('n_params', 'median'))
    cohorts = []
    for name in RB_DATASETS:
        b = _rb_bundle(name)
        cohorts.append(dict(dataset=name, task=b.task, channels=int(b.X.shape[1]), samples=int(b.X.shape[2]), sfreq=float(b.sfreq), trials=int(b.X.shape[0]), patients=int(len(np.unique(b.subjects)))))
    out['cohorts'] = pd.DataFrame(cohorts)
    T = int(max((c['samples'] for c in cohorts)))
    bench = []
    for q in RB_BENCH_Q:
        edges = _rb_bench_graph(q)
        model = QRGEEG(q, edges, edges, 3, int(cfg.L_DEFAULT), 'A').to(DEVICE)
        batch = {'x_latent': torch.randn(RB_BENCH_BATCH, q, T, device=DEVICE), 'edges': torch.randn(RB_BENCH_BATCH, len(edges), device=DEVICE)}
        y = torch.randint(0, 3, (RB_BENCH_BATCH,), device=DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.eval()
        with torch.no_grad():
            infer = _rb_time_call(lambda: model(batch))

        def step():
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(batch), y).backward()
            opt.step()
        model.train()
        train = _rb_time_call(step)
        bench.append(dict(q=q, edges=len(edges), two_qubit_gates=int(cfg.L_DEFAULT) * len(edges), readout=q + len(edges), params=int(sum((p.numel() for p in model.parameters()))), infer_ms_per_trial=1000.0 * infer / RB_BENCH_BATCH, train_ms_per_batch=1000.0 * train, peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == 'cuda' else np.nan))
        del model, opt, batch
    out['bench_q'] = pd.DataFrame(bench)
    proj = []
    for C in RB_BENCH_C:
        X = torch.randn(RB_BENCH_BATCH, C, T, device=DEVICE)
        P = torch.randn(C, 8, device=DEVICE)
        proj.append(dict(channels=C, projection_ms_per_trial=1000.0 * _rb_time_call(lambda: torch.einsum('cq,bct->bqt', P, X)) / RB_BENCH_BATCH))
    out['bench_c'] = pd.DataFrame(proj)
    for key, frame in out.items():
        frame.to_csv(RB_ROOT / f'efficiency_{key}.csv')
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        for key, frame in out.items():
            print(f'\n[efficiency] {key}:\n{frame.to_string()}')
    return out

def save_experiment_artifacts(row, fold, selected, plan, fit):
    """Persist measured held-out predictions and their source-selected plan."""
    folder = RUN_DIR / 'evaluation_artifacts' / f"{row['dataset']}__test_{row['test_subject']}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{row['model']}__seed{int(row['seed'])}"
    probability, threshold = _v12_apply_decision(np.asarray(fit['prob_raw']), fold.task, plan['calibration'], fit['source_priors'])
    fixed_atomic_npz(folder / (stem + '.npz'), {'y_true': np.asarray(fit['y'], dtype=np.int64), 'prob_raw': np.asarray(fit['prob_raw']), 'prob': np.asarray(probability), 'source_priors': np.asarray(fit['source_priors']), 'decision_threshold': np.asarray(float(threshold) if threshold is not None else np.nan), 'subject': np.asarray(fold.test.subject_names).astype(str)})
    fixed_atomic_json(folder / (stem + '.json'), _v11_json_ready({'row': row, 'selected': selected, 'plan': plan, 'fit_audit': fold.fit_audit, 'outer_test_labels_used_for_selection': False}))
    if fit.get('model') is not None:
        destination = folder / (stem + '.pt')
        temporary = destination.with_suffix('.pt.tmp')
        torch.save({'state_dict': fit['model'].state_dict(), 'selected': selected, 'plan': plan, 'row': row, 'graph_edges': fold.graph_edges}, temporary)
        os.replace(temporary, destination)

def initialize_experiment():
    global RUN_DIR, V12_OUT, store, _FIXED_ROOT, _FIXED_LOCK_HANDLE
    global EXPERIMENT_FINGERPRINT, RUN_MANIFEST, _FIXED_RUNTIME_MANIFEST
    from importlib.metadata import version as package_version
    packages = {name: package_version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'mne', 'mat73', 'pyriemann', 'networkx', 'h5py', 'pandas')}
    _FIXED_RUNTIME_MANIFEST = {'python': sys.version.split()[0], 'packages': packages, 'cuda': torch.version.cuda, 'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None}
    excluded = {'PROJECT_DIR', 'DATA_ROOT', 'RESULTS_ROOT', 'RUN_NAME'}
    scientific = {key: value for key, value in asdict(cfg).items() if key not in excluded}
    contract = {'source_sha256': SOURCE_SHA256, 'configuration': scientific, 'plan': experiment_plan(ARGS), 'environment': _FIXED_RUNTIME_MANIFEST}
    EXPERIMENT_FINGERPRINT = fixed_digest(contract)
    RUN_MANIFEST = {'experiment_fingerprint': EXPERIMENT_FINGERPRINT, **contract}
    RUN_DIR = cfg.RESULTS_ROOT / 'runtime_state'
    V12_OUT = RUN_DIR / 'paper_inputs'
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _FIXED_ROOT = RUN_DIR / 'runtime'
    _FIXED_ROOT.mkdir(exist_ok=True)
    RB_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RB_ROOT / 'experiment_manifest.json'
    expected = json.loads(fixed_canonical(RUN_MANIFEST))
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected:
        raise RuntimeError('The output directory belongs to a different experiment configuration.')
    fixed_atomic_json(manifest_path, expected)
    handle = (_FIXED_ROOT / 'execution.lock').open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another process is using this output directory.')
    _FIXED_LOCK_HANDLE = handle
    store = ResultStore(RUN_DIR)
    print('Device:', DEVICE, '| participants:', ARGS.scope, '| seeds:', RB_SEEDS, flush=True)
    print('Outputs:', RB_ROOT, flush=True)
cfg.MAX_SUBJECTS = None
cfg.MAX_OUTER_FOLDS = None
cfg.MAX_TRIALS_PER_SUBJECT = None
BASELINE_CODES = GRAPH_MODELS
"""Complete participant-level evaluation and resumable artifact exports."""

ROOT_EXPERIMENT_FINGERPRINT = None
CANONICAL_QRG_ROOT = RB_ROOT / 'primary_checkpoints'


def _v12_qrg_checkpoint(bundle_name, test_subject, seed):
    return CANONICAL_QRG_ROOT / f'{bundle_name}__test_{test_subject}' / f'B8_QRG_EEG__seed{int(seed)}.pt'


def all_subjects(bundle):
    subjects = sorted(set(map(str, bundle.subjects)), key=str)
    required = EXPECTED_SUBJECTS[bundle.name]
    if len(subjects) != required:
        raise DatasetSchemaError(f'{bundle.name}: expected {required} participants, found {len(subjects)}.')
    return subjects


@contextlib.contextmanager
def configuration_workspace(configuration_id):
    global RUN_DIR, _FIXED_ROOT, EXPERIMENT_FINGERPRINT
    old = RUN_DIR, _FIXED_ROOT, EXPERIMENT_FINGERPRINT
    RUN_DIR = RB_ROOT / 'configurations' / configuration_id
    _FIXED_ROOT = RUN_DIR / 'runtime'
    _FIXED_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_FINGERPRINT = fixed_digest({'root': ROOT_EXPERIMENT_FINGERPRINT,
                                         'configuration': configuration_id})
    try:
        yield
    finally:
        RUN_DIR, _FIXED_ROOT, EXPERIMENT_FINGERPRINT = old


def prepare_all_subject_experiment():
    global ROOT_EXPERIMENT_FINGERPRINT
    initialize_experiment()
    ROOT_EXPERIMENT_FINGERPRINT = EXPERIMENT_FINGERPRINT
    session = {'subjects': {}, 'rows': {}, 'failures': {}, 'source_plans': {}}
    try:
        for dataset in DATASET_CHOICES:
            bundle = _rb_bundle(dataset)
            subjects = all_subjects(bundle)
            session['subjects'][dataset] = subjects
            print(f'{dataset}: {len(subjects)} participants; {len(bundle.y)} trials; '
                  f'{len(bundle.class_names)} classes', flush=True)
        total = sum(map(len, session['subjects'].values()))
        if total != 30:
            raise DatasetSchemaError(f'Expected 30 participants across the four datasets, found {total}.')
        session['expected'] = {(spec['id'], dataset, subject, int(seed))
            for spec in MODEL_REGISTRY for dataset, subjects in session['subjects'].items()
            for subject in subjects for seed in SEEDS}
        planned = experiment_plan(ARGS)
        if len(session['expected']) != planned['final_evaluations']:
            raise RuntimeError('The loaded participant registry does not match the declared experiment plan.')
        fixed_atomic_json(RB_ROOT / 'participant_registry.json', session['subjects'])
        pd.DataFrame(MODEL_REGISTRY).to_csv(RB_ROOT / 'model_registry.csv', index=False)
        print(f'READY: {len(MODEL_REGISTRY)} configurations × 30 participants × 3 seeds '
              f'= {len(session["expected"])} evaluations.', flush=True)
        return session
    except BaseException:
        fixed_release_lock()
        raise
    finally:
        # The preflight stage may be rerun; the training command acquires its own lock.
        fixed_release_lock()


def job_directory(key):
    configuration, dataset, subject, seed = key
    safe_subject = re.sub(r'[^A-Za-z0-9_.-]', '_', subject)
    return RB_ROOT / 'evaluations' / configuration / dataset / ('subject_' + safe_subject) / f'seed_{int(seed)}'


def job_contract(key):
    return {'experiment': ROOT_EXPERIMENT_FINGERPRINT,
            'configuration': key[0], 'dataset': key[1], 'test_subject': key[2], 'seed': key[3],
            'dataset_identity': fixed_bundle_identity(loaded_datasets[key[1]])}


def read_completed_job(key):
    folder = job_directory(key)
    record = folder / 'result.json'
    if not record.is_file():
        return None
    payload = json.loads(record.read_text())
    if payload.get('status') != 'success' or payload.get('contract') != job_contract(key):
        raise RuntimeError(f'Incompatible completed evaluation: {key}. Use a separate output directory.')
    for artifact in payload['artifacts']:
        path = RB_ROOT / artifact['path']
        if not path.is_file() or fixed_sha256(path) != artifact['sha256']:
            raise RuntimeError(f'A saved evaluation artifact is missing or corrupt: {key}.')
    return payload['row']


def persist_evaluation(key, spec, fold, selected, plan, fit):
    folder = job_directory(key)
    folder.mkdir(parents=True, exist_ok=True)
    probability, threshold = _v12_apply_decision(np.asarray(fit['prob_raw']), fold.task,
                                               plan['calibration'], fit['source_priors'])
    row = {'configuration': spec['id'], 'part': spec['part'], 'model': spec['model'],
           'label': spec['label'], 'dataset': key[1], 'task': fold.task,
           'test_subject': key[2], 'seed': key[3], 'status': 'success',
           'evaluation_type': 'saved_checkpoint_inference' if spec['part'] == 'test_only' else 'fitted',
           'q': int(fold.q), 'layers': int(selected.get('layers', 2)),
           'graph_multiplier': float(selected['graph_multiplier']),
           'projection_mode': selected['projection_mode'], 'edge_mode': selected['edge_mode'],
           'n_test': len(fit['y']), 'n_params': int(fit['n_params']),
           'parameter_match_gap': float(fit.get('parameter_match_gap', np.nan)),
           'wall_seconds': float(fit.get('wall_seconds', np.nan)),
           'peak_mem_mb': float(fit.get('peak_mem_mb', np.nan)),
           'decision_threshold': float(threshold) if threshold is not None else np.nan,
           'prior_tau': float(plan['calibration'].get('prior_tau', 0.0) or 0.0),
           'transductive': spec['model'] == 'EA_EEGNET',
           'outer_test_labels_used_for_selection': False}
    row.update(_rb_metrics(fit['y'], fit['prob_raw'], fold.task,
                           plan['calibration'], fit['source_priors']))
    for metric in ('macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'auroc'):
        if not np.isfinite(row[metric]):
            raise FloatingPointError(f'Non-finite held-out {metric}: {key}')
    pred = (np.asarray(probability)[:, 1] >= threshold).astype(int) if threshold is not None else np.asarray(probability).argmax(1)
    row['class_f1'] = json.dumps(f1_score(fit['y'], pred, labels=np.arange(probability.shape[1]),
                                        average=None, zero_division=0).tolist())
    prediction_path = folder / 'predictions.npz'
    fixed_atomic_npz(prediction_path, {'y_true': np.asarray(fit['y'], np.int64),
        'prob_raw': np.asarray(fit['prob_raw'], np.float64), 'prob': np.asarray(probability, np.float64),
        'y_pred': pred.astype(np.int64), 'source_priors': np.asarray(fit['source_priors']),
        'decision_threshold': np.asarray(float(threshold) if threshold is not None else np.nan),
        'subject': np.asarray(fold.test.subject_names).astype(str)})
    artifacts = [prediction_path]
    if cfg.SAVE_CHECKPOINTS and fit.get('model') is not None:
        destination = _v12_qrg_checkpoint(key[1], key[2], key[3]) if spec['id'] == 'B8_QRG_EEG' else folder / 'model.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.pt.tmp')
        torch.save({'state_dict': fit['model'].state_dict(), 'selected': selected, 'plan': plan,
                    'row': row, 'graph_edges': fold.graph_edges,
                    'experiment_fingerprint': ROOT_EXPERIMENT_FINGERPRINT}, temporary)
        os.replace(temporary, destination)
        artifacts.append(destination)
    if fit.get('estimator') is not None:
        import joblib
        destination = folder / 'estimator.joblib'
        joblib.dump(fit['estimator'], destination)
        artifacts.append(destination)
    payload = {'status': 'success', 'contract': job_contract(key), 'row': row,
               'selected': selected, 'plan': plan,
               'artifacts': [{'path': str(path.relative_to(RB_ROOT)), 'sha256': fixed_sha256(path)}
                             for path in artifacts]}
    fixed_atomic_json(folder / 'result.json', _v11_json_ready(payload))
    (folder / 'failure.json').unlink(missing_ok=True)
    return row


def choose_configuration(spec, bundle, base_selected):
    selected = dict(base_selected)
    model = spec['model']
    if spec['part'] == 'representation':
        selected.update(projection_mode='xdawn' if bundle.task == 'p300' else 'pca', edge_mode='plain')
    if spec['part'] == 'qubits_q6':
        selected['q'] = 6
    if model in ('QRG_DIAG_Q6', 'QRG_DIAG_Q8'):
        selected['q'] = 6 if model.endswith('Q6') else 8
    if model in ('QRG_DIAG_L2', 'QRG_DIAG_L4'):
        selected['layers'] = 2 if model.endswith('L2') else 4
    if selected['q'] > bundle.X.shape[1]:
        raise ValueError(f'{bundle.name}/{spec["id"]}: q exceeds the available channel count.')
    return selected


def shared_source_plan(session, bundle, subject, selected):
    key = (bundle.name, subject, int(selected['q']), str(selected['projection_mode']),
           str(selected['edge_mode']), int(selected['layers']),
           float(selected['graph_multiplier']), str(selected.get('pool', 'avg')))
    if key not in session['source_plans']:
        with configuration_workspace('shared_source_plans'):
            session['source_plans'][key] = _rb_qrg_source_plan(bundle, subject, selected)
    return copy.deepcopy(session['source_plans'][key])


def source_plan_for_model(spec, bundle, subject, selected, qrg_plan):
    model = spec['model']
    if model in ('B8_QRG_EEG', 'QRG_DIAG_Q6', 'QRG_DIAG_Q8', 'QRG_DIAG_L2', 'QRG_DIAG_L4'):
        return copy.deepcopy(qrg_plan)
    return _v12_plan(bundle, subject, model, selected, qrg_plan['epochs'], {})


def fit_evaluation(spec, bundle, subject, seed, fold, selected, plan, qrg_plan):
    model = spec['model']
    seed_everything(int(seed))
    _v13_activate(selected)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if spec['part'] == 'test_only':
        fit = _v12_run_test_only(model, fold, bundle.name, subject, seed, qrg_plan['calibration'])
        plan = fit['plan']
    elif _v12_family(model, bundle.task) == 'frozen':
        fit = _v12_run_frozen(model, fold, bundle.name, subject, seed)
        plan = fit['plan']
    else:
        extra = {}
        if model.startswith('C7_SHUFFLED_TT_P'):
            permutations, excluded, _ = _v12_edge_permutations(
                fold.graph_edges, fold.q, cfg.V12_N_PERMUTATIONS,
                f'{bundle.name}|{subject}|final')
            extra['permutation'] = permutations[_v12_index(model) % len(permutations)]
        fit = _v12_fit(model, fold, seed, plan['hp'], 'final',
                      epochs=plan['epochs'], qrg_plan=qrg_plan['epochs'], extra=extra)
        if model in {'B6_SAME_GRAPH_GAT', 'B7_MATCHED_CLASSICAL_INTERACTION', 'B12_TRIG_SURROGATE'} | set(RB_NEW_GRAPH_MODELS):
            gap = float(fit.get('parameter_match_gap', np.nan))
            if not np.isfinite(gap) or gap > cfg.PARAM_MATCH_TOLERANCE:
                raise RuntimeError(f'{model}: parameter-match tolerance exceeded ({gap}).')
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    fit['wall_seconds'] = time.perf_counter() - start
    fit['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == 'cuda' else np.nan
    return fit, plan


def export_all_subject_results(session):
    records = [dict(row) for _, row in sorted(session['rows'].items())]
    frame = pd.DataFrame(records)
    key_columns = ['configuration', 'dataset', 'test_subject', 'seed']
    if not frame.empty and frame.duplicated(key_columns).any():
        raise RuntimeError('Duplicate participant/seed result keys.')
    for key in session['rows']:
        if key not in session['expected']:
            raise RuntimeError('An unexpected result is present in the output directory.')
    if frame.empty:
        frame = pd.DataFrame(columns=key_columns + ['model', 'status'] + list(EXPORTED_METRICS))
    for metric in EXPORTED_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors='coerce').astype(float)
    _atomic_csv(RB_ROOT / 'per_seed_results.csv', frame)
    per_subject = []
    if not frame.empty:
        for key, group in frame.groupby(['configuration', 'dataset', 'test_subject'], sort=True):
            row = dict(zip(['configuration', 'dataset', 'test_subject'], key))
            row.update(n_seeds=len(group), complete=set(group.seed.astype(int)) == set(SEEDS))
            for metric in EXPORTED_METRICS:
                values = pd.to_numeric(group[metric], errors='coerce').to_numpy(float)
                valid = row['complete'] and len(values) == 3 and np.isfinite(values).all()
                row[metric + '_mean'] = float(values.mean()) if valid else np.nan
                row[metric + '_sd'] = float(values.std(ddof=1)) if valid else np.nan
            per_subject.append(row)
    subject_summary = pd.DataFrame(per_subject)
    _atomic_csv(RB_ROOT / 'per_subject_mean_sd.csv', subject_summary)
    dataset_per_seed, participant_summary, seed_summary = [], [], []
    for spec in MODEL_REGISTRY:
        for dataset in DATASET_CHOICES:
            subset = frame[(frame.configuration == spec['id']) & (frame.dataset == dataset)]
            n_subjects = len(session['subjects'][dataset])
            for seed in SEEDS:
                group = subset[subset.seed.astype(int) == seed]
                entry = dict(configuration=spec['id'], dataset=dataset, seed=seed,
                             n_subjects=len(group), complete=len(group) == n_subjects)
                for metric in EXPORTED_METRICS:
                    values = pd.to_numeric(group[metric], errors='coerce').to_numpy(float)
                    entry[metric] = float(values.mean()) if entry['complete'] and np.isfinite(values).all() else np.nan
                dataset_per_seed.append(entry)
            summary = dict(configuration=spec['id'], dataset=dataset, n_subjects=n_subjects,
                           complete=len(subset) == n_subjects * 3)
            by_subject = subset.groupby('test_subject')[list(EXPORTED_METRICS)].mean(numeric_only=True)
            for metric in EXPORTED_METRICS:
                values = by_subject[metric].to_numpy(float)
                valid = summary['complete'] and len(values) == n_subjects and np.isfinite(values).all()
                summary[metric + '_mean'] = float(values.mean()) if valid else np.nan
                summary[metric + '_sd_across_subjects'] = float(values.std(ddof=1)) if valid else np.nan
            participant_summary.append(summary)
            entry = dict(configuration=spec['id'], dataset=dataset, n_seeds=3,
                         n_subjects=n_subjects, complete=summary['complete'])
            for metric in EXPORTED_METRICS:
                values = np.array([r[metric] for r in dataset_per_seed[-3:]], float)
                valid = summary['complete'] and np.isfinite(values).all()
                entry[metric + '_mean'] = float(values.mean()) if valid else np.nan
                entry[metric + '_sd_across_seed_means'] = float(values.std(ddof=1)) if valid else np.nan
            seed_summary.append(entry)
    _atomic_csv(RB_ROOT / 'dataset_per_seed_means.csv', pd.DataFrame(dataset_per_seed))
    _atomic_csv(RB_ROOT / 'dataset_subject_mean_sd.csv', pd.DataFrame(participant_summary))
    _atomic_csv(RB_ROOT / 'dataset_seed_mean_sd.csv', pd.DataFrame(seed_summary))
    statuses = []
    for key in sorted(session['expected']):
        row = dict(zip(key_columns, key))
        row['status'] = 'success' if key in session['rows'] else 'failed' if key in session['failures'] else 'pending'
        row['error'] = session['failures'].get(key, '')
        statuses.append(row)
    _atomic_csv(RB_ROOT / 'job_status.csv', pd.DataFrame(statuses))
    coverage = []
    for spec in MODEL_REGISTRY:
        count = sum(k[0] == spec['id'] for k in session['rows'])
        coverage.append(dict(configuration=spec['id'], expected=90, successful=count, complete=count == 90))
    _atomic_csv(RB_ROOT / 'model_coverage.csv', pd.DataFrame(coverage))
    missing = sorted(session['expected'] - set(session['rows']))
    fixed_atomic_json(RB_ROOT / 'completion.json', {'complete': not missing,
        'expected_evaluations': len(session['expected']), 'successful_evaluations': len(session['rows']),
        'participants': 30, 'seeds': list(SEEDS), 'evaluations_per_configuration': 90,
        'configurations': len(MODEL_REGISTRY), 'missing': missing,
        'seed_sd_ddof': 1, 'subject_sd_ddof': 1})
    lines = ['# Measured dataset results', '',
             'Values use a 0–1 scale. The SD below is across participant means after averaging three seeds per participant.',
             'Seed variation is exported separately in dataset_seed_mean_sd.csv.', '',
             '| Configuration | Dataset | Macro-F1 | Accuracy | Balanced accuracy | AUROC |',
             '|---|---|---|---|---|---|']
    for row in participant_summary:
        def cell(metric):
            mean, sd = row[metric + '_mean'], row[metric + '_sd_across_subjects']
            return f'{mean:.4f} ± {sd:.4f}' if np.isfinite(mean) and np.isfinite(sd) else 'incomplete or not applicable'
        lines.append('| ' + ' | '.join([row['configuration'], row['dataset']] +
            [cell(m) for m in ('macro_f1', 'accuracy', 'balanced_accuracy', 'auroc')]) + ' |')
    _atomic_text(RB_ROOT / 'results.md', '\n'.join(lines) + '\n')
    return frame


def record_failure(session, key, error):
    message = f'{type(error).__name__}: {error}'
    session['failures'][key] = message
    folder = job_directory(key)
    folder.mkdir(parents=True, exist_ok=True)
    fixed_atomic_json(folder / 'failure.json', {'status': 'failed', 'contract': job_contract(key),
                                             'error': message, 'traceback': traceback.format_exc()})
    print(f'FAILED {key}: {message}', flush=True)


def run_all_subject_experiment(session):
    global ROOT_EXPERIMENT_FINGERPRINT, RB_BASELINES, RB_FORCED, RB_REPORTED_DIR
    initialize_experiment()
    ROOT_EXPERIMENT_FINGERPRINT = EXPERIMENT_FINGERPRINT
    try:
        for key in sorted(session['expected']):
            row = read_completed_job(key)
            if row is not None:
                session['rows'][key] = row
                session['failures'].pop(key, None)
        export_all_subject_results(session)
        print(f'Resuming: {len(session["rows"])}/{len(session["expected"])} evaluations already complete.', flush=True)
        for dataset in DATASET_CHOICES:
            bundle = _rb_bundle(dataset)
            for subject in session['subjects'][dataset]:
                if all((spec['id'], dataset, subject, seed) in session['rows'] for spec in MODEL_REGISTRY for seed in SEEDS):
                    continue
                try:
                    if SUITE == 'full':
                        selected, epochs = select_qrg_source_only(bundle, subject)
                        main_plan = dict(hp={}, epochs=epochs, calibration={
                            'decision_threshold': float(selected.get('decision_threshold', np.nan)),
                            'prior_tau': float(selected.get('prior_tau', 0.0) or 0.0)})
                    else:
                        selected, _ = _rb_selected(bundle, subject)
                        main_plan = None
                except Exception as error:
                    for spec in MODEL_REGISTRY:
                        for seed in SEEDS:
                            key = spec['id'], dataset, subject, seed
                            if key not in session['rows']:
                                record_failure(session, key, error)
                    export_all_subject_results(session)
                    continue
                for spec in MODEL_REGISTRY:
                    pending = [seed for seed in SEEDS if (spec['id'], dataset, subject, seed) not in session['rows']]
                    if not pending:
                        continue
                    try:
                        chosen = choose_configuration(spec, bundle, selected)
                        needs_qrg = SUITE == 'full' or spec['part'] != 'baselines'
                        changed = any(chosen.get(k) != selected.get(k) for k in
                                      ('q', 'layers', 'graph_multiplier', 'pool', 'projection_mode', 'edge_mode'))
                        qrg_plan = shared_source_plan(session, bundle, subject, chosen) if changed or main_plan is None and needs_qrg else copy.deepcopy(main_plan)
                        if qrg_plan is None:
                            qrg_plan = dict(hp={}, epochs=dict(pretrain=cfg.PRETRAIN_EPOCHS,
                                stage2=cfg.QUANTUM_EPOCHS, finetune=cfg.FINETUNE_EPOCHS), calibration={})
                        with configuration_workspace(spec['id']):
                            fold = _rb_outer_fold(bundle, subject, chosen)
                            fold = _v12_transform_fold(spec['model'], fold, f'{dataset}|{subject}')
                            if spec['model'] in GRAPH_MODELS or spec['part'] in ('representation', 'qubits_q6'):
                                _rb_input_check(_v12_build(spec['model'], fold, len(fold.class_names), {}, {}), fold, spec['id'])
                            plan = None if spec['part'] == 'test_only' or _v12_family(spec['model'], bundle.task) == 'frozen' else source_plan_for_model(spec, bundle, subject, chosen, qrg_plan)
                            for seed in pending:
                                key = spec['id'], dataset, subject, seed
                                print(f'[{len(session["rows"])+1}/{len(session["expected"])}] '
                                      f'{dataset} / {subject} / {spec["id"]} / seed {seed}', flush=True)
                                try:
                                    with fixed_operation('held_out_evaluation', dataset=dataset,
                                                         test_subject=subject, model=spec['model'], seed=int(seed)):
                                        fit, used_plan = fit_evaluation(spec, bundle, subject, seed, fold, chosen, plan, qrg_plan)
                                        row = persist_evaluation(key, spec, fold, chosen, used_plan, fit)
                                    session['rows'][key] = row
                                    session['failures'].pop(key, None)
                                    print(f'  Macro-F1={row["macro_f1"]:.4f}  BAcc={row["balanced_accuracy"]:.4f}  AUROC={row["auroc"]:.4f}', flush=True)
                                    del fit
                                except Exception as error:
                                    record_failure(session, key, error)
                                finally:
                                    gc.collect()
                                    if DEVICE.type == 'cuda':
                                        torch.cuda.empty_cache()
                    except Exception as error:
                        for seed in pending:
                            key = spec['id'], dataset, subject, seed
                            if key not in session['rows']:
                                record_failure(session, key, error)
                    export_all_subject_results(session)
                # Retain one cohort's fold inputs at a time; completed models remain on disk.
                _FIXED_FOLD_CACHE.clear()
                global _FIXED_FOLD_CACHE_BYTES
                _FIXED_FOLD_CACHE_BYTES = 0
        frame = export_all_subject_results(session)
        missing = session['expected'] - set(session['rows'])
        if missing:
            raise RuntimeError(f'Incomplete: {len(session["rows"])}/{len(session["expected"])}. '
                               'See job_status.csv, correct the logged issue, and rerun the training command; successful jobs are reused.')
        if RUN_FINITE_SHOTS:
            if SUITE == 'full':
                RB_REPORTED_DIR = CANONICAL_QRG_ROOT
            if str(RB_REPORTED_DIR) in ('', '.'):
                raise ValueError('Finite-shot inference requires REFERENCE_CHECKPOINTS in the six-configuration notebook.')
            absent = [(d, s, seed) for d, subjects in session['subjects'].items() for s in subjects for seed in SEEDS
                      if _rb_reported_path(d, s, RB_QRG, seed) is None]
            if absent:
                raise FileNotFoundError('Finite-shot checkpoints are missing for: ' + str(absent))
            rb_run_shots()
        if RUN_EFFICIENCY:
            RB_BASELINES = frame[frame.part == 'baselines'].copy()
            RB_FORCED = frame[frame.part != 'baselines'].copy()
            RB_BASELINES['peak_mem_mb'] = RB_BASELINES.get('peak_mem_mb', np.nan)
            rb_run_efficiency()
        print(f'COMPLETE: {len(frame)}/{len(session["expected"])}; '
              f'{len(MODEL_REGISTRY)} configurations, 30 participants, three seeds.', flush=True)
        print('Results:', RB_ROOT, flush=True)
        return frame
    finally:
        export_all_subject_results(session)
        fixed_release_lock()


def main():
    global RUN_DIR
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not ARGS.check_only and visible != '0':
        raise RuntimeError('Use CUDA_VISIBLE_DEVICES=0 to select physical GPU 0.')
    if ARGS.check_only:
        RUN_DIR = Path(cfg.RESULTS_ROOT)/'input_checks'
        RUN_DIR.mkdir(parents=True,exist_ok=True)
        for name in DATASET_CHOICES:
            bundle = _rb_bundle(name)
            people = all_subjects(bundle)
            print(f'{name}: {len(people)} participants; {len(bundle.y)} trials; {bundle.X.shape[1:]} EEG shape',flush=True)
        print('Local dataset content checks passed.',flush=True)
        return
    session = prepare_all_subject_experiment()
    return run_all_subject_experiment(session)


if __name__=='__main__':
    main()
