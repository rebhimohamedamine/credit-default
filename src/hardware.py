import importlib.util
import os
import platform
import shutil
import subprocess
import time

import numpy as np


def hardware_report() -> dict:
    info = {'os': platform.platform(), 'cpu_count': os.cpu_count()}

    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info['ram_gb'] = None

    if shutil.which('nvidia-smi'):
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True,
        )
        info['nvidia_smi'] = out.stdout.strip()
    else:
        info['nvidia_smi'] = None

    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = bool(torch.cuda.is_available())
        if info['cuda_available']:
            props = torch.cuda.get_device_properties(0)
            info['gpu'] = props.name
            info['vram_gb'] = round(props.total_memory / 1e9, 2)
        else:
            info['gpu'] = None
            info['vram_gb'] = None
        mps = getattr(torch.backends, 'mps', None)
        info['mps_available'] = bool(mps and mps.is_available())
    except ImportError:
        info['torch'] = None
        info['cuda_available'] = False
        info['gpu'] = None
        info['vram_gb'] = None
        info['mps_available'] = False

    return info


def probe_gbm_gpu() -> dict:
    rng = np.random.default_rng(42)
    X = rng.random((2000, 10))
    y = (X[:, 0] > 0.5).astype(int)
    results = {}

    def attempt(name, fn):
        try:
            start = time.time()
            fn()
            results[name] = round(time.time() - start, 2)
        except Exception as exc:
            results[name] = f'unavailable: {type(exc).__name__}'

    if importlib.util.find_spec('xgboost'):
        import xgboost as xgb
        attempt('xgboost_gpu', lambda: xgb.XGBClassifier(n_estimators=5, device='cuda').fit(X, y))

    if importlib.util.find_spec('lightgbm'):
        import lightgbm as lgb
        attempt('lightgbm_cuda', lambda: lgb.LGBMClassifier(n_estimators=5, device_type='cuda', verbose=-1).fit(X, y))
        attempt('lightgbm_gpu', lambda: lgb.LGBMClassifier(n_estimators=5, device_type='gpu', verbose=-1).fit(X, y))

    if importlib.util.find_spec('catboost'):
        from catboost import CatBoostClassifier
        attempt('catboost_gpu', lambda: CatBoostClassifier(iterations=5, task_type='GPU', verbose=0).fit(X, y))

    return results


def time_lightgbm_cpu_vs_gpu(X, y, sample_size: int = 50_000) -> dict:
    import lightgbm as lgb

    n = min(sample_size, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample, y_sample = X[idx], y[idx]

    timings = {}
    for device in ('cpu', 'gpu', 'cuda'):
        try:
            start = time.time()
            lgb.LGBMClassifier(n_estimators=100, device_type=device, verbose=-1).fit(X_sample, y_sample)
            timings[device] = round(time.time() - start, 2)
        except Exception as exc:
            timings[device] = f'unavailable: {type(exc).__name__}'
    return timings
