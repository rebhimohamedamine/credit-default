import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.hardware import hardware_report, probe_gbm_gpu, time_lightgbm_cpu_vs_gpu

ROOT = Path(__file__).resolve().parent.parent


def main():
    hw = hardware_report()
    hw['gbm_gpu_probe'] = probe_gbm_gpu()

    rng = np.random.default_rng(42)
    X = rng.random((50_000, 10))
    y = (X[:, 0] > 0.5).astype(int)
    hw['lightgbm_cpu_vs_gpu_50k_rows'] = time_lightgbm_cpu_vs_gpu(X, y)

    (ROOT / 'hardware.json').write_text(json.dumps(hw, indent=2))

    with open(ROOT / 'requirements_freeze.txt', 'w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=False)

    print('=' * 60)
    print('HARDWARE BANNER')
    print('=' * 60)
    print(f"OS: {hw['os']}  |  CPU cores: {hw['cpu_count']}  |  RAM: {hw.get('ram_gb')} GB")
    if hw['cuda_available']:
        print(f"GPU: {hw['gpu']}  |  VRAM: {hw['vram_gb']} GB  |  torch cuda: available")
    else:
        print('GPU: none detected or torch built without CUDA support.')
        print('  -> Consequence: foundation models limited/skipped, fewer Optuna trials, smaller MLP batch.')
    print('GBM GPU probe:', json.dumps(hw['gbm_gpu_probe'], indent=2))
    print('LightGBM CPU vs GPU (50k-row sample):', json.dumps(hw['lightgbm_cpu_vs_gpu_50k_rows'], indent=2))
    print('=' * 60)
    print(f"Wrote {ROOT / 'hardware.json'}")


if __name__ == '__main__':
    main()
