from src.hardware import hardware_report, probe_gbm_gpu


def test_hardware_report_has_required_keys():
    hw = hardware_report()
    for key in ['os', 'cpu_count', 'torch', 'cuda_available']:
        assert key in hw


def test_gbm_gpu_probe_returns_dict_of_known_libs():
    result = probe_gbm_gpu()
    assert set(result) <= {'xgboost_gpu', 'lightgbm_cuda', 'lightgbm_gpu', 'catboost_gpu'}
    for v in result.values():
        assert isinstance(v, float) or (isinstance(v, str) and v.startswith('unavailable'))
