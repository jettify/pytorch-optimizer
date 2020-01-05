import torch


def assert_dict_equal(a, b, precision=0.000001):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            assert_dict_equal(a[k], b[k], precision)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for v1, v2 in zip(a, b):
            assert_dict_equal(v1, v2, precision)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert torch.allclose(a, b, atol=precision)
    else:
        assert a == b
    return True
