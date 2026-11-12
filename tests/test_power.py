import pytest

from infer_lab.power import Sample, energy_joules


def _s(t, gpu, watts):
    return Sample(t_s=t, gpu=gpu, power_w=watts, util_pct=0.0, mem_used_gb=0.0)


def test_constant_power():
    assert energy_joules([_s(0, 0, 100), _s(1, 0, 100), _s(2, 0, 100)]) == pytest.approx(200.0)


def test_linear_ramp_uses_trapezoids():
    assert energy_joules([_s(0, 0, 0), _s(1, 0, 100)]) == pytest.approx(50.0)


def test_gpus_are_integrated_separately_then_summed():
    samples = [_s(t, gpu, 100) for t in (0, 1, 2) for gpu in (0, 1)]
    assert energy_joules(samples) == pytest.approx(400.0)


def test_unsorted_samples():
    assert energy_joules([_s(2, 0, 100), _s(0, 0, 100), _s(1, 0, 100)]) == pytest.approx(200.0)


def test_too_few_samples_is_zero():
    assert energy_joules([]) == 0
    assert energy_joules([_s(0, 0, 100)]) == 0
