import pytest

from inheritance_math import CoherenceLayer, LayerStatus, average, default_layers
from inheritance_math.layers import LAYER_NAMES


def test_status_derivation():
    assert CoherenceLayer("data", 0.95).status == LayerStatus.INTACT
    assert CoherenceLayer("method", 0.60).status == LayerStatus.DEGRADED
    assert CoherenceLayer("intent", 0.10).status == LayerStatus.LOST


def test_fidelity_bounds_validation():
    with pytest.raises(ValueError):
        CoherenceLayer("data", 1.5)
    with pytest.raises(ValueError):
        CoherenceLayer("data", -0.1)


def test_layer_index_and_fragility():
    assert CoherenceLayer("physical", 1.0).index == 1
    assert CoherenceLayer("physical", 1.0).is_fragile is False
    assert CoherenceLayer("method", 1.0).is_fragile is True
    assert CoherenceLayer("semantic", 1.0).is_fragile is True
    assert CoherenceLayer("unknown", 1.0).index == -1


def test_recover_tags_recovered_and_raises_on_decrease():
    layer = CoherenceLayer("semantic", 0.2)
    recovered = layer.recover(0.7)
    assert recovered.status == LayerStatus.RECOVERED
    assert recovered.fidelity == pytest.approx(0.7)
    with pytest.raises(ValueError):
        layer.recover(0.1)


def test_average_of_layers():
    layers = default_layers([0.2, 0.4, 0.6, 0.8, 1.0])
    assert average(layers) == pytest.approx(0.6, abs=1e-9)


def test_sum_builtin_works_via_radd():
    layers = [CoherenceLayer("a", 0.3), CoherenceLayer("b", 0.4)]
    total = sum(layers)  # uses __radd__ starting from 0
    assert total.fidelity == pytest.approx(0.7, abs=1e-9)


def test_default_layers_names_and_length():
    layers = default_layers([0.9, 0.9, 0.9, 0.9, 0.9])
    assert [l.name for l in layers] == list(LAYER_NAMES)
    with pytest.raises(ValueError):
        default_layers([0.9, 0.9])
