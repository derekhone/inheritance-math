import numpy as np
import pytest

from inheritance_math import (
    CoherenceLayer,
    boundary_leakage,
    coherence_score,
    critical_layers,
    default_layers,
    recovery_index,
    run_sim,
)
from inheritance_math.metrics import DEFAULT_WEIGHTS


def test_coherence_score_all_perfect():
    layers = default_layers([1.0, 1.0, 1.0, 1.0, 1.0])
    assert coherence_score(layers) == pytest.approx(1.0, abs=1e-9)


def test_coherence_score_weighted_computation():
    layers = default_layers([1.0, 1.0, 1.0, 0.0, 0.0])
    # weights: 1,1,1.5,2,2.5 total=8 ; numerator = 1+1+1.5 = 3.5
    expected = 3.5 / 8.0
    assert coherence_score(layers) == pytest.approx(expected, abs=1e-9)


def test_fragile_layers_weighted_higher():
    # losing semantic (weight 2.5) hurts more than losing physical (weight 1.0)
    lose_semantic = default_layers([1.0, 1.0, 1.0, 1.0, 0.0])
    lose_physical = default_layers([0.0, 1.0, 1.0, 1.0, 1.0])
    assert coherence_score(lose_semantic) < coherence_score(lose_physical)
    assert DEFAULT_WEIGHTS["semantic"] > DEFAULT_WEIGHTS["physical"]


def test_boundary_leakage_value():
    before = default_layers([1.0, 1.0, 1.0, 1.0, 1.0])
    after = default_layers([0.5, 0.5, 0.5, 0.5, 0.5])
    # score halves -> 50% leakage
    assert boundary_leakage(before, after) == pytest.approx(0.5, abs=1e-9)


def test_boundary_leakage_never_negative():
    before = default_layers([0.5, 0.5, 0.5, 0.5, 0.5])
    after = default_layers([0.9, 0.9, 0.9, 0.9, 0.9])
    assert boundary_leakage(before, after) == pytest.approx(0.0, abs=1e-9)


def test_recovery_index():
    lost = CoherenceLayer("intent", 0.0)
    recovered = lost.recover(0.6)  # deficit 0.4 now recovered
    still_lost = CoherenceLayer("semantic", 0.0)  # deficit 1.0
    # total deficit = (1-0.6) + (1-0.0) = 1.4 ; recovered deficit = 0.4
    idx = recovery_index([recovered, still_lost])
    assert idx == pytest.approx(0.4 / 1.4, abs=1e-9)


def test_recovery_index_zero_when_perfect():
    assert recovery_index(default_layers([1.0] * 5)) == pytest.approx(0.0)


def test_critical_layers():
    layers = default_layers([0.9, 0.9, 0.3, 0.2, 0.1])
    assert critical_layers(layers, threshold=0.5) == ["method", "intent", "semantic"]


def test_run_sim_conserves_and_converges_to_consensus():
    # path graph of 3 nodes, no decay -> total conserved, consensus = mean
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    x0 = np.array([3.0, 0.0, 0.0])
    res = run_sim(W, x0, steps=500, dt=0.2, decay=0.0)
    assert res["converged"] is True
    # total coherence conserved
    assert res["total_coherence"][-1] == pytest.approx(3.0, abs=1e-3)
    # consensus is the mean of the initial state
    assert res["consensus"] == pytest.approx(1.0, abs=1e-3)
    # all nodes agree at convergence
    assert np.allclose(res["final_state"], res["final_state"][0], atol=1e-3)


def test_run_sim_decay_reduces_total():
    W = np.array([[0, 1], [1, 0]], dtype=float)
    x0 = np.array([1.0, 1.0])
    res = run_sim(W, x0, steps=50, dt=0.1, decay=0.05)
    assert res["total_coherence"][-1] < res["total_coherence"][0]
