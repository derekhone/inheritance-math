import numpy as np
import pytest

from inheritance_math import InheritanceChain, InheritanceMap


def test_uniform_chain_length():
    ch = InheritanceChain.uniform(np.eye(1), hops=3)
    assert len(ch) == 3


def test_coherence_decay_multiplicative():
    # 0.8 gain per hop over 2 hops: [1, 0.8, 0.64]
    ch = InheritanceChain.uniform(np.array([[0.8]]), hops=2)
    decay = ch.coherence_decay(np.array([1.0]))
    assert decay == pytest.approx([1.0, 0.8, 0.64], abs=1e-9)


def test_unbounded_authority_expands():
    ch = InheritanceChain.uniform(np.array([[1.25]]), hops=3)
    decay = ch.coherence_decay(np.array([1.0]))
    # final scope should exceed the original grant
    assert decay[-1] > 1.0
    assert decay[-1] == pytest.approx(1.25 ** 3, abs=1e-9)


def test_bounded_authority_never_exceeds_grant_axiom5():
    ch = InheritanceChain.uniform(np.array([[1.25]]), hops=3, bound=1.0)
    assert ch.is_bounded_by(np.array([1.0]), 1.0) is True
    final = ch.transmit(np.array([1.0]), add_noise=False)
    assert float(np.linalg.norm(final)) <= 1.0 + 1e-9


def test_cumulative_leakage_matches_decay():
    ch = InheritanceChain.uniform(np.array([[0.5]]), hops=2)
    # final ratio 0.25 -> leakage 0.75
    assert ch.cumulative_leakage(np.array([1.0])) == pytest.approx(0.75, abs=1e-9)


def test_trace_length_is_hops_plus_one():
    ch = InheritanceChain.uniform(np.eye(2), hops=4)
    trace = ch.trace(np.array([1.0, 1.0]))
    assert len(trace) == 5


def test_empty_chain_raises():
    with pytest.raises(ValueError):
        InheritanceChain([])


def test_mixed_maps_chain():
    m1 = InheritanceMap(np.array([[0.9]]))
    m2 = InheritanceMap(lambda x: x * 0.5)
    ch = InheritanceChain([m1, m2])
    final = ch.transmit(np.array([1.0]), add_noise=False)
    assert final[0] == pytest.approx(0.45, abs=1e-9)
