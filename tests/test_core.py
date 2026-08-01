import numpy as np
import pytest

from inheritance_math import InheritanceMap


def test_transmit_linear_decay():
    A = np.array([[1.0, 0.0], [0.0, 0.5]])
    m = InheritanceMap(A)
    received = m.transmit(np.array([2.0, 2.0]), add_noise=False)
    assert np.allclose(received, [2.0, 1.0])


def test_transmit_callable_operator():
    m = InheritanceMap(lambda x: x * 0.9, name="scale")
    received = m.transmit(np.array([1.0, 1.0]), add_noise=False)
    assert np.allclose(received, [0.9, 0.9])


def test_leakage_in_unit_interval_and_value():
    A = np.array([[0.6]])
    m = InheritanceMap(A)
    # ||received|| / ||source|| = 0.6, so leakage = 0.4
    assert m.leakage(np.array([1.0])) == pytest.approx(0.4, abs=1e-9)


def test_leakage_zero_for_amplification():
    A = np.array([[2.0]])
    m = InheritanceMap(A)
    assert m.leakage(np.array([1.0])) == pytest.approx(0.0, abs=1e-9)


def test_gain_amplification():
    A = np.array([[1.5]])
    m = InheritanceMap(A)
    assert m.gain(np.array([1.0])) == pytest.approx(1.5, abs=1e-9)


def test_bound_clips_state_axiom5():
    A = np.array([[10.0]])
    m = InheritanceMap(A, bound=1.0)
    received = m.transmit(np.array([1.0]), add_noise=False)
    assert received[0] == pytest.approx(1.0, abs=1e-9)


def test_fixed_point_identity_eigenvector():
    # Identity map: every direction is a fixed point; expect converged.
    m = InheritanceMap(np.eye(2))
    fp = m.fixed_point()
    assert fp["converged"] is True
    assert fp["residual"] == pytest.approx(0.0, abs=1e-8)


def test_fixed_point_contraction_is_origin():
    m = InheritanceMap(np.array([[0.5, 0.0], [0.0, 0.5]]))
    fp = m.fixed_point()
    assert np.allclose(fp["fixed_point"], [0.0, 0.0])
    assert fp["converged"] is True


def test_fixed_point_callable_iteration():
    # x_{k+1} = 0.5 x + 1  ->  fixed point x* = 2
    m = InheritanceMap(lambda x: 0.5 * x + 1.0)
    fp = m.fixed_point(x0=np.array([0.0]))
    assert fp["converged"] is True
    assert fp["fixed_point"][0] == pytest.approx(2.0, abs=1e-6)


def test_noise_is_reproducible_with_seed():
    A = np.eye(1)
    a = InheritanceMap(A, noise=0.5, seed=42).transmit(np.array([1.0]))
    b = InheritanceMap(A, noise=0.5, seed=42).transmit(np.array([1.0]))
    assert np.allclose(a, b)


def test_invalid_operator_raises():
    with pytest.raises(ValueError):
        InheritanceMap(np.array([[1.0, 2.0]]))  # non-square
    with pytest.raises(TypeError):
        InheritanceMap(123)  # not matrix or callable
