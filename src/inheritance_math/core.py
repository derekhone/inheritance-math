"""Core transmission primitive: :class:`InheritanceMap`.

An :class:`InheritanceMap` represents a *single transmission boundary* — one
event in which a source state is carried across a boundary and received on the
other side, possibly with loss, drift, or amplification.

It embodies several of the five axioms of Inheritance Mathematics:

* **Axiom 1 (Field Dependency):** the map operates on a state *vector*, i.e.
  within a structured field, not on a scalar in isolation.
* **Axiom 2 (Matching):** transfer quality depends on how well the operator
  matches the source state (mismatch shows up as leakage).
* **Axiom 3 (Recoverable Drift):** injected noise is treated as *drift*; some
  of it may carry recoverable structure rather than pure destruction.
* **Axiom 5 (Bounded Safety):** an optional ``bound`` hard-clips the received
  state so a transmission can never exceed a safe envelope.

The operator may be a matrix (linear map) or a callable (nonlinear map),
supporting both the discrete and continuous update maps in Derek's 2025 notes.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

Operator = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


class InheritanceMap:
    """A single transmission boundary.

    Parameters
    ----------
    operator:
        Either a square matrix ``A`` (the received state is ``A @ source``) or a
        callable ``f`` (the received state is ``f(source)``). Matrices model the
        linear/discrete update maps; callables model nonlinear/continuous ones.
    noise:
        Standard deviation of zero-mean Gaussian drift added on transmission.
        Interpreted under Axiom 3 as *drift*, not merely destruction.
    bound:
        Optional per-component magnitude cap applied to the received state
        (Axiom 5, Bounded Safety). ``None`` disables bounding.
    seed:
        Optional RNG seed for reproducible drift.
    name:
        Optional human-readable label for the boundary.
    """

    def __init__(
        self,
        operator: Operator,
        noise: float = 0.0,
        bound: Optional[float] = None,
        seed: Optional[int] = None,
        name: str = "boundary",
    ) -> None:
        if isinstance(operator, np.ndarray):
            if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
                raise ValueError("matrix operator must be square (n x n)")
        elif not callable(operator):
            raise TypeError("operator must be a square numpy matrix or a callable")
        if noise < 0:
            raise ValueError("noise std must be non-negative")
        if bound is not None and bound <= 0:
            raise ValueError("bound must be positive when provided")

        self.operator = operator
        self.noise = float(noise)
        self.bound = None if bound is None else float(bound)
        self.name = name
        self._rng = np.random.default_rng(seed)

    # --- application ---------------------------------------------------------
    def _apply_operator(self, state: np.ndarray) -> np.ndarray:
        if isinstance(self.operator, np.ndarray):
            return self.operator @ state
        return np.asarray(self.operator(state), dtype=float)

    def transmit(self, source: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Carry ``source`` across the boundary and return the received state.

        Applies the operator, optionally adds Gaussian drift, then applies the
        safety bound if one is set. Does not mutate ``source``.
        """
        source = np.asarray(source, dtype=float)
        received = self._apply_operator(source)
        if add_noise and self.noise > 0:
            received = received + self._rng.normal(0.0, self.noise, size=received.shape)
        if self.bound is not None:
            received = np.clip(received, -self.bound, self.bound)
        return received

    # --- diagnostics ---------------------------------------------------------
    def leakage(self, source: np.ndarray) -> float:
        """Fraction of source magnitude lost across the boundary, in ``[0, 1]``.

        Computed noise-free (deterministic operator + bound) so it measures
        *structural* loss, not stochastic drift. Defined as
        ``1 - ||received|| / ||source||`` clipped to ``[0, 1]``. Amplification
        (received larger than source) reports ``0.0`` leakage.
        """
        source = np.asarray(source, dtype=float)
        src_norm = float(np.linalg.norm(source))
        if src_norm == 0.0:
            return 0.0
        received = self.transmit(source, add_noise=False)
        recv_norm = float(np.linalg.norm(received))
        return float(np.clip(1.0 - recv_norm / src_norm, 0.0, 1.0))

    def gain(self, source: np.ndarray) -> float:
        """Signed magnitude ratio ``||received|| / ||source||`` (noise-free).

        ``> 1`` means amplification, ``< 1`` means decay, ``1`` means preserved.
        """
        source = np.asarray(source, dtype=float)
        src_norm = float(np.linalg.norm(source))
        if src_norm == 0.0:
            return 0.0
        received = self.transmit(source, add_noise=False)
        return float(np.linalg.norm(received) / src_norm)

    def fixed_point(
        self,
        x0: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> dict:
        """Find a fixed point of the noise-free map ``x* = T(x*)``.

        For linear operators the fixed point is found in closed form when
        possible (``(I - A) x = 0`` → the eigenvector for eigenvalue 1, or the
        origin). For nonlinear/callable operators, simple fixed-point iteration
        ``x_{k+1} = T(x_k)`` is used.

        Returns a dict with keys ``fixed_point`` (array or ``None``),
        ``converged`` (bool), ``iterations`` (int), and ``residual`` (float).
        """
        if isinstance(self.operator, np.ndarray):
            n = self.operator.shape[0]
            identity = np.eye(n)
            # Solve (I - A) x = 0 for a nontrivial fixed point via eigen-analysis.
            eigvals, eigvecs = np.linalg.eig(self.operator)
            ones = np.isclose(eigvals.real, 1.0, atol=1e-8) & (np.abs(eigvals.imag) < 1e-8)
            if np.any(ones):
                vec = eigvecs[:, np.argmax(ones)].real
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                residual = float(np.linalg.norm((self.operator - identity) @ vec))
                return {
                    "fixed_point": vec,
                    "converged": True,
                    "iterations": 0,
                    "residual": residual,
                }
            # Otherwise the unique fixed point of a contraction is the origin.
            spectral_radius = float(np.max(np.abs(eigvals)))
            origin = np.zeros(n)
            return {
                "fixed_point": origin,
                "converged": spectral_radius < 1.0,
                "iterations": 0,
                "residual": 0.0,
            }

        # Nonlinear: fixed-point iteration.
        if x0 is None:
            if dim is None:
                raise ValueError("provide x0 or dim for a callable operator")
            x0 = np.zeros(dim)
        x = np.asarray(x0, dtype=float)
        for i in range(1, max_iter + 1):
            x_next = self._apply_operator(x)
            residual = float(np.linalg.norm(x_next - x))
            x = x_next
            if residual < tol:
                return {
                    "fixed_point": x,
                    "converged": True,
                    "iterations": i,
                    "residual": residual,
                }
        return {
            "fixed_point": x,
            "converged": False,
            "iterations": max_iter,
            "residual": residual,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "matrix" if isinstance(self.operator, np.ndarray) else "callable"
        return (
            f"InheritanceMap(name={self.name!r}, operator={kind}, "
            f"noise={self.noise}, bound={self.bound})"
        )
