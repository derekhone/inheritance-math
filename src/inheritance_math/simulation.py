"""Graph-Laplacian diffusion simulation: :func:`run_sim`.

This is the graph-spatial extension of Inheritance Mathematics. A coherence
property lives on the nodes of a network of connected systems and evolves by
Laplacian diffusion:

    x_{t+1} = x_t - dt * L @ x_t          (discrete heat / consensus equation)

where ``L = D - W`` is the (combinatorial) graph Laplacian of a weighted
adjacency matrix ``W``. Diffusion drives the network toward a coherent
consensus (Axiom 4, Coherence as Dynamic Stability): the property spreads
until all connected nodes agree, while total coherence is conserved on a
connected graph. An optional per-step ``decay`` models leakage to the
environment (coherence is not perfectly conserved).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the combinatorial Laplacian ``L = D - W`` of ``adjacency``."""
    W = np.asarray(adjacency, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    degree = np.diag(W.sum(axis=1))
    return degree - W


def run_sim(
    adjacency: np.ndarray,
    initial_state: np.ndarray,
    steps: int = 100,
    dt: float = 0.1,
    decay: float = 0.0,
    tol: float = 1e-6,
) -> dict:
    """Run Laplacian diffusion of a coherence property over a network.

    Parameters
    ----------
    adjacency:
        Square weighted adjacency matrix ``W`` (``W[i, j]`` = coupling strength
        between systems ``i`` and ``j``). Symmetric for undirected coupling.
    initial_state:
        Initial coherence value per node, length ``n``.
    steps:
        Maximum number of time steps.
    dt:
        Time-step size. For stability keep ``dt * max_degree < 1``.
    decay:
        Optional uniform per-step leakage rate in ``[0, 1)`` (Axiom 5 boundary
        loss to the environment). ``0`` conserves total coherence on a
        connected graph.
    tol:
        Convergence tolerance on successive-state change (L2 norm).

    Returns
    -------
    dict
        ``time_series`` (array ``(t+1, n)`` of states per recorded step),
        ``final_state`` (array ``n``), ``converged`` (bool), ``steps_run``
        (int), ``consensus`` (float mean of the final state), and
        ``total_coherence`` (array of summed coherence per step).
    """
    W = np.asarray(adjacency, dtype=float)
    x = np.asarray(initial_state, dtype=float).copy()
    n = W.shape[0]
    if x.shape[0] != n:
        raise ValueError("initial_state length must match adjacency dimension")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")

    L = graph_laplacian(W)
    max_degree = float(W.sum(axis=1).max()) if n else 0.0
    if dt * max_degree >= 1.0:
        # Not fatal, but warn the caller via the returned dict.
        stability_warning = (
            f"dt*max_degree = {dt * max_degree:.3f} >= 1; diffusion may be unstable"
        )
    else:
        stability_warning = None

    history = [x.copy()]
    totals = [float(x.sum())]
    converged = False
    steps_run = 0
    for _ in range(steps):
        x_next = x - dt * (L @ x)
        if decay > 0.0:
            x_next = x_next * (1.0 - decay)
        change = float(np.linalg.norm(x_next - x))
        x = x_next
        history.append(x.copy())
        totals.append(float(x.sum()))
        steps_run += 1
        if change < tol:
            converged = True
            break

    return {
        "time_series": np.array(history),
        "final_state": x,
        "converged": converged,
        "steps_run": steps_run,
        "consensus": float(x.mean()),
        "total_coherence": np.array(totals),
        "stability_warning": stability_warning,
    }
