"""Inheritance Mathematics — the Coherent Inheritance Framework (CIF).

A mathematical modeling library for studying what survives when knowledge,
authority, constraints, and meaning pass across a system boundary — a
fine-tuning step, an agent delegation, or a generation of copying.

This is a *modeling* library, not a physics theory. CIF is a candidate
framework built on established tools (nonlinear dynamics, graph theory,
diffusion, fixed-point analysis) organized around five axioms:

1. Field Dependency — every system operates within a structured field.
2. Matching — productive transfer depends on appropriate state matching.
3. Recoverable Drift — noise may contain recoverable information.
4. Coherence as Dynamic Stability — coherence = organized continuity under change.
5. Bounded Safety — safety is a hard constraint, not a feature.

© 2026 Remnant Fieldworks Inc. MIT License.
"""

from __future__ import annotations

from .chain import InheritanceChain
from .core import InheritanceMap
from .layers import (
    LAYER_NAMES,
    CoherenceLayer,
    LayerStatus,
    average,
    default_layers,
)
from .metrics import (
    boundary_leakage,
    coherence_score,
    critical_layers,
    recovery_index,
)
from .simulation import graph_laplacian, run_sim

__version__ = "0.1.0a1"

__all__ = [
    "InheritanceMap",
    "CoherenceLayer",
    "LayerStatus",
    "InheritanceChain",
    "coherence_score",
    "boundary_leakage",
    "recovery_index",
    "critical_layers",
    "run_sim",
    "graph_laplacian",
    "average",
    "default_layers",
    "LAYER_NAMES",
    "__version__",
]
