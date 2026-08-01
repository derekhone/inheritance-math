"""Multi-hop transmission: :class:`InheritanceChain`.

A chain is an ordered sequence of :class:`~inheritance_math.core.InheritanceMap`
boundaries. It models multi-hop inheritance — for example a 3-hop agent
delegation, or a document copied across three generations — and tracks how
coherence (magnitude of the carried state) decays or amplifies at each hop.

Key modeled finding: without a safety bound (Axiom 5), authority/coherence can
*expand* or persist unchecked across hops; with a per-hop bound, the carried
state can never exceed the originally granted envelope.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .core import InheritanceMap


class InheritanceChain:
    """An ordered sequence of transmission boundaries.

    Parameters
    ----------
    maps:
        The ordered list of :class:`InheritanceMap` hops.
    name:
        Optional label.
    """

    def __init__(self, maps: List[InheritanceMap], name: str = "chain") -> None:
        if not maps:
            raise ValueError("an InheritanceChain needs at least one map")
        self.maps = list(maps)
        self.name = name

    def __len__(self) -> int:
        return len(self.maps)

    def transmit(self, source: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Run ``source`` through every hop and return the final received state."""
        state = np.asarray(source, dtype=float)
        for hop in self.maps:
            state = hop.transmit(state, add_noise=add_noise)
        return state

    def trace(self, source: np.ndarray, add_noise: bool = False) -> List[np.ndarray]:
        """Return the state at each stage: ``[source, after_hop1, ...]``.

        Defaults to noise-free so the trace is deterministic and comparable.
        """
        state = np.asarray(source, dtype=float)
        states = [state.copy()]
        for hop in self.maps:
            state = hop.transmit(state, add_noise=add_noise)
            states.append(state.copy())
        return states

    def coherence_decay(self, source: np.ndarray) -> List[float]:
        """Cumulative coherence ratio ``||state_k|| / ||source||`` per stage.

        Returns one value per stage including the source (which is ``1.0``).
        Values < 1 indicate decay; > 1 indicate amplification / scope creep.
        """
        states = self.trace(source, add_noise=False)
        src_norm = float(np.linalg.norm(states[0]))
        if src_norm == 0.0:
            return [0.0 for _ in states]
        return [float(np.linalg.norm(s) / src_norm) for s in states]

    def cumulative_leakage(self, source: np.ndarray) -> float:
        """Total fraction of magnitude lost from source to final state.

        ``1 - ||final|| / ||source||`` clipped to ``[0, 1]`` (noise-free).
        """
        ratios = self.coherence_decay(source)
        return float(np.clip(1.0 - ratios[-1], 0.0, 1.0))

    def is_bounded_by(self, source: np.ndarray, envelope: float) -> bool:
        """True if every stage stays within ``envelope`` in magnitude.

        Used to verify Axiom 5: a bounded chain never lets the carried state
        (e.g. delegated authority scope) exceed the granted envelope.
        """
        states = self.trace(source, add_noise=False)
        return all(float(np.linalg.norm(s)) <= envelope + 1e-9 for s in states)

    @classmethod
    def uniform(
        cls,
        operator: np.ndarray,
        hops: int,
        noise: float = 0.0,
        bound: Optional[float] = None,
        name: str = "chain",
    ) -> "InheritanceChain":
        """Build a chain of ``hops`` identical boundaries sharing an operator."""
        maps = [
            InheritanceMap(operator, noise=noise, bound=bound, name=f"{name}_hop{i+1}")
            for i in range(hops)
        ]
        return cls(maps, name=name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"InheritanceChain(name={self.name!r}, hops={len(self.maps)})"
