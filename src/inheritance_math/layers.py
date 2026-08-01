"""Coherence layers for the Coherent Inheritance Framework (CIF).

Inheritance Mathematics models transmission across five *inheritance layers*.
Each layer captures a distinct thing that must survive when something passes
across a boundary (a fine-tuning step, an agent delegation, a generation):

    1. physical  — the physical form / substrate (weights, ink, stone, DNA)
    2. data      — the raw data / symbols carried by that substrate
    3. method     — the procedure / know-how required to *use* the data
    4. intent     — the purpose / goal / alignment behind the transmission
    5. semantic   — the continued ability to *interpret* meaning over time

Empirically (and by construction in CIF) layers 3-5 are the most fragile:
data can survive intact while method, intent, and semantic continuity are lost.
This module provides :class:`CoherenceLayer`, a small value object carrying a
``fidelity`` in ``[0, 1]`` and a derived :class:`LayerStatus`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

# Canonical layer names, in order (index 0 == layer 1).
LAYER_NAMES = ("physical", "data", "method", "intent", "semantic")

# Default fidelity thresholds used to derive a discrete status from a
# continuous fidelity score. These are intentionally simple and documented.
INTACT_THRESHOLD = 0.85
DEGRADED_THRESHOLD = 0.40


class LayerStatus(str, Enum):
    """Discrete health of a coherence layer, derived from its fidelity.

    ``RECOVERED`` is never inferred automatically from fidelity alone; it is an
    explicit annotation meaning "information was lost then recovered" (Axiom 3,
    Recoverable Drift). Callers set it via :meth:`CoherenceLayer.recover`.
    """

    INTACT = "INTACT"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    RECOVERED = "RECOVERED"


def status_from_fidelity(fidelity: float) -> LayerStatus:
    """Map a continuous fidelity in ``[0, 1]`` to a discrete status."""
    if fidelity >= INTACT_THRESHOLD:
        return LayerStatus.INTACT
    if fidelity >= DEGRADED_THRESHOLD:
        return LayerStatus.DEGRADED
    return LayerStatus.LOST


@dataclass
class CoherenceLayer:
    """A single inheritance layer with a fidelity score in ``[0, 1]``.

    Parameters
    ----------
    name:
        One of :data:`LAYER_NAMES` (``physical``, ``data``, ``method``,
        ``intent``, ``semantic``). Arbitrary names are allowed but a warning is
        avoided for flexibility in research use.
    fidelity:
        How faithfully this layer survived transmission, in ``[0, 1]``.
        ``1.0`` means perfect preservation; ``0.0`` means total loss.
    status:
        Optional explicit status. If omitted it is derived from ``fidelity``.

    Notes
    -----
    ``CoherenceLayer`` supports arithmetic on fidelity so a list of layers can
    be averaged conveniently (e.g. ``sum(layers, start) / n``). Addition of two
    layers returns a new ``combined`` layer whose fidelity is the sum (used
    internally by :func:`average`); use :func:`average` for the mean.
    """

    name: str
    fidelity: float
    status: LayerStatus = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.fidelity) <= 1.0:
            raise ValueError(
                f"fidelity must be in [0, 1], got {self.fidelity!r} for layer {self.name!r}"
            )
        self.fidelity = float(self.fidelity)
        if self.status is None:
            self.status = status_from_fidelity(self.fidelity)

    # --- layer index helpers -------------------------------------------------
    @property
    def index(self) -> int:
        """1-based canonical layer index, or ``-1`` if the name is unknown."""
        try:
            return LAYER_NAMES.index(self.name) + 1
        except ValueError:
            return -1

    @property
    def is_fragile(self) -> bool:
        """True for layers 3-5 (method, intent, semantic), the fragile ones."""
        return self.index >= 3

    # --- recovery (Axiom 3: Recoverable Drift) -------------------------------
    def recover(self, new_fidelity: float) -> "CoherenceLayer":
        """Return a recovered copy of this layer at ``new_fidelity``.

        Recovery models Axiom 3 (Recoverable Drift): information that appeared
        lost may be recoverable. The returned layer is tagged ``RECOVERED``.
        """
        if new_fidelity < self.fidelity:
            raise ValueError("recovery cannot decrease fidelity")
        return CoherenceLayer(self.name, new_fidelity, LayerStatus.RECOVERED)

    # --- arithmetic ----------------------------------------------------------
    def __add__(self, other: "CoherenceLayer | float") -> "CoherenceLayer":
        other_fid = other.fidelity if isinstance(other, CoherenceLayer) else float(other)
        total = min(1.0, self.fidelity + other_fid)
        return CoherenceLayer("combined", total)

    def __radd__(self, other: float) -> "CoherenceLayer":
        # Supports sum(layers) with a numeric start value.
        return self.__add__(other)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CoherenceLayer(name={self.name!r}, fidelity={self.fidelity:.3f}, status={self.status.value})"


def average(layers: Iterable[CoherenceLayer]) -> float:
    """Return the plain (unweighted) mean fidelity of ``layers``."""
    layers = list(layers)
    if not layers:
        raise ValueError("cannot average an empty list of layers")
    return sum(layer.fidelity for layer in layers) / len(layers)


def default_layers(fidelities: Iterable[float]) -> list[CoherenceLayer]:
    """Build the five canonical layers from an iterable of five fidelities."""
    fids = list(fidelities)
    if len(fids) != len(LAYER_NAMES):
        raise ValueError(f"expected {len(LAYER_NAMES)} fidelities, got {len(fids)}")
    return [CoherenceLayer(name, fid) for name, fid in zip(LAYER_NAMES, fids)]
