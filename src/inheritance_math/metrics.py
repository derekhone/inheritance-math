"""Coherence metrics for CIF.

These functions summarize the health of a set of :class:`CoherenceLayer`
objects, or of a transmission, into single interpretable numbers.

* :func:`coherence_score` — weighted mean fidelity across layers. Fragile
  layers (4-5: intent, semantic) are weighted higher because they are hardest
  to recover, so their loss is penalized more.
* :func:`boundary_leakage` — fraction of coherence lost across a boundary.
* :func:`recovery_index` — how much recoverable structure remains in drift
  (Axiom 3): the share of "lost" coherence that is marked RECOVERED.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from .layers import LAYER_NAMES, CoherenceLayer, LayerStatus

# Default weights per canonical layer (physical..semantic). Layers 4-5 are
# weighted higher because intent and semantic continuity are the hardest to
# recover once lost; losing them should hurt the score more.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "physical": 1.0,
    "data": 1.0,
    "method": 1.5,
    "intent": 2.0,
    "semantic": 2.5,
}


def coherence_score(
    layers: Sequence[CoherenceLayer],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted mean fidelity across ``layers``, in ``[0, 1]``.

    Parameters
    ----------
    layers:
        The coherence layers to score.
    weights:
        Optional mapping ``layer name -> weight``. Unknown layer names default
        to weight ``1.0``. Defaults to :data:`DEFAULT_WEIGHTS`, which weights
        the fragile intent/semantic layers more heavily.
    """
    if not layers:
        raise ValueError("cannot score an empty list of layers")
    weights = DEFAULT_WEIGHTS if weights is None else weights
    total_w = 0.0
    acc = 0.0
    for layer in layers:
        w = weights.get(layer.name, 1.0)
        acc += w * layer.fidelity
        total_w += w
    if total_w == 0.0:
        raise ValueError("total weight is zero")
    return acc / total_w


def boundary_leakage(
    before: Sequence[CoherenceLayer],
    after: Sequence[CoherenceLayer],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Fraction of weighted coherence lost from ``before`` to ``after``.

    Returns ``max(0, score_before - score_after) / score_before`` in ``[0, 1]``.
    A value of ``0`` means no coherence was lost across the boundary.
    """
    score_before = coherence_score(before, weights)
    score_after = coherence_score(after, weights)
    if score_before == 0.0:
        return 0.0
    return max(0.0, (score_before - score_after) / score_before)


def recovery_index(layers: Iterable[CoherenceLayer]) -> float:
    """Share of lost coherence that has been recovered (Axiom 3), in ``[0, 1]``.

    Defined as ``recovered_deficit / total_deficit`` where a layer's deficit is
    ``1 - fidelity``. Layers tagged ``RECOVERED`` contribute their deficit to
    the numerator. Returns ``0.0`` when there is no deficit at all.
    """
    total_deficit = 0.0
    recovered_deficit = 0.0
    for layer in layers:
        deficit = 1.0 - layer.fidelity
        total_deficit += deficit
        if layer.status == LayerStatus.RECOVERED:
            recovered_deficit += deficit
    if total_deficit == 0.0:
        return 0.0
    return recovered_deficit / total_deficit


def critical_layers(
    layers: Sequence[CoherenceLayer],
    threshold: float = 0.5,
) -> list[str]:
    """Return names of layers whose fidelity falls below ``threshold``.

    Useful for reporting *which* layers caused a low coherence score.
    """
    return [layer.name for layer in layers if layer.fidelity < threshold]
