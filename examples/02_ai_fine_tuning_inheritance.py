"""Example 02 — Does alignment survive fine-tuning?

Models a foundation model being fine-tuned as a single inheritance boundary
across the five CIF layers:

    physical (weights)          — inherited almost perfectly
    data (training corpus)      — inherited well
    method (training procedure) — only partially carried across
    intent (alignment goal)     — degraded by the new objective
    semantic (meaning/interpretability) — most uncertain of all

We score coherence before and after fine-tuning and identify which layers are
critical failure points. Because the intent/semantic layers are weighted most
heavily (they are hardest to recover), their degradation dominates the drop.

Run:
    python examples/02_ai_fine_tuning_inheritance.py
"""

from inheritance_math import (
    boundary_leakage,
    coherence_score,
    critical_layers,
    default_layers,
)


def main() -> None:
    # Foundation model: all five layers high fidelity.
    before = default_layers([0.98, 0.95, 0.92, 0.90, 0.88])

    # After a fine-tuning event: weights & data survive, but method is only
    # partially inherited and alignment intent / semantic continuity degrade.
    after = default_layers([0.97, 0.90, 0.55, 0.35, 0.25])

    score_before = coherence_score(before)
    score_after = coherence_score(after)
    leak = boundary_leakage(before, after)

    print("=== Example 02: AI Fine-Tuning Inheritance ===")
    print("layer      fidelity_before  fidelity_after")
    for b, a in zip(before, after):
        print(f"  {b.name:<9} {b.fidelity:>10.2f}     {a.fidelity:>10.2f}"
              f"   [{a.status.value}]")

    print(f"\ncoherence score (before) : {score_before:.3f}")
    print(f"coherence score (after)  : {score_after:.3f}")
    print(f"boundary leakage         : {leak:.1%} of weighted coherence lost")
    print(f"critical failure layers  : {critical_layers(after)}")
    print(
        "\nReading: weights and data are inherited, so a naive check would say "
        "'the model transferred fine.' But CIF weights the fragile intent and "
        "semantic layers most heavily — their degradation is where alignment "
        "actually fails to survive the boundary."
    )


if __name__ == "__main__":
    main()
