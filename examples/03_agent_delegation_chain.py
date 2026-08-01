"""Example 03 — Does authority scope survive a delegation chain?

An orchestrator delegates a task to agent A, which delegates to agent B, which
delegates to agent C — a 3-hop chain. Each hop is a transmission boundary. The
carried state is the agent's *authority scope* (its magnitude = how much it is
allowed to do).

We compare two chains over the same hops:

  * UNBOUNDED — each hop applies a >1 operator (scope creep). Authority
    expands at every hop and ends up larger than what was originally granted.
  * BOUNDED (Axiom 5, Bounded Safety) — each hop clips authority to the
    originally granted envelope. Scope can never exceed the grant.

Run:
    python examples/03_agent_delegation_chain.py
"""

import numpy as np

from inheritance_math import InheritanceChain


def main() -> None:
    granted = np.array([1.0])  # originally granted authority scope (envelope = 1.0)
    creep = np.array([[1.25]])  # each hop tends to expand scope by 25%

    unbounded = InheritanceChain.uniform(creep, hops=3, name="unbounded")
    bounded = InheritanceChain.uniform(creep, hops=3, bound=1.0, name="bounded")

    unbounded_trace = [float(np.linalg.norm(s)) for s in unbounded.trace(granted)]
    bounded_trace = [float(np.linalg.norm(s)) for s in bounded.trace(granted)]

    print("=== Example 03: Agent Delegation Chain (3 hops) ===")
    print(f"originally granted authority scope: {float(granted[0]):.2f}\n")

    print("hop           unbounded_scope   bounded_scope")
    labels = ["grant", "-> agent A", "-> agent B", "-> agent C"]
    for label, u, b in zip(labels, unbounded_trace, bounded_trace):
        print(f"  {label:<12} {u:>12.3f}   {b:>12.3f}")

    print(f"\nunbounded final scope : {unbounded_trace[-1]:.3f}  "
          f"(exceeds grant by {unbounded_trace[-1] - 1.0:+.1%})")
    print(f"bounded final scope   : {bounded_trace[-1]:.3f}  "
          f"(within grant: {bounded.is_bounded_by(granted, 1.0)})")
    print(
        "\nReading: without a hard bound, delegated authority silently expands "
        "at each hop — the classic agent scope-creep failure. Axiom 5 makes "
        "the bound a hard constraint, so no downstream agent can ever act "
        "beyond what was originally granted."
    )


if __name__ == "__main__":
    main()
