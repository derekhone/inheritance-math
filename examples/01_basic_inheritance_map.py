"""Example 01 — Basic inheritance map: ALLOW / decay across one boundary.

Models a single transmission boundary where a two-component state (think:
"signal" and "constraint") is carried across a boundary that partially
preserves the signal but decays the constraint. We inspect what is received,
how much leaked, and whether the boundary has a stable fixed point.

Run:
    python examples/01_basic_inheritance_map.py
"""

import numpy as np

from inheritance_math import InheritanceMap


def main() -> None:
    # Component 0 = "signal" (preserved at 100%), component 1 = "constraint"
    # (decays to 60% each time it crosses this boundary).
    operator = np.array(
        [
            [1.0, 0.0],  # signal: ALLOW, fully inherited
            [0.0, 0.6],  # constraint: decays to 60%
        ]
    )
    boundary = InheritanceMap(operator, noise=0.0, name="allow-decay")

    source = np.array([1.0, 1.0])  # full signal, full constraint
    received = boundary.transmit(source, add_noise=False)

    print("=== Example 01: Basic Inheritance Map ===")
    print(f"source state   : {source}")
    print(f"received state : {received}")
    print(f"leakage        : {boundary.leakage(source):.3f}  "
          "(fraction of magnitude lost)")
    print(f"gain           : {boundary.gain(source):.3f}  "
          "(<1 = decay, >1 = amplification)")

    fp = boundary.fixed_point()
    print(f"fixed point    : {np.round(fp['fixed_point'], 4)}  "
          f"(converged={fp['converged']})")
    print(
        "\nReading: the signal survives (ALLOW) while the constraint decays. "
        "The only stable state is the one with zero constraint — i.e. left "
        "unbounded, the constraint washes out entirely."
    )


if __name__ == "__main__":
    main()
