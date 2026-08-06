> **Repository Role:** Research Library (Alpha) — Inheritance Math Framework (IMF) Python Library + Whitepaper · Part of the [Remnant Fieldworks](https://remnantfieldworks.com) research and product ecosystem

**Where this fits:** This repository is part of the Coherent Inheritance Framework (CIF) open-science program at Remnant Fieldworks. CIF investigates coherence dynamics across inheritance boundaries using preregistered, evidence-constrained methodology. This repository provides the Inheritance Math Framework (IMF) Python library and whitepaper for modeling what survives when knowledge, authority, and meaning cross a system boundary. For the commercial product, see [ExecutionProof](https://executionproof.io).

---

# inheritance-math

**Coherent Inheritance Framework (CIF)** — a small, dependency-light Python
library for modeling *what survives when knowledge, authority, constraints, and
meaning pass across a system boundary*: a fine-tuning step, an agent
delegation, a copy from one generation to the next.

Its basic question: **what survives a transmission — and what decays,
amplifies, leaks, or changes?**

> **Honest scope.** This is a *mathematical modeling library, not a physics
> theory.* CIF is a **candidate framework**. Its machinery is built on
> established tools — linear/nonlinear dynamics, graph theory, diffusion, and
> fixed-point analysis — organized around five axioms. Use it to make
> inheritance a measurable *process*, not an assumed outcome.

By Derek Hone · [github.com/derekhone](https://github.com/derekhone) ·
[executionproof.io](https://executionproof.io)

---

## Install

```bash
pip install inheritance-math
```

Import name is `inheritance_math`:

```python
import inheritance_math as im
```

Requires Python 3.9+ and NumPy.

---

## The five axioms

1. **Field Dependency** — every system operates within a structured field.
2. **Matching** — productive transfer depends on appropriate state matching.
3. **Recoverable Drift** — noise may contain recoverable information.
4. **Coherence as Dynamic Stability** — coherence = organized continuity under change.
5. **Bounded Safety** — safety is a hard constraint, not a feature.

## The five inheritance layers

Transmission is scored across five layers. Layers **3–5 are the most fragile**:
data often survives intact while method, intent, and meaning are lost.

| # | layer | what must survive |
|---|----------|--------------------------------------------|
| 1 | physical | the substrate (weights, ink, stone, DNA)   |
| 2 | data     | the raw symbols carried by the substrate   |
| 3 | method   | the know-how needed to *use* the data      |
| 4 | intent   | the purpose / alignment behind the transfer|
| 5 | semantic | the continued ability to *interpret* it    |

---

## Quickstart

```python
import numpy as np
import inheritance_math as im

# A single transmission boundary that decays the second component.
boundary = im.InheritanceMap(np.array([[1.0, 0.0], [0.0, 0.6]]))
print(boundary.transmit(np.array([1.0, 1.0]), add_noise=False))  # [1.  0.6]
print(round(boundary.leakage(np.array([1.0, 1.0])), 3))          # 0.175
```

---

## Three examples

### 1. Coherence score across the five layers

```python
import inheritance_math as im

layers = im.default_layers([0.98, 0.95, 0.55, 0.35, 0.25])
print(round(im.coherence_score(layers), 3))   # 0.503
print(im.critical_layers(layers))              # ['method', 'intent', 'semantic']
```

Expected output:

```
0.503
['method', 'intent', 'semantic']
```

### 2. Does alignment survive fine-tuning?

```python
import inheritance_math as im

before = im.default_layers([0.98, 0.95, 0.92, 0.90, 0.88])  # foundation model
after  = im.default_layers([0.97, 0.90, 0.55, 0.35, 0.25])  # after fine-tuning

print(round(im.coherence_score(before), 3))       # 0.914
print(round(im.coherence_score(after), 3))         # 0.503
print(round(im.boundary_leakage(before, after), 3))# 0.45
print(im.critical_layers(after))                   # ['intent', 'semantic']
```

Expected output:

```
0.914
0.503
0.45
['intent', 'semantic']
```

Weights and data are inherited, so a naive check says "the model transferred
fine." CIF weights the fragile intent/semantic layers most heavily — and that
is where alignment actually fails to survive the boundary.

### 3. Does authority scope survive a 3-hop delegation chain?

```python
import numpy as np
import inheritance_math as im

grant = np.array([1.0])
creep = np.array([[1.25]])  # each hop tends to expand scope 25%

unbounded = im.InheritanceChain.uniform(creep, hops=3)
bounded   = im.InheritanceChain.uniform(creep, hops=3, bound=1.0)  # Axiom 5

print(round(unbounded.coherence_decay(grant)[-1], 3))  # 1.953  (scope creep!)
print(bounded.is_bounded_by(grant, 1.0))               # True   (never exceeds grant)
```

Expected output:

```
1.953
True
```

Without a hard bound, delegated authority silently expands at every hop — the
classic agent scope-creep failure. Axiom 5 makes the bound a hard constraint,
so no downstream agent can ever act beyond what was originally granted.

Full runnable versions live in [`examples/`](examples/).

---

## API at a glance

| symbol | purpose |
|--------|---------|
| `InheritanceMap` | a single transmission boundary — `transmit()`, `fixed_point()`, `leakage()`, `gain()` |
| `CoherenceLayer` | one of the five layers with a `fidelity` and derived `status` |
| `InheritanceChain` | a multi-hop sequence of maps — tracks `coherence_decay()`, `is_bounded_by()` |
| `coherence_score()` | weighted mean fidelity (fragile layers weighted higher) |
| `boundary_leakage()` | coherence lost across a boundary |
| `recovery_index()` | share of lost coherence recovered (Axiom 3) |
| `run_sim()` | graph-Laplacian diffusion of coherence across a network |

---

## Development

```bash
git clone https://github.com/derekhone/inheritance-math.git
cd inheritance-math
pip install -e ".[dev]"
pytest
```

---

## License

MIT © 2026 Remnant Fieldworks Inc.
