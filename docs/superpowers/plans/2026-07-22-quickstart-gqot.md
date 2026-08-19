# Quickstart GQOT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `quickstart.py` with interpretable transformed data and balanced, unbalanced, and partial GQOT examples.

**Architecture:** Keep `quickstart.py` as a small script-style example using the public `CompositeOT` API. Use `CompositeOT.utils.make_gqot_example_data` only to obtain `composite_groups` and `composite_weights`, mirroring `examples/example_gqot.py` while sharing the quickstart's own cost and histograms across all models.

**Tech Stack:** Python 3.11+, NumPy, pytest, CompositeOT public modeling API.

## Global Constraints

- Runtime code remains in `quickstart.py`; no CLI, plotting, backend comparison, or benchmark reporting layer.
- GQOT grouping follows `examples/example_gqot.py` by using `make_gqot_example_data(...).composite_groups` and `.composite_weights`.
- The quickstart keeps one shared transformed source-target instance for all seven model sections.
- Tests must avoid long numerical convergence by monkeypatching solver execution.

---

### Task 1: Quickstart Smoke Test

**Files:**
- Create: `tests/test_quickstart.py`
- Modify: none
- Test: `tests/test_quickstart.py`

**Interfaces:**
- Consumes: `CompositeOT.CompositeOTProblem.solve`
- Produces: A smoke test that records problem names and regularizer types used by `quickstart.py`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import runpy
from types import SimpleNamespace

import numpy as np

from CompositeOT.model import CompositeOTProblem
from CompositeOT.regularizers import GroupQuadraticRegularizer


def test_quickstart_builds_all_models(monkeypatch):
    solved = []

    def fake_solve(self, *args, **kwargs):
        solved.append(self)
        return SimpleNamespace(problem=self)

    monkeypatch.setattr(CompositeOTProblem, "solve", fake_solve)

    namespace = runpy.run_path("quickstart.py")

    assert namespace["source"].shape == (100, 2)
    assert namespace["target"].shape == (100, 2)
    assert namespace["C"].shape == (100, 100)
    assert np.allclose(namespace["alpha"].sum(), 1.0)
    assert np.allclose(namespace["beta"].sum(), 1.0)
    assert not np.allclose(namespace["source"], namespace["target"])

    assert [problem.name for problem in solved] == [
        "OT",
        "QROT",
        "Unbalanced QROT",
        "Partial QROT",
        "GQOT",
        "Unbalanced GQOT",
        "Partial GQOT",
    ]
    assert sum(
        isinstance(problem.transport_regularizer, GroupQuadraticRegularizer)
        for problem in solved
    ) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_quickstart.py`
Expected: FAIL because `quickstart.py` references undefined `target` and does not build the GQOT models yet.

- [ ] **Step 3: Leave test unchanged**

No production code changes in this task.

### Task 2: Complete Quickstart

**Files:**
- Modify: `quickstart.py`
- Test: `tests/test_quickstart.py`

**Interfaces:**
- Consumes: `make_gqot_example_data(m, n, seed, n_groups)` and `GroupQuadraticRegularizer(group_weight, quadratic_weight, groups, weights)`.
- Produces: Script variables `source`, `target`, `C`, `alpha`, `beta`, `groups`, `weights`, and seven solved model sections.

- [ ] **Step 1: Implement target generation**

Add deterministic rotation, scaling, translation, and small noise after `source`:

```python
angle = np.deg2rad(35.0)
rotation = np.array(
    [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ],
    dtype=np.float64,
)
scaling = np.diag([1.4, 0.7])
translation = np.array([1.0, -0.5], dtype=np.float64)
target = source @ (rotation @ scaling).T + translation
target += 0.05 * rng.normal(size=target.shape)
```

- [ ] **Step 2: Implement GQOT grouping and sections**

Import `GroupQuadraticRegularizer` and `make_gqot_example_data`. Build helper data with `(100, 100, 0, 10)`, assign `groups = data.composite_groups` and `weights = data.composite_weights`, then add balanced, unbalanced, and partial GQOT sections matching the QROT pattern.

- [ ] **Step 3: Run focused test**

Run: `python -m pytest -q tests/test_quickstart.py`
Expected: PASS.

- [ ] **Step 4: Run syntax check**

Run: `python -m compileall -q quickstart.py tests/test_quickstart.py`
Expected: exit code 0.
