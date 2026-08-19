# Quickstart GQOT Design

## Goal

Complete `quickstart.py` as a small, readable entry point that demonstrates balanced OT, balanced QROT, unbalanced QROT, partial QROT, balanced GQOT, unbalanced GQOT, and partial GQOT with one shared synthetic instance.

## Scope

The quickstart remains a script-style example. It should not grow a CLI, benchmark reporting layer, plotting dependency, or backend comparison. It should import through the public `CompositeOT` API wherever possible.

## Data Construction

Use `rng = np.random.default_rng(0)` and generate `source` as a `(100, 2)` float64 array. Generate `target` from `source` through a deterministic linear transformation: rotation, anisotropic scaling, translation, and a small fixed-seed noise term. This makes the transport geometry easier to inspect than two unrelated point clouds.

`C`, `alpha`, and `beta` are shared by all models. `C` is the squared Euclidean pairwise cost. Both histograms are uniform and have mass one.

## GQOT Grouping

GQOT grouping must follow the public helper pattern used by `examples/example_gqot.py`: import `make_gqot_example_data` from `CompositeOT.utils`, call it with the quickstart dimensions and a small group count, and use `data.composite_groups` and `data.composite_weights` for `GroupQuadraticRegularizer`.

Only the groups and weights are borrowed from the helper. The quickstart keeps its own transformed `source`, `target`, `C`, `alpha`, and `beta` so all seven models solve the same interpretable instance.

## Model Pattern

Each example follows the existing `Model(C, alpha, beta, name=...)` builder pattern:

- Balanced OT uses `NonnegativeIndicator`.
- Balanced QROT uses `NonnegativeQuadraticRegularizer(1.0)`.
- Unbalanced QROT uses `NonnegativeQuadraticRegularizer(1.0)` plus squared L2 row and column penalties.
- Partial QROT adds scalar transported-mass side constraint with nonnegative row and column slacks.
- Balanced GQOT uses `GroupQuadraticRegularizer(1.0, 1.0, groups, weights)`.
- Unbalanced GQOT combines the same GQOT transport regularizer with squared L2 row and column penalties.
- Partial GQOT combines the same GQOT transport regularizer with the scalar mass side constraint and nonnegative row and column slacks.

## Validation

Add a focused smoke test for `quickstart.py` that executes the script with monkeypatched solver methods. The test should verify that all model sections are exercised, including the three GQOT sections, without depending on long numerical convergence.
