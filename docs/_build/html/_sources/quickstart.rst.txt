Quickstart
==========

CompositeOT is easiest to install in a conda-forge environment because its
solver stack depends on ``scikit-sparse``, which wraps SuiteSparse/CHOLMOD.
Numba is not required, but it is recommended for larger examples.

Recommended Conda Environment
-----------------------------

.. code-block:: bash

   conda create -n compositeot -c conda-forge python=3.12 numpy scipy scikit-sparse numba
   conda activate compositeot
   python -m pip install -e .

If SuiteSparse is already available on your system, an advanced pip-oriented
workflow is also possible:

.. code-block:: bash

   python -m pip install -e .

Data Setup
----------

The examples below use one synthetic point cloud and a transformed noisy copy
of it. The cost matrix ``C`` stores squared Euclidean distances, and ``alpha``
and ``beta`` are uniform marginals.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(2026)

   m = 500
   source = rng.normal(size=(m, 2))

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

   C = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
   alpha = np.full(C.shape[0], 1.0 / C.shape[0])
   beta = np.full(C.shape[1], 1.0 / C.shape[1])

Balanced OT
-----------

Balanced OT keeps the transport plan nonnegative and enforces both marginals
as hard equalities. This is the baseline model before adding quadratic,
unbalanced, partial, or group structure.

.. code-block:: python

   import time

   from CompositeOT import Model, NonnegativeIndicator
   from CompositeOT.verbose import print_solver_result

   model = Model(C, alpha, beta, name="OT")
   model.add_transport_regularizer(NonnegativeIndicator())
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("OT", result, elapsed)

Balanced QROT
-------------

Balanced QROT adds a nonnegative quadratic regularizer to the transport plan.
The row and column marginals remain hard, but the quadratic term makes the
transport objective smoother.

.. code-block:: python

   import time

   from CompositeOT import Model, NonnegativeQuadraticRegularizer
   from CompositeOT.verbose import print_solver_result

   model = Model(C, alpha, beta, name="QROT")
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("QROT", result, elapsed)

The solution is returned as a ``SolverResult``:

.. code-block:: python

   X = result.primal.X
   u = result.dual.u
   v = result.dual.v
   print(result.objective_value, result.kkt_residual)

Unbalanced QROT
---------------

Unbalanced QROT keeps the quadratic transport regularizer and relaxes the row
and column equations with squared-l2 penalties. The slack variables let the
solver absorb mass mismatch at a controlled cost.

.. code-block:: python

   import time

   from CompositeOT import Model, NonnegativeQuadraticRegularizer, SquaredL2Regularizer
   from CompositeOT.verbose import print_solver_result

   model = Model(C, alpha, beta, name="Unbalanced QROT")
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   model.add_row_regularizer(SquaredL2Regularizer(10.0))
   model.add_column_regularizer(SquaredL2Regularizer(10.0))
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("Unbalanced QROT", result, elapsed)

Partial QROT
------------

Partial QROT fixes a transported mass through the side relation
``A @ X @ B + D = S``. Nonnegative row and column slacks allow unused mass to
remain outside the transported plan.

.. code-block:: python

   import time

   from CompositeOT import Model, NonnegativeIndicator, NonnegativeQuadraticRegularizer, ZeroIndicator
   from CompositeOT.verbose import print_solver_result

   mass = 0.8 * min(np.sum(alpha), np.sum(beta))
   A = np.ones((1, C.shape[0]))
   B = np.ones((C.shape[1], 1))
   S = np.array([[mass]])

   model = Model(C, alpha, beta, name="Partial QROT")
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   model.add_side_constraint(A, B, S, ZeroIndicator())
   model.add_row_regularizer(NonnegativeIndicator())
   model.add_column_regularizer(NonnegativeIndicator())
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("Partial QROT", result, elapsed)

Balanced GQOT
-------------

Balanced GQOT uses group structure in the transport regularizer. Here the flat
transport indices are split by source rows, so the regularizer can penalize
grouped transport mass in addition to the quadratic term.

.. code-block:: python

   import time

   from CompositeOT import Model, GroupQuadraticRegularizer
   from CompositeOT.verbose import print_solver_result

   flat_indices = np.arange(C.size).reshape(C.shape, order="F")
   split = source.shape[0] // 2
   groups = (
       flat_indices[:split, :].ravel(order="F"),
       flat_indices[split:, :].ravel(order="F"),
   )
   weights = (1.0, 1.0)

   model = Model(C, alpha, beta, name="GQOT")
   model.add_transport_regularizer(
       GroupQuadraticRegularizer(1.0, 1.0, groups, weights)
   )
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("GQOT", result, elapsed)

Unbalanced GQOT
---------------

Unbalanced GQOT combines group-quadratic transport regularization with soft
row and column penalties. This is useful when the grouped structure matters
but exact marginal matching is too restrictive.

.. code-block:: python

   import time

   from CompositeOT import Model, GroupQuadraticRegularizer, SquaredL2Regularizer
   from CompositeOT.verbose import print_solver_result

   flat_indices = np.arange(C.size).reshape(C.shape, order="F")
   split = source.shape[0] // 2
   groups = (
       flat_indices[:split, :].ravel(order="F"),
       flat_indices[split:, :].ravel(order="F"),
   )
   weights = (1.0, 1.0)

   model = Model(C, alpha, beta, name="Unbalanced GQOT")
   model.add_transport_regularizer(
       GroupQuadraticRegularizer(1.0, 1.0, groups, weights)
   )
   model.add_row_regularizer(SquaredL2Regularizer(10.0))
   model.add_column_regularizer(SquaredL2Regularizer(10.0))
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("Unbalanced GQOT", result, elapsed)

Partial GQOT
------------

Partial GQOT combines the transported-mass side constraint from partial QROT
with the group-quadratic transport regularizer. The variables ``A``, ``B``,
and ``S`` are the same mass constraint arrays defined in the partial QROT
example.

.. code-block:: python

   import time

   from CompositeOT import Model, GroupQuadraticRegularizer, NonnegativeIndicator, ZeroIndicator
   from CompositeOT.verbose import print_solver_result

   flat_indices = np.arange(C.size).reshape(C.shape, order="F")
   split = source.shape[0] // 2
   groups = (
       flat_indices[:split, :].ravel(order="F"),
       flat_indices[split:, :].ravel(order="F"),
   )
   weights = (1.0, 1.0)

   model = Model(C, alpha, beta, name="Partial GQOT")
   model.add_transport_regularizer(
       GroupQuadraticRegularizer(1.0, 1.0, groups, weights)
   )
   model.add_side_constraint(A, B, S, ZeroIndicator())
   model.add_row_regularizer(NonnegativeIndicator())
   model.add_column_regularizer(NonnegativeIndicator())
   start = time.perf_counter()
   result = model.solve()
   elapsed = time.perf_counter() - start
   print_solver_result("Partial GQOT", result, elapsed)

Solver Options
--------------

Options are plain dataclasses. By default ``model.solve`` uses a first-order
warm start and then a PALM refinement step. Most users only need to set the
iteration limits, tolerances, and whether verbose iteration tables should be
printed.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Meaning
   * - ``PALMOptions.max_iterations``
     - ``100``
     - Maximum number of outer PALM iterations.
   * - ``PALMOptions.tolerance``
     - ``1e-6``
     - Target KKT tolerance for the refined solve.
   * - ``PALMOptions.sigma``
     - ``1.0``
     - Initial augmented-Lagrangian penalty.
   * - ``PALMOptions.tau``
     - ``5.0``
     - Proximal regularization used in the PALM subproblem.
   * - ``PALMOptions.sigma_growth``
     - ``1.5``
     - Multiplicative growth factor when ``sigma`` is updated.
   * - ``PALMOptions.verbose``
     - ``False``
     - Print the PALM iteration table.
   * - ``PALMOptions.newton.max_iterations``
     - ``100``
     - Maximum semismooth Newton iterations per PALM subproblem.
   * - ``PALMOptions.newton.tolerance``
     - ``1e-8``
     - Inner semismooth Newton stopping tolerance.
   * - ``ADMMOptions.max_iterations``
     - ``300``
     - Maximum SGS-ADMM warm-start iterations. Set this to ``0`` to skip the
       warm start and run PALM directly.
   * - ``ADMMOptions.tolerance``
     - ``1e-3``
     - KKT tolerance used by the warm-start phase.
   * - ``ADMMOptions.penalty``
     - ``None``
     - Initial SGS-ADMM penalty. ``None`` lets the solver choose one from the
       cost scale.
   * - ``ADMMOptions.verbose``
     - ``False``
     - Print the SGS-ADMM iteration table.

.. code-block:: python

   from CompositeOT import ADMMOptions, PALMOptions, SemismoothNewtonOptions

   newton = SemismoothNewtonOptions(max_iterations=80, tolerance=1e-8)
   palm = PALMOptions(max_iterations=100, tolerance=1e-6, tau=5.0, newton=newton)
   warm_start = ADMMOptions(max_iterations=300, tolerance=1e-3)
   result = model.solve(palm_options=palm, admm_options=warm_start)

To start PALM directly, skip the warm start:

.. code-block:: python

   palm_only = ADMMOptions(max_iterations=0)
   result = model.solve(palm_options=palm, admm_options=palm_only)

To run only the warm-start solver, use:

.. code-block:: python

   warm_result = model.solve_sgsadmm(warm_start)

Verbose Output
--------------

Both solver phases use the same diagnostic columns:

``pobj``, ``dobj``, ``gap``
   Primal objective, dual objective, and relative gap evaluated on the original
   unscaled data.

``pres`` and ``dres``
   Scaled primal feasibility and stationarity residuals.

``kkt``
   The maximum of the reported residuals used for termination.

``sigma``, ``tau``, ``inner``
   Solver parameters and the cumulative number of inner iterations reported by
   the PALM phase. The warm-start phase prints only the columns that apply to
   it.
