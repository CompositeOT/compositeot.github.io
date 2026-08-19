User Guide
==========

Problem Form
------------

CompositeOT models a transport plan ``X`` with optional side slack ``D`` and
marginal slacks ``y`` and ``z``:

.. math::

   \begin{aligned}
   \min_{X,D,y,z}\quad
      & \langle C, X\rangle + R(X) + h(D) + p_r(y) + p_c(z) \\
   \mathrm{s.t.}\quad
      & A X B + D = S, \\
      & X\mathbf{1} + y = \alpha, \\
      & X^\top\mathbf{1} + z = \beta.
   \end{aligned}

The side relation ``A X B + D = S`` is optional.  If no side constraint is
added, the model only enforces the row and column marginal equations.

Model Components
----------------

A model has four regularizer slots:

``transport``
   The regularizer ``R(X)`` on the transport plan. Typical choices are
   ``NonnegativeIndicator``, ``NonnegativeQuadraticRegularizer``, and
   ``GroupQuadraticRegularizer``.

``side``
   The regularizer ``h(D)`` for side-constraint slack. ``ZeroIndicator`` makes
   the side relation hard; ``SquaredL2Regularizer`` makes it soft.

``row`` and ``column``
   The marginal slack regularizers ``p_r(y)`` and ``p_c(z)``. ``ZeroIndicator``
   gives hard balanced marginals, while nonnegative or squared-l2 choices
   produce partial and unbalanced variants.

Common Models
-------------

Balanced OT
   Use the default transport nonnegativity and default hard marginals.

QROT
   Add ``NonnegativeQuadraticRegularizer(lambda_q)`` on the transport plan.

UOT
   Add row and column penalties, for example ``SquaredL2Regularizer``.

Partial OT
   Add a transported-mass side constraint and nonnegative marginal slacks.

GQOT
   Add ``GroupQuadraticRegularizer(lambda1, lambda2, groups, weights)``.

Martingale OT
   Add a side relation ``A X B = S`` representing the martingale moment
   condition.  Use ``ZeroIndicator`` for a hard constraint or
   ``SquaredL2Regularizer`` for a soft constraint.

Scaling
-------

By default ``solve`` scales the problem internally and unscales the returned
solution and diagnostics. Users should normally pass the original data and let
the solver handle scaling.

Solving
-------

``model.solve()`` is the recommended entry point.  It uses a first-order warm
start followed by a PALM refinement step.  Set
``ADMMOptions(max_iterations=0)`` when you want to start PALM directly.

.. code-block:: python

   from CompositeOT import ADMMOptions, PALMOptions

   palm = PALMOptions(max_iterations=100, tolerance=1e-6, tau=5.0)
   warm = ADMMOptions(max_iterations=300, tolerance=1e-3)
   result = model.solve(palm_options=palm, admm_options=warm)

Results
-------

The returned ``SolverResult`` contains:

* ``result.primal`` with ``X``, ``D``, ``y``, and ``z``.
* ``result.dual`` with ``W``, ``u``, and ``v``.
* ``result.objective_value`` and ``result.dual_objective_value``.
* ``result.relative_gap``, ``result.primal_residual``,
  ``result.dual_residual``, and ``result.kkt_residual``.
* ``result.history`` for iteration diagnostics.
