CompositeOT
===========

CompositeOT is a Python solver for optimal transport models with
transport regularization, marginal relaxation, and optional side constraints.
It is meant to make several OT variants available through one modeling
interface instead of separate problem-specific scripts.

The package solves problems of the form

.. math::

   \begin{aligned}
   \min_{X,D,y,z}\quad
      & \langle C, X\rangle + R(X) + h(D) + p_r(y) + p_c(z) \\
   \mathrm{s.t.}\quad
      & A X B + D = S, \\
      & X\mathbf{1} + y = \alpha, \\
      & X^\top\mathbf{1} + z = \beta.
   \end{aligned}

Here ``X`` is the transport plan, ``C`` is the transport cost, ``alpha`` and
``beta`` are the input marginals, and ``A X B = S`` is an optional linear side
relation.  Setting different regularizers recovers balanced OT, unbalanced OT,
partial OT, group-quadratic OT, and regularized martingale OT.

The modeling interface follows the style of optimization packages: construct a
model, add regularizers or constraints, then call ``model.solve()``.

.. code-block:: python

   import numpy as np

   from CompositeOT import Model, NonnegativeQuadraticRegularizer

   rng = np.random.default_rng(0)
   source = rng.normal(size=(100, 2))
   target = rng.normal(size=(100, 2))
   C = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
   alpha = np.full(100, 1.0 / 100)
   beta = np.full(100, 1.0 / 100)

   model = Model(C, alpha, beta)
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   result = model.solve()

Supported Regularizers
----------------------

The same regularizer interface is used for the transport term ``R(X)``, the
side slack term ``h(D)``, and the marginal slack terms ``p_r(y)`` and
``p_c(z)``.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Regularizer
     - Mathematical form
     - Typical use
   * - ``NonnegativeIndicator``
     - ``delta_{x >= 0}(x)``
     - Default transport domain; nonnegative slacks in partial OT.
   * - ``NonnegativeQuadraticRegularizer(lambda)``
     - ``delta_{x >= 0}(x) + lambda / 2 * ||x||_2^2``
     - QROT and partial OT transport regularization.
   * - ``GroupQuadraticRegularizer(lambda1, lambda2, groups, weights)``
     - ``delta_{x >= 0}(x) + lambda1 * sum_G w_G ||x_G||_2 + lambda2 / 2 * ||x||_2^2``
     - Group-quadratic transport regularization.
   * - ``SquaredL2Regularizer(lambda)``
     - ``lambda / 2 * ||x||_2^2``
     - Soft marginal relaxation and soft side constraints.
   * - ``L1Regularizer(lambda)``
     - ``lambda * ||x||_1``
     - Sparse marginal or side slack penalties.
   * - ``BoxIndicator(lower, upper)``
     - ``delta_{lower <= x <= upper}(x)``
     - Bounded transport, side, or marginal variables.
   * - ``L2BallIndicator(radius)``
     - ``delta_{||x||_2 <= radius}(x)``
     - Euclidean-ball slack constraints.
   * - ``ZeroIndicator``
     - ``delta_{x = 0}(x)``
     - Hard side constraints such as transported-mass equalities.
   * - ``ZeroRegularizer``
     - ``0``
     - Free slack variables without additional penalty.

Typical Workflow
----------------

1. Build or load ``C``, ``alpha``, and ``beta``.
2. Create ``Model(C, alpha, beta)``.
3. Add transport, row, column, or side regularizers as needed.
4. Call ``model.solve()``.
5. Inspect ``result.primal.X`` and the reported diagnostics.

Citations
---------

If you use CompositeOT or its ripALM-based solver components in academic work,
please cite:

.. code-block:: bibtex

   @article{yang2025convergence,
     title={Convergence of a Relative-type Inexact Proximal ALM for Convex Nonlinear Programming},
     author={Yang, Lei and Zhu, Jiayi and Liang, Ling and Toh, Kim-Chuan},
     journal={arXiv preprint arXiv:2510.25261},
     year={2025}
   }

   @article{zhu2024ripalm,
     title={ripALM: A Relative-Type Inexact Proximal Augmented Lagrangian Method for Linearly Constrained Convex Optimization},
     author={Zhu, Jiayi and Liang, Ling and Yang, Lei and Toh, Kim-Chuan},
     journal={arXiv preprint arXiv:2411.13267},
     year={2024}
   }

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   user_guide
   examples
   api
