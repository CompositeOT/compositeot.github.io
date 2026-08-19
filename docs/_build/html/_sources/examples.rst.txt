Examples
========

The examples are designed to be runnable from the repository root.  Each one
follows the same workflow:

1. generate ``C``, ``alpha``, and ``beta``;
2. add any extra data such as groups, transported mass, or side constraints;
3. construct a ``Model``;
4. call ``solve`` and inspect the printed diagnostics.

Generating Data
---------------

Most examples start from the same three arrays:

``C``
   the ``m`` by ``n`` transport cost matrix;
``alpha``
   the source marginal with shape ``(m,)``;
``beta``
   the target marginal with shape ``(n,)``.

The source and target point clouds are only used to build these arrays.  The
solver receives ``C``, ``alpha``, and ``beta``.

A complete construction is:

.. code-block:: python

   import numpy as np

   m = 1000
   n = 1000
   dimension = 3
   seed = 0

   rng = np.random.default_rng(seed)
   means = np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float64)
   sigma = 5.0

   def sample_cloud(size, dimension, rng):
       weights = rng.random(means.size)
       weights /= weights.sum()

       components = rng.choice(means.size, size=dimension * size, p=weights)
       support = rng.normal(means[components], sigma).reshape(dimension, size).T

       mass = rng.random(size).astype(np.float64)
       mass /= mass.sum()
       return support.astype(np.float64), mass

   def normalized_squared_distance(source, target):
       source_sq = np.sum(source * source, axis=1)[:, None]
       target_sq = np.sum(target * target, axis=1)[None, :]
       C = source_sq + target_sq - 2.0 * (source @ target.T)
       np.maximum(C, 0.0, out=C)

       max_cost = float(np.max(C))
       if max_cost > 0.0:
           C /= max_cost
       return C.astype(np.float64)

   source, alpha = sample_cloud(m, dimension, rng)
   target, beta = sample_cloud(n, dimension, rng)
   C = normalized_squared_distance(source, target)

After this block, the arrays passed to ``Model`` are:

.. code-block:: python

   C.shape      # (m, n)
   alpha.shape  # (m,)
   beta.shape   # (n,)
   alpha.sum()  # 1.0
   beta.sum()   # 1.0

Here ``C[i, j]`` is the normalized squared distance between source point ``i``
and target point ``j``.  The entries ``alpha[i]`` and ``beta[j]`` are
nonnegative masses, normalized so that each marginal sums to one.

The examples below show real solver transcripts from the commands listed in
each section.  Wall-clock times can change across machines.

Benchmark Methodology
---------------------

The following local benchmark uses the six public example scripts with their
default CLI settings and ``m = n`` in ``{1000, 2000, 3000, 4000, 5000}``. Each
script was warmed once on a small instance before timing. Times are wall-clock
seconds on the local development machine and should be read as implementation
checks, not portable performance guarantees. All runs below converged.

QROT
----

Problem:

.. math::

   \begin{aligned}
   \min_X\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)
      + \frac{1}{2}\|X\|_F^2}_{R(X)} \\
   \mathrm{s.t.}\quad
      & X\mathbf{1} = \alpha, \\
      & X^\top\mathbf{1} = \beta.
   \end{aligned}

The displayed formula matches the example default
``NonnegativeQuadraticRegularizer(1.0)``.

Data generation:

.. code-block:: python

   source, alpha = sample_cloud(m, dimension, rng)
   target, beta = sample_cloud(n, dimension, rng)
   C = normalized_squared_distance(source, target)

Model and solve:

.. code-block:: python

   from CompositeOT import Model, NonnegativeQuadraticRegularizer

   model = Model(C, alpha, beta, name="QROT")
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_qrot.py --m 30 --n 30 --max-iterations 50 \
       --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  7.613867e-02 |  7.738603e-02 | 1.1e-03 | 4.0e-04 | 1.2e-03 | 1.2e-03 | 3.5e-02 | 1.3e-01
     200 |  7.740318e-02 |  7.741253e-02 | 8.1e-06 | 1.6e-05 | 7.3e-06 | 1.6e-05 | 6.9e-02 | 1.4e-01
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  7.740318e-02 |  7.741253e-02 | 8.1e-06 | 1.6e-05 | 7.3e-06 | 1.6e-05 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  7.742561e-02 |  7.741254e-02 | 1.1e-05 | 1.3e-05 | 3.8e-05 | 3.8e-05 | 1.3e-02 | 1.0e+00 | 5.0e+00 |       1
       2 |  7.742796e-02 |  7.741255e-02 | 1.3e-05 | 7.9e-06 | 4.6e-05 | 4.6e-05 | 1.4e-02 | 1.5e+00 | 5.0e+00 |       2
       3 |  7.742946e-02 |  7.741257e-02 | 1.5e-05 | 3.1e-06 | 4.8e-05 | 4.8e-05 | 1.6e-02 | 2.2e+00 | 5.0e+00 |       4
       4 |  7.742787e-02 |  7.741259e-02 | 1.3e-05 | 3.1e-06 | 4.0e-05 | 4.0e-05 | 1.7e-02 | 3.4e+00 | 5.0e+00 |       5
       5 |  7.742270e-02 |  7.741259e-02 | 8.7e-06 | 4.4e-06 | 2.3e-05 | 2.3e-05 | 1.9e-02 | 5.1e+00 | 5.0e+00 |       6
       6 |  7.741637e-02 |  7.741259e-02 | 3.3e-06 | 3.0e-06 | 6.7e-06 | 6.7e-06 | 2.0e-02 | 7.6e+00 | 5.0e+00 |       7
       7 |  7.741267e-02 |  7.741259e-02 | 6.7e-08 | 9.3e-07 | 1.4e-06 | 1.4e-06 | 2.1e-02 | 1.1e+01 | 5.0e+00 |       8
       8 |  7.741211e-02 |  7.741259e-02 | 4.2e-07 | 8.9e-08 | 1.1e-06 | 1.1e-06 | 2.2e-02 | 1.7e+01 | 5.0e+00 |       9
       9 |  7.741245e-02 |  7.741259e-02 | 1.3e-07 | 4.7e-08 | 2.5e-07 | 2.5e-07 | 2.3e-02 | 2.6e+01 | 5.0e+00 |      10

   QROT
     converged: True (Converged with max(KKT residual, relative gap).)
     iterations: 9 outer, 200 ADMM
     objective:  7.741245e-02
     dual objective:  7.741259e-02
     relative gap: 1.3e-07
     feasibility: 2.5e-07
     stationarity: 4.7e-08
     KKT residual: 2.5e-07
     time: 1.7e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 2.5, 16, 200, 3.5e-07, 1.158346e-02
   2000, 8.7, 16, 200, 6.4e-07, 1.943852e-02
   3000, 16.7, 16, 200, 7.3e-07, 2.872860e-02
   4000, 29.5, 16, 200, 8.8e-07, 2.946718e-02
   5000, 47.4, 17, 200, 4.5e-07, 1.219550e-02

UOT
---

The UOT example adds noisy source support points and reduces target mass, then
relaxes the marginal equations with squared-l2 slack penalties.

Problem:

.. math::

   \begin{aligned}
   \min_{X,y,z}\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)
      + \frac{1}{2}\|X\|_F^2}_{R(X)}
      + \underbrace{\frac{1}{2}\|y\|_2^2}_{p_r(y)}
      + \underbrace{\frac{1}{2}\|z\|_2^2}_{p_c(z)} \\
   \mathrm{s.t.}\quad
      & X\mathbf{1} + y = \alpha, \\
      & X^\top\mathbf{1} + z = \beta.
   \end{aligned}

The displayed formula uses the example defaults
``lambda_q = lambda_r = lambda_c = 1``.

Additional data generation:

.. code-block:: python

   source, alpha = sample_cloud(m, dimension, rng)

   noise_count = int(np.ceil(noise_ratio * m))
   indices = rng.choice(m, size=noise_count, replace=False)
   center = np.mean(source[indices], axis=0, keepdims=True)
   spread = np.std(source[indices], axis=0, keepdims=True) + 1e-6
   noise = rng.normal(
       loc=center + 2.5 * spread,
       scale=1.2 * spread,
       size=(noise_count, dimension),
   )
   source = np.vstack([source, noise]).astype(np.float64, copy=False)
   alpha = np.concatenate([alpha, np.full(noise_count, 1.0 / m)])

   target, beta = sample_cloud(n, dimension, rng)
   beta *= target_mass
   C = normalized_squared_distance(source, target)

Model and solve:

.. code-block:: python

   from CompositeOT import Model, NonnegativeQuadraticRegularizer, SquaredL2Regularizer

   model = Model(C, alpha, beta, name="UOT-sqL2")
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(1.0))
   model.add_row_regularizer(SquaredL2Regularizer(1.0))
   model.add_column_regularizer(SquaredL2Regularizer(1.0))
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_uot.py --m 30 --n 30 --max-iterations 50 \
       --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  2.295842e-02 |  2.369224e-02 | 7.0e-04 | 4.5e-04 | 1.2e-03 | 1.2e-03 | 6.9e-02 | 1.6e-01
     200 |  2.370868e-02 |  2.371484e-02 | 5.9e-06 | 2.5e-06 | 7.1e-06 | 7.1e-06 | 1.2e-01 | 1.8e-01
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  2.370868e-02 |  2.371484e-02 | 5.9e-06 | 2.5e-06 | 7.1e-06 | 7.1e-06 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  2.371451e-02 |  2.371484e-02 | 3.1e-07 | 1.6e-06 | 5.2e-06 | 5.2e-06 | 1.2e-02 | 1.0e+00 | 5.0e+00 |       1
       2 |  2.371457e-02 |  2.371484e-02 | 2.6e-07 | 5.9e-07 | 5.9e-06 | 5.9e-06 | 1.4e-02 | 1.5e+00 | 5.0e+00 |       2
       3 |  2.371472e-02 |  2.371484e-02 | 1.1e-07 | 4.5e-07 | 4.1e-06 | 4.1e-06 | 1.5e-02 | 2.2e+00 | 5.0e+00 |       3
       4 |  2.371481e-02 |  2.371484e-02 | 3.2e-08 | 4.9e-07 | 1.2e-06 | 1.2e-06 | 1.6e-02 | 3.4e+00 | 5.0e+00 |       4
       5 |  2.371484e-02 |  2.371484e-02 | 3.0e-09 | 1.6e-07 | 3.9e-07 | 3.9e-07 | 1.8e-02 | 5.1e+00 | 5.0e+00 |       5

   UOT-sqL2
     converged: True (Converged with max(KKT residual, relative gap).)
     iterations: 5 outer, 200 ADMM
     objective:  2.371484e-02
     dual objective:  2.371484e-02
     relative gap: 3.0e-09
     feasibility: 3.9e-07
     stationarity: 1.6e-07
     KKT residual: 3.9e-07
     time: 2.0e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 2.8, 15, 300, 4.0e-07, 8.644641e-04
   2000, 10.7, 15, 300, 9.3e-07, 5.268401e-04
   3000, 21.8, 16, 300, 5.4e-07, 2.893390e-04
   4000, 36.4, 16, 300, 5.9e-07, 2.550819e-04
   5000, 57.6, 16, 300, 7.6e-07, 2.042758e-04

GQOT
----

GQOT uses a group-quadratic transport regularizer.  With the example defaults
``lambda1 = lambda2 = 1``, the problem is

.. math::

   \begin{aligned}
   \min_X\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)
      + \sum_G w_G\|X_G\|_2
      + \frac{1}{2}\|X\|_F^2}_{R(X)} \\
   \mathrm{s.t.}\quad
      & X\mathbf{1} = \alpha, \\
      & X^\top\mathbf{1} = \beta.
   \end{aligned}

Additional data generation:

.. code-block:: python

   alpha = np.ones(m, dtype=np.float64) / m
   beta = np.ones(n, dtype=np.float64) / n

   source_1 = rng.multivariate_normal([-1.0, 2.0], 0.25 * np.eye(2), m // 2)
   source_2 = rng.multivariate_normal([1.0, 2.0], 0.25 * np.eye(2), m // 2)
   source = np.vstack([source_1, source_2])

   components = rng.integers(0, 2, size=n)
   target_1 = rng.multivariate_normal(
       [-2.0, 2.0], 0.5 * np.eye(2), int((components == 0).sum())
   )
   target_2 = rng.multivariate_normal(
       [2.0, 3.0], 0.5 * np.eye(2), int((components == 1).sum())
   )
   target = np.vstack([target_1, target_2])
   target = target[np.argsort(target[:, 0])]

   C = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
   np.maximum(C, 0.0, out=C)

   row_groups, row_group_weights = build_row_groups(m, n, n_groups)
   groups, weights = build_composite_groups(row_groups, row_group_weights, m, n)

Model and solve:

.. code-block:: python

   from CompositeOT import GroupQuadraticRegularizer, Model

   model = Model(C, alpha, beta, name="GQOT")
   model.add_transport_regularizer(
       GroupQuadraticRegularizer(1.0, 1.0, groups, weights)
   )
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_gqot.py --m 30 --n 30 --max-iterations 50 \
       --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  2.921846e+00 |  2.676532e+00 | 3.7e-02 | 2.6e-04 | 1.1e-03 | 1.1e-03 | 6.3e-02 | 3.9e-03
     200 |  2.940436e+00 |  2.781999e+00 | 2.4e-02 | 1.1e-03 | 6.6e-06 | 1.1e-03 | 9.3e-02 | 7.9e-03
     300 |  2.933704e+00 |  2.889504e+00 | 6.5e-03 | 5.8e-04 | 3.9e-08 | 5.8e-04 | 1.3e-01 | 1.6e-02
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  2.933704e+00 |  2.889504e+00 | 6.5e-03 | 5.8e-04 | 3.9e-08 | 5.8e-04 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  2.937066e+00 |  2.912616e+00 | 3.6e-03 | 3.9e-04 | 6.5e-04 | 6.5e-04 | 7.7e-03 | 1.0e+00 | 5.0e+00 |       1
       2 |  2.937361e+00 |  2.932253e+00 | 7.4e-04 | 6.7e-05 | 8.1e-04 | 8.1e-04 | 9.8e-03 | 1.5e+00 | 5.0e+00 |       3
       3 |  2.933667e+00 |  2.932610e+00 | 1.5e-04 | 4.4e-05 | 2.0e-04 | 2.0e-04 | 1.2e-02 | 2.2e+00 | 5.0e+00 |       5
       4 |  2.932672e+00 |  2.932820e+00 | 2.1e-05 | 1.5e-05 | 3.7e-05 | 3.7e-05 | 1.4e-02 | 3.4e+00 | 5.0e+00 |       7
       5 |  2.932804e+00 |  2.932848e+00 | 6.4e-06 | 5.3e-07 | 1.2e-05 | 1.2e-05 | 1.5e-02 | 5.1e+00 | 5.0e+00 |       9
       6 |  2.932848e+00 |  2.932848e+00 | 7.9e-08 | 3.6e-07 | 2.4e-07 | 3.6e-07 | 1.6e-02 | 7.6e+00 | 5.0e+00 |      10

   GQOT
     converged: True (Converged with max(KKT residual, relative gap).)
     iterations: 6 outer, 300 ADMM
     objective:  2.932848e+00
     dual objective:  2.932848e+00
     relative gap: 7.9e-08
     feasibility: 2.4e-07
     stationarity: 3.6e-07
     KKT residual: 3.6e-07
     time: 2.1e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 5.5, 6, 100, 1.9e-07, 3.523607e+00
   2000, 20.8, 6, 100, 1.1e-07, 3.599602e+00
   3000, 45.2, 6, 100, 1.3e-07, 3.423097e+00
   4000, 95.8, 6, 100, 1.0e-08, 3.500836e+00
   5000, 142.7, 5, 100, 6.8e-08, 3.505960e+00

Partial OT
----------

Partial OT adds a transported-mass equality and nonnegative marginal slacks.

Problem:

.. math::

   \begin{aligned}
   \min_{X,D,y,z}\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)}_{R(X)}
      + \underbrace{\delta_{\{D=0\}}(D)}_{h(D)}
      + \underbrace{\delta_{\{y \ge 0\}}(y)}_{p_r(y)}
      + \underbrace{\delta_{\{z \ge 0\}}(z)}_{p_c(z)} \\
   \mathrm{s.t.}\quad
      & \mathbf{1}^\top X\mathbf{1} + D = S, \\
      & X\mathbf{1} + y = \alpha, \\
      & X^\top\mathbf{1} + z = \beta.
   \end{aligned}

The displayed formula uses the example default ``lambda_q = 0``, so
``NonnegativeQuadraticRegularizer(0.0)`` reduces to the nonnegativity
indicator.

Additional data generation:

.. code-block:: python

   source, alpha = sample_cloud(m, dimension, rng)
   target, beta = sample_cloud(n, dimension, rng)
   beta *= target_mass
   C = normalized_squared_distance(source, target)

   mass = partial_fraction * min(alpha.sum(), beta.sum())
   A = np.ones((1, C.shape[0]), dtype=np.float64)
   B = np.ones((C.shape[1], 1), dtype=np.float64)
   S = np.array([[mass]], dtype=np.float64)

Here ``S`` is a ``1`` by ``1`` array containing the transported mass.

Model and solve:

.. code-block:: python

   from CompositeOT import Model, NonnegativeIndicator, NonnegativeQuadraticRegularizer
   from CompositeOT import ZeroIndicator

   model = Model(C, alpha, beta, name="partialOT")
   model.add_side_constraint(A, B, S, regularizer=ZeroIndicator())
   model.add_transport_regularizer(NonnegativeQuadraticRegularizer(0.0))
   model.add_row_regularizer(NonnegativeIndicator())
   model.add_column_regularizer(NonnegativeIndicator())
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_partial_ot.py --m 30 --n 30 --max-iterations 50 \
       --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  1.495809e-02 |  1.029147e-02 | 4.6e-03 | 4.0e-03 | 2.1e-03 | 4.0e-03 | 3.3e-02 | 1.6e-01
     200 |  1.211361e-02 |  1.201502e-02 | 9.6e-05 | 1.1e-03 | 2.9e-04 | 1.1e-03 | 6.4e-02 | 1.8e-01
     300 |  1.271181e-02 |  1.237412e-02 | 3.3e-04 | 4.3e-04 | 4.0e-04 | 4.3e-04 | 8.9e-02 | 2.0e-01
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  1.271181e-02 |  1.237412e-02 | 3.3e-04 | 4.3e-04 | 4.0e-04 | 4.3e-04 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  1.220788e-02 |  1.261389e-02 | 4.0e-04 | 5.2e-04 | 5.8e-04 | 5.8e-04 | 1.3e-02 | 1.0e+00 | 5.0e+00 |       1
       2 |  1.221506e-02 |  1.236338e-02 | 1.4e-04 | 3.8e-04 | 5.5e-04 | 5.5e-04 | 1.5e-02 | 1.5e+00 | 5.0e+00 |       2
       3 |  1.222131e-02 |  1.229473e-02 | 7.2e-05 | 3.4e-04 | 4.9e-04 | 4.9e-04 | 1.6e-02 | 2.2e+00 | 5.0e+00 |       3
       4 |  1.224285e-02 |  1.229965e-02 | 5.5e-05 | 3.2e-04 | 4.1e-04 | 4.1e-04 | 1.8e-02 | 3.4e+00 | 5.0e+00 |       4
       5 |  1.227615e-02 |  1.232814e-02 | 5.1e-05 | 3.1e-04 | 2.6e-04 | 3.1e-04 | 2.1e-02 | 5.1e+00 | 5.0e+00 |       6
       6 |  1.231037e-02 |  1.234306e-02 | 3.2e-05 | 2.9e-04 | 7.8e-05 | 2.9e-04 | 2.2e-02 | 7.6e+00 | 5.0e+00 |       7
       7 |  1.232623e-02 |  1.234534e-02 | 1.9e-05 | 2.2e-04 | 1.5e-04 | 2.2e-04 | 2.4e-02 | 1.1e+01 | 5.0e+00 |       8
       8 |  1.231086e-02 |  1.234526e-02 | 3.4e-05 | 2.0e-04 | 1.1e-04 | 2.0e-04 | 2.7e-02 | 1.7e+01 | 5.0e+00 |      10
       9 |  1.228021e-02 |  1.233521e-02 | 5.4e-05 | 1.7e-04 | 1.6e-05 | 1.7e-04 | 2.9e-02 | 2.6e+01 | 5.0e+00 |      12
      10 |  1.226433e-02 |  1.231150e-02 | 4.6e-05 | 1.3e-04 | 5.2e-05 | 1.3e-04 | 3.2e-02 | 3.8e+01 | 5.0e+00 |      14
      11 |  1.226895e-02 |  1.228843e-02 | 1.9e-05 | 6.1e-05 | 1.9e-05 | 6.1e-05 | 3.5e-02 | 5.8e+01 | 5.0e+00 |      16
      12 |  1.227114e-02 |  1.227055e-02 | 5.7e-07 | 2.1e-06 | 8.6e-06 | 8.6e-06 | 3.6e-02 | 8.6e+01 | 5.0e+00 |      17
      13 |  1.227094e-02 |  1.227082e-02 | 1.2e-07 | 6.3e-07 | 3.2e-07 | 6.3e-07 | 3.8e-02 | 1.3e+02 | 5.0e+00 |      18

   partialOT
     converged: True (Converged with KKT residual below tolerance.)
     iterations: 13 outer, 300 ADMM
     objective:  1.227094e-02
     dual objective:  1.227082e-02
     relative gap: 1.2e-07
     feasibility: 3.2e-07
     stationarity: 6.3e-07
     KKT residual: 6.3e-07
     time: 1.9e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 5.9, 19, 300, 8.3e-07, 1.787773e-03
   2000, 20.5, 19, 300, 6.8e-07, 3.371333e-03
   3000, 40.7, 19, 300, 7.2e-07, 7.765692e-04
   4000, 75.7, 19, 300, 6.0e-07, 1.539107e-03
   5000, 121.9, 19, 300, 5.9e-07, 2.014684e-03

Partial GQOT
------------

Partial GQOT combines the transported-mass equality with the group-quadratic
transport regularizer.

Problem:

.. math::

   \begin{aligned}
   \min_{X,D,y,z}\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)
      + \sum_G w_G\|X_G\|_2
      + \frac{1}{2}\|X\|_F^2}_{R(X)}
      + \underbrace{\delta_{\{D=0\}}(D)}_{h(D)}
      + \underbrace{\delta_{\{y \ge 0\}}(y)}_{p_r(y)}
      + \underbrace{\delta_{\{z \ge 0\}}(z)}_{p_c(z)} \\
   \mathrm{s.t.}\quad
      & \mathbf{1}^\top X\mathbf{1} + D = S, \\
      & X\mathbf{1} + y = \alpha, \\
      & X^\top\mathbf{1} + z = \beta.
   \end{aligned}

The displayed formula matches the example defaults
``GroupQuadraticRegularizer(1.0, 1.0, groups, weights)``.

Additional data generation:

.. code-block:: python

   # Use the same two-cluster support, cost, and groups as GQOT.
   mass = partial_fraction * min(alpha.sum(), beta.sum())
   A = np.ones((1, C.shape[0]), dtype=np.float64)
   B = np.ones((C.shape[1], 1), dtype=np.float64)
   S = np.array([[mass]], dtype=np.float64)

Here ``S`` is a ``1`` by ``1`` array containing the transported mass.

Model and solve:

.. code-block:: python

   from CompositeOT import GroupQuadraticRegularizer, Model, NonnegativeIndicator
   from CompositeOT import ZeroIndicator

   model = Model(C, alpha, beta, name="partialGQOT")
   model.add_side_constraint(A, B, S, regularizer=ZeroIndicator())
   model.add_transport_regularizer(
       GroupQuadraticRegularizer(1.0, 1.0, groups, weights)
   )
   model.add_row_regularizer(NonnegativeIndicator())
   model.add_column_regularizer(NonnegativeIndicator())
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_partial_gqot.py --m 30 --n 30 --max-iterations 50 \
       --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  1.785726e+00 | -1.287959e+01 | 9.4e-01 | 9.3e-03 | 5.5e-03 | 9.3e-03 | 3.6e-02 | 3.9e-03
     200 |  1.918703e+00 |  2.039848e+00 | 2.4e-02 | 2.9e-02 | 2.6e-03 | 2.9e-02 | 7.9e-02 | 7.9e-03
     300 |  1.754797e+00 |  1.836680e+00 | 1.8e-02 | 8.5e-03 | 2.2e-04 | 8.5e-03 | 1.2e-01 | 1.2e-02
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  1.754797e+00 |  1.836680e+00 | 1.8e-02 | 8.5e-03 | 2.2e-04 | 8.5e-03 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  1.715029e+00 |  1.816955e+00 | 2.2e-02 | 7.8e-03 | 3.0e-03 | 7.8e-03 | 1.8e-02 | 1.0e+00 | 5.0e+00 |       1
       2 |  1.688271e+00 |  1.797810e+00 | 2.4e-02 | 5.8e-03 | 5.9e-03 | 5.9e-03 | 2.1e-02 | 1.5e+00 | 5.0e+00 |       2
       3 |  1.666377e+00 |  1.758886e+00 | 2.1e-02 | 2.6e-03 | 6.8e-03 | 6.8e-03 | 2.3e-02 | 2.2e+00 | 5.0e+00 |       3
       4 |  1.658572e+00 |  1.716444e+00 | 1.3e-02 | 1.3e-03 | 5.2e-03 | 5.2e-03 | 2.6e-02 | 3.4e+00 | 5.0e+00 |       4
       5 |  1.681173e+00 |  1.702907e+00 | 5.0e-03 | 6.3e-04 | 2.7e-03 | 2.7e-03 | 3.0e-02 | 5.1e+00 | 5.0e+00 |       6
       6 |  1.721739e+00 |  1.710142e+00 | 2.6e-03 | 3.5e-04 | 5.5e-04 | 5.5e-04 | 3.4e-02 | 7.6e+00 | 5.0e+00 |       8
       7 |  1.735164e+00 |  1.724764e+00 | 2.3e-03 | 2.5e-04 | 7.0e-04 | 7.0e-04 | 4.0e-02 | 1.1e+01 | 5.0e+00 |      11
       8 |  1.728020e+00 |  1.726737e+00 | 2.9e-04 | 2.4e-04 | 2.2e-04 | 2.4e-04 | 4.6e-02 | 1.7e+01 | 5.0e+00 |      14
       9 |  1.723977e+00 |  1.724980e+00 | 2.3e-04 | 8.8e-05 | 4.1e-05 | 8.8e-05 | 5.0e-02 | 2.6e+01 | 5.0e+00 |      16
      10 |  1.724148e+00 |  1.724583e+00 | 9.8e-05 | 2.8e-05 | 2.3e-05 | 2.8e-05 | 5.4e-02 | 3.8e+01 | 5.0e+00 |      18
      11 |  1.724380e+00 |  1.724467e+00 | 2.0e-05 | 6.9e-06 | 4.8e-06 | 6.9e-06 | 5.8e-02 | 5.8e+01 | 5.0e+00 |      20
      12 |  1.724446e+00 |  1.724450e+00 | 8.7e-07 | 1.2e-06 | 8.2e-07 | 1.2e-06 | 6.2e-02 | 8.6e+01 | 5.0e+00 |      22
      13 |  1.724457e+00 |  1.724455e+00 | 6.1e-07 | 2.1e-07 | 3.4e-07 | 3.4e-07 | 6.5e-02 | 1.3e+02 | 5.0e+00 |      23

   partialGQOT
     converged: True (Converged with KKT residual below tolerance.)
     iterations: 13 outer, 300 ADMM
     objective:  1.724457e+00
     dual objective:  1.724455e+00
     relative gap: 6.1e-07
     feasibility: 3.4e-07
     stationarity: 2.1e-07
     KKT residual: 3.4e-07
     time: 2.5e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 23.1, 14, 300, 4.0e-07, 2.034433e+00
   2000, 106.4, 15, 300, 7.1e-07, 2.082043e+00
   3000, 209.8, 15, 300, 3.7e-07, 1.990608e+00
   4000, 465.6, 14, 300, 5.6e-07, 2.033895e+00
   5000, 1404.4, 14, 300, 5.8e-07, 2.031234e+00

Martingale OT
-------------

For a source grid ``x`` and target grid ``y``, the row-wise martingale
condition is

.. math::

   \sum_j X_{ij} y_j = \alpha_i x_i.

The default example uses the nonnegativity indicator for transport and a soft
squared-l2 penalty with weight ``10`` for the martingale side residual:

.. math::

   \begin{aligned}
   \min_{X,D}\quad
      & \langle C, X\rangle
      + \underbrace{\delta_{\mathbb R_+^{m\times n}}(X)}_{R(X)}
      + \underbrace{5\|D\|_F^2}_{h(D)} \\
   \mathrm{s.t.}\quad
      & X\mathbf{1} = \alpha, \\
      & X^\top\mathbf{1} = \beta, \\
      & A X B + D = S.
   \end{aligned}

Additional data generation:

.. code-block:: python

   grid = np.linspace(-1.0, 1.0, size, dtype=np.float64)
   alpha = np.full(size, 1.0 / size, dtype=np.float64)
   beta = alpha.copy()
   C = (grid[:, None] - grid[None, :]) ** 2

   A = np.eye(size, dtype=np.float64)
   B = grid[:, None]
   S = alpha[:, None] * grid[:, None]

Model and solve:

.. code-block:: python

   from CompositeOT import Model, NonnegativeIndicator, SquaredL2Regularizer

   model = Model(C, alpha, beta, name="martingaleOT")
   model.add_side_constraint(A, B, S, regularizer=SquaredL2Regularizer(10.0))
   model.add_transport_regularizer(NonnegativeIndicator())
   result = model.solve()

Run:

.. code-block:: bash

   python examples/example_martingale_ot.py --m 30 --n 30 \
       --max-iterations 50 --admm-iterations 300 --tolerance 1e-6

Output:

.. code-block:: text

    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma
     100 |  8.431897e-03 | -2.910307e-03 | 1.1e-02 | 1.8e-03 | 1.1e-03 | 1.8e-03 | 4.4e-02 | 3.0e-02
     200 |  5.228534e-03 |  9.127654e-03 | 3.8e-03 | 1.9e-03 | 6.7e-06 | 1.9e-03 | 7.4e-02 | 3.3e-02
     300 |  2.538075e-03 |  3.735982e-03 | 1.2e-03 | 8.1e-04 | 3.9e-08 | 8.1e-04 | 1.1e-01 | 5.0e-02
    iter |          pobj |          dobj |     gap |    pres |    dres |     kkt |    time |   sigma |     tau |   inner
       0 |  2.538075e-03 |  3.735982e-03 | 1.2e-03 | 8.1e-04 | 3.9e-08 | 8.1e-04 | 0.0e+00 | 1.0e+00 | 5.0e+00 |       0
       1 |  2.752968e-03 |  3.735982e-03 | 9.8e-04 | 8.1e-04 | 2.1e-04 | 8.1e-04 | 2.1e-03 | 1.0e+00 | 5.0e+00 |       0
       2 |  2.710355e-03 |  3.476042e-03 | 7.6e-04 | 7.1e-04 | 3.9e-04 | 7.1e-04 | 1.1e-02 | 1.5e+00 | 5.0e+00 |       1
       3 |  2.671802e-03 |  3.129956e-03 | 4.6e-04 | 6.3e-04 | 4.0e-04 | 6.3e-04 | 1.3e-02 | 2.2e+00 | 5.0e+00 |       2
       4 |  2.615871e-03 |  3.081132e-03 | 4.6e-04 | 6.2e-04 | 2.1e-04 | 6.2e-04 | 1.5e-02 | 3.4e+00 | 5.0e+00 |       3
       5 |  2.533586e-03 |  3.109927e-03 | 5.7e-04 | 6.2e-04 | 1.4e-04 | 6.2e-04 | 1.6e-02 | 5.1e+00 | 5.0e+00 |       4
       6 |  2.409122e-03 |  3.109927e-03 | 7.0e-04 | 6.2e-04 | 2.4e-04 | 6.2e-04 | 1.7e-02 | 7.6e+00 | 5.0e+00 |       4
       7 |  2.220604e-03 |  3.124237e-03 | 9.0e-04 | 6.2e-04 | 7.0e-05 | 6.2e-04 | 1.8e-02 | 1.1e+01 | 5.0e+00 |       5
       8 |  1.938234e-03 |  3.118465e-03 | 1.2e-03 | 6.2e-04 | 1.9e-05 | 6.2e-04 | 2.0e-02 | 1.7e+01 | 5.0e+00 |       6
       9 |  1.516953e-03 |  3.118465e-03 | 1.6e-03 | 6.2e-04 | 2.1e-04 | 6.2e-04 | 2.0e-02 | 2.6e+01 | 5.0e+00 |       6
      10 |  8.839857e-04 |  3.113663e-03 | 2.2e-03 | 6.2e-04 | 2.8e-05 | 6.2e-04 | 2.2e-02 | 3.8e+01 | 5.0e+00 |       7
      11 |  1.185572e-08 |  2.910703e-03 | 2.9e-03 | 4.7e-04 | 8.6e-06 | 4.7e-04 | 2.5e-02 | 5.8e+01 | 5.0e+00 |       9
      12 |  4.539665e-10 |  9.650556e-07 | 9.6e-07 | 1.6e-07 | 1.6e-05 | 1.6e-05 | 2.6e-02 | 8.6e+01 | 5.0e+00 |      10
      13 |  1.755685e-13 | -6.443327e-07 | 6.4e-07 | 1.0e-07 | 1.1e-07 | 1.1e-07 | 2.8e-02 | 1.3e+02 | 5.0e+00 |      11

   martingaleOT
     converged: True (Converged with max(KKT residual, relative gap).)
     iterations: 13 outer, 300 ADMM
     objective:  1.755685e-13
     dual objective: -6.443327e-07
     relative gap: 6.4e-07
     feasibility: 1.1e-07
     stationarity: 1.0e-07
     KKT residual: 1.1e-07
     time: 2.0e-01 s

.. csv-table:: Test results
   :header: "n=m", "time (s)", "outer", "ADMM", "KKT", "objective"
   :align: left

   1000, 6.6, 20, 300, 8.0e-07, 6.983665e-06
   2000, 33.2, 18, 300, 7.5e-07, 8.102010e-06
   3000, 66.2, 20, 300, 2.9e-07, 3.613601e-06
   4000, 114.1, 19, 100, 3.0e-07, 3.877028e-06
   5000, 189.5, 15, 100, 8.8e-07, 9.910263e-06
