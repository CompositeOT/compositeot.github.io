# CompositeOT Python Project Architecture Reading

生成时间: 2026-07-12

范围: 本报告只根据当前仓库源码做软件架构层面的解读；不评价算法正确性。

## 0. 阅读边界

### 代码明确表明的事实

- 项目包名是 `CompositeOT`，由 `pyproject.toml` 的 `[project] name = "CompositeOT"` 声明；运行时主包目录是 `CompositeOT/`。锚点: `pyproject.toml` `[project]`, `CompositeOT/__init__.py` `__all__`.
- 核心建模问题写成
  `min <C, X> + R(X) + h(D) + p_r(y) + p_c(z)`，约束为 `A X B + D = S`, `X 1 + y = alpha`, `X.T 1 + z = beta`。锚点: `CompositeOT/model.py` `CompositeOTProblem`, `README.md` "Quick Example".
- 默认用户工作流是构造 `Model(C, alpha, beta)`，添加 regularizer 或 side constraint，然后调用 `model.solve()`。锚点: `CompositeOT/model.py` `CompositeOTModel.solve`, `README.md` "Quick Example", `examples/example_qrot.py` `main`.

### 我的推测

- 这个仓库的设计目标是提供一个小型建模层加一个结构化求解器内核，而不是只暴露单个函数式 OT solver。依据是 `CompositeOTModel` 与 `CompositeOTProblem` 的分离，以及 regularizer/side/operator plan 的可扩展接口。锚点: `CompositeOT/model.py` `CompositeOTModel`, `CompositeOT/model.py` `CompositeOTProblem`, `CompositeOT/plan.py` `SolverPlan`.
- `old/` 目录应该是历史原型或非发布材料，不属于当前公共包职责。依据是 `README.md` 明确说 `old/` 被 Git 忽略且不属于 package distribution。锚点: `README.md` "Repository Layout".

## 1. 目录结构和主要模块职责

### 代码明确表明的事实

- `CompositeOT/` 是公共 Python 包。它集中导出建模类、选项类、求解结果、regularizer、scaling 和 solver 入口。锚点: `CompositeOT/__init__.py` `__all__`.
- `CompositeOT/model.py` 负责问题表示，分成不可变求解器输入 `CompositeOTProblem` 和可变用户建模器 `CompositeOTModel`，别名 `Model = CompositeOTModel`。锚点: `CompositeOT/model.py` `CompositeOTProblem`, `CompositeOT/model.py` `CompositeOTModel`, `CompositeOT/model.py` `Model`.
- `CompositeOT/regularizers/` 负责 transport、side、row、column 四类正则项/指示函数的统一接口。共同接口由 `RegularizerBase` 和 `Regularizer` 协议描述。锚点: `CompositeOT/regularizers/base.py` `RegularizerBase`, `CompositeOT/regularizers/base.py` `Regularizer`.
- `CompositeOT/solver.py` 是高层求解入口，负责自动缩放、可选 SGS-ADMM warm start、PALM 高精度阶段，以及结果反缩放。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/solver.py` `solve_scaled`.
- `CompositeOT/sgsadmm.py` 负责 symmetric Gauss-Seidel ADMM warm-start 循环。锚点: `CompositeOT/sgsadmm.py` `sgsadmm`.
- `CompositeOT/palm.py` 负责 PALM/ripALM 外层高精度循环。锚点: `CompositeOT/palm.py` `solve_palm`.
- `CompositeOT/newton.py` 负责 PALM 内层 semismooth Newton 子问题循环。锚点: `CompositeOT/newton.py` `solve_semismooth_newton`.
- `CompositeOT/subproblem.py` 负责把当前 dual iterate 组装成 prox 参数、primal 变量、gradient、Newton objective 和 generalized Jacobian。锚点: `CompositeOT/subproblem.py` `evaluate_newton_data`, `CompositeOT/subproblem.py` `ProxArgument`.
- `CompositeOT/plan.py` 负责编译问题结构，缓存 side operator、transport regularizer、normal equation 所需维度和结构。锚点: `CompositeOT/plan.py` `SolverPlan`, `CompositeOT/plan.py` `SidePlan`, `CompositeOT/plan.py` `TransportPlan`, `CompositeOT/plan.py` `NormalPlan`.
- `CompositeOT/normal.py`, `CompositeOT/normalassemble.py`, `CompositeOT/normallowrank.py` 组成 Newton normal equation 的装配和求解层。锚点: `CompositeOT/normal.py` `solve_normal`, `CompositeOT/normalassemble.py` `assemble_normal`, `CompositeOT/normallowrank.py` `_add_low_rank_update`.
- `CompositeOT/side.py` 和 `CompositeOT/marginal.py` 是 ADMM block solve 的结构化子模块。锚点: `CompositeOT/side.py` `SideBlockSolver`, `CompositeOT/side.py` `solve_side_block_into`, `CompositeOT/marginal.py` `solve_uv_block_into`.
- `CompositeOT/results.py` 定义 primal/dual/result/history 数据结构。锚点: `CompositeOT/results.py` `PrimalSolution`, `CompositeOT/results.py` `DualSolution`, `CompositeOT/results.py` `SolverResult`, `CompositeOT/results.py` `ADMMResult`.
- `CompositeOT/options.py` 定义用户可调 solver options。锚点: `CompositeOT/options.py` `ADMMOptions`, `CompositeOT/options.py` `PALMOptions`, `CompositeOT/options.py` `SemismoothNewtonOptions`.
- `CompositeOT/utils.py` 负责 dtype/shape 验证、可选 Numba 包装、示例数据生成和分组构造。锚点: `CompositeOT/utils.py` `require_float64_array`, `CompositeOT/utils.py` `njit`, `CompositeOT/utils.py` `ExampleData`, `CompositeOT/utils.py` `make_qrot_example_data`.
- `examples/` 是脚本入口层，每个脚本构造一个具体 OT 变体并调用公共建模 API。锚点: `examples/example_qrot.py` `main`, `examples/example_uot.py` `main`, `examples/example_partial_ot.py` `main`, `examples/example_martingale_ot.py` `main`.
- `tests/` 既有公共 API smoke test，也有 DOTmark 数据集驱动测试/benchmark 脚本。锚点: `tests/test_public_api.py` `test_public_import_and_small_qrot_solve`, `tests/test_dotmark.py` `run_dotmark`.

### 我的推测

- `CompositeOT/ops.py` 是低层数值 kernel 汇集层，面向内存复用和可选 Numba 加速。依据是函数多为 `*_into` 或 inplace 风格，并被 `model.py`, `sgsadmm.py`, `regularizers/` 调用。锚点: `CompositeOT/ops.py` `scaled_add`, `CompositeOT/ops.py` `dual_residual_scalar_into`.
- `docs/generated/` 是 Sphinx API 自动生成文档，不是运行时依赖。依据是路径名和 README 的文档构建说明。锚点: `README.md` "Documentation", `docs/generated/CompositeOT.solver.rst` module page.

## 2. 公共 API 和用户入口

### 代码明确表明的事实

- 顶层 import API 由 `CompositeOT/__init__.py` 的 `__all__` 明确声明。核心用户类包括 `Model`, `CompositeOTModel`, `CompositeOTProblem`, `ADMMOptions`, `PALMOptions`, `SemismoothNewtonOptions`, regularizer 类和 result 类。锚点: `CompositeOT/__init__.py` `__all__`.
- 最主要的用户建模入口是 `Model(C, alpha, beta)`，实际指向 `CompositeOTModel`。锚点: `CompositeOT/model.py` `Model`, `CompositeOT/model.py` `CompositeOTModel.__init__`.
- 用户通过 `CompositeOTModel.add_transport_regularizer`, `add_row_regularizer`, `add_column_regularizer`, `add_side_constraint` 配置模型。锚点: `CompositeOT/model.py` `CompositeOTModel.add_transport_regularizer`, `CompositeOT/model.py` `CompositeOTModel.add_row_regularizer`, `CompositeOT/model.py` `CompositeOTModel.add_side_constraint`.
- 用户求解入口有 `CompositeOTModel.solve`, `CompositeOTProblem.solve`, 顶层函数 `solve`，以及较低层的 `sgsadmm`。锚点: `CompositeOT/model.py` `CompositeOTModel.solve`, `CompositeOT/model.py` `CompositeOTProblem.solve`, `CompositeOT/solver.py` `solve`, `CompositeOT/sgsadmm.py` `sgsadmm`.
- `CompositeOTModel.optimize` 是 `solve` 的别名。锚点: `CompositeOT/model.py` `CompositeOTModel.optimize`.
- 示例脚本是可直接运行的用户入口，例如 QROT、UOT、partial OT、GQOT、partial GQOT、martingale OT。锚点: `examples/example_qrot.py` `main`, `examples/example_uot.py` `main`, `examples/example_partial_ot.py` `main`, `examples/example_gqot.py` `main`, `examples/example_partial_gqot.py` `main`, `examples/example_martingale_ot.py` `main`.
- `tests/test_dotmark.py` 也是一个脚本式入口，`main` 根据文件顶部的 `RUN_*` 配置运行 DOTmark 问题。锚点: `tests/test_dotmark.py` `main`, `tests/test_dotmark.py` `run_dotmark`.

### 我的推测

- 对普通用户最稳定的 API 应该是 `Model`, built-in regularizers, `ADMMOptions`, `PALMOptions`, `SolverResult`，而不是 `subproblem.py`, `normalassemble.py` 这类内核模块。依据是 `CompositeOT/__init__.py` 选择性导出，以及 README 示例只使用顶层对象。锚点: `CompositeOT/__init__.py` `__all__`, `README.md` "Quick Example".

## 3. 与模型、求解器、后端、数据结构、测试相关的模块

### 3.1 最优传输模型相关

#### 代码明确表明的事实

- `CompositeOTProblem` 保存 `C`, `alpha`, `beta`, transport/side/row/column regularizer，以及可选 `A`, `B`, `S` side relation。锚点: `CompositeOT/model.py` `CompositeOTProblem`.
- `CompositeOTProblem` 提供模型残差和目标计算: `objective_value`, `reporting_objective_value`, `row_residual`, `column_residual`, `dual_residual`, `marginal_dual_rhs`。锚点: `CompositeOT/model.py` `CompositeOTProblem.objective_value`, `CompositeOT/model.py` `CompositeOTProblem.row_residual`, `CompositeOT/model.py` `CompositeOTProblem.dual_residual`, `CompositeOT/model.py` `CompositeOTProblem.marginal_dual_rhs`.
- `CompositeOTModel` 是 mutable builder，`compile` 返回 immutable `CompositeOTProblem`。锚点: `CompositeOT/model.py` `CompositeOTModel`, `CompositeOT/model.py` `CompositeOTModel.compile`.
- built-in regularizers 包括 nonnegative indicator、nonnegative quadratic、squared L2、L1、L2 ball、box、zero、zero indicator、group quadratic。锚点: `CompositeOT/regularizers/__init__.py` `__all__`, `CompositeOT/regularizers/nonnegative_quadratic.py` `NonnegativeQuadraticRegularizer`, `CompositeOT/regularizers/group_quadratic.py` `GroupQuadraticRegularizer`.

#### 我的推测

- `CompositeOTProblem` 是求解器内部的 canonical representation；所有具体 OT 变体都通过 regularizer 和 side constraint 组合表达，而不是通过不同 problem subclass 表达。锚点: `CompositeOT/model.py` `CompositeOTProblem`, `examples/example_uot.py` `main`, `examples/example_partial_ot.py` `main`.

### 3.2 求解器相关

#### 代码明确表明的事实

- `solve` 默认对原问题做 `scale_problem`，再调用 `solve_scaled`，最后 `unscale_solver_result`。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/scaling.py` `scale_problem`, `CompositeOT/scaling.py` `unscale_solver_result`.
- `solve_scaled` 先编译 `SolverPlan`，若 `ADMMOptions.max_iterations == 0` 就直接进入 `solve_palm`；否则先跑 `sgsadmm`，再以 ADMM 输出作为 PALM warm start。锚点: `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/plan.py` `SolverPlan.compile`, `CompositeOT/sgsadmm.py` `sgsadmm`, `CompositeOT/palm.py` `solve_palm`.
- `sgsadmm` 的核心循环更新 side block `W`、marginal block `(u, v)`、四个 prox split blocks `(Q, G, r, c)`，然后更新 primal multiplier `(X, D, y, z)`。锚点: `CompositeOT/sgsadmm.py` `sgsadmm`, `CompositeOT/side.py` `solve_side_block_into`, `CompositeOT/marginal.py` `solve_uv_block_into`.
- `solve_palm` 外层循环每次设置 `sigma` 和 `tau`，调用 `solve_semismooth_newton`，更新 `auxiliary_error`, `primal`, `dual`，并通过 KKT diagnostics 判断终止。锚点: `CompositeOT/palm.py` `solve_palm`, `CompositeOT/hyperparams.py` `palm_sigma_at_iteration`, `CompositeOT/termination.py` `termination_residual`.
- `solve_semismooth_newton` 内层循环调用 `evaluate_newton_data`，用 `solve_normal` 解 normal equation，然后 `armijo_line_search` 接受步长。锚点: `CompositeOT/newton.py` `solve_semismooth_newton`, `CompositeOT/subproblem.py` `evaluate_newton_data`, `CompositeOT/normal.py` `solve_normal`, `CompositeOT/linesearch.py` `armijo_line_search`.

#### 我的推测

- 运行时主要求解策略是“两阶段”: ADMM 做较便宜的 warm start，PALM/Newton 做高精度收敛。依据是 `solver.py` 文档字符串和 `solve_scaled` 的控制流。锚点: `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/palm.py` `solve_palm`.

### 3.3 数值后端相关

#### 代码明确表明的事实

- 基础数组后端是 NumPy，输入验证要求 `numpy.ndarray` 且 `float64`。锚点: `CompositeOT/utils.py` `require_float64_array`.
- 可选加速后端是 Numba；如果无法 import `numba`，`njit` 返回原函数。锚点: `CompositeOT/utils.py` `njit`, `CompositeOT/utils.py` `prange`.
- 线性代数后端使用 SciPy dense/sparse routine；normal equation 若为 dense 使用 `scipy.linalg.solve`，若为 sparse 优先尝试 `sksparse.cholmod.cholesky`，失败后回退 `scipy.sparse.linalg.splu`。锚点: `CompositeOT/normal.py` `solve_normal`.
- side block solver 对 scalar_sum、row_weight、dense、sparse 四类情况选择不同实现。锚点: `CompositeOT/side.py` `SideBlockSolver.from_plan`, `CompositeOT/side.py` `SideBlockSolver.solve_flat`.
- normal assembly 使用 sparse/dense/low-rank 表示 `ProxJacobian`，并在 `normalassemble.py` 和 `normallowrank.py` 中构造矩阵。锚点: `CompositeOT/regularizers/base.py` `ProxJacobian`, `CompositeOT/normalassemble.py` `assemble_normal`, `CompositeOT/normallowrank.py` `_add_low_rank_update`.

#### 我的推测

- 项目目前没有抽象的 "backend interface" 来替换 NumPy/SciPy；所谓 backend 更像是固定 NumPy/SciPy 计算栈加可选 Numba/SuiteSparse 加速。锚点: `CompositeOT/utils.py` `njit`, `CompositeOT/normal.py` `solve_normal`.

### 3.4 数据结构相关

#### 代码明确表明的事实

- `PrimalSolution` 持有 `(X, D, y, z)`，并支持 `zeros`, `validate`, `pack`, `from_vector`, `norm`, `squared_distance`。锚点: `CompositeOT/results.py` `PrimalSolution`.
- `DualSolution` 持有 `(W, u, v)`，并支持 `zeros`, `pack`, `from_vector`, `update`, `dot`。锚点: `CompositeOT/results.py` `DualSolution`.
- `SolverResult` 持有最终 primal/dual、诊断指标、迭代数、runtime 和 history。锚点: `CompositeOT/results.py` `SolverResult`.
- `ADMMResult` 持有 ADMM warm-start 或 standalone 结果。锚点: `CompositeOT/results.py` `ADMMResult`.
- `ExampleData` 持有 synthetic examples 所需的 `C`, `alpha`, `beta`, side mass relation 和 group metadata。锚点: `CompositeOT/utils.py` `ExampleData`.
- `DOTmarkProblem` 和 `PythonOTResult` 是 DOTmark 测试/benchmark 层的数据结构。锚点: `tests/test_dotmark.py` `DOTmarkProblem`, `tests/test_dotmark.py` `PythonOTResult`.

#### 我的推测

- `PrimalSolution.pack` 和 `DualSolution.pack` 的向量化顺序是为了和 Newton normal equation 变量顺序对齐。依据是 `DualSolution` 文档字符串和 `solve_semismooth_newton` 里 `DualSolution.from_vector` 的使用。锚点: `CompositeOT/results.py` `DualSolution.pack`, `CompositeOT/newton.py` `solve_semismooth_newton`.

### 3.5 测试和示例相关

#### 代码明确表明的事实

- 最小公共 API 测试构造 2x2 QROT 问题，调用 `model.solve(admm_options=ADMMOptions(max_iterations=0))`，断言输出形状和有限数值。锚点: `tests/test_public_api.py` `test_public_import_and_small_qrot_solve`.
- DOTmark 脚本负责加载本地 CSV 图像、构造 normalized squared Euclidean cost、配置不同 problem type，并调用 `model.solve`。锚点: `tests/test_dotmark.py` `load_dotmark_marginal`, `tests/test_dotmark.py` `make_dotmark_cost`, `tests/test_dotmark.py` `make_dotmark_model`, `tests/test_dotmark.py` `run_dotmark`.
- DOTmark 脚本包含可选 PythonOT/POT QROT 对照入口。锚点: `tests/test_dotmark.py` `run_pythonot_qrot`.
- 示例公共参数、option 构造和打印逻辑集中在 `examples/common.py`。锚点: `examples/common.py` `make_parser`, `examples/common.py` `make_options`, `examples/common.py` `solve_and_report`.

#### 我的推测

- `tests/test_dotmark.py` 更像 benchmark/experiment driver，而不是传统 pytest 单元测试，因为文件顶部有 `RUN_*` 配置块和 `main()` 入口。锚点: `tests/test_dotmark.py` `main`, `tests/test_dotmark.py` `RUN_PROBLEM_OPTIONS`.

## 4. 典型 API 调用到核心求解循环的候选调用链

### 候选链 A: README/QROT 风格默认求解

1. 用户构造模型: `Model(C, alpha, beta)`。锚点: `README.md` "Quick Example", `CompositeOT/model.py` `CompositeOTModel.__init__`.
2. 用户添加 transport regularizer: `model.add_transport_regularizer(NonnegativeQuadraticRegularizer(...))`。锚点: `README.md` "Quick Example", `CompositeOT/model.py` `CompositeOTModel.add_transport_regularizer`, `CompositeOT/regularizers/nonnegative_quadratic.py` `NonnegativeQuadraticRegularizer`.
3. 用户调用 `model.solve(...)`。锚点: `CompositeOT/model.py` `CompositeOTModel.solve`.
4. `CompositeOTModel.solve` 调用 `self.compile().solve(...)`，生成 `CompositeOTProblem`。锚点: `CompositeOT/model.py` `CompositeOTModel.compile`, `CompositeOT/model.py` `CompositeOTProblem.solve`.
5. `CompositeOTProblem.solve` 调用高层 `solve(problem, ...)`。锚点: `CompositeOT/model.py` `CompositeOTProblem.solve`, `CompositeOT/solver.py` `solve`.
6. `solve` 调用 `scale_problem`, `scale_primal`, `scale_dual`，然后进入 `solve_scaled`。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/scaling.py` `scale_problem`.
7. `solve_scaled` 调用 `SolverPlan.compile(problem)`。锚点: `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/plan.py` `SolverPlan.compile`.
8. 如果 `admm_options.max_iterations > 0`，`solve_scaled` 先调用 `sgsadmm(problem, admm_options, plan=plan)`。锚点: `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/sgsadmm.py` `sgsadmm`.
9. `sgsadmm` 进入 `for iteration in range(...)`，在循环里调用 `solve_side_block_into`, `problem.marginal_dual_rhs`, `solve_uv_block_into`, regularizer 的 `prox_into`，最后更新 `(X, D, y, z)`。锚点: `CompositeOT/sgsadmm.py` `sgsadmm`, `CompositeOT/side.py` `solve_side_block_into`, `CompositeOT/marginal.py` `solve_uv_block_into`, `CompositeOT/regularizers/base.py` `RegularizerBase.prox_into`.
10. `solve_scaled` 把 ADMM 输出作为 warm start 调用 `solve_palm`。锚点: `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/palm.py` `solve_palm`.
11. `solve_palm` 进入 PALM outer loop，调用 `solve_semismooth_newton`。锚点: `CompositeOT/palm.py` `solve_palm`, `CompositeOT/newton.py` `solve_semismooth_newton`.
12. `solve_semismooth_newton` 调用 `evaluate_newton_data` 得到 primal、gradient、objective、Jacobians。锚点: `CompositeOT/newton.py` `solve_semismooth_newton`, `CompositeOT/subproblem.py` `evaluate_newton_data`.
13. `solve_semismooth_newton` 调用 `solve_normal` 解 Newton direction，再用 `armijo_line_search` 接受 dual step。锚点: `CompositeOT/newton.py` `solve_semismooth_newton`, `CompositeOT/normal.py` `solve_normal`, `CompositeOT/linesearch.py` `armijo_line_search`.
14. `solve_normal` 经 `plan.normal.assemble(...)` 调到 `assemble_normal`，再用 SciPy dense/sparse solver 或 CHOLMOD/SuperLU 求解。锚点: `CompositeOT/normal.py` `solve_normal`, `CompositeOT/plan.py` `NormalPlan.assemble`, `CompositeOT/normalassemble.py` `assemble_normal`.
15. `solve_palm` 根据 `evaluate_reporting_kkt` 和 `termination_residual` 终止并返回 `SolverResult`。锚点: `CompositeOT/palm.py` `solve_palm`, `CompositeOT/termination.py` `evaluate_reporting_kkt`, `CompositeOT/results.py` `make_solver_result`.
16. `solve` 对结果 `unscale_solver_result` 并返回用户。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/scaling.py` `unscale_solver_result`.

### 候选链 B: 用户跳过 ADMM warm start

1. 用户传入 `ADMMOptions(max_iterations=0)`。锚点: `README.md` "Quick Example", `CompositeOT/options.py` `ADMMOptions`.
2. `solve_scaled` 判断 `admm_options.max_iterations == 0`，直接调用 `solve_palm`。锚点: `CompositeOT/solver.py` `solve_scaled`.
3. 后续进入 PALM/Newton/normal equation 路径，同候选链 A 的第 11 至第 16 步。锚点: `CompositeOT/palm.py` `solve_palm`, `CompositeOT/newton.py` `solve_semismooth_newton`, `CompositeOT/normal.py` `solve_normal`.

### 候选链 C: UOT 示例

1. `examples/example_uot.py` 通过 `make_uot_example_data` 构造不平衡 synthetic data。锚点: `examples/example_uot.py` `main`, `CompositeOT/utils.py` `make_uot_example_data`.
2. `main` 构造 `Model(data.C, data.alpha, data.beta, name="UOT-sqL2")`。锚点: `examples/example_uot.py` `main`, `CompositeOT/model.py` `CompositeOTModel.__init__`.
3. `main` 添加 `NonnegativeQuadraticRegularizer` 到 transport，添加 `SquaredL2Regularizer` 到 row/column slack。锚点: `examples/example_uot.py` `main`, `CompositeOT/model.py` `CompositeOTModel.add_transport_regularizer`, `CompositeOT/model.py` `CompositeOTModel.add_row_regularizer`, `CompositeOT/model.py` `CompositeOTModel.add_column_regularizer`.
4. `solve_and_report` 调用 `model.solve(palm_options=..., admm_options=...)`。锚点: `examples/common.py` `solve_and_report`, `CompositeOT/model.py` `CompositeOTModel.solve`.
5. 后续进入 `solver.solve -> solve_scaled -> sgsadmm -> solve_palm -> solve_semismooth_newton`。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/solver.py` `solve_scaled`, `CompositeOT/sgsadmm.py` `sgsadmm`, `CompositeOT/palm.py` `solve_palm`, `CompositeOT/newton.py` `solve_semismooth_newton`.

## 5. 模块间依赖观察

### 代码明确表明的事实

- 建模层依赖 regularizer 和低层 ops，但为了避免循环导入，`CompositeOTProblem.solve` 在方法内部 import `solver.solve`。锚点: `CompositeOT/model.py` `CompositeOTProblem.solve`.
- Solver control flow 依赖 `SolverPlan`、`scaling`、`sgsadmm`、`solve_palm` 和 result containers。锚点: `CompositeOT/solver.py` `solve`, `CompositeOT/solver.py` `solve_scaled`.
- PALM 依赖 Newton、hyperparams、termination 和 result history。锚点: `CompositeOT/palm.py` `solve_palm`.
- Newton 依赖 subproblem evaluator、normal equation solver 和 line search。锚点: `CompositeOT/newton.py` `solve_semismooth_newton`.
- Subproblem evaluator 依赖 regularizer 的 `prox_into`, `conjugate_moreau_envelope_value`, `generalized_jacobian`。锚点: `CompositeOT/subproblem.py` `primal_from_args`, `CompositeOT/subproblem.py` `objective_from_args`, `CompositeOT/subproblem.py` `evaluate_jacobians`.

### 我的推测

- 代码把“可变建模状态”和“求解器输入状态”拆开，是为了让求解器能假设 problem immutable、shape/dtype 已验证，从而简化 hot path。锚点: `CompositeOT/model.py` `CompositeOTModel.compile`, `CompositeOT/model.py` `CompositeOTProblem.__post_init__`.
- `SolverPlan` 的存在说明性能关键路径不希望每轮重新识别 side/transport 结构。锚点: `CompositeOT/plan.py` `SolverPlan.compile`, `CompositeOT/plan.py` `SolverPlan.validate_for`.

## 6. 暂不覆盖的内容

- 本报告不判断 `sgsadmm`, `solve_palm`, `solve_semismooth_newton`, `assemble_normal` 的数学实现是否正确。锚点: `CompositeOT/sgsadmm.py` `sgsadmm`, `CompositeOT/palm.py` `solve_palm`, `CompositeOT/newton.py` `solve_semismooth_newton`, `CompositeOT/normalassemble.py` `assemble_normal`.
- 本报告不比较 CompositeOT 与 POT/PythonOT 的数值质量。锚点: `tests/test_dotmark.py` `run_pythonot_qrot`.
- 本报告不建议重构，只描述当前架构。锚点: `CompositeOT/__init__.py` `__all__`, `CompositeOT/solver.py` `solve`.

## 7. `examples/example_qrot.py` 的完整求解路径追踪

本节追踪默认 QROT 示例，即直接运行 `python examples/example_qrot.py` 时的路径。默认参数来自 `examples/example_qrot.py` 和 `examples/common.py`: `m=1000`, `n=1000`, `seed=0`, `dimension=3`, `lambda_q=1.0`, `PALMOptions(max_iterations=100, tolerance=1e-6, stoptype=1, ...)`, `ADMMOptions(max_iterations=300, tolerance=1e-3, ...)`。若用户传入 `--admm-iterations 0`，会跳过 SGS-ADMM warm start，直接进入 PALM；下面先写默认路径。

### 7.1 从脚本入口到模型对象

#### 代码明确表明的事实

1. `examples/example_qrot.py:11` `main()` 创建 parser，并设置 `m=1000`, `n=1000`, `verbose=True`，再添加 `--dimension` 和 `--lambda-q`。输入是命令行字符串；输出是 `argparse.Namespace`。数学对象尚未创建。
2. `examples/example_qrot.py:20` `make_qrot_example_data(args.m, args.n, args.seed, args.dimension)` 返回 `CompositeOT/utils.py:45` `ExampleData`。其中 `C` 是 `numpy.ndarray[float64]`，形状 `(m, n)`；`alpha` 是 `(m,)`；`beta` 是 `(n,)`。锚点: `CompositeOT/utils.py:153` `make_qrot_example_data`, `CompositeOT/utils.py:343` `_make_balanced_gmm`, `CompositeOT/utils.py:383` `_normalized_squared_distance`, `CompositeOT/utils.py:401` `_make_example_data_from_cost`.
3. 数学对象: `C_ij = ||source_i - target_j||^2 / max(C)`，`alpha` 和 `beta` 是归一化边际质量。锚点: `CompositeOT/utils.py:383` `_normalized_squared_distance`, `CompositeOT/utils.py:392` `_squared_distance`.
4. `examples/example_qrot.py:25` `Model(data.C, data.alpha, data.beta, name="QROT")` 实际调用 `CompositeOT/model.py:900` `CompositeOTModel`。输入 `C: (m,n) float64`, `alpha: (m,) float64`, `beta: (n,) float64`；输出是 mutable builder `CompositeOTModel`。锚点: `CompositeOT/model.py:909` `CompositeOTModel.__init__`.
5. `CompositeOTModel.__init__` 默认设置 `transport_regularizer=NonnegativeIndicator()`, `side_regularizer=ZeroIndicator()`, `row_regularizer=ZeroIndicator()`, `column_regularizer=ZeroIndicator()`，并调用 `compile()` 做一次验证。锚点: `CompositeOT/model.py:909` `CompositeOTModel.__init__`, `CompositeOT/model.py:1040` `CompositeOTModel.compile`.
6. `examples/example_qrot.py:26` `model.add_transport_regularizer(NonnegativeQuadraticRegularizer(args.lambda_q))` 将 transport regularizer 改为 QROT 正则。`NonnegativeQuadraticRegularizer` 输入 `quadratic_weight: float`；输出 regularizer 对象。锚点: `CompositeOT/regularizers/nonnegative_quadratic.py:15` `NonnegativeQuadraticRegularizer`, `CompositeOT/model.py:969` `CompositeOTModel.add_transport_regularizer`.
7. 数学对象: transport 正则变为
   `R(X) = 0.5 * lambda_q * ||X||_F^2 + indicator_{X >= 0}(X)`。锚点: `CompositeOT/regularizers/nonnegative_quadratic.py:15` `NonnegativeQuadraticRegularizer`.
8. `examples/common.py:116` `make_options(args, stoptype=1)` 生成 `ExampleSolverOptions(palm, admm)`。输出类型是 `examples/common.py:27` `ExampleSolverOptions`，字段 `palm: PALMOptions`, `admm: ADMMOptions`。锚点: `CompositeOT/options.py:18` `ADMMOptions`, `CompositeOT/options.py:98` `PALMOptions`.
9. `examples/common.py:146` `solve_and_report("QROT", model, options)` 调用 `model.solve(palm_options=options.palm, admm_options=options.admm)`，返回 `CompositeOT/results.py:225` `SolverResult`。锚点: `examples/common.py:146` `solve_and_report`, `CompositeOT/model.py:1057` `CompositeOTModel.solve`.

### 7.2 编译后的 QROT 问题

#### 代码明确表明的事实

1. `CompositeOT/model.py:1057` `CompositeOTModel.solve` 不直接求解 builder，而是调用 `self.compile().solve(...)`。中间分派不能跳过。锚点: `CompositeOT/model.py:1040` `CompositeOTModel.compile`, `CompositeOT/model.py:197` `CompositeOTProblem.solve`.
2. `CompositeOTModel.compile` 输出 `CompositeOTProblem`，字段类型和形状为:
   - `C: np.ndarray float64 (m,n)`
   - `alpha: np.ndarray float64 (m,)`
   - `beta: np.ndarray float64 (n,)`
   - `transport_regularizer: NonnegativeQuadraticRegularizer`
   - `side_regularizer: ZeroIndicator`
   - `row_regularizer: ZeroIndicator`
   - `column_regularizer: ZeroIndicator`
   - `A=None`, `B=None`, `S=None`, 所以 `side_shape == (0,0)`。锚点: `CompositeOT/model.py:67` `CompositeOTProblem`, `CompositeOT/model.py:1040` `CompositeOTModel.compile`.
3. 数学问题是无 side constraint 的 QROT:
   ```text
   min_X,D,y,z <C, X> + 0.5*lambda_q*||X||_F^2 + indicator_{X>=0}(X)
                + indicator_{0}(y) + indicator_{0}(z)
   s.t. X 1 + y = alpha,
        X.T 1 + z = beta.
   ```
   因为 `row_regularizer` 和 `column_regularizer` 是 `ZeroIndicator`，数学上等价于硬边际 `X1=alpha`, `X.T1=beta`。锚点: `CompositeOT/model.py:67` `CompositeOTProblem`, `CompositeOT/regularizers/zero_indicator.py:14` `ZeroIndicator`.
4. `CompositeOTProblem.solve` 再分派到 `CompositeOT/solver.py:35` `solve(problem, ...)`。锚点: `CompositeOT/model.py:197` `CompositeOTProblem.solve`, `CompositeOT/solver.py:35` `solve`.

### 7.3 高层 solver 分派和缩放

#### 代码明确表明的事实

1. `CompositeOT/solver.py:35` `solve` 输入 `problem: CompositeOTProblem`, `palm_options: PALMOptions`, `admm_options: ADMMOptions`, `initial_primal=None`, `initial_dual=None`；输出 `SolverResult`。锚点: `CompositeOT/solver.py:35` `solve`.
2. `solve` 先调用 `admm_options.validate()`，然后 `CompositeOT/scaling.py:56` `scale_problem(problem)`。输出 `ProblemScaling(raw_problem, scaled_problem, pscale, dscale)`。锚点: `CompositeOT/options.py:18` `ADMMOptions.validate`, `CompositeOT/scaling.py:43` `ProblemScaling`, `CompositeOT/scaling.py:56` `scale_problem`.
3. 缩放后的数据类型与形状不变:
   - `scaled_problem.C = C / dscale`, 形状 `(m,n) float64`
   - `scaled_problem.alpha = alpha / pscale`, 形状 `(m,) float64`
   - `scaled_problem.beta = beta / pscale`, 形状 `(n,) float64`
   - `lambda_scaled = lambda_q * pscale / dscale`。锚点: `CompositeOT/scaling.py:56` `scale_problem`, `CompositeOT/scaling.py:105` `scale_regularizer`.
4. 数学对应关系:
   `X_raw = pscale * X_scaled`, `D_raw = pscale * D_scaled`, `y_raw = pscale * y_scaled`, `z_raw = pscale * z_scaled`；`W_raw = dscale * W_scaled`, `u_raw = dscale * u_scaled`, `v_raw = dscale * v_scaled`。锚点: `CompositeOT/scaling.py:151` `scale_primal`, `CompositeOT/scaling.py:180` `unscale_primal`, `CompositeOT/scaling.py:190` `unscale_dual`.
5. `solve` 设置默认 ADMM penalty: 如果 `admm_options.penalty is None`，则变成 `1.0 / max(||problem.C||_F, 1.0)`。锚点: `CompositeOT/solver.py:35` `solve`.
6. `solve` 调用 `CompositeOT/solver.py:72` `solve_scaled(scaled_problem, ...)`。`initial_primal` 和 `initial_dual` 仍为 `None`。锚点: `CompositeOT/solver.py:72` `solve_scaled`, `CompositeOT/scaling.py:151` `scale_primal`, `CompositeOT/scaling.py:166` `scale_dual`.
7. `solve_scaled` 调用 `CompositeOT/plan.py:341` `SolverPlan.compile(problem)`。对 QROT 无 side 的结构，结果是:
   - `SidePlan.kind == "none"`, `side.dim == 0`, `side.shape == (0,0)`。锚点: `CompositeOT/plan.py:32` `SidePlan`, `CompositeOT/plan.py:91` `SidePlan.compile`.
   - `TransportPlan.kind == "nonnegative_quadratic"`，并保存 `nonnegative_quadratic_weight=lambda_scaled`。锚点: `CompositeOT/plan.py:161` `TransportPlan`, `CompositeOT/plan.py:173` `TransportPlan.compile`.
   - `NormalPlan.total_dim == m+n`, `side_dim == 0`, `row_dim == m`, `column_dim == n`。锚点: `CompositeOT/plan.py:186` `NormalPlan`, `CompositeOT/plan.py:212` `NormalPlan.compile`.
8. 默认 `ADMMOptions.max_iterations=300`，所以 `solve_scaled` 先进入 `CompositeOT/sgsadmm.py:40` `sgsadmm`，再进入 `CompositeOT/palm.py:34` `solve_palm`。如果 `--admm-iterations 0`，`solve_scaled` 会直接返回 `solve_palm(...)`。锚点: `CompositeOT/solver.py:72` `solve_scaled`.

### 7.4 SGS-ADMM warm start 的变量和公式

#### 代码明确表明的事实

1. `CompositeOT/sgsadmm.py:40` `sgsadmm` 输入 scaled `CompositeOTProblem`, `ADMMOptions`, `plan`；输出 `CompositeOT/results.py:254` `ADMMResult`。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm`, `CompositeOT/results.py:254` `ADMMResult`.
2. 对 QROT 无 side 问题，ADMM 中真实迭代的数组变量是:
   - primal multiplier / primal candidate: `X (m,n) float64`, `D (0,0)`, `y (m,)`, `z (n,)`
   - dual variables: `W (0,0)`, `u (m,)`, `v (n,)`
   - split variables: `Q (m,n)`, `G (0,0)`, `r (m,)`, `c (n,)`
   - work buffers: `Q_old`, `r_old`, `c_old`, `affine_Q`, `Q_argument`, `r_argument`, `c_argument` 等。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm`.
3. 数学对象:
   - `u` 和 `v` 是 row/column equality 的 dual variables。
   - `Q` 是 transport conjugate argument 的 split variable，对应 `Q = u 1.T + 1 v.T - C`。
   - `r` 和 `c` 是 `u` 与 `v` 的 split variables，对应 row/column regularizer 的 conjugate arguments。
   - `X,y,z` 是 split equations 的 multipliers，同时在收敛时作为 primal blocks。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm` docstring.
4. 无 side 时 `CompositeOT/side.py:186` `solve_side_block_into` 收到 `solver=None` 后直接返回；`W` 和 `D` 保持空形状 `(0,0)`。锚点: `CompositeOT/side.py:186` `solve_side_block_into`, `CompositeOT/side.py:39` `SideBlockSolver.from_plan`.
5. 每轮 ADMM 先调用 `problem.marginal_dual_rhs(...)` 组装 RHS，再调用 `CompositeOT/marginal.py:24` `solve_uv_block_into(u, v, u, v)` 解 coupled marginal block。输入输出均为 `u: (m,) float64`, `v: (n,) float64`，允许原地覆盖。锚点: `CompositeOT/model.py` `CompositeOTProblem.marginal_dual_rhs`, `CompositeOT/marginal.py:24` `solve_uv_block_into`.
6. `solve_uv_block_into` 解的线性系统是
   ```text
   (n + 1) u_i + sum(v) = row_rhs_i
   sum(u) + (m + 1) v_j = column_rhs_j.
   ```
   锚点: `CompositeOT/marginal.py:24` `solve_uv_block_into`.
7. transport affine expression 由 `CompositeOTProblem.dual_residual` 形成:
   ```text
   affine_Q_ij = u_i + v_j - C_ij
   ```
   形状 `(m,n) float64`。锚点: `CompositeOT/model.py` `CompositeOTProblem.dual_residual`.
8. QROT transport prox 使用 `CompositeOT/regularizers/nonnegative_quadratic.py:68` `NonnegativeQuadraticRegularizer.prox_into`:
   ```text
   prox_{sigma R}(a)_ij = max(a_ij, 0) / (1 + sigma * lambda_scaled).
   ```
   在 ADMM 代码里 `sigma` 对应变量名 `penalty`。锚点: `CompositeOT/regularizers/nonnegative_quadratic.py:68` `prox_into`, `CompositeOT/sgsadmm.py:40` `sgsadmm`.
9. ADMM split prox 代码对应 Moreau 形式:
   ```text
   Q_arg = penalty * affine_Q + X
   Q     = affine_Q + X/penalty - prox_{penalty R}(Q_arg)/penalty
   r_arg = penalty * u + y
   r     = u + y/penalty - prox_{penalty p_r}(r_arg)/penalty
   c_arg = penalty * v + z
   c     = v + z/penalty - prox_{penalty p_c}(c_arg)/penalty
   ```
   对 QROT，`p_r` 和 `p_c` 是 `ZeroIndicator`，所以 row/column prox 输出零向量。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm`, `CompositeOT/regularizers/zero_indicator.py:14` `ZeroIndicator`.
10. primal multiplier 更新公式:
    ```text
    X += relaxation * penalty * (affine_Q - Q)
    y += relaxation * penalty * (u - r)
    z += relaxation * penalty * (v - c)
    ```
    `D` 的公式同样存在，但 QROT 无 side，形状是 `(0,0)`。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm`, `CompositeOT/ops.py` `add_scaled_difference`.
11. ADMM 停止准则: 只在 `check_interval` 或最后一轮评估 `evaluate_reporting_kkt`；若 `kkt_residual <= admm_options.tolerance` 则 `converged=True` 并停止。默认 tolerance 是 `1e-3`。锚点: `CompositeOT/sgsadmm.py:40` `sgsadmm`, `CompositeOT/termination.py:135` `evaluate_reporting_kkt`.
12. ADMM 输出 `ADMMResult(primal=PrimalSolution(X,D,y,z), dual=DualSolution(W,u,v), ...)`，作为 PALM warm start。锚点: `CompositeOT/results.py:26` `PrimalSolution`, `CompositeOT/results.py:124` `DualSolution`, `CompositeOT/results.py:254` `ADMMResult`.

### 7.5 PALM 外层循环的变量和停止准则

#### 代码明确表明的事实

1. `CompositeOT/solver.py:72` `solve_scaled` 把 ADMM 的 `warm_start_result.primal` 和 `warm_start_result.dual` 传给 `CompositeOT/palm.py:34` `solve_palm`。锚点: `CompositeOT/solver.py:72` `solve_scaled`, `CompositeOT/palm.py:34` `solve_palm`.
2. `solve_palm` 真正迭代的外层变量是:
   - `primal: PrimalSolution(X,D,y,z)`，QROT 中形状为 `X(m,n)`, `D(0,0)`, `y(m,)`, `z(n,)`
   - `dual: DualSolution(W,u,v)`，QROT 中形状为 `W(0,0)`, `u(m,)`, `v(n,)`
   - `auxiliary_error: DualSolution(W,u,v)`，同 dual 形状
   - 标量参数 `sigma`, `tau`。锚点: `CompositeOT/palm.py:34` `solve_palm`, `CompositeOT/results.py:26` `PrimalSolution`, `CompositeOT/results.py:124` `DualSolution`.
3. 每轮 PALM 由 `CompositeOT/hyperparams.py:15` `palm_sigma_at_iteration` 和 `CompositeOT/hyperparams.py:36` `palm_tau_at_iteration` 计算 `sigma` 与 `tau`。锚点: `CompositeOT/palm.py:34` `solve_palm`.
4. PALM 核心更新分派:
   ```text
   dual_next, primal_next, gradient, inner_iterations, inner_converged
       = solve_semismooth_newton(...)
   auxiliary_error = auxiliary_error.update(gradient, -sigma)
   primal = primal_next
   dual = dual_next
   ```
   锚点: `CompositeOT/palm.py:34` `solve_palm`, `CompositeOT/results.py:124` `DualSolution.update`.
5. PALM 的外层诊断由 `CompositeOT/termination.py:135` `evaluate_reporting_kkt` 计算。若当前 scaled problem 的 metadata 里有 raw reporting problem，它会先映射回 raw units 再计算诊断。锚点: `CompositeOT/termination.py:135` `evaluate_reporting_kkt`, `CompositeOT/scaling.py:56` `scale_problem`.
6. PALM 默认 `stoptype=1`，所以 `CompositeOT/termination.py:180` `termination_residual` 返回 `diagnostics.optimality_residual`。而 `evaluate_kkt` 中 `optimality_residual = max(kkt_residual, 1e-1 * relative_gap)`。因此默认停止条件是:
   ```text
   max(kkt_residual, 0.1 * relative_gap) <= PALMOptions.tolerance
   ```
   默认 tolerance 是 `1e-6`。锚点: `CompositeOT/termination.py:42` `evaluate_kkt`, `CompositeOT/termination.py:180` `termination_residual`, `examples/common.py:116` `make_options`.
7. 另一个停止条件是运行时间超过 `PALMOptions.maxtime` 或达到 `PALMOptions.max_iterations`。锚点: `CompositeOT/palm.py:34` `solve_palm`, `CompositeOT/options.py:98` `PALMOptions`.

### 7.6 Semismooth Newton 内层循环和 QROT fast path

#### 代码明确表明的事实

1. `CompositeOT/newton.py:29` `solve_semismooth_newton` 输入当前 PALM center:
   - `center_primal: PrimalSolution`
   - `center_dual: DualSolution`
   - `initial_dual: DualSolution`
   - `auxiliary_error: DualSolution`
   - `sigma: float`, `tau: float`
   - `options: SemismoothNewtonOptions`
   - `plan: SolverPlan`
   输出 `(dual, primal, gradient, iterations, converged)`。锚点: `CompositeOT/newton.py:29` `solve_semismooth_newton`.
2. Newton 真正迭代的变量是 packed dual vector `[vec(W); u; v]`。对 QROT 无 side，`W` 是空矩阵，所以 Newton 变量维度是 `m+n`，由 `u (m,)` 和 `v (n,)` 构成。锚点: `CompositeOT/newton.py:29` `solve_semismooth_newton`, `CompositeOT/results.py:124` `DualSolution.pack`, `CompositeOT/results.py:124` `DualSolution.from_vector`.
3. `make_subproblem_workspace(problem, plan)` 对 QROT 无 side 且 transport 是 nonnegative quadratic 时返回 fast path buffers:
   - `X (m,n) float64`
   - `transport_active (m,n) bool`
   - `row_sum (m,) float64`
   - `column_sum (n,) float64`
   - `row_argument (m,)`, `column_argument (n,)`
   - `y (m,)`, `z (n,)`
   - `gradient_W (0,0)`, `gradient_vector (m+n,)`。锚点: `CompositeOT/subproblem.py:98` `make_subproblem_workspace`, `CompositeOT/plan.py:274` `SolverPlan.uses_fused_transport_subproblem`.
4. `CompositeOT/subproblem.py:136` `evaluate_newton_data` 对 QROT 进入 `_nonnegative_quadratic_fast_data`。锚点: `CompositeOT/subproblem.py:136` `evaluate_newton_data`, `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`.
5. fast path 的 transport prox 公式:
   ```text
   scale = 1 / (1 + sigma * lambda_scaled)
   a_ij = center_X_ij - sigma * C_ij + sigma * (u_i + v_j)
   X_ij = max(a_ij, 0) * scale
   ```
   其中 `X` 是本 Newton dual iterate 对应恢复出的 primal coupling，形状 `(m,n)`。锚点: `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`, `CompositeOT/subproblem.py:709` `_transport_nq_prox_rows`.
6. QROT 无 side 时 `D=center_primal.D`，形状 `(0,0)`；row/column slack 由 zero-indicator prox 得到 `y=0`, `z=0`。锚点: `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`, `CompositeOT/regularizers/zero_indicator.py:14` `ZeroIndicator`.
7. fast path 的 gradient:
   ```text
   gradient_u = X 1 + y - alpha + (tau/sigma) * (u - center_u)
   gradient_v = X.T 1 + z - beta + (tau/sigma) * (v - center_v)
   ```
   输出 `gradient: DualSolution(W(0,0), u(m,), v(n,))` 和 `gradient_vector: (m+n,) float64`。锚点: `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`.
8. fast path 的 smooth subproblem objective transport 部分写成:
   ```text
   0.5 * (1/sigma + lambda_scaled) * ||X||_F^2
   ```
   再加 row/column Moreau terms、`-<alpha,u> - <beta,v>`、PALM 常数项和 dual proximal term。锚点: `CompositeOT/subproblem.py:646` `_nonnegative_quadratic_objective_from_square`.
9. fast path 的 generalized Jacobian:
   - transport block 是 active-set diagonal，`diagonal=transport_active`, `diagonal_scale=scale`，形状 `(m,n)`。
   - side block 是空 `(0,0)` zero Jacobian。
   - row/column block 来自 zero-indicator regularizer。锚点: `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`, `CompositeOT/regularizers/nonnegative_quadratic.py:81` `generalized_jacobian`.
10. `CompositeOT/newton.py:29` `solve_semismooth_newton` 每轮调用 `CompositeOT/normal.py:24` `solve_normal(plan, sigma, tau, *jacobians, rhs=-gradient_vector)`。输出是一维 `np.ndarray float64`，形状 `(m+n,)`。锚点: `CompositeOT/normal.py:24` `solve_normal`.
11. `solve_normal` 通过 `plan.normal.assemble(...)` 调用 `CompositeOT/normalassemble.py` `assemble_normal`，形成 Newton normal matrix `H`，再解
    ```text
    H d = -gradient_vector.
    ```
    如果矩阵是 dense，走 `scipy.linalg.solve`; 如果是 sparse，优先 `sksparse.cholmod.cholesky`，失败后 `scipy.sparse.linalg.splu`。锚点: `CompositeOT/normal.py:24` `solve_normal`, `CompositeOT/plan.py:186` `NormalPlan.assemble`, `CompositeOT/normalassemble.py` `assemble_normal`.
12. 解出的方向向量用 `DualSolution.from_vector(problem, direction_vector)` 还原为 `direction.W(0,0)`, `direction.u(m,)`, `direction.v(n,)`。锚点: `CompositeOT/results.py:124` `DualSolution.from_vector`, `CompositeOT/newton.py:29` `solve_semismooth_newton`.
13. `CompositeOT/linesearch.py:20` `armijo_line_search` 在 dual variable 上做 backtracking，候选点是 `dual + step * direction`，目标函数由 `CompositeOT/subproblem.py:207` `objective_at_dual` 重新计算。锚点: `CompositeOT/linesearch.py:20` `armijo_line_search`, `CompositeOT/subproblem.py:207` `objective_at_dual`.
14. Newton 内层停止准则有三种:
    ```text
    relative_inexactness lhs <= rhs
    or gradient.norm() <= SemismoothNewtonOptions.tolerance
    or max(lhs, rhs) < 1e-12
    ```
    其中 relative inexactness 是
    ```text
    lhs = 2*abs(sigma*(auxiliary_error.dot(gradient) - dual.dot(gradient)))
          + sigma^2 * ||gradient||^2
    rhs = rho * (||primal-center_primal||^2
                 + tau * ||dual-center_dual||^2)
    ```
    锚点: `CompositeOT/newton.py:29` `solve_semismooth_newton`, `CompositeOT/termination.py:187` `relative_inexactness`.
15. 若方向不是下降方向或 line search 失败，Newton 返回 `inner_converged=False`，PALM 仍会记录这个状态并继续或最后报告。锚点: `CompositeOT/newton.py:29` `solve_semismooth_newton`, `CompositeOT/palm.py:34` `solve_palm`.

### 7.7 最终 coupling 如何恢复

#### 代码明确表明的事实

1. 在 PALM/Newton 阶段，coupling `X` 不是最后另行解线性方程得到的；它在每次 `evaluate_newton_data` 中由当前 dual iterate 的 prox 公式恢复，并作为 `PrimalSolution.X` 返回。对 QROT fast path:
   ```text
   X_scaled_ij = max(center_X_ij - sigma*C_scaled_ij
                     + sigma*(u_i + v_j), 0)
                 / (1 + sigma*lambda_scaled)
   ```
   锚点: `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`, `CompositeOT/subproblem.py:709` `_transport_nq_prox_rows`.
2. `solve_semismooth_newton` 返回的 `primal_next` 被 `solve_palm` 赋给外层 `primal`。因此最终 PALM result 中的 scaled coupling 是最后一次接受的 Newton dual iterate 对应的 prox output。锚点: `CompositeOT/newton.py:29` `solve_semismooth_newton`, `CompositeOT/palm.py:34` `solve_palm`.
3. `solve_palm` 用 `CompositeOT/results.py:332` `make_solver_result` 封装 scaled result。锚点: `CompositeOT/results.py:332` `make_solver_result`.
4. `CompositeOT/solver.py:35` `solve` 最后调用 `CompositeOT/scaling.py:199` `unscale_solver_result`。其中 `unscale_primal` 做:
   ```text
   X_raw = pscale * X_scaled
   D_raw = pscale * D_scaled
   y_raw = pscale * y_scaled
   z_raw = pscale * z_scaled
   ```
   所以用户看到的 `result.primal.X` 就是 raw units coupling，形状 `(m,n) float64`。锚点: `CompositeOT/scaling.py:180` `unscale_primal`, `CompositeOT/scaling.py:199` `unscale_solver_result`.
5. `examples/common.py:146` `solve_and_report` 返回这个 `SolverResult`；`examples/example_qrot.py:11` `main` 本身没有保存变量，交互使用时用户可通过直接调用同一 API 得到 `result.primal.X`。锚点: `examples/common.py:146` `solve_and_report`, `CompositeOT/results.py:225` `SolverResult`.

### 7.8 本节的推测

- 对 `example_qrot.py` 这条路径，性能关键分支大概率是 `SolverPlan.uses_fused_transport_subproblem == True` 后的 `_nonnegative_quadratic_fast_data`，因为 QROT 同时满足 `transport.is_nonnegative_quadratic` 和 `side.has_scalar_transport_shift`，无 side 时 side shift 是 `0.0`。锚点: `CompositeOT/plan.py:274` `SolverPlan.uses_fused_transport_subproblem`, `CompositeOT/subproblem.py:428` `_nonnegative_quadratic_fast_data`.
- 对默认参数 `m=n=1000`，Newton normal variable 维度是 `2000`，transport coupling 是 `1000 x 1000`。这个维度结论来自默认参数和 `NormalPlan.total_dim = side_dim + row_dim + column_dim`，不是运行时实测。锚点: `examples/example_qrot.py:11` `main`, `CompositeOT/plan.py:186` `NormalPlan`.
