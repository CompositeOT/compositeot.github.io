# CompositeOT Time Profiling Manual

本文档面向这个 repo 的实际求解链路，目标是帮助你回答:

- `examples/example_qrot.py` 一次运行总耗时是多少？
- 数据生成、建模、缩放、ADMM warm start、PALM outer loop、SSN inner loop 分别耗时多少？
- SSN 中 `evaluate_newton_data`, `solve_normal`, `armijo_line_search` 谁最贵？
- 未来优化代码时，如何稳定复现实验并对比改动前后的耗时？

## 1. 先理解当前调用链

以 `examples/example_qrot.py` 为例，主路径是:

```text
examples/example_qrot.py::main
  -> CompositeOT.utils.make_qrot_example_data
  -> CompositeOT.model.Model / CompositeOTModel
  -> CompositeOTModel.add_transport_regularizer
  -> examples.common.solve_and_report
  -> CompositeOTModel.solve
  -> CompositeOTModel.compile
  -> CompositeOTProblem.solve
  -> CompositeOT.solver.solve
  -> CompositeOT.solver.solve_scaled
      -> SolverPlan.compile
      -> sgsadmm              # if ADMMOptions.max_iterations > 0
      -> solve_palm
          -> solve_semismooth_newton
              -> evaluate_newton_data
              -> solve_normal
              -> armijo_line_search
  -> unscale_solver_result
```

关键文件:

- `examples/example_qrot.py`: QROT 示例入口。
- `examples/common.py`: 示例 CLI options 和总耗时打印。
- `CompositeOT/solver.py`: 高层调度，缩放、ADMM warm start、PALM。
- `CompositeOT/sgsadmm.py`: SGS-ADMM warm start 主循环。
- `CompositeOT/palm.py`: PALM outer loop。
- `CompositeOT/newton.py`: semismooth Newton inner loop。
- `CompositeOT/subproblem.py`: Newton 数据、prox 恢复 `X`、gradient、Jacobian。
- `CompositeOT/normal.py`: normal equation 求解。
- `CompositeOT/linesearch.py`: Armijo line search。

## 2. 最低成本: 使用已有计时信息

### 2.1 跑示例并开启 verbose

```powershell
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --verbose
```

`examples/common.py::solve_and_report` 已经用 `time.perf_counter()` 包住了 `model.solve(...)`:

```python
start = time.perf_counter()
result = model.solve(...)
elapsed = time.perf_counter() - start
print_solver_result(title, result, elapsed)
```

你会看到最终 summary:

```text
QROT
  converged: ...
  iterations: ...
  objective: ...
  time: ... s
```

这个 `time` 是从 `model.solve()` 开始到返回的总耗时，不包含 `make_qrot_example_data(...)` 的数据生成时间。

### 2.2 看 PALM history

`CompositeOT/results.py::new_solver_history` 里已有 `history["time"]`。`CompositeOT/palm.py::solve_palm` 每次外层迭代 append 当前累计时间。

交互式使用时:

```python
result = model.solve(...)
print(result.runtime)
print(result.history["iteration"])
print(result.history["time"])
print(result.history["inner_iterations"])
```

含义:

- `result.runtime`: solver 内部统计的总 runtime。
- `result.history["time"]`: PALM 迭代检查点累计时间；如果有 ADMM warm start，`solver.py` 会把 ADMM warm-start elapsed 加到 PALM history time 上。
- `result.history["inner_iterations"]`: 每个 PALM outer iteration 中 SSN 的 inner iteration 数。

注意: 这仍然不是函数级 profiling，只能看阶段累计。

## 3. 推荐基准命令

为了减少噪音，建议先用中小规模并固定参数:

```powershell
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --seed 0 --dimension 3 --admm-iterations 0 --max-iterations 20 --verbose
```

这条命令是 PALM/SSN only，适合看 Newton 子问题耗时。

默认 warm-start 路径:

```powershell
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --seed 0 --dimension 3 --admm-iterations 300 --max-iterations 20 --verbose
```

只跑 SGS-ADMM 的方式不是命令行默认路径，需要写一个很小的脚本调用:

```python
result = model.solve_sgsadmm(options=admm_options)
```

原因: `model.solve(...)` 的语义是:

```text
ADMM max_iterations > 0: SGS-ADMM warm start + PALM
ADMM max_iterations = 0: PALM only
```

而不是 ADMM only。

## 4. 方法一: 外部总耗时 profiling

如果你只想测脚本总耗时，用 PowerShell:

```powershell
Measure-Command { .\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 }
```

这个包括:

- Python 启动时间
- import 时间
- 数据生成时间
- model 构造时间
- solver 时间
- 打印时间

适合粗略端到端比较，不适合定位热点函数。

## 5. 方法二: `cProfile` 函数级统计

Python 标准库自带 `cProfile`，不用额外安装。

```powershell
.\.venv\python.exe -m cProfile -o qrot.prof examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 --max-iterations 20
```

查看前 40 个 cumulative time 函数:

```powershell
.\.venv\python.exe -m pstats qrot.prof
```

进入交互后输入:

```text
sort cumtime
stats 40
```

或者用一行脚本:

```powershell
.\.venv\python.exe -c "import pstats; p=pstats.Stats('qrot.prof'); p.strip_dirs().sort_stats('cumtime').print_stats(40)"
```

重点看这些函数:

- `CompositeOT/solver.py::solve`
- `CompositeOT/solver.py::solve_scaled`
- `CompositeOT/sgsadmm.py::sgsadmm`
- `CompositeOT/palm.py::solve_palm`
- `CompositeOT/newton.py::solve_semismooth_newton`
- `CompositeOT/subproblem.py::evaluate_newton_data`
- `CompositeOT/subproblem.py::_nonnegative_quadratic_fast_data`
- `CompositeOT/normal.py::solve_normal`
- `CompositeOT/normalassemble.py::assemble_normal`
- `CompositeOT/linesearch.py::armijo_line_search`

解释:

- `tottime`: 函数自身耗时，不含子函数。
- `cumtime`: 函数加所有子函数总耗时。
- `ncalls`: 调用次数。

对这个 repo，通常先看 `cumtime` 找大阶段，再看 `tottime` 找真正热点。

### 5.1 查看 `.prof` 文件的其它方法

`qrot.prof` 是 `cProfile` 生成的二进制统计文件。除了 `python -m pstats`，还可以用下面几种方式查看。

#### 5.1.1 `snakeviz`

这是最推荐的可视化方式之一，会在浏览器中展示调用树/火焰图。

```powershell
.\.venv\python.exe -m pip install snakeviz
.\.venv\python.exe -m snakeviz qrot.prof
```

适合快速回答:

```text
总时间主要被哪条调用链吃掉？
某个大函数下面最贵的子函数是谁？
```

#### 5.1.2 `tuna`

另一个轻量可视化工具:

```powershell
.\.venv\python.exe -m pip install tuna
.\.venv\python.exe -m tuna qrot.prof
```

`tuna` 的交互视图也适合快速定位累计耗时热点。

#### 5.1.3 导出文本报告

按 cumulative time 导出前 80 个函数:

```powershell
.\.venv\python.exe -c "import pstats; p=pstats.Stats('qrot.prof'); p.strip_dirs().sort_stats('cumtime').print_stats(80)" > qrot_profile_cumtime.txt
```

按 self time / total time 导出前 80 个函数:

```powershell
.\.venv\python.exe -c "import pstats; p=pstats.Stats('qrot.prof'); p.strip_dirs().sort_stats('tottime').print_stats(80)" > qrot_profile_tottime.txt
```

建议两个都保存:

- `cumtime`: 找阶段瓶颈，例如 `solve_semismooth_newton` 总体贵不贵。
- `tottime`: 找函数自身热点，例如某个 Python 循环是否真的贵。

#### 5.1.4 交互式 `pstats`

```powershell
.\.venv\python.exe -m pstats qrot.prof
```

进入后常用命令:

```text
sort cumtime
stats 50
sort tottime
stats 50
callers solve_normal
callees solve_semismooth_newton
```

常见用法:

- `callers <函数名>`: 看是谁调用了这个函数。
- `callees <函数名>`: 看这个函数内部又调用了谁。
- `stats <N>`: 打印前 N 条结果。

#### 5.1.5 `gprof2dot` 调用图

可以把 `.prof` 转成图片:

```powershell
.\.venv\python.exe -m pip install gprof2dot
gprof2dot -f pstats qrot.prof | dot -Tpng -o qrot_profile.png
```

注意: 这个方法还需要系统安装 Graphviz，并确保 `dot.exe` 在 `PATH` 中。

如果只是快速看，优先级建议是:

```text
snakeviz > 文本 cumtime/tottime 报告 > pstats 交互 > gprof2dot
```

## 6. 方法三: 写一个专用 profiling driver

建议不要直接改 `examples/example_qrot.py`。新建临时脚本，比如 `scripts/profile_qrot.py`，专门测阶段耗时。

示例:

```python
from __future__ import annotations

import time

from CompositeOT import ADMMOptions, Model, NonnegativeQuadraticRegularizer, PALMOptions
from CompositeOT.utils import make_qrot_example_data


def timed(label: str, func):
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    print(f"{label:30s} {elapsed:10.6f} s")
    return value


def main() -> None:
    m = 200
    n = 200
    seed = 0
    dimension = 3
    lambda_q = 1.0

    data = timed(
        "make_qrot_example_data",
        lambda: make_qrot_example_data(m, n, seed, dimension),
    )
    model = timed(
        "Model init",
        lambda: Model(data.C, data.alpha, data.beta, name="QROT-profile"),
    )
    timed(
        "add QROT regularizer",
        lambda: model.add_transport_regularizer(
            NonnegativeQuadraticRegularizer(lambda_q)
        ),
    )
    problem = timed("compile", model.compile)

    palm_options = PALMOptions(max_iterations=20, tolerance=1e-6, verbose=False)
    admm_options = ADMMOptions(max_iterations=0, verbose=False)

    result = timed(
        "problem.solve",
        lambda: problem.solve(
            palm_options=palm_options,
            admm_options=admm_options,
        ),
    )
    print("runtime reported:", result.runtime)
    print("outer iterations:", result.iterations)
    print("admm iterations:", result.admm_iterations)


if __name__ == "__main__":
    main()
```

运行:

```powershell
.\.venv\python.exe scripts\profile_qrot.py
```

这个脚本适合回答:

- 数据生成是否占比明显？
- `Model(...)` 和 `compile()` 是否可忽略？
- `problem.solve(...)` 本身耗时是多少？

## 7. 方法四: 手动插桩 solver 阶段

当你要知道 ADMM、PALM、SSN 的阶段耗时时，最直接的是临时加 `perf_counter`。

### 7.1 `CompositeOT/solver.py`

目标: 分离缩放、plan compile、ADMM warm start、PALM。

可临时加在:

- `solve(...)`: `scale_problem`, `solve_scaled`, `unscale_solver_result`
- `solve_scaled(...)`: `SolverPlan.compile`, `sgsadmm`, `solve_palm`

建议输出字段:

```text
scale_problem
SolverPlan.compile
sgsadmm
solve_palm
unscale_solver_result
```

注意: 这些 print 会污染示例输出。长期建议做成 option 或环境变量，比如:

```python
import os
PROFILE_TIMING = os.environ.get("COMPOSITEOT_PROFILE_TIMING") == "1"
```

运行时:

```powershell
$env:COMPOSITEOT_PROFILE_TIMING='1'
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200
```

### 7.2 `CompositeOT/palm.py`

目标: 每个 PALM outer iteration 的耗时和 SSN 耗时。

插桩点:

```python
inner_start = time.perf_counter()
dual_next, primal_next, gradient, inner_iterations, inner_converged = (
    solve_semismooth_newton(...)
)
inner_elapsed = time.perf_counter() - inner_start
```

建议记录:

```text
outer iteration
sigma
tau
inner_iterations
inner_elapsed
diagnostics evaluation elapsed
history append elapsed
```

### 7.3 `CompositeOT/newton.py`

目标: 拆 SSN inner iteration。

关键耗时点:

```text
make_subproblem_workspace
initial evaluate_newton_data
relative_inexactness
solve_normal
armijo_line_search
post-line-search evaluate_newton_data
```

在 `solve_semismooth_newton` 里，一轮 SSN 大致是:

```text
evaluate_newton_data
relative_inexactness
solve_normal
armijo_line_search
evaluate_newton_data
```

最值得单独计时的是:

- `evaluate_newton_data`: prox, gradient, Jacobian。
- `solve_normal`: normal matrix assembly + linear solve。
- `armijo_line_search`: 多次 objective evaluation。

### 7.4 `CompositeOT/subproblem.py`

对 QROT，重点是 fast path:

- `evaluate_newton_data`
- `_nonnegative_quadratic_fast_data`
- `_transport_nq_prox_rows`
- `_nonnegative_quadratic_fast_objective`
- `_transport_nq_prox_square`

如果 `numba` 已安装，第一次调用 Numba kernel 会包含 JIT 编译时间。正式计时前应先 warm up 一次。

## 8. Numba profiling 注意事项

这个 repo 的 `CompositeOT/utils.py::njit` 是可选 Numba:

```python
try:
    from numba import njit
except ImportError:
    ...
```

如果安装了 `numba`:

- 第一次运行会触发 JIT 编译，耗时会显著偏大。
- 第二次同样参数运行通常更接近真实计算耗时。
- 有 `cache=True` 的 kernel 会缓存编译产物，但首次仍可能慢。

建议:

```powershell
# 第一次 warm-up，不记录
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 --max-iterations 3

# 第二次开始记录
.\.venv\python.exe -m cProfile -o qrot.prof examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 --max-iterations 20
```

如果你要比较 Numba 与纯 Python fallback，需要在两个干净环境分别测，不建议在同一环境中临时 monkey patch。

## 9. 推荐 profiling 层级

### 第一层: 端到端

问题:

```text
整个脚本要多久？
```

方法:

```powershell
Measure-Command { .\.venv\python.exe examples\example_qrot.py --m 200 --n 200 }
```

### 第二层: solver 总耗时

问题:

```text
model.solve() 要多久？
```

方法:

- 直接看 `examples/common.py::solve_and_report` 输出。
- 或使用专用 driver 包住 `problem.solve(...)`。

### 第三层: 算法阶段

问题:

```text
ADMM 和 PALM 分别多久？
```

方法:

- 对比 `--admm-iterations 0` 和默认路径。
- 临时插桩 `CompositeOT/solver.py::solve_scaled`。

### 第四层: PALM/SSN 内部

问题:

```text
SSN 每轮谁最贵？
```

方法:

- 插桩 `CompositeOT/newton.py::solve_semismooth_newton`。
- 分别记录 `evaluate_newton_data`, `solve_normal`, `armijo_line_search`。

### 第五层: kernel / linear algebra

问题:

```text
normal assembly 慢，还是 linear solve 慢？
transport prox 慢，还是 row/column reductions 慢？
```

方法:

- 插桩 `CompositeOT/normal.py::solve_normal`，把 `plan.normal.assemble(...)` 和 `dense_linalg.solve/splu/cholesky` 分开。
- 插桩 `CompositeOT/subproblem.py::_nonnegative_quadratic_fast_data`，把 transport prox rows、column sum、row/column prox、Jacobian 构造分开。

## 10. 推荐输出格式

建议将 profiling 输出做成 CSV 风格，方便后续画图:

```text
phase,iteration,elapsed,extra
solve_scaled.plan_compile,0,0.00123,
solve_scaled.sgsadmm,0,0.45678,iters=300
palm.newton,1,0.12345,inner=5
newton.evaluate_newton_data,1,0.03000,
newton.solve_normal,1,0.08000,
newton.line_search,1,0.01000,
```

如果未来要正式保留 profiling 功能，建议新增一个轻量 helper，比如:

```python
class TimerLog:
    def __init__(self):
        self.rows = []

    def record(self, phase, elapsed, **extra):
        self.rows.append((phase, elapsed, extra))
```

但在开始阶段，直接 `time.perf_counter()` 加 print 更快。

## 11. 对 `example_qrot.py` 的具体建议

### 测 PALM/SSN only

```powershell
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 --max-iterations 20 --verbose
```

用途:

- 直接观察 PALM outer iterations。
- 聚焦 SSN 与 normal equation。

### 测 ADMM warm start + PALM

```powershell
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --admm-iterations 300 --max-iterations 20 --verbose
```

用途:

- 看 warm start 是否减少 PALM iterations。
- 比较总 runtime 是否更好。

### 测规模增长

```powershell
.\.venv\python.exe examples\example_qrot.py --m 100 --n 100 --admm-iterations 0 --max-iterations 10
.\.venv\python.exe examples\example_qrot.py --m 200 --n 200 --admm-iterations 0 --max-iterations 10
.\.venv\python.exe examples\example_qrot.py --m 400 --n 400 --admm-iterations 0 --max-iterations 10
```

记录:

```text
m,n,total_time,outer_iterations,total_inner_iterations,result.runtime
```

## 12. 常见误区

1. 第一次 Numba 运行不能代表稳定耗时。
2. `verbose=True` 会产生打印开销，小规模时可能影响比例。
3. `Measure-Command` 包含 Python 启动和 import 时间。
4. `result.runtime` 是 solver 内部时间，不等于完整脚本时间。
5. `model.solve(admm_options=ADMMOptions(max_iterations=0))` 是 PALM only，不是 ADMM only。
6. ADMM only 要调用 `model.solve_sgsadmm(...)`。
7. `cProfile` 对 NumPy/SciPy 底层 C/Fortran 时间只能看到外层 Python 调用，无法展开 BLAS/LAPACK 内部。

## 13. 一个建议的优化工作流

1. 固定环境: Python/NumPy/SciPy/Numba 版本不变。
2. 固定问题: `m,n,seed,dimension,lambda_q,options` 全固定。
3. warm up 一次，尤其是安装了 Numba 时。
4. 运行 3 次以上，记录 median，而不是只看一次。
5. 先用 `cProfile` 找大阶段。
6. 对大阶段加 `perf_counter` 手动插桩。
7. 只优化最大耗时阶段。
8. 每次优化后跑同样命令对比:
   - total time
   - `result.iterations`
   - `result.admm_iterations`
   - `result.objective_value`
   - `result.kkt_residual`
9. 如果耗时下降但迭代数或 KKT 质量明显变化，要分开判断是算法行为改变还是代码效率提升。

## 14. 建议优先观察的热点

按这个 repo 的结构，QROT 路径优先看:

1. `CompositeOT/subproblem.py::_nonnegative_quadratic_fast_data`
   - transport prox + row sums + active set。
2. `CompositeOT/normalassemble.py::assemble_normal`
   - Newton normal matrix assembly。
3. `CompositeOT/normal.py::solve_normal`
   - dense/sparse direct solve。
4. `CompositeOT/linesearch.py::armijo_line_search`
   - backtracking objective evaluations。
5. `CompositeOT/sgsadmm.py::sgsadmm`
   - 如果启用 warm start，看 ADMM sweep 是否值得。

这五个点基本覆盖了未来优化 QROT 示例最可能受益的位置。
