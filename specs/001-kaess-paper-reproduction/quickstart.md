# Quickstart: Review, Verify, Then Implement

本文件给出当前 worktree 的可执行入口。规格仍为 Draft，因此默认只执行
文档检查和已有自动测试；不要直接把现有 phase2 或 medium 输出登记为
论文正式结果。

当前只建立了 Spec Kit-compatible artifacts，官方 `specify` CLI、
scripts 和 templates 尚未安装。不要尝试运行 `/speckit.*` 命令；完整
初始化属于后续需用户批准的工具链变更，详见 `.specify/README.md`。

## 1. Enter the Verified WSL Environment

从 PowerShell 进入 WSL：

```powershell
wsl.exe -d Ubuntu
```

在 WSL 中执行：

```bash
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
cd "/mnt/c/Users/user/Documents/New project/jax-fem-r3-opt"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

已核对环境：

- worktree branch: `codex/r3-optimization`
- baseline commit before this specification:
  `af1ff8fcf8f79f279631613a386abd2ce68f292e`
- Python: `3.13.13`
- pytest: `9.1.1`

Windows 原生 `git` 无法识别这个 worktree，因为 `.git` 指向 WSL 内的
主仓库元数据。Git 命令必须在 WSL 中运行。

## 2. Review the Specification

```bash
git status --short --branch
find .specify specs/001-kaess-paper-reproduction -maxdepth 3 -type f | sort
```

校验 JSON 合法性：

```bash
python -m json.tool \
  specs/001-kaess-paper-reproduction/contracts/run-manifest.schema.json \
  >/dev/null
python -m json.tool \
  specs/001-kaess-paper-reproduction/contracts/paper-comparison.schema.json \
  >/dev/null
python -m json.tool \
  specs/001-kaess-paper-reproduction/contracts/backend-qualification.schema.json \
  >/dev/null
python -m json.tool \
  specs/001-kaess-paper-reproduction/contracts/backend-qualification-validation.schema.json \
  >/dev/null
```

审阅顺序：

1. `.specify/memory/constitution.md`
2. `spec.md`
3. `research.md`
4. `plan.md`
5. `data-model.md`
6. `checklists/requirements.md`
7. `checklists/paper-parity.md`
8. `tasks.md`

JSON syntax 只是第一层；正式实现还必须运行 Draft 2020-12 meta-schema、
正/反例和跨文件语义验证测试。

## 3. Existing Test Baseline

### Physics and Solver Tests

```bash
python -m pytest -q \
  tests/unit/test_newton_acceptance.py \
  tests/unit/test_newton_acceptance_compatibility.py \
  tests/unit/test_pardiso_phase23.py \
  tests/unit/test_v03_bbar_hex8.py \
  tests/unit/test_v03_thermal_mass_lumping.py \
  tests/unit/test_v03_weak_solid_powder.py \
  tests/unit/test_v06_j2_kernel.py \
  tests/unit/test_v06_lifecycle.py \
  tests/unit/test_v06_material_validation.py \
  tests/unit/test_v06_thermal_balance.py \
  tests/unit/test_v06_thermal_ledger.py \
  tests/integration/test_v03_mechanics_cutback.py \
  tests/integration/test_v03_mechanics_temperature_floor.py \
  tests/integration/test_v03_physics_fixes.py \
  tests/integration/test_v06_release_anchor_box.py
```

### Evidence-Pipeline Tests

```bash
python -m pytest -q \
  tests/unit/test_v06_mesh_audit.py \
  tests/unit/test_v06_verification.py \
  tests/integration/test_v06_provenance.py \
  tests/integration/test_v06_response_gate.py \
  tests/integration/test_v06_run_audit.py \
  tests/integration/test_v06_screening.py \
  tests/integration/test_v06_validation.py \
  tests/integration/test_v06_xrd_geometry.py \
  tests/integration/test_v06_xrd_vtu.py
```

完整非 benchmark 测试：

```bash
python -m pytest -q \
  tests/unit tests/contract tests/integration tests/regression
```

若某些目录尚不存在，先使用：

```bash
python -m pytest -q tests/unit tests/integration
```

## 4. Current Cases: Classification Only

无求解预览：

```bash
WORK_ROOT=/home/user/work/159 \
bash cases/kaess_2023/run_kaess_medium_fullheight.sh --print-plan
```

中尺度回归：

```bash
WORK_ROOT=/home/user/work/159 \
bash cases/kaess_2023/run_kaess_medium_fullheight.sh
```

该命令是 `3×100 µm` regression，不是论文正式复现。

当前 phase2 入口可以保留为 legacy baseline，但仍包含本规格列出的 P0
差异，禁止作为新 CPU reference 或论文正式结果：

```bash
RUN_ID="legacy10x30_$(date -u +%Y%m%dT%H%M%SZ)" \
WORK_ROOT=/home/user/work/159 \
PLATE_TEMP_C=150 \
POWER_TAG=P250 \
ELEMENT_TYPE=c3d8 \
POWDER_SOLID=1 \
BUILD_LAYERS=10 \
LAYER_THICKNESS=3.0e-5 \
XLA_PLATFORM=cpu \
LINEAR_SOLVER=pardiso \
bash cases/kaess_2023/run_kaess_phase2.sh
```

现有后处理：

```bash
python cases/kaess_2023/analyze_kaess.py \
  <run-dir> \
  --json <run-dir>/kaess_summary.json
```

## 5. Planned Formal Commands

以下入口将在对应任务完成后才存在。

CPU scientific reference（以下入口将在任务完成后存在）：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORK_ROOT=/home/user/work/159 \
XLA_PLATFORM=cpu \
LINEAR_SOLVER=pardiso \
bash cases/kaess_2023/run_kaess_cpu_reference.sh \
  --suite kernel,small-domain,real-dof-prefix,1layer-release,multilayer-mini \
  --repeat 2
```

Hybrid GPU qualification（必须复用相同 commit、输入、float64 checkpoint
和 acceptance model）：

```bash
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORK_ROOT=/home/user/work/159 \
bash cases/kaess_2023/run_kaess_gpu_qualification.sh \
  --mode hybrid_gpu_assembly_cpu_pardiso \
  --reference-suite latest-approved \
  --repeat 2
```

资格通过后的加速梯度：

```bash
WORK_ROOT=/home/user/work/159 \
bash cases/kaess_2023/run_kaess_accelerated_formal.sh \
  --mode hybrid_gpu_assembly_cpu_pardiso \
  --ladder 3x30,5x60,10x30
```

正式 launcher 必须先检查上游 gate artifacts，不允许用环境变量静默覆盖
物理模型和验收阈值。

`full_gpu` 不是上述 hybrid 的别名。当前代码仍为
`full_loop_xla=false` 且线性解使用 CPU PARDISO；在 GPU sparse solver、
状态驻留和 release 门禁实现前，不提供可执行的 `full_gpu` 正式命令。

## 6. Material Input Warning

当前 Kaess 材料配置位于仓库外：

```text
/home/user/work/159/materials/316L/ss316l_material_config_kaess.json
```

在 G0 冻结完成前，不要把依赖这个外部可变文件的运行登记为第三方可复跑
结果。实现任务将选择仓库内冻结副本或强制 SHA-256 方案。

## 7. Stop Rules

- Review Gate 未批准：不修改正式求解器。
- P0 physics 未通过：不启动 1层正式验证。
- CPU 小尺度 reference 未重复通过：不启动 GPU/hybrid 长算。
- CPU/GPU hot-scan slice 通过但 cooling/release 未通过：只允许继续诊断，
  不启动论文翘曲正式算例。
- 10层标准例的 `paper-comparison.verdict` 为 `partial` 或 `fail`：保存
  证据，但拒绝正式参数矩阵；仅可显式运行不进入论文结论的 diagnostic
  matrix。
- 加速后端资格未通过：不启动该后端的 3层。
- 3层加速桥接未通过：不启动 5×60 µm。
- 5×60 未通过：不启动 10×30 µm。
- 构建/冷却状态未通过：不执行正式 release。
- `backend_parity=pass` 但 energy/convergence gate 失败：不得将运行标记为
  accepted。
- 任何 CPU/GPU 配对 commit、dirty diff、物理输入、mask 或 acceptance
  model 不同：资格记录无效。
- CPU PARDISO 非零、任一正式阶段的 global sparse/linear solve/state
  未驻留 GPU，或存在意外 CPU fallback：不得使用正式 `full_gpu` 标签。
  `host_python + PETSc CUDA/AMGX` 可保持 `full_loop_xla=false`；只有
  `xla_loop` 路线必须为 `true`。
