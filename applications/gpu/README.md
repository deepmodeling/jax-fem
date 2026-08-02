# Single-GPU PETSc/AMGX experiment

This directory records an experimental single-GPU sparse assembly and solve
pipeline. The benchmark was run on one NVIDIA Quadro RTX 8000 with 48 GB of
GPU memory, using double precision and a 3D HEX8 hyperelasticity problem.

## Commands

Old pipeline, in the existing environment that provides `pyamgx`:

```bash
conda activate jax-fem-env

env \
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_FEM_GPU_SOLVER=amgx \
  python applications/gpu/hyperelastic3d_benchmark.py --n 80
```

Experimental pipeline, in the environment containing the custom
CUDA/AMGX-enabled PETSc 3.25.3 build (`pyamgx` is not required):

```bash
conda activate jax-fem-gpu

export PETSC_GPU_PREFIX=/home/user/Documents/tianjuxue/software/petsc-gpu-stack-3.25.3/install

env \
  PETSC_DIR="$PETSC_GPU_PREFIX" \
  PETSC_ARCH="" \
  LD_LIBRARY_PATH="$PETSC_GPU_PREFIX/lib" \
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_FEM_GPU_SOLVER=petsc_amgx \
  python applications/gpu/hyperelastic3d_benchmark.py --n 80
```

## Result for `n=80`

The mesh has 512,000 HEX8 cells and 1,594,323 degrees of freedom.

| Pipeline | First solve | Repeat solve | Solution L2 norm |
| --- | ---: | ---: | ---: |
| Old `pyamgx` | 121.294 s | 93.518 s | 42.27962954818 |
| PETSc GPU COO + native AMGX | 89.928 s | 66.434 s | 42.27962953993 |

For this case, the experimental route reduced first-solve time by about 26%
and repeat-solve time by about 29%. The two solution norms agree to roughly
`2e-10` relative difference.

## Data flow and current limitation

Old pipeline:

```text
JAX element assembly on GPU
  -> tangent values copied to CPU
  -> PETSc builds CPU CSR
  -> pyamgx uploads CSR and vectors to GPU
  -> AMGX solve
  -> solution downloaded to CPU and returned to JAX
```

Experimental pipeline:

```text
JAX element assembly on GPU
  -> PETSc MatSetValuesCOO consumes the JAX device values
  -> PETSc forms CUDA CSR on GPU
  -> native AMGX consumes the PETSc device CSR
  -> AMGX solve on GPU
```

The old pipeline also completed `n=100` on the same 48 GB GPU. The
experimental pipeline ran out of GPU memory at `n=100`; its simultaneous JAX
COO, PETSc COO/CSR, and AMGX matrix/hierarchy storage currently has a much
larger peak-memory footprint. The experiment is therefore promising for
`n=80`, but is not yet stable enough for the main branch.
