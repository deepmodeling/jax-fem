# Single-GPU device-resident PETSc experiment

This directory records an experimental JAX-FEM pipeline that keeps the bulk
finite-element assembly and linear solve data on one NVIDIA GPU.  It was tested
on one Quadro RTX 8000 (48 GB) with a three-dimensional HEX8 Neo-Hookean
problem in double precision.

The experiment is preserved on the `petsc_gpu` branch.  It is intentionally
not part of `main`: the speedup was substantial for `n=50`, nearly disappeared
for `n=80`, and the tested `n=100` case ran out of memory.

## Implementation

The standard solver supports several CPU and GPU backends and is left
unchanged.  The experiment consists of:

- `jax_fem/petsc_gpu_solver.py`: an opt-in, single-process PETSc GPU Newton
  solver;
- a small opt-in hook in `jax_fem/problem.py` that prevents the tangent from
  being converted to a host NumPy array;
- `applications/gpu/hyperelastic3d.py`: the benchmark and solution comparison.

The PETSc route uses:

- `MATSEQAIJCUSPARSE` and `VECSEQCUDA`;
- PETSc COO preallocation followed by device `MatSetValuesCOO` updates;
- DLPack for JAX/PETSc vector handoffs;
- symmetric row-and-column Dirichlet elimination;
- CG preconditioned by cuSPARSE ICC(0), with factor ordering reuse.

Only convergence scalars, one-time sparsity indices, mesh construction, and
optional output use the CPU.  The element residual/tangent, matrix values,
linear right-hand side, and solution remain device resident.

## Data flow

Existing `pyamgx` route:

```text
JAX element residual/tangent on GPU
  -> tangent copied to CPU
  -> PETSc/CPU COO-to-CSR assembly
  -> CSR matrix and vectors uploaded by pyamgx
  -> AMGX solve on GPU
  -> solution downloaded and returned to JAX
```

Experimental PETSc route:

```text
JAX element residual/tangent on GPU
  -> PETSc MatSetValuesCOO consumes device tangent values
  -> PETSc forms AIJCUSPARSE on GPU
  -> cuSPARSE ICC(0) + PETSc CG on GPU
  -> PETSc CUDA Vec returned to JAX through DLPack
```

## Server environment

The AMGX reference used the existing `jax-fem-env`; the PETSc route used
`jax-fem-gpu`.  The latter contained JAX 0.10.1, petsc4py/PETSc 3.25.3,
real `float64` scalars, 32-bit PETSc indices, CUDA support, and DLPack-enabled
PETSc vectors.  HYPRE was not available.

The CUDA-enabled PETSc installation was:

```bash
export PETSC_GPU_PREFIX=/home/user/Documents/tianjuxue/software/petsc-gpu-stack-3.25.3/install
```

The old environment had a conflicting CUDA library search path, so its AMGX
command explicitly removes `LD_LIBRARY_PATH`.  Both commands disable JAX's GPU
preallocation because JAX and the external sparse solver need to share the
same 48 GB device.

## Benchmark commands

Run AMGX first to write the timing and solution reference:

```bash
conda activate jax-fem-env

env -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.gpu.hyperelastic3d \
  --backend amgx --gpu 0 --n 50 \
  --json /tmp/jax_fem_amgx.json \
  --solution /tmp/jax_fem_amgx_solution.npy
```

Then run PETSc in its environment:

```bash
conda activate jax-fem-gpu
export PETSC_GPU_PREFIX=/home/user/Documents/tianjuxue/software/petsc-gpu-stack-3.25.3/install

env \
  PETSC_DIR="$PETSC_GPU_PREFIX" \
  PETSC_ARCH="" \
  LD_LIBRARY_PATH="$PETSC_GPU_PREFIX/lib" \
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.gpu.hyperelastic3d \
  --backend petsc_gpu --gpu 0 --n 50 \
  --reference-json /tmp/jax_fem_amgx.json \
  --reference-solution /tmp/jax_fem_amgx_solution.npy \
  --json /tmp/jax_fem_petsc_gpu.json
```

Use the same commands with `--n 80` for the larger measured case.  Optional
Krylov experiments include `--cg-single-reduction` and
`--ksp-type pipecg`.  `JAX_FEM_GPU_SOLVER` is not used by this benchmark; the
backend is selected by `--backend`.

Each invocation solves twice.  The first measurement includes compilation and
first-use caches; the second is the representative warm-cache result.

## Results

### End-to-end timings

| Mesh | Cells / DOFs | Backend | First solve | Warm solve | AMGX / backend |
| --- | ---: | --- | ---: | ---: | ---: |
| `50^3` | 125,000 / 397,953 | pyamgx | 24.538 s | 15.872 s | 1.000x |
| `50^3` | 125,000 / 397,953 | PETSc CG + ICC(0) | 19.097 s | 9.884 s | 1.606x |
| `50^3` | 125,000 / 397,953 | PETSc single-reduction CG + ICC(0) | 19.783 s | 9.658 s | 1.643x |
| `50^3` | 125,000 / 397,953 | PETSc PIPECG + ICC(0) | 19.083 s | 10.055 s | 1.578x |
| `80^3` | 512,000 / 1,594,323 | pyamgx | 67.732 s | 44.640 s | 1.000x |
| `80^3` | 512,000 / 1,594,323 | PETSc CG + ICC(0) | 68.568 s | 43.913 s | 1.017x |

The PETSc/AMGX relative solution differences were `4.62e-11` for `n=50`
and `9.36e-11` for `n=80` in the vector L2 norm.

### PETSc warm-run breakdown

| Mesh | JAX local assembly | PETSc matrix update | PETSc linear phase | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `50^3` | 3.537 s | 1.426 s | 4.917 s | 9.884 s |
| `80^3` | 28.148 s | 5.314 s | 10.447 s | 43.913 s |

The cell count grew by 4.096x from `n=50` to `n=80`, but local assembly grew
by about 7.96x and became 64% of the PETSc wall time.  The dominant candidate
is `split_and_compute_cell`: it evaluates a 24-direction forward-mode element
Jacobian and concatenates 20 tangent batches.  At `n=80`, one complete element
tangent array is about 2.36 GB.  This was not profiled finely enough to assign
an exact compute-versus-memory percentage.

## Limitations and conclusion

- The `n=80` gain is only 1.7%, which is close enough to normal run-to-run
  variation that the two routes should be considered performance-equivalent.
- The tested `n=100` case exceeded available memory.  Simultaneous JAX element
  tangents, PETSc COO metadata/CSR, ICC factors, and automatic-differentiation
  temporaries create a much larger peak footprint than the staged route.
- The implementation is restricted to one process and one visible CUDA GPU.
- Multipoint constraints and line search are not implemented in this narrow
  solver.
- The experiment does not establish a generally faster or more memory-scalable
  replacement for the existing AMGX path.

The experiment demonstrates that bulk JAX/PETSc handoffs can remain on the
GPU and can be faster at moderate size.  Its scaling and peak-memory behavior
are not strong enough to justify adding the extra solver path to `main`, so the
work is archived on the `petsc_gpu` branch for possible future profiling or
redesign.
