# coding: utf-8
"""V07 experimental PARDISO solver variants for the ablation study.

Selected via the ``V07_PARDISO_MODE`` environment variable read by the v04
wrapper's ``_PardisoCustomSolver``. Modes form a ladder where each level adds
exactly one optimization on top of the previous, so bench deltas attribute
cleanly:

    base       -- (env unset) untouched v04 path: pypardiso.spsolve per call
                  (matrix copy + phase 12 + full compare + phase 33 + double
                  ia/ja conversion)
    nocmp      -- bypass pypardiso bookkeeping: one phase-13 call per solve,
                  fresh 1-based ia/ja each call
    cache-idx  -- nocmp + reuse the 1-based int32 ia/ja across calls while the
                  sparsity pattern is unchanged (exact indptr/indices check)
    phase23    -- cache-idx + reuse the symbolic factorization: phase 13 on
                  the first solve of a pattern, phase 23 afterwards
    fp32ir     -- phase23 with single-precision factorization (iparm(28)=1)
                  plus manual float64 iterative refinement to recover
                  direct-solver accuracy

Multiple PARDISO internal handles are kept per sparsity pattern (thermal and
mechanics systems have different sizes), keyed by (n, nnz).
"""

from __future__ import annotations

import atexit
import ctypes
import sys

import numpy as np
import scipy.sparse

_LADDER = ("nocmp", "cache-idx", "phase23", "fp32ir")

# fp32ir manual iterative refinement controls.
_IR_MAX_STEPS = 10
_IR_RTOL = 1e-13

# Keep at most this many per-pattern PARDISO handles (thermal + mechanics
# plus headroom for activation-driven pattern changes).
_MAX_STATES = 4


class _RawPardiso:
    """Direct MKL PARDISO caller reusing pypardiso's library discovery.

    pypardiso's ``_call_pardiso`` rebuilds the 1-based index arrays on every
    call and its ``solve``/``factorize`` bookkeeping copies and compares the
    full matrix; this class calls the raw entry point with caller-owned
    arrays instead.
    """

    def __init__(self, single_precision: bool = False):
        from pypardiso import PyPardisoSolver

        # Borrow library handle + ctypes signature from a throwaway instance.
        inner = PyPardisoSolver(mtype=11)
        self._mkl_pardiso = inner._mkl_pardiso
        self._pt_type = inner._pt_type
        self.pt = np.zeros(64, dtype=inner._pt_type[1])
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)
        self.mtype = 11
        self.single = bool(single_precision)
        if self.single:
            # iparm(1)=0 would make MKL ignore all inputs, so switching on
            # iparm(28) (single precision) requires spelling out the defaults
            # for mtype=11 explicitly (MKL PARDISO iparm reference).
            ip = self.iparm
            ip[0] = 1    # iparm(1): user-supplied iparm
            ip[1] = 2    # iparm(2): METIS nested dissection
            ip[7] = 0    # iparm(8): no internal refinement (done manually)
            ip[9] = 13   # iparm(10): pivot perturbation 1e-13
            ip[10] = 1   # iparm(11): scaling (default for mtype=11)
            ip[12] = 1   # iparm(13): weighted matching (default for mtype=11)
            ip[17] = -1  # iparm(18): report nnz(L+U)
            ip[27] = 1   # iparm(28): single-precision factorization
            ip[34] = 0   # iparm(35): 1-based indexing

    def call(self, n, a_data, ia, ja, b, phase):
        """One raw pardiso call; b must be float64/float32 to match `single`."""
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        void_p = ctypes.c_void_p
        self._mkl_pardiso(
            self.pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),
            ctypes.byref(ctypes.c_int32(1)),            # maxfct
            ctypes.byref(ctypes.c_int32(1)),            # mnum
            ctypes.byref(ctypes.c_int32(self.mtype)),   # mtype
            ctypes.byref(ctypes.c_int32(phase)),        # phase
            ctypes.byref(ctypes.c_int32(n)),            # n
            void_p(a_data.ctypes.data),                 # a
            ia.ctypes.data_as(c_int32_p),               # ia (1-based)
            ja.ctypes.data_as(c_int32_p),               # ja (1-based)
            self.perm.ctypes.data_as(c_int32_p),        # perm
            ctypes.byref(ctypes.c_int32(1)),            # nrhs
            self.iparm.ctypes.data_as(c_int32_p),       # iparm
            ctypes.byref(ctypes.c_int32(0)),            # msglvl
            void_p(b.ctypes.data),                      # b
            void_p(x.ctypes.data),                      # x
            ctypes.byref(err),
        )
        if err.value != 0:
            raise RuntimeError(
                f"MKL PARDISO failed with error {err.value} (phase {phase})"
            )
        return x

    def release(self):
        try:
            dummy = np.zeros(0)
            ia = np.ones(1, dtype=np.int32)
            ja = np.zeros(0, dtype=np.int32)
            self.call(0, dummy, ia, ja, dummy, -1)
        except Exception:
            pass


class _PatternState:
    """Cached per-sparsity-pattern data: index arrays + PARDISO handle."""

    __slots__ = ("indptr", "indices", "ia", "ja", "raw", "analyzed", "data")

    def __init__(self, indptr, indices, single):
        self.indptr = indptr
        self.indices = indices
        self.ia = indptr.astype(np.int32)
        self.ia += 1
        self.ja = indices.astype(np.int32)
        self.ja += 1
        self.raw = _RawPardiso(single_precision=single)
        self.analyzed = False
        self.data = None  # values of the currently factorized matrix


class VariantSolver:
    """Callable used by the v04 wrapper when V07_PARDISO_MODE is set."""

    def __init__(self, mode: str):
        mode = mode.strip()
        if mode not in _LADDER:
            raise ValueError(
                f"V07_PARDISO_MODE={mode!r}; expected one of {_LADDER}"
            )
        self.mode = mode
        self.level = _LADDER.index(mode)
        self.label = f"pardiso_v07({mode})"
        self._states = {}  # (n, nnz) -> _PatternState
        self._stats = {
            "solves": 0,
            "analyze_calls": 0,
            "pattern_rebuilds": 0,
            "backsolve_hits": 0,
            "ir_steps_total": 0,
            "ir_steps_max": 0,
        }
        atexit.register(self._report)

    def _report(self):
        print(f"[v07-pardiso {self.mode}] {self._stats}", file=sys.stderr)

    def _get_state(self, n, indptr, indices):
        key = (n, indices.shape[0])
        st = self._states.get(key)
        if st is not None and (
            np.array_equal(st.indptr, indptr)
            and np.array_equal(st.indices, indices)
        ):
            return st
        if st is not None:
            st.raw.release()
            self._stats["pattern_rebuilds"] += 1
        if len(self._states) >= _MAX_STATES and key not in self._states:
            _, old = self._states.popitem()
            old.raw.release()
        st = _PatternState(indptr, indices, single=(self.mode == "fp32ir"))
        self._states[key] = st
        return st

    def __call__(self, A, b, x0=None, linear_options=None):
        indptr, indices, data = A.getValuesCSR()
        rhs = np.asarray(b, dtype=np.float64)
        n = indptr.shape[0] - 1
        self._stats["solves"] += 1

        if self.mode == "nocmp":
            # Fresh index conversion every call, single phase-13 solve.
            ia = indptr.astype(np.int32)
            ia += 1
            ja = indices.astype(np.int32)
            ja += 1
            raw = self._raw_singleton()
            self._stats["analyze_calls"] += 1
            return raw.call(n, data, ia, ja, rhs, phase=13)

        st = self._get_state(n, indptr, indices)

        if self.mode == "cache-idx":
            self._stats["analyze_calls"] += 1
            return st.raw.call(n, data, st.ia, st.ja, rhs, phase=13)

        if self.mode == "phase23":
            if not st.analyzed:
                self._stats["analyze_calls"] += 1
                x = st.raw.call(n, data, st.ia, st.ja, rhs, phase=13)
                st.analyzed = True
                st.data = data.copy()
                return x
            # Unchanged values (e.g. constant-matrix time stepping): the
            # existing factorization is still valid, backsolve only.
            if st.data is not None and np.array_equal(st.data, data):
                self._stats["backsolve_hits"] += 1
                return st.raw.call(n, data, st.ia, st.ja, rhs, phase=33)
            x = st.raw.call(n, data, st.ia, st.ja, rhs, phase=23)
            st.data = data.copy()
            return x

        # fp32ir: single-precision factorize+solve, float64 refinement.
        data32 = data.astype(np.float32)
        rhs32 = rhs.astype(np.float32)
        if not st.analyzed:
            self._stats["analyze_calls"] += 1
            x32 = st.raw.call(n, data32, st.ia, st.ja, rhs32, phase=13)
            st.analyzed = True
        else:
            x32 = st.raw.call(n, data32, st.ia, st.ja, rhs32, phase=23)

        A64 = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(n, n), copy=False
        )
        x = x32.astype(np.float64)
        bnorm = np.linalg.norm(rhs)
        tol = _IR_RTOL * bnorm if bnorm > 0 else _IR_RTOL
        steps = 0
        for _ in range(_IR_MAX_STEPS):
            r = rhs - A64 @ x
            if np.linalg.norm(r) <= tol:
                break
            dx32 = st.raw.call(
                n, data32, st.ia, st.ja, r.astype(np.float32), phase=33
            )
            x += dx32.astype(np.float64)
            steps += 1
        self._stats["ir_steps_total"] += steps
        self._stats["ir_steps_max"] = max(self._stats["ir_steps_max"], steps)
        return x

    def _raw_singleton(self):
        st = self._states.get("nocmp")
        if st is None:
            st = _RawPardiso(single_precision=False)
            self._states["nocmp"] = st
        return st
