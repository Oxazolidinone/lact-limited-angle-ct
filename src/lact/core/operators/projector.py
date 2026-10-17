"""Fan-beam projection operator as a sparse CSR matrix in PyTorch.

``forward`` and ``adjoint`` use the stored A and A^T, so A^T is the EXACT transpose of A. The gradient of ``forward``
is defined explicitly through A^T (and vice versa), so the operator works inside autograd without PyTorch support for
sparse-tensor derivatives.

Operations are NORMALISED by default: forward(x) = (A / ||A||) x, adjoint(y) = (A / ||A||)^T y, so that ||A_n|| = 1
and a gradient step alpha = 1 is safe. Measurements must be normalised accordingly with
``normalize_data(y) = y / ||A||``. ``raw_forward`` returns the unnormalised A x (units: mm).

Backends: ``torch`` (PyTorch CSR, for GPUs and x86 CPUs with MKL) or ``scipy`` (SciPy CSR split into row blocks and
multiplied in parallel threads; CPU only). On ARM CPUs (e.g. Apple Silicon) PyTorch's CSR product is ~40x slower, so
``backend="auto"`` picks ``scipy`` there.
"""
from __future__ import annotations

import platform
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
import torch

from ..imaging.geometry import FanBeamGeometry
from .opnorm import opnorm_sparse

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*")


def _to_torch_csr(M: sp.csr_matrix, device, dtype=torch.float32, index_dtype=torch.int32) -> torch.Tensor:
    M = M.tocsr()
    crow = torch.from_numpy(M.indptr.astype(np.int64 if index_dtype == torch.int64 else np.int32))
    col = torch.from_numpy(M.indices.astype(np.int64 if index_dtype == torch.int64 else np.int32))
    val = torch.from_numpy(M.data.astype(np.float32))
    t = torch.sparse_csr_tensor(crow, col, val, size=M.shape, dtype=dtype)
    return t.to(device)


def _spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """A (M x N, CSR) @ X (N x B, dense)."""
    return A @ X


class _SparseLinear(torch.autograd.Function):
    """y = x @ M^T for a sparse M; the gradient uses the stored MT (= M^T)."""

    @staticmethod
    def forward(ctx, x2d: torch.Tensor, M: torch.Tensor, MT: torch.Tensor) -> torch.Tensor:
        ctx.MT = MT
        return _spmm(M, x2d.t().contiguous()).t().contiguous()

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        gx = _spmm(ctx.MT, g.t().contiguous()).t().contiguous()
        return gx, None, None


class _RowBlocksCSR:
    """SciPy CSR matrix split into row blocks; each block is multiplied in its own thread (SciPy releases the GIL)."""

    def __init__(self, M: sp.csr_matrix, threads: int):
        k = max(1, min(int(threads), M.shape[0]))
        b = np.linspace(0, M.shape[0], k + 1).astype(np.int64)
        self.blocks = [M[b[i]:b[i + 1]] for i in range(k)]
        self.offsets = b[:-1].tolist()
        self.shape = M.shape
        self.pool = ThreadPoolExecutor(k) if k > 1 else None

    def matmul_rows(self, x2d: torch.Tensor) -> torch.Tensor:
        """x2d (B, N) -> (B, M) = x2d @ M^T."""
        X = x2d.detach().to("cpu", torch.float32).contiguous().numpy()
        out = np.empty((X.shape[0], self.shape[0]), np.float32)

        def run(i_blk):
            i, blk = i_blk
            r0 = self.offsets[i]
            for b in range(X.shape[0]):
                out[b, r0:r0 + blk.shape[0]] = blk @ X[b]

        items = list(enumerate(self.blocks))
        if self.pool is None:
            for it in items:
                run(it)
        else:
            list(self.pool.map(run, items))
        return torch.from_numpy(out)


class _ScipyLinear(torch.autograd.Function):
    """y = x @ M^T where M is a _RowBlocksCSR; the gradient uses the stored MT (= M^T)."""

    @staticmethod
    def forward(ctx, x2d: torch.Tensor, M: _RowBlocksCSR, MT: _RowBlocksCSR | None) -> torch.Tensor:
        ctx.MT = MT
        return M.matmul_rows(x2d)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return ctx.MT.matmul_rows(g), None, None


def default_backend(device) -> str:
    dev = torch.device(device)
    return "scipy" if dev.type == "cpu" and platform.machine().lower() in ("arm64", "aarch64") else "torch"


class SparseFanBeam:
    """The operator A_Omega (deliberately not an nn.Module, so it never ends up in a model state_dict)."""

    _A: torch.Tensor | _RowBlocksCSR                  # sparse torch tensor, or row blocks for the scipy backend
    _AT: torch.Tensor | _RowBlocksCSR | None          # None when built with with_adjoint=False

    def __init__(self, A: sp.csr_matrix, geom: FanBeamGeometry, img_size: int,
                 norm: float | None = None, device: str | torch.device = "cpu",
                 index_dtype: torch.dtype = torch.int32, with_adjoint: bool = True,
                 backend: str = "auto", threads: int | None = None):
        self.geom = geom
        self.img_size = int(img_size)
        self.n_views = geom.n_views
        self.n_det = geom.n_det
        if A.shape != (self.n_views * self.n_det, self.img_size ** 2):
            raise ValueError(f"A has shape {A.shape}, which does not match the geometry/image size")
        self.norm = float(norm) if norm and np.isfinite(norm) else opnorm_sparse(A)[0]
        self.scale = 1.0 / self.norm
        self._A_sp = A.astype(np.float32)
        self.index_dtype = index_dtype
        self.with_adjoint = bool(with_adjoint)
        if backend not in ("auto", "torch", "scipy"):
            raise ValueError("backend must be auto, torch or scipy")
        self.backend_req = backend
        self.threads = int(threads) if threads else max(1, torch.get_num_threads())
        self.device = torch.device("cpu")
        self._build(device)

    # ------------------------------------------------------------------ construction
    def _build(self, device):
        device = torch.device(device)
        A = self._A_sp
        AT = A.T.tocsr() if self.with_adjoint else None
        self.backend = default_backend(device) if self.backend_req == "auto" else self.backend_req
        if self.backend == "scipy":
            if device.type != "cpu":
                raise ValueError("the scipy backend runs on the CPU only")
            self._A = _RowBlocksCSR(A, self.threads)
            self._AT = _RowBlocksCSR(AT, self.threads) if AT is not None else None
            self.device = device
            return
        try:
            A_t = _to_torch_csr(A, device, index_dtype=self.index_dtype)
            AT_t = _to_torch_csr(AT, device, index_dtype=self.index_dtype) if AT is not None else None
            # quick check that the product works with this index dtype
            _ = A_t @ torch.zeros(A.shape[1], 1, device=device)
        except (RuntimeError, NotImplementedError):
            self.index_dtype = torch.int64
            A_t = _to_torch_csr(A, device, index_dtype=torch.int64)
            AT_t = _to_torch_csr(AT, device, index_dtype=torch.int64) if AT is not None else None
        self._A, self._AT = A_t, AT_t
        self.device = device

    def to(self, device) -> SparseFanBeam:
        if torch.device(device) != self.device:
            self._build(device)
        return self

    # ------------------------------------------------------------------ shapes
    @property
    def sino_shape(self) -> tuple[int, int]:
        return self.n_views, self.n_det

    @property
    def img_shape(self) -> tuple[int, int]:
        return self.img_size, self.img_size

    @property
    def nnz(self) -> int:
        return int(self._A_sp.nnz)

    # ------------------------------------------------------------------ operations
    def _apply(self, x: torch.Tensor, M, MT, out_shape) -> torch.Tensor:
        B = x.shape[0]
        x2d = x.reshape(B, -1)
        if x2d.dtype != torch.float32:
            x2d = x2d.float()
        fn = _ScipyLinear if self.backend == "scipy" else _SparseLinear
        y2d = fn.apply(x2d, M, MT)
        return y2d.reshape(B, 1, *out_shape)

    def raw_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised A x. x: (B,1,H,W) -> (B,1,V,D)."""
        if self._AT is None:  # forward projection only, no gradient needed
            with torch.no_grad():
                B = x.shape[0]
                A = self._A
                if isinstance(A, _RowBlocksCSR):          # scipy backend
                    return A.matmul_rows(x.reshape(B, -1)).reshape(B, 1, *self.sino_shape)
                return (A @ x.reshape(B, -1).float().t().contiguous()).t().reshape(B, 1, *self.sino_shape)
        return self._apply(x, self._A, self._AT, self.sino_shape)

    def raw_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Unnormalised A^T y. y: (B,1,V,D) -> (B,1,H,W)."""
        if self._AT is None:
            raise RuntimeError("Projector was created with with_adjoint=False")
        return self._apply(y, self._AT, self._A, self.img_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw_forward(x) * self.scale

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.raw_adjoint(y) * self.scale

    __call__ = forward

    def normalize_data(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.scale

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """A_n^T A_n x."""
        return self.adjoint(self.forward(x))

    def data_residual(self, x: torch.Tensor, y_raw: torch.Tensor) -> torch.Tensor:
        """||A x - y|| / ||y|| per image of the batch (unnormalised, no gradient)."""
        with torch.no_grad():
            r = self.raw_forward(x) - y_raw
            num = r.flatten(1).norm(dim=1)
            den = y_raw.flatten(1).norm(dim=1).clamp_min(1e-12)
        return num / den

    def __repr__(self) -> str:
        return (f"SparseFanBeam({self.geom.name}, img={self.img_size}, views={self.n_views}, "
                f"det={self.n_det}, nnz={self.nnz / 1e6:.1f}M, ||A||={self.norm:.3f}, device={self.device}, "
                f"backend={self.backend})")
