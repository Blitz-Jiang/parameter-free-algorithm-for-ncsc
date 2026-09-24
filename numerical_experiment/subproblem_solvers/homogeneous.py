"""Dense HSDA homogeneous subproblem (HSDA paper, equation (2.1))."""
from dataclasses import dataclass
import math
import torch


@dataclass
class HomogeneousResult:
    u: torch.Tensor
    v: float
    eigenvalue: float
    residual: float
    converged: bool


class HomogeneousSubproblemSolver:
    def __init__(self, tol=1e-12):
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite")
        self.tol = float(tol)

    def solve(self, g, H, alpha):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        n = g.numel()
        if g.ndim != 1 or H.shape != (n, n):
            raise ValueError("Expected vector g and square H")
        G = torch.empty((n + 1, n + 1), dtype=g.dtype, device=g.device)
        G[:n, :n] = (H + H.T) / 2
        G[:n, n] = G[n, :n] = g
        G[n, n] = -alpha
        if not torch.isfinite(G).all():
            raise ValueError("Non-finite homogeneous matrix")
        values, vectors = torch.linalg.eigh(G)
        w = vectors[:, 0]
        # Deterministic orientation; u/v and the signed negative-curvature
        # direction are otherwise invariant under eigenvector sign changes.
        pivot = int(torch.argmax(torch.abs(w)).item())
        if w[pivot] < 0:
            w = -w
        residual = torch.linalg.vector_norm(G @ w - values[0] * w).item()
        scale = max(1., torch.linalg.matrix_norm(G).item())
        tol = max(self.tol, 32 * torch.finfo(g.dtype).eps)
        return HomogeneousResult(w[:-1].detach(), w[-1].item(),
                                 values[0].item(), residual,
                                 residual <= tol * scale)
