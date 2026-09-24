"""Dense spectral cubic minimization, including singular boundary solutions."""

import math
from dataclasses import dataclass
import torch


@dataclass
class CubicSubproblemResult:
    s: torch.Tensor
    multiplier: float
    model_value: float
    n_iterations: int
    converged: bool


class CubicSubproblemSolver:
    """Minimize g's + s'Hs/2 + M ||s||^3/6; certify scaled KKT residuals.

    ``tol`` is a relative numerical tolerance, floored at 64 machine eps.
    A converged result satisfies numerical global-optimality conditions;
    no exact-arithmetic guarantee is claimed for rounded input/output.
    """

    def __init__(self, tol: float = 1e-14, max_iter: int = 1000):
        if not math.isfinite(tol) or tol <= 0 or max_iter < 1:
            raise ValueError("Require positive finite tol and max_iter >= 1")
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, g: torch.Tensor, H: torch.Tensor, M: float) -> CubicSubproblemResult:
        M = float(M)
        if not math.isfinite(M) or M <= 0:
            raise ValueError("M must be positive and finite")
        if g.ndim != 1 or g.numel() == 0 or H.shape != (g.numel(), g.numel()):
            raise ValueError("Expected nonempty vector g and square H")
        if g.dtype not in (torch.float32, torch.float64) or H.dtype != g.dtype or H.device != g.device:
            raise ValueError("g and H must share a float32/float64 dtype and device")
        if not torch.isfinite(g).all() or not torch.isfinite(H).all():
            raise ValueError("g and H must be finite")
        H = H * .5 + H.T * .5
        theta, V = torch.linalg.eigh(H)
        a = V.T @ g
        hnorm = theta.abs().max().item()
        gnorm = torch.linalg.vector_norm(g).item()
        finfo = torch.finfo(g.dtype)
        eta = max(self.tol, 64 * finfo.eps)
        lam0 = max(0., -theta[0].item())
        # Compute the eigenvalue gaps BEFORE adding delta: lam0+delta may
        # round to lam0 even when delta itself is accurately representable.
        gaps = theta - theta[0] if lam0 > 0 else theta

        def result(z, lam, iterations):
            s = V @ z
            r = torch.linalg.vector_norm(s).item()
            stat = torch.linalg.vector_norm(H @ s + lam * s + g).item()
            cubic = abs(lam - .5 * M * r)
            psd = max(0., -(theta[0].item() + lam))
            value = (g @ s + .5 * s @ H @ s).item() + (M / 6.) * r**3
            finite = torch.isfinite(s).all().item() and all(
                math.isfinite(t) for t in (lam, r, stat, cubic, psd, value))
            ok = finite and lam >= 0 and (
                stat <= eta * max(gnorm + (hnorm + abs(lam)) * r, finfo.tiny)
                and psd <= eta * max(hnorm + abs(lam), finfo.tiny)
                and cubic <= eta * max(abs(lam), .5 * M * r, finfo.tiny))
            return CubicSubproblemResult(s, lam, value, iterations, bool(ok))

        # Do not erase a small but nonzero gradient in a PSD model.
        if lam0 == 0 and gnorm == 0:
            return result(torch.zeros_like(g), 0., 0)

        if lam0 > 0:
            minimal = gaps <= eta * max(hnorm, finfo.tiny)
            if torch.linalg.vector_norm(a[minimal]).item() <= eta * gnorm:
                z0 = torch.zeros_like(a)
                z0[~minimal] = -a[~minimal] / gaps[~minimal]
                r0 = torch.linalg.vector_norm(z0).item()
                target = 2 * (lam0 / M)
                if math.isfinite(target) and target > 0 and r0 <= target * (1 + eta):
                    # Scaled square root avoids squaring a large radius.
                    alpha = target * math.sqrt(max(0., 1 - (r0 / target)**2))
                    z0[0] = alpha
                    candidate = result(z0, lam0, 0)
                    if candidate.converged:
                        return candidate
                    # Approximate eigenspace classification was insufficient;
                    # fall through to the ordinary root using actual gaps.

        def secular(delta):
            z = -a / (gaps + delta)
            r = torch.linalg.vector_norm(z).item()
            return r - 2 * ((lam0 + delta) / M), z

        # phi(0+) is positive for a genuine ordinary case. There is no
        # fixed positive lower cutoff that could skip a near-boundary root.
        left = 0.
        right = max(lam0, hnorm, math.sqrt(gnorm) * math.sqrt(M), finfo.tiny)
        for _ in range(self.max_iter):
            phi, _ = secular(right)
            if math.isfinite(phi) and phi <= 0:
                break
            right *= 2
            if not math.isfinite(right):
                raise RuntimeError("Cubic secular bracket overflow")
        else:
            raise RuntimeError("Failed to bracket cubic secular root within budget")

        last = None
        for k in range(1, self.max_iter + 1):
            mid = left + (right - left) / 2
            if mid == left or mid == right:
                break
            phi, z = secular(mid)
            # Only a small relative norm mismatch warrants the matrix KKT
            # check; ordinary bisection iterations stay in the eigenbasis.
            target = 2 * ((lam0 + mid) / M)
            if math.isfinite(phi) and abs(phi) <= eta * max(target, finfo.tiny):
                last = result(z, lam0 + mid, k)
                if last.converged:
                    return last
            if phi > 0:
                left = mid
            else:
                right = mid
        # Budget exhaustion must not become an unchecked success.
        delta = left + (right - left) / 2
        _, z = secular(delta)
        last = result(z, lam0 + delta, k)
        return last
