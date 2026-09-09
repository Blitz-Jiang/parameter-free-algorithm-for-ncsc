# subproblem_solvers/cubic.py for cubic newton subproblem

import torch
from dataclasses import dataclass

@dataclass
class CubicSubproblemResult:
    s: torch.Tensor
    multiplier: float
    model_value: float
    n_iterations: int
    converged: bool


class CubicSubproblemSolver:
    def __init__(self, tol: float = 1e-14, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter
    
    def solve(
            self,
            g: torch.Tensor,
            H: torch.Tensor,
            M: float,
    ) -> CubicSubproblemResult:
        eigvals, eigvecs = torch.linalg.eigh(H)
        lambda_min = eigvals[0].item()
        u_min = eigvecs[:, 0]

        g_hat = eigvecs.T @ g

        lam_low = max(0.0, -lambda_min)

        g_norm = torch.linalg.vector_norm(g).item()

        if g_norm <= self.tol and lambda_min >= 0:
            return CubicSubproblemResult(
                s=torch.zeros_like(g),
                multiplier=0.0,
                model_value=0.0,
                n_iterations=0,
                converged=True
            )

        if g_norm <= self.tol and lambda_min < 0:
            lam = - lambda_min

            radius = 2.0 * lam / M
            s = radius * u_min

            model_value = (
                g @ s + 0.5 * s @ H @ s
                + (M / 6.0) * torch.linalg.vector_norm(s) ** 3
            ).item()

            return CubicSubproblemResult(
                s=s,
                multiplier=lam,
                model_value=model_value,
                n_iterations=0,
                converged=True
            )


        def secular_eq(lam: float) -> float:
            demon = eigvals + lam

            s_hat = - g_hat / demon

            s_norm = torch.linalg.vector_norm(s_hat).item()

            return s_norm - (2.0 * lam / M)

        tiny = max(self.tol, 1e-14)
        lam_left = lam_low + tiny
        
        phi_left = secular_eq(lam_left)

        lam_right = max(1.0, 2.0 * lam_left)

        backet_iter = 0
        while secular_eq(lam_right) > 0:
            lam_right *= 2.0
            backet_iter += 1
        if backet_iter > 100:
            raise RuntimeError("Failed to bracket the root of secular equation.")

        converged = False
        n_iters = 0

        for k in range(self.max_iter):
            lam_mid = 0.5 * (lam_left + lam_right)

            phi_mid = secular_eq(lam_mid)

            n_iters += 1

            if abs(phi_mid) <= self.tol:
                converged = True
                break

            if phi_mid > 0:
                lam_left = lam_mid
            else:
                lam_right = lam_mid
            if n_iters >= self.max_iter:
                import warnings
                warnings.warn("Cubic subproblem solver did not converge within the maximum number of iterations.")
                break

        lam = lam_mid

        demon = eigvals + lam
        s_hat = - g_hat / demon
        s = eigvecs @ s_hat

        s_norm = torch.linalg.vector_norm(s).item()

        model_value = (
            g @ s + 0.5 * s @ H @ s
            + (M / 6.0) * s_norm ** 3
        ).item()

        return CubicSubproblemResult(
            s=s,
            multiplier=lam,
            model_value=model_value,
            n_iterations=n_iters,
            converged=converged
        )
    