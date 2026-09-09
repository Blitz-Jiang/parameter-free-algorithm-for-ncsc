import torch
import warnings
from dataclasses import dataclass

@dataclass
class TRSubproblemResult:
    s: torch.Tensor
    multiplier: float
    model_value: float
    n_iterations: int
    converged: bool


class TRSubproblemSolver:
    def __init__(self, tol: float = 1e-14, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(
            self,
            g: torch.Tensor,
            H: torch.Tensor,
            radius: float,
    ) -> TRSubproblemResult:
        if radius <= 0:
            raise ValueError("radius must be positive.")

        H = 0.5 * (H + H.T)

        eigvals, eigvecs = torch.linalg.eigh(H)
        lambda_min = eigvals[0].item()

        g_hat = eigvecs.T @ g

        eig_tol = max(self.tol, 1e-14)

        if lambda_min >= -eig_tol:
            positive = torch.abs(eigvals) > eig_tol

            null_component_norm = (
                torch.linalg.vector_norm(g_hat[~positive]).item()
                if torch.any(~positive) else 0.0
            )

            if null_component_norm <= eig_tol:
                s_hat = torch.zeros_like(g_hat)
                s_hat[positive] = -g_hat[positive] / eigvals[positive]
                s = eigvecs @ s_hat

                s_norm = torch.linalg.vector_norm(s).item()

                if s_norm <= radius + self.tol:
                    model_value = (g @ s + 0.5 * s @ H @ s).item()

                    return TRSubproblemResult(
                        s=s.detach(),
                        multiplier=0.0,
                        model_value=model_value,
                        n_iterations=0,
                        converged=True
                    )

        lam_low = max(0.0, -lambda_min)

        if lam_low > 0.0:
            shifted = eigvals + lam_low

            nonzero = torch.abs(shifted) > eig_tol
            zero = ~nonzero

            g_null_norm = (
                torch.linalg.vector_norm(g_hat[zero]).item()
                if torch.any(zero) else 0.0
            )

            if g_null_norm <= eig_tol:
                s_hat_bar = torch.zeros_like(g_hat)
                s_hat_bar[nonzero] = -g_hat[nonzero] / shifted[nonzero]
                s_bar = eigvecs @ s_hat_bar

                s_bar_norm = torch.linalg.vector_norm(s_bar).item()

                if s_bar_norm <= radius + self.tol:
                    remaining_sq = max(0.0, radius**2 - s_bar_norm**2)
                    alpha = remaining_sq**0.5
                    u_min = eigvecs[:, 0]
                    s = s_bar + alpha * u_min

                    model_value = (g @ s + 0.5 * s @ H @ s).item()

                    return TRSubproblemResult(
                        s=s.detach(),
                        multiplier=lam_low,
                        model_value=model_value,
                        n_iterations=0,
                        converged=True
                    )

        def secular_eq(lam: float) -> float:
            denom = eigvals + lam

            s_hat = -g_hat / denom

            s_norm = torch.linalg.vector_norm(s_hat).item()

            return s_norm - radius

        tiny = max(self.tol, 1e-14)
        lam_left = lam_low + tiny

        phi_left = secular_eq(lam_left)

        if phi_left <= 0:
            raise RuntimeError("Unexpected trust-region secular equation state near lambda lower bound.")

        lam_right = max(1.0, 2.0 * lam_left)

        bracket_iter = 0
        while secular_eq(lam_right) > 0:
            lam_right *= 2.0
            bracket_iter += 1

            if bracket_iter > 100:
                raise RuntimeError("Failed to bracket the root of trust-region secular equation.")

        converged = False
        n_iters = 0

        for _ in range(self.max_iter):
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

        if not converged:
            warnings.warn("Trust-region subproblem solver did not converge within max_iter.")

        lam = lam_mid

        denom = eigvals + lam
        s_hat = -g_hat / denom
        s = eigvecs @ s_hat

        model_value = (g @ s + 0.5 * s @ H @ s).item()

        return TRSubproblemResult(
            s=s.detach(),
            multiplier=lam,
            model_value=model_value,
            n_iterations=n_iters,
            converged=converged
        )
