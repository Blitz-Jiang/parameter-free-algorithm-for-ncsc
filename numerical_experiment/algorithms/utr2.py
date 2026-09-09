import torch
import math

from .base import NCSCAlgorithm, AlgorithmResult

class UTR2(NCSCAlgorithm):
    def __init__(
            self,
            problem,
            inner_solver,
            trust_region_solver,
            epsilon: float,
            max_iterations: int = 10_000,
    ):
        super().__init__(problem)

        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self.inner_solver = inner_solver
        self.trust_region_solver = trust_region_solver

        self.epsilon = float(epsilon)
        self.max_iterations = int(max_iterations)

        self.ell = float(problem.ell)
        self.mu = float(problem.mu)
        self.rho = float(problem.rho)

        self.kappa = self.ell / self.mu

        self.MP = (
            4.0 * math.sqrt(2.0) * self.kappa ** 3 * self.rho
        )

        self.sigma = math.sqrt(self.MP / 6.0)

        self.tau_out = (
            self.mu * math.sqrt(self.MP * self.epsilon / 6.0)
            / (64.0 * (1.0 + self.kappa) ** 2 * self.rho)
        )

    def _build_corrected_oracle(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            grad_y: torch.Tensor,
    ):
        gx = self.problem.grad_x(x, y)
        H_xx, H_xy, H_yx, H_yy = self.problem.hessian_blocks(x, y)

        v = torch.linalg.solve(H_yy, grad_y)
        g = gx - H_xy @ v

        W = torch.linalg.solve(H_yy, H_yx)
        H = H_xx - H_xy @ W

        H = 0.5 * (H + H.T)

        inner_residual = torch.linalg.vector_norm(grad_y).item()

        return g, H, inner_residual

    def _hessian_error_certification(
            self,
            inner_residual: float,
    ) -> float:
        return (
            (1.0 + self.kappa) ** 2 * self.rho * inner_residual / self.mu
        )

    def _check_stopping(
            self,
            g: torch.Tensor,
            H: torch.Tensor,
            xi: float,
    ) -> bool:
        grad_norm = torch.linalg.vector_norm(g).item()
        lambda_min_H = torch.linalg.eigvalsh(H)[0].item()

        grad_ok = grad_norm <= self.epsilon
        curvature_ok = lambda_min_H >= -math.sqrt(self.MP * self.epsilon) + xi

        return grad_ok and curvature_ok

    def _build_tr_subproblem(
            self,
            g: torch.Tensor,
            H: torch.Tensor,
            xi: float,
    ):
        grad_norm = torch.linalg.vector_norm(g).item()
        h = max(grad_norm, self.epsilon)

        I = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        B = H + xi * I + 2.0 * self.sigma * math.sqrt(h) * I
        B = 0.5 * (B + B.T)

        radius = math.sqrt(h) / (4.0 * self.sigma)

        return B, radius, h

    def run(
            self,
            x0: torch.Tensor,
            y_minus1: torch.Tensor,
    ) -> AlgorithmResult:
        x = x0.clone().detach()
        y_prev = y_minus1.clone().detach()

        total_inner_grad_evals = 0
        n_subproblem_solves = 0

        history = {
            "inner_steps": [],
            "inner_grad_evals": [],
            "inner_residual": [],
            "xi": [],
            "grad_norm": [],
            "lambda_min_H": [],
            "h": [],
            "radius": [],
            "step_norm": [],
            "tr_multiplier": [],
            "model_value": [],
        }

        for k in range(self.max_iterations + 1):
            inner_result = self.inner_solver.run(
                self.problem,
                x,
                y_prev,
                stop_rule="gradient_norm",
                target=self.tau_out,
            )

            if not inner_result.converged:
                raise RuntimeError(
                    f"Inner solver failed at UTR2 iteration {k}. "
                    f"Residual = {inner_result.residual:.3e}, "
                    f"target = {self.tau_out:.3e}."
                )

            y = inner_result.y
            total_inner_grad_evals += inner_result.n_grad_evals

            g, H, inner_residual = self._build_corrected_oracle(
                x, y, inner_result.grad_y,
            )
            xi = self._hessian_error_certification(inner_residual)

            grad_norm = torch.linalg.vector_norm(g).item()
            lambda_min_H = torch.linalg.eigvalsh(H)[0].item()

            history["inner_steps"].append(inner_result.n_steps)
            history["inner_grad_evals"].append(inner_result.n_grad_evals)
            history["inner_residual"].append(inner_residual)
            history["xi"].append(xi)
            history["grad_norm"].append(grad_norm)
            history["lambda_min_H"].append(lambda_min_H)

            if self._check_stopping(g, H, xi):
                return AlgorithmResult(
                    x=x.detach(),
                    y=y.detach(),
                    n_iterations=n_subproblem_solves,
                    n_inner_grad_evals=total_inner_grad_evals,
                    n_subproblem_solves=n_subproblem_solves,
                    converged=True,
                    history=history,
                )

            if n_subproblem_solves >= self.max_iterations:
                break

            B, radius, h = self._build_tr_subproblem(g, H, xi)

            tr_result = self.trust_region_solver.solve(
                g=g,
                H=B,
                radius=radius,
            )

            if not tr_result.converged:
                raise RuntimeError(
                    f"Trust-region subproblem did not converge at UTR2 iteration {k}."
                )

            d = tr_result.s
            n_subproblem_solves += 1

            step_norm = torch.linalg.vector_norm(d).item()

            history["h"].append(h)
            history["radius"].append(radius)
            history["step_norm"].append(step_norm)
            history["tr_multiplier"].append(tr_result.multiplier)
            history["model_value"].append(tr_result.model_value)

            x = (x + d).detach()
            y_prev = y.detach()

        return AlgorithmResult(
            x=x.detach(),
            y=y.detach(),
            n_iterations=n_subproblem_solves,
            n_inner_grad_evals=total_inner_grad_evals,
            n_subproblem_solves=n_subproblem_solves,
            converged=False,
            history=history,
        )
