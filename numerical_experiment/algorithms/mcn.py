import torch
import math

from .base import NCSCAlgorithm, AlgorithmResult

class MCN(NCSCAlgorithm):

    def __init__(
            self, 
            problem, 
            inner_solver,
            cubic_solver,
            epsilon: float,
            K0: int,
            max_iterations: int = 10_000,
    ):
        super().__init__(problem)

        self.inner_solver = inner_solver
        self.cubic_solver = cubic_solver

        self.epsilon = float(epsilon)
        self.K0 = int(K0)
        self.max_iterations = int(max_iterations)

        self.ell = float(problem.ell)
        self.mu = float(problem.mu)
        self.rho = float(problem.rho)

        self.kappa = self.ell / self.mu

        self.M = (
            4.0 * math.sqrt(2.0) * self.kappa ** 3 * self.rho 
        )

        self.Cg = 1.0 / 192.0
        self.Ch = 1.0 / 48.0

        self.inner_distance_tol = min(
            self.Cg * self.epsilon / self.ell,
            self.Ch * math.sqrt(self.M * self.epsilon) / self.rho,
        )

        self.threshold = 1 / 2.0 * math.sqrt(self.epsilon / self.M)

    def _compute_Kt(
            self,
            k:int,
            s_prev: torch.Tensor,
    ) -> int:
        if k == 0:
            return self.K0

        s_norm = torch.linalg.vector_norm(s_prev).item()

        Kt = math.ceil(
            2.0 
            * math.sqrt(self.kappa) 
            * math.log(
                math.sqrt(self.kappa + 1.0) / self.inner_distance_tol
                * (self.inner_distance_tol + self.kappa * s_norm)
            )
        )

        return Kt

    def _build_oracle(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ):
        g = self.problem.grad_x(x, y)
        H_xx, H_xy, H_yx, H_yy = self.problem.hessian_blocks(x, y)

        W = torch.linalg.solve(H_yy, H_yx)
        H = H_xx - H_xy @ W

        H = 0.5 * (H + H.T)

        return g, H
    
    def run(
            self,
            x0: torch.Tensor,
            y_minus1: torch.Tensor,
    ) -> AlgorithmResult:
        x = x0.clone().detach()
        y_prev = y_minus1.clone().detach()
        s_prev = None

        total_inner_grad_evals = 0
        n_subproblem_solves = 0

        history = {
            "K_t": [],
            "inner_steps": [],
            "inner_grad_evals": [],
            "inner_residual": [],
            "grad_norm": [],
            "lambda_min_H": [],
            "step_norm": [],
            "model_value": [],
        }

        for k in range(self.max_iterations):
            Kt = self._compute_Kt(k, s_prev=s_prev)

            inner_result = self.inner_solver.run(
                self.problem,
                x,
                y_prev,
                stop_rule="steps",
                target=Kt,
            )
            y = inner_result.y
            total_inner_grad_evals += inner_result.n_grad_evals

            g, H = self._build_oracle(x, y)

            grad_norm = torch.linalg.vector_norm(g).item()
            lambda_min_H = torch.linalg.eigvalsh(H)[0].item()

            cubic_result = self.cubic_solver.solve(
                g=g,
                H=H,
                M=self.M,
            )

            if not cubic_result.converged:
                raise RuntimeError(f"Cubic subproblem solver did not converge at iteration {k}.")

            s = cubic_result.s

            n_subproblem_solves += 1

            step_norm = torch.linalg.vector_norm(s).item()

            history["K_t"].append(Kt)
            history["inner_steps"].append(inner_result.n_steps)
            history["inner_grad_evals"].append(inner_result.n_grad_evals)
            history["inner_residual"].append(inner_result.residual)
            history["grad_norm"].append(grad_norm)
            history["lambda_min_H"].append(lambda_min_H)
            history["step_norm"].append(step_norm)
            history["model_value"].append(cubic_result.model_value)

            if step_norm <= self.threshold:
                x_out = (x + s).detach()
                return AlgorithmResult(
                    x=x_out,
                    y=y.detach(),
                    n_iterations=k + 1,
                    n_inner_grad_evals=total_inner_grad_evals,
                    n_subproblem_solves=n_subproblem_solves,
                    converged=True,
                    history=history,
                )

            x = (x + s).detach()
            y_prev = y.detach()
            s_prev = s.detach()

            