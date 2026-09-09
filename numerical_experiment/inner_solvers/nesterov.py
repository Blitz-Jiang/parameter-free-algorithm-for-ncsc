from .base import InnerSolver, InnerSolverResult
import torch
import math
from typing import Literal

class NesterovAGD(InnerSolver):
    def __init__(self, ell, mu):
        self.ell = ell
        self.mu = mu

        self.kappa = self.ell / self.mu

        sqrt_kappa = math.sqrt(self.kappa)
        self.beta = ((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0))

        self.step_size = 1.0 / self.ell

    def _step(self, problem, x, y, v):
        grad_v = problem.grad_y(x, v)

        y_next = (v + self.step_size * grad_v)
        v_next = (y_next + self.beta * (y_next - y))

        return y_next, v_next

    def run(
        self, 
        problem, 
        x: torch.Tensor,
        y0: torch.Tensor,
        *, 
        stop_rule: Literal["steps", "gradient_norm"], 
        target: float,
        max_steps: int = 100_000,
    ) -> InnerSolverResult:

        y = y0.clone().detach()
        v = y.clone().detach()

        if stop_rule == "steps":
            return self._run_steps(problem, x, y, v, target)
        elif stop_rule == "gradient_norm":
            return self._run_gradient_norm(problem, x, y, v, target, max_steps)
        else:
            raise ValueError(f"Unknown stop_rule: {stop_rule}")

    def _run_steps(self, problem, x, y, v, target):
        n_steps = 0
        n_grad_evals = 0

        if target < 0 or not isinstance(target, int):
            raise ValueError("For 'steps' stop_rule, target must be a non-negative integer.")
        
        K = int(target)
        for _ in range(K):
            y, v = self._step(problem, x, y, v)
            n_grad_evals += 1
            n_steps += 1

        grad_y = problem.grad_y(x, y)
        residual = torch.linalg.vector_norm(grad_y)
        n_grad_evals += 1

        ##检查一下是否真的收敛了
        if residual > 1e-6:
            print(f"Warning: NesterovAGD did not converge. Residual: {residual.item()}")
        
        return InnerSolverResult(
            y=y,
            n_steps=n_steps,
            n_grad_evals=n_grad_evals,
            grad_y=grad_y.detach(),
            converged=True,
        )

    def _run_gradient_norm(self, problem, x, y, v, target, max_steps):
        n_steps = 0
        n_grad_evals = 0

        tau = float(target)
        grad_y = problem.grad_y(x, y)
        residual = torch.linalg.vector_norm(grad_y)
        n_grad_evals += 1
        
        if residual <= tau:
            return InnerSolverResult(
                y=y,
                n_steps=n_steps,
                n_grad_evals=n_grad_evals,
                grad_y=grad_y.detach(),
                converged=True,
            )
        
        while n_steps < max_steps:
            y, v = self._step(problem, x, y, v)
            n_grad_evals += 1
            n_steps += 1
            tau = float(target)
            grad_y = problem.grad_y(x, y)
            residual = torch.linalg.vector_norm(grad_y)
            n_grad_evals += 1

            if residual <= tau:
                return InnerSolverResult(
                    y=y,
                    n_steps=n_steps,
                    n_grad_evals=n_grad_evals,
                    grad_y=grad_y.detach(),
                    converged=True,
                )

        return InnerSolverResult(
            y=y,
            n_steps=n_steps,
            n_grad_evals=n_grad_evals,
            grad_y=grad_y.detach(),
            converged=False,
        )
