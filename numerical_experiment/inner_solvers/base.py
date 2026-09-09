import torch
from dataclasses import dataclass
from typing import Literal
from abc import ABC, abstractmethod

@dataclass
class InnerSolverResult:
    y:torch.Tensor
    n_steps:int
    n_grad_evals:int
    grad_y: torch.Tensor
    converged:bool

    @property
    def residual(self) -> float:
        return torch.linalg.vector_norm(self.grad_y).item()


class InnerSolver(ABC):
    @abstractmethod
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
        pass
