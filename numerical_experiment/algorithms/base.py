import torch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class AlgorithmResult:
    x: torch.Tensor
    y: torch.Tensor

    n_iterations: int
    n_inner_grad_evals: int
    n_subproblem_solves: int

    converged: bool

    history: dict = field(default_factory=dict)

class NCSCAlgorithm(ABC):

    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def run(
        self, 
        x0: torch.Tensor, 
        y0: torch.Tensor,
    ) -> AlgorithmResult:
        pass