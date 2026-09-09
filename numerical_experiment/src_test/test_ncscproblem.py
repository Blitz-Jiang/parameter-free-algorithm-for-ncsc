import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from problems.NCSC import NCSCProblem
from problems.CosLogCosh import CosLogCosh
import torch
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    torch.manual_seed(1)
    ell = 10.0
    mu = 1.0
    rho = 0.1
    omega = 1.0
    d = 10
    Q = torch.randn(d, d)
    beta = 0.5
    problem = CosLogCosh(ell, mu, rho, omega, Q, beta)

    x = torch.randn(d)
    y = torch.randn(d)

    gy_auto = problem.grad_y(x, y)
    gy_true = (
        0.5
        * torch.sqrt(
            torch.tensor(
                problem.mu,
                dtype=x.dtype,
                device=x.device,
            )
        )
        * (problem.Q.T @ x)
        - problem.mu * y
        - problem.beta * torch.tanh(y)
    )

    
    print("Gradient w.r.t y (auto):", gy_auto)
    print("Gradient w.r.t y (true):", gy_true)
    print("Difference:", torch.norm(gy_auto - gy_true))
