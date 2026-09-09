import argparse
import math
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

from inner_solvers.scar import SCAR
from problems.CosLogCosh import CosLogCosh


class CountedCosLogCosh(CosLogCosh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_calls = 0
        self.function_calls = 0

    def grad_y(self, x, y):
        self.grad_calls += 1
        return super().grad_y(x, y)

    def f(self, x, y):
        self.function_calls += 1
        return super().f(x, y)


def analytic_grad_y(problem, x, y):
    return (
        problem.a_mu * (problem.Q.T @ x)
        - problem.mu_y * y
        - problem.beta * torch.tanh(y)
    )


def timeout_handler(signum, frame):
    raise TimeoutError("SCAR exceeded the test timeout.")


def check_solve(problem, x, y0, target, max_steps, timeout):
    problem.grad_calls = 0
    problem.function_calls = 0
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    start = time.perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout)
        result = SCAR().run(
            problem, x, y0,
            stop_rule="gradient_norm",
            target=target,
            max_steps=max_steps,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        print(f"  elapsed: {time.perf_counter() - start:.6f}s", flush=True)
        print(f"  actual grad_y calls: {problem.grad_calls}", flush=True)
        print(f"  actual f calls (including autograd): {problem.function_calls}", flush=True)

    true_grad = analytic_grad_y(problem, x, result.y)
    true_residual = torch.linalg.vector_norm(true_grad).item()
    difference = torch.linalg.vector_norm(result.grad_y - true_grad).item()

    print(f"  converged: {result.converged}")
    print(f"  SCAR stages: {result.n_steps}")
    print(f"  reported grad_y calls: {result.n_grad_evals}")
    print(f"  reported residual: {result.residual:.6e}")
    print(f"  analytic residual: {true_residual:.6e}")
    print(f"  cached/analytic gradient difference: {difference:.6e}")

    assert result.n_grad_evals == problem.grad_calls, "Gradient evaluation count mismatch"
    assert torch.isfinite(result.y).all(), "Non-finite returned point"
    assert torch.isfinite(result.grad_y).all(), "Non-finite cached gradient"
    torch.testing.assert_close(result.grad_y, true_grad, rtol=1e-10, atol=1e-12)
    assert result.converged, "SCAR did not converge"
    assert result.residual <= target, "Reported residual exceeds target"
    assert true_residual <= target + 1e-12, "Analytic residual exceeds target"
    return result


def main():
    p = argparse.ArgumentParser(description="Test SCAR on CosLogCosh")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--mu-y", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--target", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("timeout must be positive and finite")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    Q = torch.randn(args.d, args.d)
    problem = CountedCosLogCosh(
        mu_y=args.mu_y, omega=args.omega, Q=Q, beta=args.beta,
    )
    x = torch.randn(args.d)
    y0 = torch.randn(args.d)

    print(f"seed={args.seed}, d={args.d}, mu_y={args.mu_y}, omega={args.omega}, beta={args.beta}")
    print(f"target={args.target}, max_steps={args.max_steps}", flush=True)
    print("\nRandom initial point:", flush=True)
    result = check_solve(problem, x, y0, args.target, args.max_steps, args.timeout)

    print("\nAlready-converged initial point:", flush=True)
    warm_result = check_solve(
        problem, x, result.y, args.target, args.max_steps, args.timeout,
    )
    assert warm_result.n_steps == 0, "Already-converged input should need no SCAR stages"
    assert warm_result.n_grad_evals == 1, "Already-converged input should need one gradient"
    torch.testing.assert_close(warm_result.y, result.y, rtol=0, atol=0)
    print("\nPASS: convergence, cached gradient, evaluation count, and early return.")


if __name__ == "__main__":
    main()
