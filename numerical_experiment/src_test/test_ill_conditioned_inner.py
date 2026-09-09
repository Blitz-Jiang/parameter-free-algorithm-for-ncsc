import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from inner_solvers.nesterov import NesterovAGD
from inner_solvers.scar import SCAR
from problems.NCSC import NCSCProblem


class IllConditionedNCSC(NCSCProblem):
    def __init__(self, eigenvalues, max_grad_calls):
        super().__init__()
        self.eigenvalues = eigenvalues
        self.max_grad_calls = max_grad_calls
        self.grad_calls = 0
        self.function_calls = 0

    def f(self, x, y):
        self.function_calls += 1
        return torch.cos(x).sum() + x @ y - 0.5 * (self.eigenvalues * y.square()).sum()

    def grad_y(self, x, y):
        if self.grad_calls >= self.max_grad_calls:
            raise RuntimeError("Shared gradient evaluation budget exhausted.")
        self.grad_calls += 1
        return super().grad_y(x, y)

    def _calculate_constants(self):
        return {"ell": max(1.0, self.eigenvalues.max().item()) + 1.0,
                "mu": self.eigenvalues.min().item(), "rho": 1.0}

    def value_remainder_y(self, x, y, z, grad_y, f_y, f_z):
        return -0.5 * (self.eigenvalues * (z - y).square()).sum()


def timeout_handler(signum, frame):
    raise TimeoutError("Per-solver time limit exceeded.")


def run_solver(name, solver, problem, x, y0, args):
    problem.grad_calls = 0
    problem.function_calls = 0
    result = None
    row = {"method": name}
    start = time.perf_counter()
    try:
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        result = solver.run(
            problem, x.clone(), y0.clone(), stop_rule="gradient_norm",
            target=args.target, max_steps=args.max_grad_calls,
        )
        row["seconds"] = time.perf_counter() - start
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - start)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    row.update(grad_calls=problem.grad_calls, function_calls=problem.function_calls)
    if result is not None:
        exact_gradient = x - problem.eigenvalues * result.y
        y_star = x / problem.eigenvalues
        error = result.y - y_star
        exact_residual = torch.linalg.vector_norm(exact_gradient).item()
        row.update(
            converged=result.converged,
            residual=result.residual,
            exact_residual=exact_residual,
            solution_error=torch.linalg.vector_norm(error).item(),
            relative_solution_error=(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(y_star)).item(),
            inner_objective_gap=(0.5 * (problem.eigenvalues * error.square()).sum()).item(),
            steps=result.n_steps,
            reported_grad_calls=result.n_grad_evals,
            count_matches=result.n_grad_evals == problem.grad_calls,
            cached_gradient_error=torch.linalg.vector_norm(result.grad_y - exact_gradient).item(),
            passed=bool(result.converged and exact_residual <= args.target
                        and result.n_grad_evals == problem.grad_calls),
        )
    else:
        row["passed"] = False
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="AGD vs SCAR on an ill-conditioned NCSC quadratic inner problem")
    p.add_argument("--conditions", type=float, nargs="+", default=[1e2, 1e4, 1e6])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target", type=float, default=1e-6)
    p.add_argument("--max-grad-calls", type=int, default=200_000)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--output", type=Path, default=Path("/tmp/ill_conditioned_inner.json"))
    args = p.parse_args()
    if any(not math.isfinite(c) or c < 1 for c in args.conditions):
        p.error("conditions must be finite and at least 1")
    if args.d < 2 or args.timeout <= 0 or not math.isfinite(args.timeout):
        p.error("d must be at least 2 and timeout must be positive and finite")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    x = torch.randn(args.d)
    x = x / torch.linalg.vector_norm(x)
    y0 = torch.zeros_like(x)
    signal.signal(signal.SIGALRM, timeout_handler)

    warm_problem = IllConditionedNCSC(torch.ones_like(x), args.max_grad_calls)
    warm_problem.grad_y(x, y0)

    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "formula": "f(x,y) = sum cos(x_i) + x^T y - 0.5 sum a_i y_i^2",
              "initial_gradient_norm": 1.0, "cases": []}
    for condition in args.conditions:
        a = torch.logspace(-math.log10(condition), 0.0, args.d)
        problem = IllConditionedNCSC(a, args.max_grad_calls)
        print(f"\nInner condition={condition:g}, mu={problem.mu:g}, joint ell={problem.ell:g}", flush=True)
        case = {"inner_condition": condition, "mu": problem.mu, "ell": problem.ell, "runs": []}
        methods = [
            ("AGD_joint_bound", NesterovAGD(problem.ell, problem.mu)),
            ("AGD_exact_inner_bound", NesterovAGD(a.max().item(), problem.mu)),
            ("SCAR", SCAR()),
        ]
        for name, solver in methods:
            case["runs"].append(run_solver(name, solver, problem, x, y0, args))
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"\nResults: {args.output}", flush=True)


if __name__ == "__main__":
    main()
