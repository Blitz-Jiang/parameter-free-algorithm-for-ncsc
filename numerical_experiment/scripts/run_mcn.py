import argparse
import math
import sys
from pathlib import Path

import torch

# Allow running this file directly: python scripts/run_mcn.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.mcn import MCN
from inner_solvers.nesterov import NesterovAGD
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.cubic import CubicSubproblemSolver


def parse_args():
    p = argparse.ArgumentParser(description="Run MCN on CosLogCosh synthetic instance")
    p.add_argument("--d", type=int, default=20, help="Dimension of x, y")
    p.add_argument("--mu-y", type=float, default=1.0, help="Quadratic strong-concavity parameter of CosLogCosh")
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    return p.parse_args()


def make_tensor(*shape, device, dtype):
    return torch.randn(*shape, device=device, dtype=dtype)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    problem = CosLogCosh(
        mu_y=args.mu_y,
        omega=args.omega,
        Q=make_tensor(args.d, args.d, device=device, dtype=dtype),
        beta=args.beta,
    )

    x0 = make_tensor(args.d, device=device, dtype=dtype)
    # The K0 formula assumes y_{-1} = 0.
    y0 = torch.zeros(args.d, device=device, dtype=dtype)

    inner_solver = NesterovAGD(ell=problem.ell, mu=problem.mu)
    cubic_solver = CubicSubproblemSolver()

    algo = MCN(
        problem=problem,
        inner_solver=inner_solver,
        cubic_solver=cubic_solver,
        epsilon=args.epsilon,
        K0=0,  # Set below using the algorithm's inner_distance_tol.
        max_iterations=args.max_iterations,
    )

    print("Estimating y*(x0) with gradient tolerance 1e-12 (excluded from complexity):")
    initial_solution = inner_solver.run(
        problem,
        x0,
        y0,
        stop_rule="gradient_norm",
        target=1e-12,
    )
    if not initial_solution.converged or not (
        math.isfinite(initial_solution.residual)
        and initial_solution.residual <= 1e-12
    ):
        raise RuntimeError(
            "Initial y*(x0) solve did not reach gradient tolerance 1e-12: "
            f"residual={initial_solution.residual:.6e}. "
            "Use --dtype float64 if running in float32."
        )

    y_star_norm = torch.linalg.vector_norm(initial_solution.y).item()
    if not math.isfinite(y_star_norm):
        raise RuntimeError("Initial y*(x0) solve returned a non-finite norm.")
    if y_star_norm == 0.0:
        algo.K0 = 0
    else:
        log_ratio = (
            0.5 * math.log1p(algo.kappa)
            + math.log(y_star_norm)
            - math.log(algo.inner_distance_tol)
        )
        algo.K0 = max(0, math.ceil(2.0 * math.sqrt(algo.kappa) * log_ratio))

    print(f"  residual={initial_solution.residual:.6e}, ||y*(x0)||={y_star_norm:.6e}")
    print(f"  tilde_epsilon={algo.inner_distance_tol:.6e}, K0={algo.K0}")
    print("Running MCN:")
    print(f"  d={args.d}, mu_y={args.mu_y}, omega={args.omega}, beta={args.beta}")
    print(f"  ell={problem.ell}, mu={problem.mu}, rho={problem.rho}, kappa={problem.kappa}")
    print(f"  K0={algo.K0}, epsilon={args.epsilon}, max_iter={args.max_iterations}")

    result = algo.run(x0, y0)

    if result is None:
        print("MCN returned None (most likely reached max_iterations without return).")
        return

    print("\nResult:")
    print(f"  converged: {result.converged}")
    print(f"  n_iterations: {result.n_iterations}")
    print(f"  n_inner_grad_evals: {result.n_inner_grad_evals}")
    print(f"  n_subproblem_solves: {result.n_subproblem_solves}")
    print(f"  x[:5]: {result.x[: min(5, result.x.numel())]}")
    print(f"  y[:5]: {result.y[: min(5, result.y.numel())]}")

    if result.history:
        print("\nFinal metrics:")
        print(f"  grad_norm[-1]: {result.history['grad_norm'][-1]:.6e}")
        print(f"  step_norm[-1]: {result.history['step_norm'][-1]:.6e}")
        print(f"  lambda_min_H[-1]: {result.history['lambda_min_H'][-1]:.6e}")
        print(f"  inner_residual[-1]: {result.history['inner_residual'][-1]:.6e}")


if __name__ == "__main__":
    main()
