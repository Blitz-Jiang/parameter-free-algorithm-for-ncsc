import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.utr2 import UTR2
from inner_solvers.nesterov import NesterovAGD
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.trs import TRSubproblemSolver


def parse_args():
    p = argparse.ArgumentParser(description="Run UTR2 on CosLogCosh synthetic instance")
    p.add_argument("--d", type=int, default=20, help="Dimension of x, y")
    p.add_argument("--mu-y", type=float, default=1.0, help="Quadratic strong-concavity parameter of CosLogCosh")
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--max-iterations", type=int, default=2000)
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
    y0 = torch.zeros(args.d, device=device, dtype=dtype)

    inner_solver = NesterovAGD(ell=problem.ell, mu=problem.mu)
    trust_region_solver = TRSubproblemSolver()

    algo = UTR2(
        problem=problem,
        inner_solver=inner_solver,
        trust_region_solver=trust_region_solver,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
    )

    print("Running UTR2:")
    print(f"  d={args.d}, mu_y={args.mu_y}, omega={args.omega}, beta={args.beta}")
    print(f"  ell={problem.ell}, mu={problem.mu}, rho={problem.rho}, kappa={problem.kappa}")
    print(f"  epsilon={args.epsilon}, max_iter={args.max_iterations}")
    print(f"  MP={algo.MP:.6e}, sigma={algo.sigma:.6e}, tau_out={algo.tau_out:.6e}")

    result = algo.run(x0, y0)

    print("\nResult:")
    print(f"  converged: {result.converged}")
    print(f"  n_iterations: {result.n_iterations}")
    print(f"  n_inner_grad_evals: {result.n_inner_grad_evals}")
    print(f"  n_subproblem_solves: {result.n_subproblem_solves}")
    print(f"  x[:5]: {result.x[: min(5, result.x.numel())]}")
    print(f"  y[:5]: {result.y[: min(5, result.y.numel())]}")

    if result.history:
        print("\nFinal oracle metrics:")
        for key in ("grad_norm", "lambda_min_H", "inner_residual", "xi"):
            if result.history[key]:
                print(f"  {key}[-1]: {result.history[key][-1]:.6e}")

        if result.history["step_norm"]:
            print("\nLast trust-region step metrics:")
            for key in ("step_norm", "radius", "h", "tr_multiplier", "model_value"):
                print(f"  {key}[-1]: {result.history[key][-1]:.6e}")


if __name__ == "__main__":
    main()
