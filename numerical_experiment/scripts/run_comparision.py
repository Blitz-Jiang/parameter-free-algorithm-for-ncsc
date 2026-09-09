import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.mcn import MCN
from algorithms.utr2 import UTR2
from inner_solvers.nesterov import NesterovAGD
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.cubic import CubicSubproblemSolver
from subproblem_solvers.trs import TRSubproblemSolver


def parse_args():
    p = argparse.ArgumentParser(description="Compare MCN and UTR2 on the same CosLogCosh instance")
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--mu-y", type=float, default=0.1)
    p.add_argument("--omega", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--max-iterations", type=int, default=1_000_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subproblem-tol", type=float, default=1e-14)
    p.add_argument("--subproblem-max-iter", type=int, default=1000)
    p.add_argument("--order", choices=["mcn-first", "utr2-first"], default="mcn-first")
    return p.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def solve_y(problem, x, y0):
    solver = NesterovAGD(ell=problem.ell, mu=problem.mu)
    result = solver.run(problem, x, y0, stop_rule="gradient_norm", target=1e-12)
    if not result.converged or not (
        math.isfinite(result.residual) and result.residual <= 1e-12
    ):
        raise RuntimeError(f"Reference inner solve failed: residual={result.residual:.6e}")
    return result


def prepare_K0(algo, problem, x0, y0):
    solution = solve_y(problem, x0, y0)
    y_norm = torch.linalg.vector_norm(solution.y).item()
    if not math.isfinite(y_norm):
        raise RuntimeError("Reference inner solution has a non-finite norm.")
    if y_norm == 0.0:
        algo.K0 = 0
    else:
        log_ratio = (
            0.5 * math.log1p(algo.kappa)
            + math.log(y_norm)
            - math.log(algo.inner_distance_tol)
        )
        algo.K0 = max(0, math.ceil(2.0 * math.sqrt(algo.kappa) * log_ratio))
    return solution


def evaluate(problem, x, y0, epsilon, M):
    solution = solve_y(problem, x, y0)
    y = solution.y
    g = problem.grad_x(x, y)
    H_xx, H_xy, H_yx, H_yy = problem.hessian_blocks(x, y)
    H = H_xx - H_xy @ torch.linalg.solve(H_yy, H_yx)
    H = 0.5 * (H + H.T)
    grad_norm = torch.linalg.vector_norm(g).item()
    lambda_min = torch.linalg.eigvalsh(H)[0].item()
    return {
        "reference_value": problem.f(x, y).item(),
        "reference_grad_norm": grad_norm,
        "reference_lambda_min": lambda_min,
        "reference_inner_residual": solution.residual,
        "common_test": grad_norm <= epsilon and lambda_min >= -math.sqrt(M * epsilon),
    }


def run_algorithm(name, algo, problem, x0, y0, device, epsilon, M):
    print(f"\nRunning {name}...", flush=True)
    x_start = x0.clone().detach()
    y_start = y0.clone().detach()
    synchronize(device)
    start = time.perf_counter()
    try:
        result = algo.run(x_start, y_start)
    except Exception as exc:
        synchronize(device)
        return {"seconds": time.perf_counter() - start, "error": str(exc)}
    synchronize(device)
    seconds = time.perf_counter() - start

    if result is None:
        return {
            "seconds": seconds,
            "error": "Algorithm returned None; no final point or complexity available.",
        }

    record = {
        "seconds": seconds,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "n_inner_grad_evals": result.n_inner_grad_evals,
        "n_subproblem_solves": result.n_subproblem_solves,
        "n_oracle_builds": len(result.history["grad_norm"]),
    }
    try:
        record.update(evaluate(problem, result.x, y0, epsilon, M))
    except Exception as exc:
        record["evaluation_error"] = str(exc)
    return record


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64

    Q = torch.randn(args.d, args.d, device=device, dtype=dtype)
    x0 = torch.randn(args.d, device=device, dtype=dtype)
    y0 = torch.zeros(args.d, device=device, dtype=dtype)
    problem = CosLogCosh(mu_y=args.mu_y, omega=args.omega, Q=Q, beta=args.beta)

    mcn = MCN(
        problem=problem,
        inner_solver=NesterovAGD(ell=problem.ell, mu=problem.mu),
        cubic_solver=CubicSubproblemSolver(
            tol=args.subproblem_tol, max_iter=args.subproblem_max_iter,
        ),
        epsilon=args.epsilon,
        K0=0,
        max_iterations=args.max_iterations,
    )
    utr2 = UTR2(
        problem=problem,
        inner_solver=NesterovAGD(ell=problem.ell, mu=problem.mu),
        trust_region_solver=TRSubproblemSolver(
            tol=args.subproblem_tol, max_iter=args.subproblem_max_iter,
        ),
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
    )

    print("Shared instance and settings:")
    print(f"  d={args.d}, seed={args.seed}, device={device}, dtype={dtype}")
    print(f"  mu_y={args.mu_y}, omega={args.omega}, beta={args.beta}")
    print(f"  ell={problem.ell}, mu={problem.mu}, rho={problem.rho}, kappa={problem.kappa}")
    print(f"  epsilon={args.epsilon}, max_iter={args.max_iterations}, y0=0")
    print(f"  subproblem_tol={args.subproblem_tol}, subproblem_max_iter={args.subproblem_max_iter}")

    print("\nPreparing MCN K0 (excluded from main-run time and complexity)...", flush=True)
    synchronize(device)
    start = time.perf_counter()
    initial_solution = prepare_K0(mcn, problem, x0, y0)
    synchronize(device)
    preparation_seconds = time.perf_counter() - start
    print(f"  K0={mcn.K0}, tilde_epsilon={mcn.inner_distance_tol:.6e}")
    print(f"  seconds={preparation_seconds:.6f}, grad_evals={initial_solution.n_grad_evals}")
    print(f"  UTR2 tau_out={utr2.tau_out:.6e}")

    algorithms = [("MCN", mcn), ("UTR2", utr2)]
    if args.order == "utr2-first":
        algorithms.reverse()
    records = {}
    for name, algo in algorithms:
        records[name] = run_algorithm(
            name, algo, problem, x0, y0, device, args.epsilon, mcn.M,
        )

    print("\nComparison:")
    print(f"{'metric':<30} {'MCN':>20} {'UTR2':>20}")
    for key in (
        "seconds", "converged", "n_iterations", "n_inner_grad_evals",
        "n_subproblem_solves", "n_oracle_builds", "reference_value",
        "reference_grad_norm", "reference_lambda_min", "reference_inner_residual",
        "common_test",
    ):
        values = []
        for name in ("MCN", "UTR2"):
            value = records[name].get(key, "N/A")
            values.append(f"{value:.6e}" if isinstance(value, float) else str(value))
        print(f"{key:<30} {values[0]:>20} {values[1]:>20}")

    for name, record in records.items():
        for key in ("error", "evaluation_error"):
            if key in record:
                print(f"\n{name} {key}: {record[key]}")

    print("\nconverged uses each algorithm's own stopping rule.")
    print("common_test uses reference ||grad Phi|| <= epsilon and lambda_min >= -sqrt(M * epsilon).")
    print("Reference metrics use an approximate y*(x) with gradient tolerance 1e-12.")
    print("Reference evaluation is excluded from main-run time and complexity.")
    print("Timing is a single run; use --order utr2-first to check order effects.")


if __name__ == "__main__":
    main()
