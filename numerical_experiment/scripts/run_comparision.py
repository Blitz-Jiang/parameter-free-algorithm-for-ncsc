import argparse
import json
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
from algorithms.utr3 import UTR3
from algorithms.utr4 import UTR4
from algorithms.utr5 import UTR5
from inner_solvers.nesterov import NesterovAGD
from inner_solvers.scar import SCAR
from inner_solvers.scar_early import SCAREarly
from inner_solvers.scar_persistent_early import SCARPersistentEarly
from problems.CosLogCosh import CosLogCosh
from subproblem_solvers.cubic import CubicSubproblemSolver
from subproblem_solvers.trs import TRSubproblemSolver


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


def parse_args(default_algorithms=None):
    names = ("mcn", "utr2", "utr3", "utr3-early", "utr4", "utr4-early", "utr5", "utr5-early")
    defaults = ("mcn", "utr2", "utr3-early", "utr4-early", "utr5", "utr5-early")
    p = argparse.ArgumentParser(description="Compare MCN and UTR methods on the same CosLogCosh instance")
    p.add_argument("--algorithms", nargs="+", choices=names, default=list(default_algorithms or defaults))
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--mu-y", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=1e-7)
    p.add_argument("--epsilon-h", type=float)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=100_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subproblem-tol", type=float, default=1e-14)
    p.add_argument("--subproblem-max-iter", type=int, default=1000)
    p.add_argument("--order", choices=["mcn-first", "utr2-first"], default="mcn-first")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    if args.epsilon_h is not None and (
        not math.isfinite(args.epsilon_h) or args.epsilon_h <= 0.0
    ):
        p.error("--epsilon-h must be positive and finite")
    return args


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


def evaluate(problem, x, y0, epsilon, M, epsilon_h=None):
    solution = solve_y(problem, x, y0)
    y = solution.y
    g = problem.grad_x(x, y)
    H_xx, H_xy, H_yx, H_yy = problem.hessian_blocks(x, y)
    H = H_xx - H_xy @ torch.linalg.solve(H_yy, H_yx)
    H = 0.5 * (H + H.T)
    grad_norm = torch.linalg.vector_norm(g).item()
    lambda_min = torch.linalg.eigvalsh(H)[0].item()
    epsilon_h = math.sqrt(epsilon) if epsilon_h is None else epsilon_h
    residual = solution.residual
    grad_upper = grad_norm + problem.kappa * residual
    lambda_lower = lambda_min - (1.0 + problem.kappa) ** 2 * problem.rho * residual / problem.mu
    return {
        "reference_value": problem.f(x, y).item(),
        "reference_grad_norm": grad_norm,
        "reference_lambda_min": lambda_min,
        "reference_inner_residual": residual,
        "reference_grad_upper": grad_upper,
        "reference_lambda_lower": lambda_lower,
        "common_test": grad_norm <= epsilon and lambda_min >= -math.sqrt(M * epsilon),
        "shared_sosp_with_inner_error": grad_upper <= epsilon and lambda_lower >= -epsilon_h,
    }


def run_algorithm(name, algo, problem, x0, y0, device, epsilon, M, epsilon_h=None):
    print(f"\nRunning {name}...", flush=True)
    x_start = x0.clone().detach()
    y_start = y0.clone().detach()
    grad_before = problem.grad_calls
    function_before = problem.function_calls
    synchronize(device)
    start = time.perf_counter()
    try:
        result = algo.run(x_start, y_start)
    except Exception as exc:
        synchronize(device)
        return {
            "seconds": time.perf_counter() - start, "error": str(exc),
            "actual_inner_grad_calls": problem.grad_calls - grad_before,
            "passed": False,
        }
    synchronize(device)
    seconds = time.perf_counter() - start

    if result is None:
        return {
            "seconds": seconds,
            "error": "Algorithm returned None; no final point or complexity available.",
            "actual_inner_grad_calls": problem.grad_calls - grad_before,
            "passed": False,
        }

    actual_grad_calls = problem.grad_calls - grad_before
    history = result.history
    record = {
        "seconds": seconds,
        "epsilon": algo.epsilon,
        "inner_solver": type(algo.inner_solver).__name__,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "n_inner_grad_evals": result.n_inner_grad_evals,
        "actual_inner_grad_calls": actual_grad_calls,
        "count_matches": result.n_inner_grad_evals == actual_grad_calls
        == sum(history["inner_grad_evals"]),
        "function_calls_including_autograd": problem.function_calls - function_before,
        "n_subproblem_solves": result.n_subproblem_solves,
        "n_oracle_builds": history.get("n_oracle_builds", len(history.get("grad_norm", []))),
        "termination_reason": history.get("termination_reason"),
        "x": result.x.detach().cpu().tolist(),
        "y": result.y.detach().cpu().tolist(),
        "history": history,
    }
    for key in (
        "n_rejected", "n_validators", "n_successful_halvings",
        "n_value_oracles", "n_derivative_oracles", "final_sigma",
    ):
        if key in history:
            record[key] = history[key]
    if "trial_status" in history:
        record["n_accepted"] = history["trial_status"].count("accepted")
    if "tr_refined" in history:
        record["n_tr_refinements"] = sum(value is True for value in history["tr_refined"])
    if "inner_phase" in history:
        phase_counts = {}
        for phase, count in zip(history["inner_phase"], history["inner_grad_evals"]):
            phase_counts[phase] = phase_counts.get(phase, 0) + count
        record["inner_grad_evals_by_phase"] = phase_counts
    try:
        record.update(evaluate(problem, result.x, y0, epsilon, M, epsilon_h))
    except Exception as exc:
        record["evaluation_error"] = str(exc)
    record["passed"] = bool(
        result.converged and record["count_matches"]
        and record.get("shared_sosp_with_inner_error", False)
    )
    return record


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(default_algorithms=None):
    args = parse_args(default_algorithms)
    names = list(dict.fromkeys(args.algorithms))
    labels = {name: name.upper().replace("-EARLY", "-early") for name in names}
    if args.order == "utr2-first" and "utr2" in names:
        names.remove("utr2")
        names.insert(0, "utr2")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    epsilon_h = math.sqrt(args.epsilon) if args.epsilon_h is None else args.epsilon_h
    device = torch.device(args.device)
    dtype = torch.float64

    Q = torch.randn(args.d, args.d, device=device, dtype=dtype)
    x0 = torch.randn(args.d, device=device, dtype=dtype)
    y0 = torch.zeros(args.d, device=device, dtype=dtype)
    problem = CountedCosLogCosh(mu_y=args.mu_y, omega=args.omega, Q=Q, beta=args.beta)

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
    algorithms = {}
    for name in names:
        if name == "mcn":
            algorithms[name] = mcn
        else:
            base_name = name.split("-", 1)[0]
            cls = {"utr2": UTR2, "utr3": UTR3, "utr4": UTR4, "utr5": UTR5}[base_name]
            if base_name == "utr2":
                inner = NesterovAGD(ell=problem.ell, mu=problem.mu)
            elif name == "utr3-early":
                inner = SCAREarly()
            elif name.endswith("-early"):
                inner = SCARPersistentEarly()
            else:
                inner = SCAR()
            algorithms[name] = cls(
                problem=problem,
                inner_solver=inner,
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
    print(f"  epsilon={args.epsilon}, epsilon_H={epsilon_h}, max_iter={args.max_iterations}, y0=0")
    print(f"  threads={args.threads}, reference_inner_target=1e-12")
    print(f"  subproblem_tol={args.subproblem_tol}, subproblem_max_iter={args.subproblem_max_iter}")
    print(f"  algorithms={', '.join(labels[name] for name in names)}")
    print("  UTR3-early uses SCAREarly; UTR4/5-early share SCARPersistentEarly.")
    print("  UTR5 uses ordinary SCAR with the original AR termination rule.")

    preparation = {}
    if "mcn" in names:
        print("\nPreparing MCN K0 (excluded from main-run time and complexity)...", flush=True)
        synchronize(device)
        start = time.perf_counter()
        initial_solution = prepare_K0(mcn, problem, x0, y0)
        synchronize(device)
        preparation = {
            "seconds": time.perf_counter() - start,
            "grad_evals": initial_solution.n_grad_evals,
            "K0": mcn.K0,
            "tilde_epsilon": mcn.inner_distance_tol,
        }
        print(f"  K0={mcn.K0}, tilde_epsilon={mcn.inner_distance_tol:.6e}")
        print(f"  seconds={preparation['seconds']:.6f}, grad_evals={initial_solution.n_grad_evals}")
    if "utr2" in algorithms:
        print(f"  UTR2 tau_out={algorithms['utr2'].tau_out:.6e}")

    records = {}
    for name, algo in algorithms.items():
        records[labels[name]] = run_algorithm(
            labels[name], algo, problem, x0, y0, device, args.epsilon, mcn.M, epsilon_h,
        )

    print("\nComparison:")
    print(f"{'metric':<30}" + "".join(f" {name:>20}" for name in records))
    for key in (
        "seconds", "epsilon", "converged", "n_iterations", "n_inner_grad_evals",
        "actual_inner_grad_calls", "count_matches",
        "n_subproblem_solves", "n_accepted", "n_rejected", "n_validators",
        "n_successful_halvings", "n_tr_refinements", "n_oracle_builds",
        "reference_value", "reference_grad_norm", "reference_lambda_min",
        "reference_inner_residual", "reference_grad_upper", "reference_lambda_lower",
        "common_test", "shared_sosp_with_inner_error", "passed",
    ):
        values = []
        for record in records.values():
            value = record.get(key, "N/A")
            values.append(f"{value:.6e}" if isinstance(value, float) else str(value))
        print(f"{key:<30}" + "".join(f" {value:>20}" for value in values))

    for name, record in records.items():
        for key in ("error", "evaluation_error", "termination_reason"):
            if record.get(key):
                print(f"\n{name} {key}: {record[key]}")

    output = args.output or ROOT / "outputs" / f"comparison_seed_{args.seed}_eps_{args.epsilon:.0e}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": vars(args),
        "algorithm_order": list(records),
        "dtype": str(dtype),
        "problem": {
            "Q": Q, "x0": x0, "y0": y0,
            "ell": problem.ell, "mu": problem.mu, "rho": problem.rho,
            "kappa": problem.kappa, "M": mcn.M,
        },
        "reference_inner_tolerance": 1e-12,
        "common_gradient_threshold": args.epsilon,
        "common_curvature_threshold": -math.sqrt(mcn.M * args.epsilon),
        "shared_curvature_threshold": -epsilon_h,
        "epsilon_H": epsilon_h,
        "mcn_preparation": preparation,
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, default=json_value) + "\n")

    print("\nconverged uses each algorithm's own stopping rule.")
    print("common_test retains the reference threshold -sqrt(M * epsilon).")
    print("shared_sosp_with_inner_error requires grad_upper <= epsilon and lambda_lower >= -epsilon_H.")
    print("passed requires convergence, matching gradient counts, and the shared error-aware SOSP test.")
    print("n_subproblem_solves counts trial subproblems; n_accepted excludes terminal validation.")
    print("Reference metrics use an approximate y*(x) with gradient tolerance 1e-12.")
    print("MCN K0 preparation and reference evaluation are excluded from main-run time and complexity.")
    print("Timing is a single run; change --algorithms order to check order effects.")
    print(f"Results and histories: {output}")


if __name__ == "__main__":
    main()
