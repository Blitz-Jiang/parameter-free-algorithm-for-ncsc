import argparse
import contextlib
import io
import json
import math
import statistics
import time
from pathlib import Path

import torch

from run_comparision import (
    MCN, UTR2, CosLogCosh, NesterovAGD, CubicSubproblemSolver,
    TRSubproblemSolver, prepare_K0, evaluate,
)


class RecordedMCN(MCN):
    def _build_oracle(self, x, y):
        self.points.append(x.detach().clone())
        return super()._build_oracle(x, y)


class RecordedUTR2(UTR2):
    def _build_corrected_oracle(self, x, y, grad_y):
        self.points.append(x.detach().clone())
        return super()._build_corrected_oracle(x, y, grad_y)

    def _build_tr_subproblem(self, g, H, xi):
        B, radius, h = super()._build_tr_subproblem(g, H, xi)
        damping = 2.0 * self.sigma * math.sqrt(h)
        row = {
            "base_radius": radius,
            "hessian_over_damping": torch.linalg.matrix_norm(H, ord=2).item() / damping,
            "xi_over_damping": xi / damping,
        }
        try:
            cubic = CubicSubproblemSolver().solve(g=g, H=H, M=self.MP)
            if cubic.converged:
                row["same_oracle_cubic_norm"] = torch.linalg.vector_norm(cubic.s).item()
                self.trust_region_solver.cubic_step = cubic.s
            else:
                row["counterfactual_error"] = "Cubic solve did not converge"
                self.trust_region_solver.cubic_step = None
        except Exception as exc:
            row["counterfactual_error"] = str(exc)
            self.trust_region_solver.cubic_step = None
        self.trust_region_solver.row = row
        return B, radius * self.radius_factor, h


class RecordedTR(TRSubproblemSolver):
    def solve(self, g, H, radius):
        result = super().solve(g=g, H=H, radius=radius)
        row = self.row
        norm = torch.linalg.vector_norm(result.s).item()
        row.update(step_norm=norm, radius=radius, step_over_radius=norm / radius,
                   multiplier=result.multiplier)
        if self.cubic_step is not None and norm > 0:
            cubic_norm = row["same_oracle_cubic_norm"]
            row["same_oracle_step_ratio"] = cubic_norm / norm
            if cubic_norm > 0:
                row["direction_cosine"] = (self.cubic_step @ result.s).item() / (cubic_norm * norm)
        self.rows.append(row)
        return result


def distribution(values):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return {}
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": values[0], "max": values[-1],
        "p10": values[int(0.1 * (len(values) - 1))],
        "p90": values[int(0.9 * (len(values) - 1))],
    }


def diagnostics(rows):
    if not rows:
        return {}
    summary = {
        "boundary_fraction": statistics.mean(abs(r["step_over_radius"] - 1.0) <= 1e-8 for r in rows),
        "positive_multiplier_fraction": statistics.mean(r["multiplier"] > 1e-10 for r in rows),
        "counterfactual_failures": sum("counterfactual_error" in r for r in rows),
    }
    for key in ("step_over_radius", "same_oracle_step_ratio", "direction_cosine",
                "hessian_over_damping", "xi_over_damping"):
        summary[key] = distribution([r[key] for r in rows if key in r])
    return summary


def reference_profile(problem, points, history, epsilon, M, samples):
    if not points or samples == 0:
        return []
    count = min(samples, len(points))
    indices = sorted(set(round(i * (len(points) - 1) / max(1, count - 1)) for i in range(count)))
    profile = []
    for i in indices:
        row = {"oracle_index": i}
        for key in ("step_norm", "radius", "tr_multiplier"):
            if i < len(history.get(key, [])):
                row[key] = history[key][i]
        try:
            row.update(evaluate(problem, points[i], torch.zeros_like(points[i]), epsilon, M))
        except Exception as exc:
            row["error"] = str(exc)
        profile.append(row)
    return profile


def run_one(name, algo, problem, x0, y0, args, M):
    algo.points = []
    record = {"name": name}
    captured = io.StringIO()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(captured):
            result = algo.run(x0.clone(), y0.clone())
        record["instrumented_seconds"] = time.perf_counter() - start
        if result is None:
            record["error"] = "Algorithm returned None at iteration limit; counts unavailable"
        else:
            record.update(
                converged=result.converged, outer=result.n_iterations,
                inner_grad_evals=result.n_inner_grad_evals,
                oracle_calls=len(result.history["grad_norm"]),
                subproblem_solves=result.n_subproblem_solves, history=result.history,
            )
            record["inner_per_oracle"] = result.n_inner_grad_evals / max(1, record["oracle_calls"])
            try:
                record["final_reference"] = evaluate(problem, result.x, y0, args.epsilon, M)
            except Exception as exc:
                record["reference_error"] = str(exc)
            record["reference_profile"] = reference_profile(
                problem, algo.points, result.history, args.epsilon, M, args.samples,
            )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["solver_stdout_tail"] = captured.getvalue()[-2000:]
    if isinstance(algo, RecordedUTR2):
        rows = algo.trust_region_solver.rows
        record["step_diagnostics"] = rows
        record["diagnostics"] = diagnostics(rows)
        n = len(rows)
        record["phase_diagnostics"] = {
            phase: diagnostics(rows[a:b])
            for phase, a, b in (("early", 0, n // 3), ("middle", n // 3, 2 * n // 3),
                                ("late", 2 * n // 3, n))
        }
    final = record.get("final_reference", {})
    print(f"  {name}: outer={record.get('outer')}, inner={record.get('inner_grad_evals')}, "
          f"common_test={final.get('common_test')}, error={record.get('error')}", flush=True)
    if "diagnostics" in record:
        d = record["diagnostics"]
        print(f"    boundary={d.get('boundary_fraction')}, "
              f"same-oracle ratio median={d.get('same_oracle_step_ratio', {}).get('median')}, "
              f"H/damping median={d.get('hessian_over_damping', {}).get('median')}", flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description="Diagnose UTR2 outer-iteration overhead without editing algorithms")
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mu-y", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1e-7)
    p.add_argument("--max-iterations", type=int, default=3000)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("/tmp/utr2_outer_diagnostics.json"))
    args = p.parse_args()
    if args.samples < 0:
        p.error("samples must be non-negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    Q = torch.randn(args.d, args.d, dtype=torch.float64)
    x0 = torch.randn(args.d, dtype=torch.float64)
    y0 = torch.zeros_like(x0)
    cases = [("baseline", args.mu_y, args.omega, args.beta)]
    if args.sweep:
        cases += [
            ("omega_half", args.mu_y, args.omega / 2, args.beta),
            ("omega_double", args.mu_y, args.omega * 2, args.beta),
            ("mu_half", args.mu_y / 2, args.omega, args.beta),
            ("mu_double", args.mu_y * 2, args.omega, args.beta),
            ("beta_plus_half", args.mu_y, args.omega, args.beta + 0.5),
        ]
    report = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "predicted_flat_model_ratio": 4 / math.sqrt(3),
        "notes": [
            "CPU float64; identical Q, x0, y0 across cases and methods.",
            "K0 preparation and reference evaluation excluded from algorithm complexity.",
            "Counterfactual cubic solves are diagnostics, excluded from algorithm counts.",
            "Instrumented timing is not a fair speed benchmark.",
            "Radius-scaled variants are ablations, not the original theoretically justified UTR2.",
            "Common reference test is approximate, using inner gradient tolerance 1e-12.",
        ],
        "cases": [],
    }
    for label, mu_y, omega, beta in cases:
        print(f"\n{label}: mu_y={mu_y}, omega={omega}, beta={beta}", flush=True)
        problem = CosLogCosh(mu_y=mu_y, omega=omega, Q=Q.clone(), beta=beta)
        mcn = RecordedMCN(
            problem, NesterovAGD(problem.ell, problem.mu), CubicSubproblemSolver(),
            epsilon=args.epsilon, K0=0, max_iterations=args.max_iterations,
        )
        preparation = prepare_K0(mcn, problem, x0, y0)
        case = {
            "label": label, "mu_y": mu_y, "omega": omega, "beta": beta,
            "ell": problem.ell, "rho": problem.rho, "kappa": problem.kappa,
            "M": mcn.M, "K0": mcn.K0, "excluded_K0_grad_evals": preparation.n_grad_evals,
            "runs": [run_one("MCN", mcn, problem, x0, y0, args, mcn.M)],
        }
        factors = [1.0, 2.0, 4 / math.sqrt(3)] if label == "baseline" else [1.0]
        for factor in factors:
            tr = RecordedTR()
            tr.rows = []
            utr = RecordedUTR2(
                problem, NesterovAGD(problem.ell, problem.mu), tr,
                epsilon=args.epsilon, max_iterations=args.max_iterations,
            )
            utr.radius_factor = factor
            run = run_one(f"UTR2_radius_x{factor:.6g}", utr, problem, x0, y0, args, mcn.M)
            run["radius_factor"] = factor
            baseline = case["runs"][0]
            if baseline.get("outer", 0) > 0 and "outer" in run:
                run["outer_ratio_to_mcn"] = run["outer"] / baseline["outer"]
                run["inner_ratio_to_mcn"] = run["inner_grad_evals"] / baseline["inner_grad_evals"]
                print(f"    outer/MCN={run['outer_ratio_to_mcn']:.6f}, "
                      f"inner/MCN={run['inner_ratio_to_mcn']:.6f}", flush=True)
            case["runs"].append(run)
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostic histories and reference profiles: {args.output}", flush=True)


if __name__ == "__main__":
    main()
