from .base import NCSCAlgorithm, AlgorithmResult

import copy
import math
import torch
from torch.overrides import TorchFunctionMode


class _StableLogCosh(TorchFunctionMode):
    def __init__(self):
        self.arguments = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.log and len(args) == 1 and not kwargs:
            pair = self.arguments.get(id(args[0]))
            if pair is not None and pair[0] is args[0]:
                y = pair[1]
                return torch.nn.functional.softplus(2.0 * y) - y - math.log(2.0)
        result = func(*args, **kwargs)
        if func is torch.cosh and len(args) == 1 and not kwargs:
            self.arguments[id(result)] = (result, args[0])
        return result


class _CountedInnerProblem:
    def __init__(self, problem):
        self.problem = problem
        self.n_grad_evals = 0

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def grad_y(self, x, y):
        self.n_grad_evals += 1
        with _StableLogCosh():
            return self.problem.grad_y(x, y)

    def f(self, x, y):
        with _StableLogCosh():
            return self.problem.f(x, y)


class UTR3(NCSCAlgorithm):
    def __init__(
            self,
            problem,
            inner_solver,
            trust_region_solver,
            epsilon: float,
            max_iterations: int = 100_000,
    ):
        super().__init__(problem)
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be finite and in (0, 1), got {epsilon}")
        if int(max_iterations) != max_iterations or max_iterations < 0:
            raise ValueError("max_iterations must be a non-negative integer.")
        if epsilon ** 2 == 0.0 or epsilon ** 3 == 0.0:
            raise ValueError("epsilon is too small to represent its squared and cubed tolerances.")

        self.inner_solver = inner_solver
        self.trust_region_solver = trust_region_solver
        self.epsilon = epsilon
        self.max_iterations = int(max_iterations)
        self.eta = 1.0 / 256.0

    def _build_corrected_tuple(self, x, y, grad_y):
        with _StableLogCosh():
            value = self.problem.f(x, y)
            grad_x = self.problem.grad_x(x, y)
            H_xx, H_xy, H_yx, H_yy = self.problem.hessian_blocks(x, y)

        v = torch.linalg.solve(H_yy, grad_y)
        P_hat = value - 0.5 * grad_y @ v
        g = grad_x - H_xy @ v
        W = torch.linalg.solve(H_yy, H_yx)
        H = H_xx - H_xy @ W
        H = 0.5 * (H + H.T)

        if not torch.isfinite(P_hat):
            raise RuntimeError("Non-finite corrected value.")
        if not torch.isfinite(g).all() or not torch.isfinite(H).all():
            raise RuntimeError("Non-finite corrected derivatives.")
        return P_hat.item(), g.detach(), H.detach()

    def _max_res(self, counted_problem, x, y):
        result = self.inner_solver.run(
            counted_problem, x, y,
            stop_rule="gradient_norm", target=self.epsilon ** 2,
        )
        residual = torch.linalg.vector_norm(result.grad_y).item()
        if not result.converged or not math.isfinite(residual) or residual > self.epsilon ** 2:
            raise RuntimeError(
                f"Inner residual contract failed: residual={residual:.6e}, "
                f"target={self.epsilon ** 2:.6e}."
            )
        if not torch.isfinite(result.y).all():
            raise RuntimeError("Inner solver returned a non-finite point.")
        return result

    def _refine_unit_tr(self, g, B, tol, max_iter):
        eigenvalues, vectors = torch.linalg.eigh(B)
        projected_g = vectors.T @ g
        lower = max(0.0, -eigenvalues[0].item())
        shifted = eigenvalues - eigenvalues[0] if lower > 0.0 else eigenvalues
        positive = shifted > 0.0

        if torch.all(projected_g[~positive] == 0.0):
            coefficients = torch.zeros_like(projected_g)
            coefficients[positive] = -projected_g[positive] / shifted[positive]
            norm = torch.linalg.vector_norm(coefficients).item()
            if norm <= 1.0:
                if lower > 0.0:
                    coefficients[0] += math.sqrt(max(0.0, 1.0 - norm ** 2))
                return vectors @ coefficients, lower

        left = 0.0
        right = torch.linalg.vector_norm(projected_g).item()
        for _ in range(max_iter):
            delta = 0.5 * (left + right)
            if delta == left or delta == right:
                break
            coefficients = -projected_g / (shifted + delta)
            norm = torch.linalg.vector_norm(coefficients).item()
            if math.isfinite(norm) and abs(norm - 1.0) <= tol:
                return vectors @ coefficients, lower + delta
            if norm > 1.0:
                left = delta
            else:
                right = delta
        raise RuntimeError("Shifted trust-region secular equation did not converge.")

    def _solve_tr(self, g, B, radius):
        if not math.isfinite(radius) or radius <= 0.0 or not torch.isfinite(B).all():
            raise RuntimeError("Invalid trust-region model or radius.")

        roundoff = 1000.0 * torch.finfo(g.dtype).eps
        scaled_g = radius * g
        scaled_B = radius * (radius * B)
        scale = max(torch.linalg.vector_norm(scaled_g).item(), torch.linalg.matrix_norm(scaled_B).item())
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("Invalid trust-region objective scaling.")

        unit_tol = roundoff
        solver = copy.copy(self.trust_region_solver)
        if hasattr(solver, "tol"):
            if not math.isfinite(solver.tol) or solver.tol <= 0.0:
                raise ValueError("Trust-region tolerance must be positive and finite.")
            unit_tol = max(unit_tol, float(solver.tol))
            solver.tol = unit_tol / 4.0
        radius_tol = unit_tol * radius

        unit_g, unit_B = scaled_g / scale, scaled_B / scale

        def restore_and_check(unit_step, unit_multiplier):
            d = radius * unit_step.detach()
            multiplier = unit_multiplier * (scale / radius) / radius
            if d.shape != g.shape or not torch.isfinite(d).all():
                raise RuntimeError("Trust-region solver returned an invalid step.")
            if not math.isfinite(multiplier) or multiplier < 0.0:
                raise RuntimeError("Trust-region solver returned an invalid multiplier.")

            r = torch.linalg.vector_norm(d).item()
            B_norm = torch.linalg.matrix_norm(B).item()
            g_norm = torch.linalg.vector_norm(g).item()
            curvature_tol = roundoff * max(1.0, B_norm, multiplier)
            stationarity_tol = roundoff * max(1.0, g_norm, (B_norm + multiplier) * r)
            stationarity = torch.linalg.vector_norm(B @ d + multiplier * d + g).item()
            lambda_min = torch.linalg.eigvalsh(B)[0].item()

            if not all(math.isfinite(v) for v in (r, stationarity, lambda_min)):
                raise RuntimeError("Non-finite trust-region optimality residual.")
            radius_error = r - radius
            if radius_error > radius_tol:
                raise RuntimeError("Trust-region solver returned an infeasible step.")
            if stationarity > stationarity_tol or lambda_min + multiplier < -curvature_tol:
                raise RuntimeError("Trust-region solution failed the global KKT checks.")
            if multiplier > 0.0 and abs(radius_error) > radius_tol:
                raise RuntimeError("Trust-region solution failed complementarity.")

            boundary = abs(radius_error) <= radius_tol
            return d, multiplier, r, boundary, radius_tol

        try:
            result = solver.solve(g=unit_g, H=unit_B, radius=1.0)
        except RuntimeError:
            result = None
        self._last_tr_refined = False
        if result is not None and result.converged:
            try:
                return restore_and_check(result.s, float(result.multiplier))
            except RuntimeError:
                pass

        self._last_tr_refined = True
        unit_step, unit_multiplier = self._refine_unit_tr(
            unit_g, unit_B, unit_tol / 4.0, getattr(solver, "max_iter", 1000),
        )
        return restore_and_check(unit_step, unit_multiplier)

    def run(self, x0, y0) -> AlgorithmResult:
        epsilon = self.epsilon
        x = x0.clone().detach()
        y = y0.clone().detach()
        sigma = epsilon
        counted_problem = _CountedInnerProblem(self.problem)
        n_subproblem_solves = 0
        n_oracle_builds = 0
        n_accepted = 0
        n_rejected = 0
        g = H = None

        history = {
            "sigma": [],
            "grad_norm": [],
            "lambda_min": [],
            "radius": [],
            "step_norm": [],
            "multiplier": [],
            "boundary": [],
            "accepted": [],
            "trial_status": [],
            "tr_tolerance": [],
            "tr_refined": [],
            "inner_steps": [],
            "inner_grad_evals": [],
            "inner_residual": [],
        }

        def finish(converged, reason):
            history["n_oracle_builds"] = n_oracle_builds
            history["n_accepted"] = n_accepted
            history["n_rejected"] = n_rejected
            history["n_failed_trials"] = n_subproblem_solves - n_accepted - n_rejected
            history["final_sigma"] = sigma
            history["termination_reason"] = reason
            if g is not None:
                history["final_grad_norm"] = torch.linalg.vector_norm(g).item()
                history["final_lambda_min"] = torch.linalg.eigvalsh(H)[0].item()
            return AlgorithmResult(
                x=x.detach(), y=y.detach(),
                n_iterations=n_subproblem_solves,
                n_inner_grad_evals=counted_problem.n_grad_evals,
                n_subproblem_solves=n_subproblem_solves,
                converged=converged, history=history,
            )

        def max_res_at(point, initial_y):
            before = counted_problem.n_grad_evals
            try:
                result = self._max_res(counted_problem, point, initial_y)
            except (RuntimeError, ValueError, OverflowError) as exc:
                history["inner_steps"].append(None)
                history["inner_grad_evals"].append(counted_problem.n_grad_evals - before)
                history["inner_residual"].append(None)
                raise RuntimeError(str(exc)) from exc
            history["inner_steps"].append(result.n_steps)
            history["inner_grad_evals"].append(counted_problem.n_grad_evals - before)
            history["inner_residual"].append(result.residual)
            return result

        try:
            inner = max_res_at(x, y)
            y = inner.y.detach()
            P_hat, g, H = self._build_corrected_tuple(x, y, inner.grad_y)
            n_oracle_builds += 1
        except (RuntimeError, ValueError, OverflowError) as exc:
            return finish(False, f"Initial oracle failed: {exc}")

        while True:
            grad_norm = torch.linalg.vector_norm(g).item()
            lambda_min = torch.linalg.eigvalsh(H)[0].item()
            if grad_norm <= epsilon and lambda_min >= -(9.0 / 8.0) * sigma * math.sqrt(epsilon):
                return finish(True, "Stopping condition satisfied.")
            if n_subproblem_solves >= self.max_iterations:
                return finish(False, "Maximum number of trust-region trials reached.")

            h = max(grad_norm, epsilon)
            identity = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
            B = H + (sigma * math.sqrt(epsilon) / 64.0 + sigma * math.sqrt(h)) * identity
            radius = math.sqrt(h) / (4.0 * sigma)

            history["sigma"].append(sigma)
            history["grad_norm"].append(grad_norm)
            history["lambda_min"].append(lambda_min)
            history["radius"].append(radius)
            for key in ("step_norm", "multiplier", "boundary", "accepted", "tr_tolerance", "tr_refined"):
                history[key].append(None)
            history["trial_status"].append("failed")
            n_subproblem_solves += 1

            try:
                d, multiplier, r, boundary, radius_tol = self._solve_tr(g, B, radius)
            except (RuntimeError, ValueError, OverflowError) as exc:
                return finish(False, f"Trust-region trial {n_subproblem_solves} failed: {exc}")

            history["step_norm"][-1] = r
            history["multiplier"][-1] = multiplier
            history["boundary"][-1] = boundary
            history["tr_tolerance"][-1] = radius_tol
            history["tr_refined"][-1] = self._last_tr_refined

            if r == 0.0:
                if grad_norm > epsilon or lambda_min < -(9.0 / 8.0) * sigma * math.sqrt(epsilon):
                    return finish(False, "Numerical zero step does not satisfy the stopping condition.")
                history["trial_status"][-1] = "zero_step"
                return finish(True, "Zero global trust-region step.")

            x_plus = (x + d).detach()
            try:
                inner_plus = max_res_at(x_plus, y)
                y_plus = inner_plus.y.detach()
                P_plus, g_plus, H_plus = self._build_corrected_tuple(x_plus, y_plus, inner_plus.grad_y)
                n_oracle_builds += 1
            except (RuntimeError, ValueError, OverflowError) as exc:
                return finish(False, f"Candidate oracle at trial {n_subproblem_solves} failed: {exc}")

            grad_plus_norm = torch.linalg.vector_norm(g_plus).item()
            if not boundary:
                accepted = P_plus <= P_hat + epsilon ** 3 and grad_plus_norm <= 0.5 * h
            else:
                required_decrease = self.eta * (h ** 1.5 / sigma + multiplier * r ** 2) + epsilon ** 3
                accepted = P_hat - P_plus >= required_decrease and grad_plus_norm <= 0.5 * h + multiplier * r

            history["accepted"][-1] = accepted
            history["trial_status"][-1] = "accepted" if accepted else "rejected"
            if accepted:
                x, y, P_hat, g, H = x_plus, y_plus, P_plus, g_plus, H_plus
                sigma = max(epsilon, sigma / 2.0)
                n_accepted += 1
            else:
                sigma *= 2.0
                n_rejected += 1
                if not math.isfinite(sigma):
                    return finish(False, "Non-finite UTR3 scale sigma.")
