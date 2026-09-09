from .base import AlgorithmResult
from .utr3 import UTR3, _CountedInnerProblem, _StableLogCosh
from inner_solvers.scar_early import SCAREarly

import math
import torch


class UTR4(UTR3):
    def __init__(
            self,
            problem,
            inner_solver,
            trust_region_solver,
            epsilon: float,
            max_iterations: int = 100_000,
    ):
        super().__init__(
            problem, inner_solver, trust_region_solver, epsilon, max_iterations,
        )
        if isinstance(inner_solver, SCAREarly):
            raise ValueError("UTR4 currently requires ordinary SCAR, not SCAREarly.")
        for name in ("initialize_persistent_state", "estimate_secant", "persistent_halving"):
            if not callable(getattr(inner_solver, name, None)):
                raise TypeError(f"Inner solver must provide {name}().")
        self.c0 = 2.0 ** -12
        self.A_epsilon = 1.0 - math.log(self.epsilon)
        self.sigma0 = 1.0 / self.A_epsilon
        self.N_w = math.ceil(math.log2((1.0 + self.c0) / self.c0))

    def _build_corrected_value(self, x, y, grad_y):
        with _StableLogCosh():
            value = self.problem.f(x, y)
            H_yy = torch.autograd.functional.hessian(
                lambda yy: self.problem.f(x, yy), y.detach(),
            )
        v = torch.linalg.solve(H_yy, grad_y)
        P_hat = value - 0.5 * grad_y @ v
        if not torch.isfinite(P_hat):
            raise RuntimeError("Non-finite corrected value.")
        return P_hat.item()

    def _build_corrected_derivatives(self, x, y, grad_y):
        with _StableLogCosh():
            grad_x = self.problem.grad_x(x, y)
            H_xx, H_xy, H_yx, H_yy = self.problem.hessian_blocks(x, y)
        v = torch.linalg.solve(H_yy, grad_y)
        g = grad_x - H_xy @ v
        H = H_xx - H_xy @ torch.linalg.solve(H_yy, H_yx)
        H = 0.5 * (H + H.T)
        if not torch.isfinite(g).all() or not torch.isfinite(H).all():
            raise RuntimeError("Non-finite corrected derivatives.")
        return g.detach(), H.detach()

    def run(self, x0, y0) -> AlgorithmResult:
        epsilon = self.epsilon
        sqrt_epsilon = math.sqrt(epsilon)
        epsilon_32 = epsilon * sqrt_epsilon
        x, y = x0.clone().detach(), y0.clone().detach()
        sigma = self.sigma0
        counted_problem = _CountedInnerProblem(self.problem)
        state = None
        grad_y = g = H = None
        n_trials = n_accepted = n_rejected = n_validators = 0
        n_value_oracles = n_derivative_oracles = n_halvings = 0
        history = {
            "sigma": [], "grad_norm": [], "lambda_min": [], "radius": [],
            "step_norm": [], "multiplier": [], "boundary": [],
            "accepted": [], "trial_status": [], "tr_tolerance": [],
            "tr_refined": [], "candidate_value": [], "value_decrease": [],
            "required_decrease": [], "validator_secant": [],
            "validator_target": [], "validator_grad_norm": [],
            "validator_lambda_min": [], "validator_residual": [],
            "inner_phase": [], "inner_trial": [], "inner_steps": [],
            "inner_grad_evals": [], "inner_residual": [],
            "inner_nu": [], "inner_M": [],
            "inner_step_unit": "successful_persistent_halving",
            "N_w": self.N_w, "initial_sigma": sigma,
        }

        def finish(converged, reason):
            history.update(
                n_accepted=n_accepted, n_rejected=n_rejected,
                n_validators=n_validators, n_successful_halvings=n_halvings,
                n_value_oracles=n_value_oracles,
                n_derivative_oracles=n_derivative_oracles,
                n_oracle_builds=n_derivative_oracles,
                final_sigma=sigma,
                persistent_state=None if state is None else dict(state),
                termination_reason=reason,
            )
            if g is not None:
                history["final_grad_norm"] = torch.linalg.vector_norm(g).item()
                history["final_lambda_min"] = torch.linalg.eigvalsh(H)[0].item()
            if grad_y is not None:
                history["final_inner_residual"] = torch.linalg.vector_norm(grad_y).item()
            return AlgorithmResult(
                x=x.detach(), y=y.detach(), n_iterations=n_trials,
                n_inner_grad_evals=counted_problem.n_grad_evals,
                n_subproblem_solves=n_trials, converged=converged, history=history,
            )

        def tracked(phase, operation):
            before = counted_problem.n_grad_evals
            try:
                result = operation()
            finally:
                history["inner_phase"].append(phase)
                history["inner_trial"].append(n_trials)
                history["inner_steps"].append(0)
                history["inner_grad_evals"].append(counted_problem.n_grad_evals - before)
                history["inner_residual"].append(None)
                history["inner_nu"].append(None if state is None else state["nu"])
                history["inner_M"].append(None if state is None else state["M"])
            return result

        def gradient_at(point, inner_point, phase):
            gradient = tracked(phase, lambda: counted_problem.grad_y(point, inner_point))
            residual = torch.linalg.vector_norm(gradient).item()
            if not math.isfinite(residual):
                raise RuntimeError("Non-finite inner gradient.")
            history["inner_residual"][-1] = residual
            return gradient.detach()

        def halve(point, inner_point, gradient, phase):
            nonlocal state, n_halvings
            old_residual = torch.linalg.vector_norm(gradient).item()
            updated_y, updated_grad, state, reported = tracked(
                phase,
                lambda: self.inner_solver.persistent_halving(
                    counted_problem, point, inner_point, gradient, state,
                ),
            )
            if reported != history["inner_grad_evals"][-1]:
                raise RuntimeError("Persistent SCAR gradient count mismatch.")
            residual = torch.linalg.vector_norm(updated_grad).item()
            if not math.isfinite(residual) or residual > 0.5 * old_residual:
                raise RuntimeError("Persistent SCAR halving contract failed.")
            if not torch.isfinite(updated_y).all():
                raise RuntimeError("Non-finite persistent SCAR return point.")
            n_halvings += 1
            history["inner_steps"][-1] = 1
            history["inner_residual"][-1] = residual
            history["inner_nu"][-1] = state["nu"]
            history["inner_M"][-1] = state["M"]
            return updated_y.detach(), updated_grad.detach()

        def refine(point, inner_point, gradient, target, phase):
            if not math.isfinite(target) or target <= 0.0:
                raise RuntimeError("Invalid persistent SCAR residual target.")
            while torch.linalg.vector_norm(gradient).item() > target:
                inner_point, gradient = halve(point, inner_point, gradient, phase)
            return inner_point, gradient

        phase = "initialization"
        try:
            grad_y = gradient_at(x, y, "initial_gradient")
            state, reported = tracked(
                "initial_secant",
                lambda: self.inner_solver.initialize_persistent_state(
                    counted_problem, x, y, grad_y,
                ),
            )
            if reported != history["inner_grad_evals"][-1]:
                raise RuntimeError("Initial secant gradient count mismatch.")
            a0 = state["nu"]
            history["initial_secant"] = a0
            history["inner_nu"][-1] = state["nu"]
            history["inner_M"][-1] = state["M"]
            target = self.c0 * a0 * sqrt_epsilon / (4.0 * sigma)
            history["initial_target"] = target
            y, grad_y = refine(x, y, grad_y, target, "initial_refinement")
            P_hat, g, H = self._build_corrected_tuple(x, y, grad_y)
            n_value_oracles += 1
            n_derivative_oracles += 1

            while n_trials < self.max_iterations:
                radius = sqrt_epsilon / (4.0 * sigma)
                I = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
                B = H + (65.0 / 64.0) * sigma * sqrt_epsilon * I
                history["sigma"].append(sigma)
                history["grad_norm"].append(torch.linalg.vector_norm(g).item())
                history["lambda_min"].append(torch.linalg.eigvalsh(H)[0].item())
                history["radius"].append(radius)
                for key in (
                    "step_norm", "multiplier", "boundary", "accepted", "tr_tolerance",
                    "tr_refined", "candidate_value", "value_decrease", "required_decrease",
                    "validator_secant", "validator_target", "validator_grad_norm",
                    "validator_lambda_min", "validator_residual",
                ):
                    history[key].append(None)
                history["trial_status"].append("failed")
                n_trials += 1
                phase = f"trust-region trial {n_trials}"
                d, multiplier, r, boundary, radius_tol = self._solve_tr(g, B, radius)
                history["step_norm"][-1] = r
                history["multiplier"][-1] = multiplier
                history["boundary"][-1] = boundary
                history["tr_tolerance"][-1] = radius_tol
                history["tr_refined"][-1] = self._last_tr_refined

                phase = f"working candidate at trial {n_trials}"
                x_plus = (x + d).detach()
                y_plus = y
                grad_plus = grad_y if torch.equal(x_plus, x) else gradient_at(
                    x_plus, y_plus, "candidate_gradient",
                )
                for _ in range(self.N_w):
                    y_plus, grad_plus = halve(x_plus, y_plus, grad_plus, "working_halving")

                if boundary:
                    phase = f"boundary value at trial {n_trials}"
                    P_plus = self._build_corrected_value(x_plus, y_plus, grad_plus)
                    n_value_oracles += 1
                    decrease = P_hat - P_plus
                    required = epsilon_32 / (256.0 * sigma)
                    history["candidate_value"][-1] = P_plus
                    history["value_decrease"][-1] = decrease
                    history["required_decrease"][-1] = required
                    if decrease >= required:
                        phase = f"accepted derivatives at trial {n_trials}"
                        g_plus, H_plus = self._build_corrected_derivatives(x_plus, y_plus, grad_plus)
                        n_derivative_oracles += 1
                        x, y, grad_y = x_plus, y_plus, grad_plus
                        P_hat, g, H = P_plus, g_plus, H_plus
                        n_accepted += 1
                        history["accepted"][-1] = True
                        history["trial_status"][-1] = "accepted"
                        continue
                    rejection = "boundary_rejected"
                else:
                    phase = f"validator at trial {n_trials}"
                    n_validators += 1
                    a_plus, reported = tracked(
                        "validator_secant",
                        lambda: self.inner_solver.estimate_secant(
                            counted_problem, x_plus, y_plus, grad_plus,
                        ),
                    )
                    if reported != history["inner_grad_evals"][-1]:
                        raise RuntimeError("Validator secant gradient count mismatch.")
                    target = self.c0 * a_plus * epsilon_32
                    history["validator_secant"][-1] = a_plus
                    history["validator_target"][-1] = target
                    y_v, grad_v = refine(x_plus, y_plus, grad_plus, target, "validation_halving")
                    g_v, H_v = self._build_corrected_derivatives(x_plus, y_v, grad_v)
                    n_derivative_oracles += 1
                    norm_v = torch.linalg.vector_norm(g_v).item()
                    lambda_v = torch.linalg.eigvalsh(H_v)[0].item()
                    history["validator_grad_norm"][-1] = norm_v
                    history["validator_lambda_min"][-1] = lambda_v
                    history["validator_residual"][-1] = torch.linalg.vector_norm(grad_v).item()
                    if norm_v <= epsilon / 2.0 and lambda_v >= -(21.0 / 8.0) * sigma * sqrt_epsilon:
                        x, y, grad_y, g, H = x_plus, y_v, grad_v, g_v, H_v
                        history["accepted"][-1] = True
                        history["trial_status"][-1] = "validated"
                        n_accepted += 1
                        return finish(True, "Interior validation satisfied.")
                    rejection = "validation_rejected"

                history["accepted"][-1] = False
                history["trial_status"][-1] = rejection
                n_rejected += 1
                sigma *= 2.0
                if not math.isfinite(sigma):
                    raise RuntimeError("UTR4 scale overflowed.")
                phase = f"current-point refresh after trial {n_trials}"
                y, grad_y = halve(x, y, grad_y, "refresh_halving")
                P_hat, g, H = self._build_corrected_tuple(x, y, grad_y)
                n_value_oracles += 1
                n_derivative_oracles += 1

            return finish(False, "Maximum number of trust-region trials reached.")
        except (RuntimeError, ValueError, OverflowError) as exc:
            return finish(False, f"{phase}: {exc}")
