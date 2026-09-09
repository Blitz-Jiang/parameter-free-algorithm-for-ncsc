from .base import InnerSolver, InnerSolverResult

import torch
import math


class SCAR(InnerSolver):
    def __init__(self):
        pass

    def _solve_ar_subproblem(
            self,
            problem,
            x,
            y0,
            y_bar,
            sigma,
            L0,
            grad_phi0=None,
            max_iters=100_000,
            max_backtracking=100,
    ):
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError(f"sigma must be positive and finite, got {sigma}")

        if L0 <= 0.0 or not math.isfinite(L0):
            raise ValueError(f"L0 must be positive and finite, got {L0}")

        y = y0.clone().detach()
        v = y.clone().detach()
        theta = 1.0
        L_est = float(L0)
        n_grad_evals = 0

        for k in range(1, max_iters + 1):
            if k == 1 and grad_phi0 is not None:
                grad_phi_v = grad_phi0.detach()
            else:
                grad_phi_v = -problem.grad_y(x, v)
                n_grad_evals += 1

            if not torch.isfinite(grad_phi_v).all():
                raise RuntimeError("Non-finite gradient in AR subproblem.")

            phi_v = -problem.f(x, v)
            if not torch.isfinite(phi_v):
                raise RuntimeError("Non-finite function value in AR subproblem.")

            for _ in range(max_backtracking):
                if not math.isfinite(L_est) or not math.isfinite(L_est + sigma):
                    raise RuntimeError("Non-finite smoothness estimate in AR subproblem.")

                y_trial = (
                    v + (-grad_phi_v + sigma * (y_bar - v)) / (L_est + sigma)
                ).detach()
                if not torch.isfinite(y_trial).all():
                    raise RuntimeError("Non-finite trial point in AR subproblem.")

                phi_trial = -problem.f(x, y_trial)
                diff = y_trial - v
                lhs = -problem.value_remainder_y(
                    x, v, y_trial, -grad_phi_v, -phi_v, -phi_trial,
                )
                rhs = 0.5 * L_est * torch.sum(diff ** 2)
                if not torch.isfinite(lhs) or not torch.isfinite(rhs):
                    raise RuntimeError("Non-finite value during AR subproblem backtracking.")
                if lhs <= rhs:
                    break

                L_est *= 2.0
            else:
                raise RuntimeError("AR subproblem backtracking exceeded limit.")

            y_next = y_trial
            L_sk = 2.0 * L_est
            required_k = math.ceil(8.0 * math.sqrt(2.0 * L_sk / sigma))

            if k >= required_k:
                return y_next.detach(), n_grad_evals

            theta_next = (1.0 + math.sqrt(1.0 + 4.0 * theta ** 2)) / 2.0
            beta = (theta - 1.0) / theta_next
            v_next = (y_next + beta * (y_next - y)).detach()

            y = y_next.detach()
            v = v_next
            theta = theta_next

        raise RuntimeError("AR subproblem solver exceeded max_iters.")

    def _ar(
            self,
            problem,
            x,
            y0,
            sigma1,
            M0,
            grad_phi0=None,
    ):
        if sigma1 <= 0.0 or not math.isfinite(sigma1):
            raise ValueError(f"sigma1 must be positive and finite, got {sigma1}")

        if M0 <= 0.0 or not math.isfinite(M0):
            raise ValueError(f"M0 must be positive and finite, got {M0}")

        y_prev = y0.clone().detach()
        y_bar_prev = y0.clone().detach()
        sigma_prev = 0.0
        M_prev = float(M0)
        grad_phi_prev = None if grad_phi0 is None else grad_phi0.detach()
        n_grad_evals = 0
        s = 1

        while True:
            if s == 1:
                sigma = float(sigma1)
            else:
                sigma = 4.0 * sigma_prev

            if not math.isfinite(sigma):
                raise RuntimeError("Non-finite regularization parameter in AR.")

            gamma = 1.0 - sigma_prev / sigma
            y_bar = ((1.0 - gamma) * y_bar_prev + gamma * y_prev).detach()

            y_new, subproblem_grad_evals = self._solve_ar_subproblem(
                problem=problem,
                x=x,
                y0=y_prev,
                y_bar=y_bar,
                sigma=sigma,
                L0=M_prev / 2.0,
                grad_phi0=grad_phi_prev,
            )
            n_grad_evals += subproblem_grad_evals

            M, grad_phi, backtracking_grad_evals = self._ar_backtracking(
                problem=problem,
                x=x,
                y=y_new,
                y_bar=y_bar,
                sigma=sigma,
                M0=M_prev / 2.0,
            )
            n_grad_evals += backtracking_grad_evals

            if sigma >= M:
                return y_new.detach(), float(M), grad_phi.detach(), n_grad_evals

            s += 1
            y_prev = y_new.detach()
            y_bar_prev = y_bar.detach()
            sigma_prev = float(sigma)
            M_prev = float(M)
            grad_phi_prev = grad_phi.detach()

    def _ar_backtracking(
            self,
            problem,
            x,
            y,
            y_bar,
            sigma,
            M0,
            max_backtracking=100,
    ):
        M = float(M0)
        if M <= 0.0 or not math.isfinite(M):
            raise ValueError(f"M must be positive and finite, got {M}")
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError(f"sigma must be positive and finite, got {sigma}")

        grad_phi = -problem.grad_y(x, y)
        if not torch.isfinite(grad_phi).all():
            raise RuntimeError("Non-finite gradient in AR backtracking.")
        grad_phi_s = grad_phi + sigma * (y - y_bar)
        if not torch.isfinite(grad_phi_s).all():
            raise RuntimeError("Non-finite regularized gradient in AR backtracking.")

        phi_y = -problem.f(x, y)
        if not torch.isfinite(phi_y):
            raise RuntimeError("Non-finite function value in AR backtracking.")

        for _ in range(max_backtracking):
            if not math.isfinite(M) or not math.isfinite(2.0 * (M + sigma)):
                raise RuntimeError("Non-finite smoothness estimate in AR backtracking.")
            y_trial = (y - grad_phi_s / (2.0 * (M + sigma))).detach()
            if not torch.isfinite(y_trial).all():
                raise RuntimeError("Non-finite trial point in AR backtracking.")

            phi_trial = -problem.f(x, y_trial)
            diff = y_trial - y
            lhs = -problem.value_remainder_y(
                x, y, y_trial, -grad_phi, -phi_y, -phi_trial,
            )
            rhs = 0.5 * M * torch.sum(diff ** 2)
            if not torch.isfinite(lhs) or not torch.isfinite(rhs):
                raise RuntimeError("Non-finite value in AR backtracking.")
            if lhs <= rhs:
                return float(M), grad_phi.detach(), 1

            M *= 2.0

        raise RuntimeError("AR backtracking exceeded limit.")

    def _initialize_guesses(
            self,
            problem,
            x,
            y,
            grad_y,
    ):
        direction = torch.ones_like(y)
        direction_norm = torch.linalg.vector_norm(direction).item()
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise RuntimeError("Invalid secant direction.")
        direction = direction / direction_norm
        z = (y + direction).detach()

        denominator = torch.linalg.vector_norm(z - y).item()
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("Invalid secant denominator.")

        grad_z = problem.grad_y(x, z)
        if not torch.isfinite(grad_z).all():
            raise RuntimeError("Non-finite secant gradient.")
        numerator = torch.linalg.vector_norm(grad_z - grad_y).item()
        if not math.isfinite(numerator):
            raise RuntimeError("Non-finite secant numerator.")
        a = numerator / denominator

        if not math.isfinite(a) or a <= 0.0:
            raise RuntimeError(f"Invalid secant estimate: {a}")

        mu0 = float(a)
        M0 = float(a)

        return mu0, M0, 1

    def estimate_secant(self, problem, x, y, grad_y):
        estimate, _, n_grad_evals = self._initialize_guesses(
            problem=problem, x=x, y=y, grad_y=grad_y,
        )
        return estimate, n_grad_evals

    def initialize_persistent_state(self, problem, x, y, grad_y):
        estimate, n_grad_evals = self.estimate_secant(problem, x, y, grad_y)
        return {"nu": estimate, "M": estimate}, n_grad_evals

    def persistent_halving(
            self,
            problem,
            x,
            y,
            grad_y,
            state,
    ):
        old_residual = torch.linalg.vector_norm(grad_y).item()
        if not math.isfinite(old_residual) or not torch.isfinite(y).all():
            raise RuntimeError("Invalid initial point or gradient in persistent SCAR.")
        for key in ("nu", "M"):
            if not math.isfinite(state[key]) or state[key] <= 0.0:
                raise ValueError(f"Persistent SCAR {key} must be positive and finite.")
        if old_residual == 0.0:
            return y.detach(), grad_y.detach(), state, 0

        n_grad_evals = 0
        while True:
            y_candidate, M_new, grad_phi_candidate, ar_evals = self._ar(
                problem=problem,
                x=x,
                y0=y,
                sigma1=state["nu"] / 10.0,
                M0=state["M"],
                grad_phi0=-grad_y,
            )
            n_grad_evals += ar_evals
            state["M"] = float(M_new)
            grad_candidate = (-grad_phi_candidate).detach()
            residual = torch.linalg.vector_norm(grad_candidate).item()
            if not math.isfinite(residual) or not torch.isfinite(y_candidate).all():
                raise RuntimeError("Non-finite persistent SCAR candidate.")
            if not math.isfinite(state["M"]) or state["M"] <= 0.0:
                raise RuntimeError("Invalid persistent SCAR smoothness estimate.")
            if residual <= 0.5 * old_residual:
                return y_candidate.detach(), grad_candidate, state, n_grad_evals
            state["nu"] /= 4.0
            if state["nu"] == 0.0:
                raise RuntimeError("Persistent SCAR curvature guess underflowed.")

    def run(
            self,
            problem,
            x,
            y0,
            *,
            stop_rule,
            target,
            max_steps=100_000,
    ) -> InnerSolverResult:
        """max_steps and n_steps count SCAR stages; n_grad_evals counts oracle calls."""
        if stop_rule != "gradient_norm":
            raise ValueError("SCAR only supports 'gradient_norm' stop rule.")

        tau = float(target)
        if tau <= 0.0 or not math.isfinite(tau):
            raise ValueError(f"target must be positive and finite, got {target}")

        y = y0.clone().detach()
        n_steps = 0
        n_grad_evals = 0

        grad_y = problem.grad_y(x, y)
        n_grad_evals += 1
        residual = torch.linalg.vector_norm(grad_y).item()
        if not math.isfinite(residual):
            raise RuntimeError("Non-finite initial gradient residual in SCAR.")

        if residual <= tau:
            return InnerSolverResult(
                y=y,
                n_steps=n_steps,
                n_grad_evals=n_grad_evals,
                grad_y=grad_y.detach(),
                converged=True,
            )

        mu_guess, M, initialization_grad_evals = self._initialize_guesses(
            problem=problem,
            x=x,
            y=y,
            grad_y=grad_y,
        )
        n_grad_evals += initialization_grad_evals

        while n_steps < max_steps:
            old_y = y
            old_grad_y = grad_y
            old_residual = residual

            y_candidate, M_new, grad_phi_candidate, ar_grad_evals = self._ar(
                problem=problem,
                x=x,
                y0=old_y,
                sigma1=mu_guess / 10.0,
                M0=M,
                grad_phi0=-old_grad_y,
            )
            n_grad_evals += ar_grad_evals
            n_steps += 1

            grad_candidate = (-grad_phi_candidate).detach()
            candidate_residual = torch.linalg.vector_norm(grad_candidate).item()
            if not math.isfinite(candidate_residual):
                raise RuntimeError("Non-finite candidate gradient residual in SCAR.")
            M = float(M_new)

            if candidate_residual > 0.5 * old_residual:
                mu_guess = mu_guess / 4.0
                y = old_y
                grad_y = old_grad_y
                residual = old_residual
            else:
                y = y_candidate.detach()
                grad_y = grad_candidate
                residual = candidate_residual

            if residual <= tau:
                return InnerSolverResult(
                    y=y,
                    n_steps=n_steps,
                    n_grad_evals=n_grad_evals,
                    grad_y=grad_y.detach(),
                    converged=True,
                )

        return InnerSolverResult(
            y=y,
            n_steps=n_steps,
            n_grad_evals=n_grad_evals,
            grad_y=grad_y.detach(),
            converged=False,
        )
