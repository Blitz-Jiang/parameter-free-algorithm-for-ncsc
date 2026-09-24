"""Fixed-count UTR5 with mpmath arithmetic for the full optimization run.

Requires an explicit high-precision problem factory. Conversion to torch occurs
only at input/output; exact decimal output is also saved in history.
"""
import math
import torch
from .base import NCSCAlgorithm, AlgorithmResult
from inner_solvers.scar_mp import SCARMP
from subproblem_solvers.trs_mp import TRSubproblemSolverMP


class UTR5FixedMP(NCSCAlgorithm):
    def __init__(self, problem, oracle_factory, epsilon, max_iterations=100_000,
                 dps=80, max_halving_attempts=8):
        super().__init__(problem)
        import mpmath
        if int(dps) != dps or dps < 40:
            raise ValueError('MP backend requires integer dps >= 40')
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError('epsilon must lie in (0,1)')
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        if int(max_halving_attempts) != max_halving_attempts or max_halving_attempts < 1:
            raise ValueError('max_halving_attempts must be a positive integer')
        self.c = mpmath.mp.clone(); self.c.dps = int(dps)
        self.epsilon, self.max_iterations = float(epsilon), int(max_iterations)
        self.oracle_factory = oracle_factory
        self.inner_solver = SCARMP(self.c,max_attempts=int(max_halving_attempts))
        self.trust_region_solver = TRSubproblemSolverMP(self.c)

    def run(self,x0,y0):
        c = self.c
        self.oracle = p = self.oracle_factory(c)
        self.inner_solver.failures = 0
        x,y = c.matrix(x0.detach().cpu().tolist()),c.matrix(y0.detach().cpu().tolist())
        eps = c.mpf(self.epsilon); root = c.sqrt(eps); eps32 = eps*root
        sigma = 1/(1-c.log(eps)); c0 = c.mpf(1)/4096
        state = None
        ntrial = naccepted = nrejected = nvalid = nhalf = 0
        history = {key: [] for key in ['inner_grad_evals','inner_phase','inner_residual',
                  'tracking_halvings','working_halvings_completed','trial_status','sigma',
                  'grad_norm','radius','step_norm','multiplier','value','W','U','T']}
        history.update(arithmetic='mpmath',decimal_digits=c.dps,tracking_mode='fixed_count',
                       oracle_backend='analytic CosLogCosh',max_halving_attempts=self.inner_solver.max_attempts)

        def tracked(phase, fn):
            before = p.grad_calls
            try: return fn()
            finally:
                history['inner_grad_evals'].append(p.grad_calls-before)
                history['inner_phase'].append(phase)

        def halve(xx,yy,gg,phase):
            nonlocal nhalf
            R = c.norm(gg)
            yn,gn = tracked(phase,lambda: self.inner_solver.halve(p,xx,yy,gg,state))
            if c.norm(gn) > R/2:
                raise RuntimeError('MP successful-halving contract failed')
            nhalf += 1
            history['inner_residual'].append(c.nstr(c.norm(gn),c.dps))
            return yn,gn

        def refine(xx,yy,gg,target,phase):
            while c.norm(gg) > target:
                yy,gg = halve(xx,yy,gg,phase)
            return yy,gg

        def finish(ok,reason):
            history.update(n_accepted=naccepted,n_rejected=nrejected,n_validators=nvalid,
                           n_successful_halvings=nhalf,n_failed_curvature_guesses=self.inner_solver.failures,
                           n_oracle_builds=p.derivative_calls,termination_reason=reason,
                           final_sigma=float(sigma),x_mp=[c.nstr(t,c.dps) for t in x],
                           y_mp=[c.nstr(t,c.dps) for t in y],
                           persistent_state=None if state is None else {k:c.nstr(v,c.dps) for k,v in state.items()})
            return AlgorithmResult(x=torch.tensor([float(t) for t in x],dtype=x0.dtype,device=x0.device),
                y=torch.tensor([float(t) for t in y],dtype=y0.dtype,device=y0.device),
                n_iterations=ntrial,n_inner_grad_evals=p.grad_calls,n_subproblem_solves=ntrial,
                converged=ok,history=history)

        try:
            gy = tracked('initial_gradient',lambda:p.grad_y(x,y))
            a = tracked('initial_secant',lambda:self.inner_solver.secant(p,x,y,gy))
            state = dict(nu=a,M=a)
            y,gy = refine(x,y,gy,c0*a*root/(4*sigma),'initial_refinement')
            val,g,H = p.corrected(x,y,gy)
            while ntrial < self.max_iterations:
                h = max(c.norm(g),eps); R = c.sqrt(h)/(4*sigma)
                B = H+(sigma*c.sqrt(h)+sigma*root/64)*c.eye(len(x))
                ntrial += 1
                history['trial_status'].append('failed')
                d,lam,r,boundary = self.trust_region_solver.solve(g,B,R)
                W,U,T = sigma*c.sqrt(h)*r*r,lam*r*r,eps32/sigma
                N = int(c.ceil(c.log1p(r/(c0*root/(4*sigma)))/c.log(2)))
                for key,v in [('sigma',sigma),('grad_norm',c.norm(g)),('radius',R),('step_norm',r),
                              ('multiplier',lam),('value',val),('W',W),('U',U),('T',T)]:
                    history[key].append(float(v))
                history['tracking_halvings'].append(N)
                history['working_halvings_completed'].append(0)
                xp,yp = x+d,y
                gp = gy if r == 0 else tracked('candidate_gradient',lambda:p.grad_y(xp,yp))
                for _ in range(N):
                    yp,gp = halve(xp,yp,gp,'working_halving')
                    history['working_halvings_completed'][-1] += 1
                if h == eps and not boundary:
                    nvalid += 1
                    av = tracked('validator_secant',lambda:self.inner_solver.secant(p,xp,yp,gp))
                    yv,gv = refine(xp,yp,gp,c0*av*eps32,'validation_halving')
                    _,vg,vH = p.corrected(xp,yv,gv,with_value=False)
                    if c.norm(vg) <= eps/2 and c.eigsy(vH,eigvals_only=True)[0] >= -c.mpf(21)/8*sigma*root:
                        x,y = xp,yv
                        history['trial_status'][-1] = 'validated'
                        return finish(True,'MP fixed-count UTR5 interior validation satisfied')
                    rejection = 'validation_rejected'
                else:
                    vp,gg,HH = p.corrected(xp,yp,gp)
                    if val-vp >= W/8+U/4-T/1024 and c.norm(gg) <= h/2+lam*r:
                        x,y,gy,val,g,H = xp,yp,gp,vp,gg,HH
                        naccepted += 1
                        history['trial_status'][-1] = 'accepted'
                        continue
                    rejection = 'rejected'
                nrejected += 1
                history['trial_status'][-1] = rejection
                sigma *= 2
                y,gy = halve(x,y,gy,'refresh_halving')
                val,g,H = p.corrected(x,y,gy)
            return finish(False,'Maximum MP UTR5 trial count reached')
        except (RuntimeError,ValueError,ZeroDivisionError,OverflowError) as exc:
            return finish(False,str(exc))
