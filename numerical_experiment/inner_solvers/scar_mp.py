"""Ordinary SCAR in an isolated mpmath context; no AR early stopping."""


class SCARMP:
    def __init__(self, ctx, max_attempts=8, max_steps=100_000):
        self.c = ctx
        self.max_attempts, self.max_steps = max_attempts, max_steps
        self.failures = 0

    def secant(self, p, x, y, g):
        c = self.c
        z = y + c.matrix([1/c.sqrt(len(y))]*len(y))
        a = c.norm(p.grad_y(x,z)-g)/c.norm(z-y)
        if not c.isfinite(a) or a <= 0:
            raise RuntimeError('Invalid MP secant')
        return a

    def _line_ok(self, p, x, y, yn, gq, L):
        # Count actual function queries; use the stable analytic remainder
        # instead of subtracting close values.
        p.f(x,y); p.f(x,yn)
        return p.q_remainder(y,yn) <= L*self.c.norm(yn-y)**2/2

    def _subsolve(self, p, x, y0, bar, sigma, L, g0):
        c = self.c
        y, v, theta = y0.copy(), y0.copy(), c.mpf(1)
        for k in range(1,self.max_steps+1):
            g = g0 if k == 1 and g0 is not None else -p.grad_y(x,v)
            for _ in range(100):
                yn = v + (-g+sigma*(bar-v))/(L+sigma)
                if self._line_ok(p,x,v,yn,g,L): break
                L *= 2
            else: raise RuntimeError('MP SCAR subproblem backtracking exhausted')
            if k >= c.ceil(8*c.sqrt(4*L/sigma)): return yn
            tn = (1+c.sqrt(1+4*theta**2))/2
            v = yn+(theta-1)/tn*(yn-y)
            y,theta = yn,tn
        raise RuntimeError('MP SCAR subproblem step budget exhausted')

    def _ar(self,p,x,u,g,nu,M):
        c = self.c
        yp,bar,previous,gp = u.copy(),u.copy(),c.mpf(0),-g
        sigma = nu/10
        for _ in range(100):
            gamma = 1-previous/sigma
            bar = (1-gamma)*bar+gamma*yp
            yn = self._subsolve(p,x,yp,bar,sigma,M/2,gp)
            gn = -p.grad_y(x,yn)
            gs = gn+sigma*(yn-bar)
            M /= 2
            for _ in range(100):
                trial = yn-gs/(2*(M+sigma))
                if self._line_ok(p,x,yn,trial,gn,M): break
                M *= 2
            else: raise RuntimeError('MP SCAR AR backtracking exhausted')
            if sigma >= M: return yn,M,-gn
            yp,gp,previous = yn,gn,sigma
            sigma *= 4
        raise RuntimeError('MP SCAR AR stage budget exhausted')

    def halve(self,p,x,y,g,state):
        R = self.c.norm(g)
        if R == 0: return y,g
        for _ in range(self.max_attempts):
            yn,M,gn = self._ar(p,x,y,g,state['nu'],state['M'])
            state['M'] = M
            if self.c.norm(gn) <= R/2: return yn,gn
            self.failures += 1
            state['nu'] /= 4
        raise RuntimeError('MP SCAR halving attempt budget exhausted; entry residual=' +
                           self.c.nstr(R,18) + '; increase precision or inspect curvature guesses')
