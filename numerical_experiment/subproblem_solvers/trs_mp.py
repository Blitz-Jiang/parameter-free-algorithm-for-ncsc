"""Dense eigenspace TR solver in the supplied mpmath context."""


class TRSubproblemSolverMP:
    def __init__(self, ctx):
        self.c = ctx
        self.tol = ctx.power(10,-min(40,ctx.dps//2))

    def solve(self,g,H,R):
        c,tol = self.c,self.tol
        vals,V = c.eigsy((H+H.T)/2)
        gh = V.T*g
        scale = max(c.mpf(1),c.norm(H))
        low = max(c.mpf(0),-vals[0])
        null = [i for i in range(len(g)) if abs(vals[i]+low) <= tol*scale]
        sh = c.matrix([0 if i in null else -gh[i]/(vals[i]+low) for i in range(len(g))])
        if c.sqrt(sum(gh[i]**2 for i in null)) <= tol*max(c.mpf(1),c.norm(g)) and c.norm(sh) <= R:
            if low > 0:
                sh[null[0]] += c.sqrt(max(c.mpf(0),R*R-c.norm(sh)**2))
            return self._verify(g,H,R,V*sh,low)
        def step(lam): return c.matrix([-gh[i]/(vals[i]+lam) for i in range(len(g))])
        high = max(c.mpf(1),2*low)
        for _ in range(1000):
            if c.norm(step(high)) <= R: break
            high *= 2
        else: raise RuntimeError('MP TR bracket failed')
        for _ in range(8*c.dps+100):
            lam = (low+high)/2
            sh = step(lam)
            if abs(c.norm(sh)-R) <= tol*R:
                return self._verify(g,H,R,V*sh,lam)
            if c.norm(sh) > R: low = lam
            else: high = lam
        raise RuntimeError('MP TR root solve exhausted')

    def _verify(self,g,H,R,s,lam):
        c,tol = self.c,self.tol*100
        norm = c.norm(s)
        shifted = H+lam*c.eye(len(g))
        residual = c.norm(shifted*s+g)/max(c.mpf(1),c.norm(g)+c.norm(shifted)*norm)
        if (residual > tol or norm > R*(1+tol) or lam < 0
            or c.eigsy(shifted,eigvals_only=True)[0] < -tol*max(c.mpf(1),c.norm(shifted))
            or abs(lam*(norm-R)) > tol*max(c.mpf(1),abs(lam)*R)):
            raise RuntimeError('MP TR KKT verification failed')
        boundary = lam > 0 or abs(norm-R) <= tol*R
        return s,lam,norm,boundary
