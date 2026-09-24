"""Explicit arbitrary-precision backend for the canonical CosLogCosh payoff.

Inputs are the exact values of the existing floating-point problem coefficients.
This is an analytic backend, not generic mpmath autograd.
"""


class CosLogCoshMP:
    def __init__(self, problem, ctx):
        self.c = ctx
        self.Q = ctx.matrix(problem.Q.detach().cpu().tolist())
        self.mu, self.beta, self.a, self.omega = map(ctx.mpf,
            [problem.mu_y,problem.beta,problem.a_mu,problem.omega])
        self.grad_calls = self.function_calls = self.derivative_calls = 0

    def grad_y(self,x,y):
        self.grad_calls += 1
        c = self.c
        return self.a*self.Q.T*x-self.mu*y-self.beta*c.matrix([c.tanh(t) for t in y])

    def f(self,x,y):
        self.function_calls += 1
        c = self.c
        # -2 sin^2(t/2) avoids cos(t)-1 cancellation near zero.
        return (sum(-2*c.sin(self.omega*t/2)**2/self.omega**2 for t in x)
                +self.a*(x.T*self.Q*y)[0]-self.mu*(y.T*y)[0]/2
                -self.beta*sum(c.log(c.cosh(t)) for t in y))

    def q_remainder(self,y,yn):
        c = self.c
        total = c.mpf(0)
        for yi,di in zip(y,yn-y):
            ai = -2*di if yi >= 0 else 2*di
            qi = 1/(1+c.exp(2*abs(yi)))
            w = qi*c.expm1(ai)
            if abs(ai) < c.mpf('0.01'):
                # Stable exp(a)-1-a and log(1+w)-w with precision-scaled sums.
                er = lr = c.mpf(0)
                et,lt = ai*ai/2,-w*w/2
                for k in range(2, 4*c.dps+100):
                    er += et; lr += lt
                    if (abs(et) <= c.eps*abs(er) and abs(lt) <= c.eps*abs(lr)):
                        break
                    et *= ai/(k+1)
                    lt *= -w*k/(k+1)
                else: raise RuntimeError('MP remainder series did not converge')
                rem = qi*er+lr
            else:
                rem = c.log1p(w)-qi*ai
            total += self.mu*di*di/2+self.beta*rem
        return total

    def corrected(self,x,y,gy,with_value=True):
        c = self.c
        self.derivative_calls += 1
        diagonal = [-self.mu-self.beta*(1-c.tanh(t)**2) for t in y]
        inverse_g = c.matrix([g/d for g,d in zip(gy,diagonal)])
        gx = -c.matrix([c.sin(self.omega*t)/self.omega for t in x])+self.a*self.Q*y
        g = gx-self.a*self.Q*inverse_g
        H = c.diag([-c.cos(self.omega*t) for t in x])-self.a**2*self.Q*c.diag([1/d for d in diagonal])*self.Q.T
        val = self.f(x,y)-(gy.T*inverse_g)[0]/2 if with_value else None
        return val,g,H
