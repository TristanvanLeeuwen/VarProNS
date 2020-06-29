from pyunlocbox import functions
import numpy as np

class pos(functions.proj):
    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(pos, self).__init__(**kwargs)
    def _prox(self,x,T):
        return np.array([_ if _ > 0 else 0 for _ in x])

class incr(functions.proj):
    def __init__(self, delta=0,**kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(incr, self).__init__(**kwargs)
        self.delta = delta
    def _prox(self,x,T):
        n = len(x)
        S = np.tril(np.ones((n,n)))
        sol = opt.lsq_linear(S,x,bounds=(self.delta,np.inf))
        return S@sol.x
    
class norm_l1_pos(functions.norm):
    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(norm_l1_pos, self).__init__(**kwargs)

    def _eval(self, x):
        sol = self.A(x) - self.y()
        return self.lambda_ * np.sum(np.abs(self.w * sol))

    def _prox(self, x, T):
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        if self.tight:
            # Nati: I've checked this code the use of 'y' seems correct
            sol = self.A(x) - self.y()
            sol[:] = functions._soft_threshold(sol, gamma * self.nu * self.w) - sol
            sol[:] = x + self.At(sol) / self.nu
            sol[sol<0] = 0
        else:
            raise NotImplementedError('Not implemented for non-tight frame.')
        return sol
    
class capped_simplex(functions.proj):
    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(capped_simplex, self).__init__(**kwargs)
        self.k = kwargs['k_']
    def _prox(self,x,T):
        return capped_simplex.project(x,self.k)
    
    @staticmethod
    def project(y,s):
        """Projection onto the capped simplex
        min_x \|x - y\|_2^2 s.t. x \in [0,1]^n, <1,x> = s
        
        input: 
            y - array of length n
            s - parameter \in [0,n]
            
        output:
            x - array of length n
            
        reference:
            Weiran Wang: "Projection onto the capped simplex". March 3, 2015, arXiv:1503.01002.
        """
        n = len(y)
        # check some base case
        if s>n:
            raise ValueError('problem is not feasible')
        elif s==0:
            return np.zeros(n)
        elif s==n:
            return np.ones(n)
        # sort and concatenate to get -oo, y_1, y_2, ..., y_{n}, oo
        idx = np.argsort(y)
        ys = np.concatenate(([-np.inf],y[idx],[np.inf]))
        x = np.zeros(n)
        # cumsum and concatenate
        T = np.concatenate(([0],np.cumsum(ys[1:])))
        # main loop a = 0, ..., n+1
        for a in range(0,n+2):
            if s == (n - a) and ys[a+1] - ys[a] >= 1:
                b = a
                x[idx] = np.concatenate((np.zeros(a),ys[a+1:b+1] + gamma,np.ones(n-b)))
                return x
            # inner loop b = a+1, ..., n
            for b in range(a+1,n+1):
                gamma = (s + b - n + T[a] - T[b])/(b - a)
                if (ys[a] + gamma <= 0) and (ys[a+1] + gamma > 0) and (ys[b] + gamma < 1) and (ys[b+1] + gamma >= 1):
                    x[idx] = np.concatenate((np.zeros(a),ys[a+1:b+1] + gamma,np.ones(n-b)))
                    return x
                
class TV(functions.norm_tv):
    def __init__(self,**kwargs):
        super(TV, self).__init__(**kwargs)
        self.n_ = kwargs['n_']
        self.verbosity = 'NONE'
    def _eval(self,x):
        val = super(TV,self)._eval(x.reshape((self.n_,self.n_)))
        return self.lambda_ * val
    def _prox(self,x,T):
        gamma = self.lambda_ * T
        return super(TV,self)._prox(x.reshape((self.n_,self.n_)),gamma).flatten()