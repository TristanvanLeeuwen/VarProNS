from pyunlocbox import functions, solvers, acceleration
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
import scipy.sparse as sparse
import scipy.linalg as linalg

class Example(object):
    """
    Abstract base class for examples

    Properties
    ----------

    Methods
    -------
    value               - compute value f(x,y)
    gradx,grady         - compute \nabla_x f(x,y), \nabla_y f(x,y)
    misfit              - returns objective f(x,y) as pyunlocbox function object
    inner_solve         - solve inner problem min_y f(x,y) + r_2(y) for given x and r_2
    plot_data           - plot data
    plot_results        - plot results
    """

    def val(self,x,y):
        raise NotImplementedError

    def gradx(self,x,y):
        raise NotImplementedError

    def grady(self,x,y):
        raise NotImplementedError

    def misfit(self,x = [],y = []):
        """
        Misfit f(x,y)

        Input
        -----
            x - array
            y - array

        Output
        ------
            f(x,[]) - pyunlocbox function object representing f(x,y) as a function of y
            f([],y) - pyunlocbox function object representing f(x,y) as a function of x
            f(x,y)  - value of f
        """

        if len(x) == 0:
            f = functions.func()
            f._eval = lambda x : self.val(x,y)
            f._grad = lambda x : self.gradx(x,y)
        elif len(y) == 0:
            f = functions.func()
            f._eval = lambda y : self.val(x,y)
            f._grad = lambda y : self.grady(x,y)
        else:
            f = self.val(x,y)
        return f

    def inner_solve(self,x,y0,Lyy = 1,r2 = functions.dummy(),rtol=1e-9, maxit=100000,verbosity='NONE'):
        """
        Solve min_y f(x,y) + r_2(y)

        Input
        -----
        x           - array
        y0          - initial iterate
        Lyy         - Lipschitz constant of f
        r2          - pyunlocbox function object
        rtol        - stopping criterion for solver
        maxit       - max iteration count for inner solve
        verbosity   - level of verbosity for inner solver

        Output
        ------
        y           - array
        """

        # get pyunlocbox function object for f(x,.)
        f = self.misfit(x = x)

        # setup solver
        accel = acceleration.fista_backtracking()
        solver = solvers.forward_backward(step=1/Lyy, accel=accel)

        # run solver
        results = solvers.solve([f, r2], y0, solver, rtol=rtol, maxit=maxit,verbosity=verbosity)

        # return result
        y = results['sol']

        return y

    def plot_data(self):
        raise NotImplementedError

    def plot_results(self):
        raise NotImplementedError


class ExpFit(Example):
    """
    # Exponential data fitting
    Measurement model
    $$b \sim \pi(A(x)y),$$
    with
    $$[A(x)]_{ij} = \exp( - x_j t_i ), \quad i = 1, \ldots m$$
    Let
    $$[G(x,y)]_{ij} = \frac{\partial b_i}{\partial x_j} = -t_iy_j\exp( - x_j t_i )$$
    and
    $$[R(x,y,r)]_{ij} = \sum_{k=1}^m\frac{\partial^2 b_k}{\partial x_i \partial x_j}r_k =
    \delta_{ij}y_i\sum_{k=1}^m t_k^2 \exp(-x_i t_k)r_k$$
    Then
    $$f(x,y) = \log(b!)^T1 + (A(x)y)^T1 - \log\left(A(x)y\right)^Tb,$$
    with
    $$\nabla_x f(x,y) = G(x,y)^T\left(1 - \text{diag}(A(x)y)^{-1}b\right)$$
    $$\nabla_y f(x,y) = A(x)^T\left(1 - \text{diag}(A(x)y)^{-1}b\right)$$

    Multiple experiments:
    $$b_k = \pi(A(x)y_k),$$
    with $t_i = (i-1) h$, $h = 5/(m-1)$, $m = 1000$, $x = (1,2,3,4)$, $y$ generated randomly.

    We identify the parameters by solving
    $$\min_{x,y} \sum_{i=k}^K\ell(A(x)y_k,b_k) + \delta_{\mathbb{R}_{+}^n}(y_k),$$
    with $\ell$ denoting the loss function.

    See:

    https://arxiv.org/pdf/1302.0441.pdf and
    https://stanford.edu/class/ee367/reading/lecture10_notes.pdf
    """
    def __init__(self,t,xt,yt,noise='gaussian',sigma=0):

        m = len(t)
        n = len(xt)
        k = len(yt)//n

        self.t = t
        self.m = m
        self.n = n
        self.k = k
        self.noise = noise
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=6)
        self.xt = xt
        self.yt = yt
        self.data = self.getData(xt,yt)

    def getA(self,x):
        """
        get matrix A(x) with elements a_{ij} = exp(-x_jt_i)

        input:
            x - array of length n
        output:
            A - matrix of size m x n
        """
        A = np.exp(np.outer(-self.t,x))
        return A

    def getG(self,x,y):
        """
        Compute d(A(x)y)/dx

        input:
            x - array of length n
            y - array of length n

        output:
            G - matrix of size m x n
        """
        A = self.getA(x)
        G = np.outer(-self.t,y)*A
        return G

    def getData(self,x,y):
        """
        Compute data b_i = A(x)y_i
        """
        A = self.getA(x)
        bt = np.zeros(self.m*self.k)
        for i in range(self.k):
            bt[i*self.m:(i+1)*self.m] = A.dot(y[i*self.n:(i+1)*self.n])
        if self.noise == 'gaussian':
            bn = self.rng.normal(bt,self.sigma)
        else:
            bn = self.rng.poisson(bt)
        return  bn

    def val(self,x,y):
        """
        Compute misfit \sum_{i=1}^k L(A(x)y_i,b_i)

        input:
            x - array of length n
            y - array of length k*n
        """
        f = 0
        A = self.getA(x)
        for i in range(self.k):
            bk = A@y[i*self.n:(i+1)*self.n]
            f += self.Loss(bk,self.data[i*self.m:(i+1)*self.m])[0]
        return f

    def gradx(self,x,y):
        """
        Compute gradient of misfit \sum_{i=1}^k L(A(x)y_i,b_i) w.r.t. x

        input:
            x - array of length n
            y - array of length k*n
        """

        g = np.zeros(self.n)
        A = self.getA(x)
        for i in range(self.k):
            Gk = self.getG(x,y[i*self.n:(i+1)*self.n])
            bk = A@y[i*self.n:(i+1)*self.n]
            g += Gk.T@self.Loss(bk,self.data[i*self.m:(i+1)*self.m])[1]
        return g

    def grady(self,x,y):
        """
        Compute gradient of misfit \sum_{i=1}^k L(A(x)y_i,b_i) w.r.t. y
        input:
            x - array of length n
            y - array of length k*n
        """
        g = np.zeros(self.n*self.k)
        A = self.getA(x)
        for i in range(self.k):
            bk = A@y[i*self.n:(i+1)*self.n]
            g[i*self.n:(i+1)*self.n] = A.T@self.Loss(bk,self.data[i*self.m:(i+1)*self.m])[1]

        return g

    def Loss(self,b1,b2):
        if self.noise == 'gaussian':
            f = 0.5*np.linalg.norm((b1-b2))**2
            g = (b1-b2)
        else:
            f = np.sum(gammaln(b2+1)) + np.sum(b1) - b2.dot(np.log(b1))
            g = (1 - b2/b1)
        return (f,g)

    def plot_data(self):
        fig,ax = plt.subplots(1,1)
        ax.plot(self.t,self.data,'k*')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$d(t)$')
        return fig,ax

    def plot_results(self,results,labels=[]):
        N = len(results)
        fig,ax = plt.subplots(2,N,sharex=True)
        if N > 1:
            i = 0
            ax[0,i].set_ylabel(r'$d(t)$')
            ax[1,i].set_ylabel(r'$\exp(-x_i t)$')
            for x,y in results:
                Ax = self.getA(x)
                ax[0,i].plot(self.t,self.data,'k*',self.t,Ax@y)
                if i > 0:
                    ax[0,i].set_yticks([])
                    ax[1,i].set_yticks([])
                for j in range(self.n):
                    if y[j] > 1e-1:
                        ax[1,i].plot(self.t,np.exp(-x[j]*self.t))
                    if self.yt[j] > 1e-1:
                        ax[1,i].plot(self.t,np.exp(-self.xt[j]*self.t),'k--')
                ax[1,i].set_xlabel(r'$t$')
                if labels:
                    ax[0,i].set_title(labels[i])
                i+=1
        else:
            x,y = results
            Ax = self.getA(x)
            for j in range(self.n):
                ax.plot(self.t,np.exp(-self.xt[j]*self.t),'k--')
                ax.plot(self.t,np.exp(-x[j]*self.t))
            ax.plot(t,data,'k*',t,Ax@y)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$d(t)$')
        return fig,ax

class Trimming(Example):
    """
    Solve $$\min_{x,y} \sum_{i=1}^n y_if_i(x) + (\beta/2)\|y\|_2^2 \quad \text{s.t.} \quad y \in \Delta_k,$$
    with
    $$\Delta_k = \{y\in[0,1]^n\,|\, 1^Ty = k\},$$
    denotes the capped simplex.

    We have
    $$f(x,y) = \sum_{i=1}^n y_if_i(x) + (\beta/2)\|y\|_2^2,$$
    with
    $$\nabla_x f(x,y) =  \sum_{i=1}^n y_i\nabla f_i(x),$$
    $$\nabla_y f(x,y) = (f_1(x), \ldots, f_n(x)).$$
    And
    $$r_2(x) = \delta_{\Delta_k}(y),$$
    where the projection solves
    $$\text{proj}_{\Delta_k}(y) = \min_{z\in\Delta_k} \|z - y\|_2^2.$$

    For the experiments we'll aim to find the mean of a number of samples, drawn from a Gaussian;

    $$f_i(x) = (1/2)\|x - d_i\|_2^2.$$

    We draw $k$ samples from one distribution while drawing $n-k$ from another to simulate outliers.
    """
    def __init__(self, n, k, mu1, mu2, sigma1, sigma2, beta):
        self.m = len(mu1)
        self.n = n
        self.k = k
        self.beta=beta
        self.rng = np.random.default_rng(seed=6)

        self.data = self.getData(mu1, mu2, sigma1, sigma2)

    def getData(self,mu1, mu2, sigma1, sigma2):
        d = np.zeros((self.n,self.m))
        d[:self.k] = self.rng.normal(mu1,sigma1,size=(self.k,self.m))
        d[self.k:] = self.rng.normal(mu2,sigma2,size=(self.n-self.k,self.m))
        return d

    def val(self,x,y):
        f = (self.beta/2)*np.linalg.norm(y)**2
        for i in range(self.n):
            f += y[i]*(1/2)*np.linalg.norm(x - self.data[i])**2
        return f

    def gradx(self,x,y):
        gx = np.zeros(self.m)
        for i in range(self.n):
            gx += y[i]*(x - self.data[i])
        return gx

    def grady(self,x,y):
        gy = self.beta*y
        for i in range(self.n):
            gy[i] += 0.5*np.linalg.norm(x - self.data[i])**2
        return gy


    def plot_results(self,results,labels=[]):
        N = len(results)
        fig,ax = plt.subplots(1,N,sharey=True)
        if N > 1:
            i = 0
            for x,y in results:
                ax[i].plot(self.data[:,0],self.data[:,1],'k*')
                ax[i].plot(self.data[y>0,0],self.data[y>0,1],'r+')
                ax[i].plot(x[0],x[1],'w+')
                ax[i].set_aspect('equal')
                if labels:
                    ax[i].set_title(labels[i])
                i+=1
        else:
            ax.plot(self.data[:,0],self.data[:,1],'k*')
            ax.plot(self.data[y>0,0],self.data[y>0,1],'r+')
            ax.plot(x[0],x[1],'w+')
            ax.set_aspect('equal')
        return fig,ax

class Tomography(Example):

    def __init__(self,s,theta,xt,yt,noise='gaussian',sigma=0):

        m = len(s)
        n = int(np.sqrt(len(yt)))
        k = len(theta)

        self.xt = xt
        self.yt = yt

        self.m = m
        self.n = n
        self.k = k

        self.theta = theta
        self.s = s

        self.noise = noise
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=6)
        self.kernel = lambda s : np.exp(-500*s**2)
        self.dkernel = lambda s : -1000*s*np.exp(-500*s**2)
        grid_x,grid_y = np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))

        grid_s,grid_theta = np.meshgrid(s,theta)

        self.grid_x = grid_x.flatten()
        self.grid_y = grid_y.flatten()

        self.grid_theta = grid_theta.flatten()
        self.grid_s = grid_s.flatten()

        self.data = self.getData()

    def getA(self,x):
        """
        get matrix A(x) with elements a_{ij} = k()

        input:
            x - array of length 2*k
        output:
            A - matrix of size k*m x n*n
        """

        A = np.zeros((self.m*self.k,self.n*self.n))

        for i in range(self.k):
            theta = self.theta + x[i]
            s = self.s + x[i + self.k]
            for j in range(self.m):
                A[len(s)*i + j,:] = self.kernel(self.grid_x*np.cos(theta[i]) + self.grid_y*np.sin(theta[i]) + s[j])
        return A

    def getG(self,x,y):
        """
        Compute d(A(x)y)/dx

        input:
            x - array of length 2*k
            y - array of length n*n

        output:
            G - matrix of size k*m x 2*k
        """

        G = np.zeros((self.k*self.m,2*self.k))

        for i in range(self.k):
            theta = self.theta + x[i]
            s = self.s + x[i + self.k]
            for j in range(self.m):
                dtheta = self.dkernel(self.grid_x*np.cos(theta[i]) + self.grid_y*np.sin(theta[i]) + s[j])*(-self.grid_x*np.sin(theta[i]) + self.grid_y*np.cos(theta[i]))
                G[len(s)*i + j,i] = dtheta.dot(y)
                ds = self.dkernel(self.grid_x*np.cos(theta[i]) + self.grid_y*np.sin(theta[i]) + s[j])
                G[len(s)*i + j,i + self.k] = ds.dot(y)

        return G

    def getData(self):
        """
        Compute data b_i = A(x)y_i
        """
        A = self.getA(self.xt)
        bt = A.dot(self.yt)

        if self.noise == 'gaussian':
            bn = self.rng.normal(bt,self.sigma)
        else:
            bn = self.rng.poisson(bt)
        return  bn

    def val(self,x,y):
        """
        Compute misfit

        input:
            x - array of length 2*k
            y - array of length n*n
        """
        A = self.getA(x)
        f = self.Loss(A.dot(y),self.data)[0]

        return f

    def gradx(self,x,y):
        """
        Compute gradient of misfit w.r.t. x

        input:
            x - array of length 2*k
            y - array of length n*n
        """
        A = self.getA(x)
        G = self.getG(x,y)
        g = G.T.dot(self.Loss(A.dot(y),self.data)[1])

        return g

    def grady(self,x,y):
        """
        Compute gradient of misfit w.r.t. y
        input:
            x - array of length 2*k
            y - array of length n*n
        """
        A = self.getA(x)
        g = A.T.dot(self.Loss(A.dot(y),self.data)[1])

        return g

    def Loss(self,b1,b2):
        if self.noise == 'gaussian':
            f = 0.5*np.linalg.norm((b1-b2))**2
            g = (b1-b2)
        else:
            f = np.sum(gammaln(b2+1)) + np.sum(b1) - b2.dot(np.log(b1))
            g = (1 - b2/b1)
        return (f,g)

    def plot_data(self):
        fig,ax = plt.subplots(1,1)
        ax.imshow(self.data.reshape((self.k,self.m)).T,extent=(self.theta[0],self.theta[-1],self.s[0],self.s[-1]))
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$s$')
        ax.set_aspect(2)
        return fig,ax

    def plot_results(self,results,labels=[]):
        N = len(results)
        fig,ax = plt.subplots(1,N,sharey=True)
        if N > 1:
            for i in range(N):
                x,y = results[i]
                ax[i].imshow(y.reshape((self.n,self.n)),extent=(0,1,0,1),vmin=0,vmax=1)
                if labels:
                    ax[i].set_title(labels[i])
        else:
            x,y = results[0]
            ax.imshow(y.reshape((self.n,self.n)),extent=(0,1,0,1),vmin=0,vmax=1)
        return fig,ax
