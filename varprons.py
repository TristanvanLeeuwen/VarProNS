import numpy as np
from pyunlocbox import functions, solvers
import matplotlib.pyplot as plt

def algorithm1(f,r1,r2,x0,y0,L = (1,1),maxit = 1000,tol = 1e-3):
    """
    Joint proximal gradient method for problems of the form

    min_{x,y} f(x,y) + r_1(x) + r_2(y)

    with $f$ Lipschitz smooth and r_1, r_2 are convex and proxible.

    Input
    -----
    f           - function returning a pyublocbox function opbject; f(x,[]) returns a function of y and f([],y) a function of x
    r_1, r_2    - pyublocbox function opbjects
    x0,y0       - initial iterates
    L           - (Lx,Ly), Lipschitz constants of \nabla_x f and \nabla_y f
    maxit       - max. number of iterations
    tol         - stopping tolerance

    Output
    -----
    result      - dictionary with fields    'objective': convergence history in terms of f;
                                            'dx': convergence history in terms of optimality
                                            'cost': commulative cost in terms of total number of iterations
                                            'sol': array with final iterates
    """
    xk = np.copy(x0)
    yk = np.copy(y0)
    res = {'objective':[],'sol':[],'dx':[],'cost':[]}
    alpha = 1/L[0]
    beta = 1/L[1]

    for k in range(maxit):


        fk = f([],yk).eval(xk) + r1.eval(xk) + r2.eval(yk)

        gx,gy = f([],yk).grad(xk),f(xk,[]).grad(yk)
        xp,yp = r1.prox(xk - alpha*gx,alpha), r2.prox(yk - beta*gy,beta)
        dx,dy = xp - xk, yp - yk
        xk,yk = xp,yp

        res['objective'].append(fk)
        res['dx'].append(np.sqrt((np.linalg.norm(dx)/alpha)**2 + (np.linalg.norm(dy)/beta)**2))
        res['cost'].append(2)

        if res['objective'][-1] <= tol:
            break

        xk = xp
        yk = yp

    res['sol'] = (xk,yk)
    return res

def algorithm2(f,r1,r2,x0,y0,L = (1,1),maxit = [1000,1000],tol = [1e-3,1e-3],warmstart=True,):
    """
    Variable projection method for problems of the form

    min_{x,y} f(x,y) + r_1(x) + r_2(y)

    with $f$ Lipschitz smooth and r_1, r_2 are convex and proxible.

    Input
    -----
    f           - function returning a pyublocbox function opbject; f(x,[]) returns a function of y and f([],y) a function of x
    r_1, r_2    - pyublocbox function opbjects
    x0,y0       - initial iterates
    L           - (Lx,Ly), Lipschitz constants of \nabla_x f and \nabla_y f
    maxit       - (maxit_outer, maxit_inner) max. number of outer/inner iterations
    tol         - (tol_outer,tol_inner) stopping tolerance for outer/inner iterations
    warmstart   - boolean, use warm start for inner iterations

    Output
    -----
    result      - dictionary with fields    'objective': convergence history in terms of f;
                                            'dx': convergence history in terms of optimality
                                            'cost': commulative cost in terms of total number of iterations
                                            'sol': array with final iterates
    """

    # initialize
    xk = np.copy(x0)
    yk = np.copy(y0)
    res = {'objective':[],'sol':[],'dx':[],'cost':[]}

    # stepsizes
    alpha = 1/L[0]
    beta = 1/L[1]

    # start outer loop
    for k in range(maxit[0]):

        # start inner loop to solve min_y f(x_k,y) + r2(y)
        yk = yk if warmstart else np.copy(y0)
        epsk = tol[1] if isinstance(tol[1],float) else tol[1](k)
        for l in range(maxit[1]):
            gy = f(xk,[]).grad(yk)
            yp = r2.prox(yk - beta*gy,beta)
            dy = yp - yk
            yk = yp
            if np.linalg.norm(dy)/beta <= epsk:
                break

        # evaluate reduced objective
        fk = f([],yk).eval(xk) + r1.eval(xk) + r2.eval(yk)
        gx = f([],yk).grad(xk)

        # update x
        xp = r1.prox(xk - alpha*gx,alpha)
        dx = xp - xk
        xk = xp

        # keep track
        res['objective'].append(fk)
        res['dx'].append(np.linalg.norm(dx)/alpha)
        res['cost'].append(l + 2)

        # check convergence
        if res['objective'][-1] <= tol[0]:
            break

    res['sol'] = (xk,yk)
    return res

def algorithm3(f,r1,r2,x0,y0,L = (1,1),rho = 1, maxit = [1000,1000],tol = 1e-3,warmstart=True,):
    """
    Adaptive variable projection method for problems of the form

    min_{x,y} f(x,y) + r_1(x) + r_2(y)

    with $f$ Lipschitz smooth and r_1, r_2 are convex and proxible.

    Input
    -----
    f           - function returning a pyublocbox function opbject; f(x,[]) returns a function of y and f([],y) a function of x
    r_1, r_2    - pyublocbox function opbjects
    x0,y0       - initial iterates
    L           - (Lx,Ly), Lipschitz constants of \nabla_x f and \nabla_y f
    rho         - parameter for adaptive stopping criterion
    maxit       - (maxit_outer, maxit_inner) max. number of outer/inner iterations
    tol         - stopping tolerance for outer iterations
    warmstart   - boolean, use warm start for inner iterations

    Output
    -----
    result      - dictionary with fields    'objective': convergence history in terms of f;
                                            'dx': convergence history in terms of optimality
                                            'cost': commulative cost in terms of total number of iterations
                                            'sol': array with final iterates
    """
    # initialize
    xk = np.copy(x0)
    yk = np.copy(y0)
    res = {'objective':[],'sol':[],'dx':[],'cost':[]}

    # stepsizes
    alpha = 1/L[0]
    beta = 1/L[1]

    # start outer loop
    for k in range(maxit[0]):

        # start inner loop to solve min_y f(x_k,y) + r2(y)
        yk = yk if warmstart else np.copy(y0)
        for l in range(maxit[1]):
            # update x
            gx = f([],yk).grad(xk)
            xp = r1.prox(xk - alpha*gx,alpha)
            dx = xp - xk

            # update y
            gy = f(xk,[]).grad(yk)
            yp = r2.prox(yk - beta*gy,beta)
            dy = yp - yk
            yk = yp

            # check convergence
            if np.linalg.norm(dy) <= rho*np.linalg.norm(dx):
                break

        # update x
        xk = xp
        # evaluate reduced objective
        fk = f([],yk).eval(xk) + r1.eval(xk) + r2.eval(yk)

        # keep track
        res['objective'].append(fk)
        res['dx'].append(np.linalg.norm(dx)/alpha)
        res['cost'].append(l + 2)

        # check convergence
        if res['objective'][-1] <= tol:
            break

    res['sol'] = (xk,yk)
    return res

def plot_convergence(results, labels, reference = np.float64(0)):
    """
    Function to plot convergence results
    """
    fig,ax = plt.subplots(1,2,sharey=True)
    for result,label in zip(results,labels):
        niter = len(result['objective'])
        k = np.linspace(0,niter-1,niter)
        ax[0].semilogy(k+1,result['objective'] - reference,label=label)
        ax[1].semilogy(np.cumsum(result['cost']),result['objective'] - reference,label=label)

    ax[0].set_xlabel('iteration')
    if np.abs(reference) > 0:
        ax[0].set_ylabel(r'$f_k - f_*$')
    else:
        ax[0].set_ylabel(r'$f_k$')
    ax[0].legend()
    ax[1].set_xlabel('cost')
    return fig, ax
