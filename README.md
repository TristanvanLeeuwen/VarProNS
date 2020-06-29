# VarProNS
Python code accompanying the paper "Non-smooth Variable Projection" by T. van Leeuwen and A. Aravkin. Submitted to SISC.

The algorithms are designed to solve problems of the form
$$
\min_{x,y} f(x,y) + r_1(x) + r_2(y),
$$
where $f$ is Liptschitz-smooth and $r_1, r_2$ are (convex) regularisation terms.

Algorithm 1 solves it using
\[
x_{k+1} = \text{prox_{\alpha r_1}}(x_k - \alpha \nabla_x f(x_k,y_k)),
\]
\[
y_{k+1} = \text{prox_{\beta r_2}}(y_k - \alpha \nabla_y f(x_k,y_k)).
\]

Algorithm 2 and 3 essentially recast the problem as
\[
\min_x \overline{f}(x) + r_1(x),
\]
where $\overline{f}(x) = \min_y f(x,y) + r_2(y)$.
We can solve this without explicitly forming \overline{f} as
\[
x_{k+1} = \text{prox_{\alpha r_1}}(x_k - \alpha \nabla_x f(x_k,y_k)),
y_{k+1} = \text{argmin_y} f(x_{k+1},y) + r_2(y).
\]
The main difference between algorithms 2 and 3 being that the latter solves the inner problem in $y$ inexactly.

The main benefits of the VP approach over the naïve joint approach are

* The conditioning of $\overline{f}$ is generally much better than that of $f$, and at least not worse. This leads to a faster convergence of algorithms 2,3 as compared to algorithm 1.
* The inexact method (algorithm 3) generally requires much less computation time than the exact version (algorithm 2).
* Efficient solvers for the inner problem can be exploited easily.

## Contents

* `varprons.py` : implementation of Algorithms 2.1, 2.2 and 2.3
* `./applicatons` : Jupyter notebooks containing numerical examples presented in the paper

## Requirements

The implementation used (amongst others) the [`pyunlocbox`](https://pyunlocbox.readthedocs.io/en/stable/index.html) module. See `requirements.txt`

## Installation

* Install the requirements:
```
$ pip install -r requirements.txt
```

* Run the notebooks in `./applications`

## Need help?

Feel free to [contact me](mailto:t.vanleeuwen@uu.nl) if you have any questions or need help adapting the code to your own problems.
