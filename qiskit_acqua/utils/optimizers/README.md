# QISKit ACQUA - Optimizers

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a set of algorithms and utilities
for use with quantum computers. 
The *optimizers* folder here contains a variety of different optimizers for use by algorithms

# Optimizers

Optimizers may be used in variational algorithms, such as [VQE](../../../qiskit_acqua#vqe)
and [VQkE](../../../qiskit_acqua#vqke). 

The following optimizers are supplied here. These are all local optimizers:

* [CG](#cg)
* [COBYLA](#cobyla)
* [L_BFGS_B](#l_bfgs_b)
* [NELDER_MEAD](#nelder_mead)
* [P_BFGS](#p_bfgs)
* [POWELL](#powell)
* [SLSQP](#slsqp)
* [SPSA](#spsa)
* [TNC](#tnc)

Further optimizers may be found in the [nlopts](./nlopts) folder that use the open-source
[NLOpt](https://nlopt.readthedocs.io) package and require NLopt to be installed to be used.

The optimizers here that are based on
[scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) have a 
common pattern for parameters exposed here. The *tol* parameter is from the *minimize* method itself and the remaining
parameters are from the 
[options dictionary](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html) which may
be referred to for further information 


## CG

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

CG algorithm: Unconstrained optimization. It will ignore bounds or constraints
Method CG uses a nonlinear conjugate gradient algorithm by Polak and Ribiere, a variant of the Fletcher-Reeves
method. Only the first derivatives are used.

The following parameters are supported:

* maxiter: *defaults to 20*

  Maximum number of iterations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* gtol: *defaults to 1e-05*

  Gradient norm must be less than gtol before successful termination.

* tol: float, optional

  Tolerance for termination


## COBYLA

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Constrained Optimization By Linear Approximation algorithm.

The following parameters are supported:

* maxiter: *defaults to 1000*

  Maximum number of iterations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* rhobeg: *defaults to 1.0*

  Reasonable initial changes to the variables.

* tol: float, optional

  Tolerance for termination


## L_BFGS_B

This utilizes the
[scipy.optimize.fmin_l_bfgs_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)
optimizer as its core.
 
The following parameters are supported:

* maxfun: *defaults to 1000*

  Maximum number of function evaluations

* factr: *defaults to 10*

  An iteration stopping parameter

* iprint: *defaults to -1*

  Controls the frequency of printed output that shows optimizer workings.

Further detailed information on *factr* and *iprint* may be found at 
[scipy.optimize.fmin_l_bfgs_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)


## NELDER_MEAD

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

NELDER_MEAD algorithm: Unconstrained optimization. It will ignore bounds or constraints
Method Nelder-Mead uses the Simplex algorithm. This algorithm is robust in many applications. However, if numerical
computation of derivative can be trusted, other algorithms using the first and/or second derivatives information might
be preferred for their better performance in general.

The following parameters are supported:

* maxiter: integer, optional

  Maximum number of iterations to perform.

* maxfev: *defaults to 1000*

  Maximum number of functional evaluations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* xatol: *defaults to 0.0001*

  Absolute error in xopt between iterations that is acceptable for convergence.

* tol: float, optional

  Tolerance for termination


## P_BFGS

This is a parallel use of [L_BFGS_B](#l_bfgs_b) that can be useful when the target hardware is Quantum Simulators
running on a classical machine. This allows the multiple processes to use simulation to potentially reach a minimum
faster. It has the same parameters as [L_BFGS_B](#l_bfgs_b) and additionally the following.

* max_processes: integer, optional, minimum value is 1

  By default P_BFGS will run one optimization in the current process and spawn additional processes up to
  the number of processor cores. An integer may be specified to limit the total number of processes (cores)
  used.
  
  Note: the parallel processes do not currently work for this optimizer on the Microsoft Windows platform. There
  it will just run the one optimization in the main process and hence the resulting behavior will be the same as
  the L_BFGS_B optimizer


## POWELL

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

POWELL algorithm: Unconstrained optimization. It will ignore bounds or constraints
Method Powell is a modification of Powell’s method which is a conjugate direction method. It performs sequential
one-dimensional minimization along each vector of the directions, which is updated at each iteration of the main
minimization loop. The function need not be differentiable, and no derivatives are taken.

The following parameters are supported:

* maxiter: integer, optional

  Maximum number of iterations to perform.

* maxfev: *defaults to 1000*

  Maximum number of functional evaluations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* xtol: *defaults to 0.0001*

  Relative error in solution xopt acceptable for convergence.

* tol: float, optional

  Tolerance for termination


## SLSQP

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method SLSQP uses Sequential Least SQuares Programming to minimize a function of several variables with any combination
of bounds, equality and inequality constraints. The method wraps the SLSQP Optimization subroutine originally
implemented by Dieter Kraft. Note that the wrapper handles infinite values in bounds by converting them into large
floating values.

The following parameters are supported:

* maxiter: *defaults to 100*

  Maximum number of iterations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* ftol: *defaults to 1e-06*

  Precision goal for the value of f in the stopping criterion.

* tol: float, optional

  Tolerance for termination


## SPSA

Simultaneous Perturbation Stochastic Approximation algorithm.

This optimizer can be used in the presence of noise, such as measurement uncertainty on a Quantum computation,
when finding a minimum. If you are using a qasm simulator or a real device this would be an optimum choice
among the optimizers provided here.

The optimization includes a calibration that will include additional functional evaluations to do this. 

The following parameters are supported:

* max_trials: *defaults to 1000*

  Maximum number of trial steps for to be taken for the optimization. There are two function evaluations per trial.

* save_steps: *defaults to 1*

  Stores optimization outcomes each 'save_steps' trial steps

* last_avg: *defaults to 1*

  The number of last updates of the variables to average on for the final objective function.


## TNC

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method TNC uses a truncated Newton algorithm to minimize a function with variables subject to bounds. This algorithm
uses gradient information; it is also called Newton Conjugate-Gradient. It differs from the Newton-CG method described
above as it wraps a C implementation and allows each variable to be given upper and lower bounds.

The following parameters are supported:

* maxiter: *defaults to 100*

  Maximum number of iterations to perform.

* disp: True|**False**

  Set to True to print convergence messages.
  
* accuracy: *defaults to 0*

  Relative precision for finite difference calculations.

* ftol: *defaults to -1*

  Precision goal for the value of f in the stopping criterion.

* xtol: *defaults to -1*

  Precision goal for the value of x in the stopping criterion (after applying x scaling factors).

* gtol: *defaults to -1*

  Precision goal for the value of the projected gradient in the stopping criterion (after applying x scaling factors).

* tol: float, optional

  Tolerance for termination
