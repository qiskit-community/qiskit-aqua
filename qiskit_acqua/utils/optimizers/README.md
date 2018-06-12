# QISKit ACQUA - Optimizers

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a set of algorithms and utilities
for use with quantum computers. 
The *optimizers* folder here contains a variety of different optimizers for use by algorithms

# Optimizers

Optimizers may be used in variational algorithms, such as [VQE](../../../qiskit_acqua#vqe). 

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

Further optimizers may be found in the [nlopts](./nlopts/README.md) folder that use the open-source
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

* `maxiter`=*integer, defaults to 20*

  Maximum number of iterations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `gtol`=*number, defaults to 1e-05*

  Gradient norm must be less than gtol before successful termination.

* `tol`=*number, optional, defaults to None*

  Tolerance for termination


## COBYLA

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Constrained Optimization By Linear Approximation algorithm.

The following parameters are supported:

* `maxiter`=*integer, defaults to 1000*

  Maximum number of iterations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `rhobeg`=*number, defaults to 1.0*

  Reasonable initial changes to the variables.

* `tol`=*number, optional, defaults to None*

  Tolerance for termination


## L_BFGS_B

This utilizes the
[scipy.optimize.fmin_l_bfgs_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)
optimizer as its core.
 
The following parameters are supported:

* `maxfun`=*integer, defaults to 1000*

  Maximum number of function evaluations

* `factr`=*integer, defaults to 10*

  An iteration stopping parameter

* `iprint`=*integer, defaults to -1*

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

* `maxiter`=*integer, optional*

  Maximum number of iterations to perform.

* `maxfev`=*integer, defaults to 1000*

  Maximum number of functional evaluations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `xatol`=*number, defaults to 0.0001*

  Absolute error in xopt between iterations that is acceptable for convergence.

* `tol`=*number, optional, defaults to None*

  Tolerance for termination


## P_BFGS

This is a parallel use of [L_BFGS_B](#l_bfgs_b) that can be useful when the target hardware is Quantum Simulators
running on a classical machine. This allows the multiple processes to use simulation to potentially reach a minimum
faster. It has the same parameters as [L_BFGS_B](#l_bfgs_b) and additionally the following.

* `max_processes`=*integer, optional, minimum value is 1*

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

* `maxiter`=*integer, optional*

  Maximum number of iterations to perform.

* `maxfev`=*integer, defaults to 1000*

  Maximum number of functional evaluations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `xtol`=*number, defaults to 0.0001*

  Relative error in solution xopt acceptable for convergence.

* `tol`=*number, optional, defaults to None*

  Tolerance for termination


## SLSQP

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method SLSQP uses Sequential Least SQuares Programming to minimize a function of several variables with any combination
of bounds, equality and inequality constraints. The method wraps the SLSQP Optimization subroutine originally
implemented by Dieter Kraft. Note that the wrapper handles infinite values in bounds by converting them into large
floating values.

The following parameters are supported:

* `maxiter`=*integer, defaults to 100*

  Maximum number of iterations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `ftol`=*number, defaults to 1e-06*

  Precision goal for the value of f in the stopping criterion.

* `tol`=*number, optional, defaults to None*

  Tolerance for termination


## SPSA

Simultaneous Perturbation Stochastic Approximation algorithm.

This optimizer can be used in the presence of noise, such as measurement uncertainty on a Quantum computation,
when finding a minimum. If you are using a qasm simulator or a real device this would be an optimum choice
among the optimizers provided here.

The optimization includes a calibration that will include additional functional evaluations to do this. 

The following parameters are supported:

* `max_trials`=*integer, defaults to 1000*

  Maximum number of trial steps for to be taken for the optimization. There are two function evaluations per trial.

* `save_steps`=*integer, defaults to 1*

  Stores optimization outcomes each 'save_steps' trial steps

* `last_avg`=*integer, defaults to 1*

  The number of last updates of the variables to average on for the final objective function.

* `parameters`=*array of 5 numbers, optional, defaults to None*

  Control parameters for SPSA. The SPSA updates the parameters (theta) for objective function (J) through the following equation at iteration k.

  > theta_{k+1} = theta_{k} + step_size * gradient,

  - step_size = c0 * (k + 1 + c4)^(-c2)
  - gradient = (J(theta_{k}+) - J(theta_{k}-)) * delta / (2 * c1 * (k+1)^(-c3))
    - theta_{k}+ = theta_{k} + c1 * (k+1)^(-c3) * delta; theta_{k}- = theta_{k} - c1 * (k+1)^(-c3) * delta
  - J(theta): objective value of theta

  c0 to c4 are the five control parameters.

  By default, c0 are calibrated through few evaluations on the objective function with
  the initial theta. c1 to c4 are set as 0.1, 0.602, 0.101, 0.0, respectively.

  
## TNC

It utilizes the scipy.optimize package: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method TNC uses a truncated Newton algorithm to minimize a function with variables subject to bounds. This algorithm
uses gradient information; it is also called Newton Conjugate-Gradient. It differs from the Newton-CG method described
above as it wraps a C implementation and allows each variable to be given upper and lower bounds.

The following parameters are supported:

* `maxiter`=*integer, defaults to 100*

  Maximum number of iterations to perform.

* `disp`=True|**False**

  Set to True to print convergence messages.
  
* `accuracy`=*number, defaults to 0*

  Relative precision for finite difference calculations.

* `ftol`=*number, defaults to -1*

  Precision goal for the value of f in the stopping criterion.

* `xtol`=*number, defaults to -1*

  Precision goal for the value of x in the stopping criterion (after applying x scaling factors).

* `gtol`=*number, defaults to -1*

  Precision goal for the value of the projected gradient in the stopping criterion (after applying x scaling factors).

* `tol`=*number, optional, defaults to None*

  Tolerance for termination
