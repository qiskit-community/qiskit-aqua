Optimizers
==========

QISKit ACQUA  contains a variety of different optimizers for
use by quantum variational algorithms, such as `VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__.  We can logically divide
optimizers into two categories:

- :ref:`Local Optimizers`: Given an optimization problem, a *local optimizer* is a function that attempts to find an optimal value
within the neighboring set of a candidate solution.

- :ref:`Global Optimizers`: Given an optimization problem, a *global optimizer* is a function that attempts to find an optimal value
among all possible solutions.


.. topic:: Extending the Optimizer Library
    Consistent with its unique  design and architecture, QISKit ACQUA has a modular and
    extensible architecture. Algorithms and their supporting objects, such as optimizers for quantum vational algorithms,
    are pluggable modules in QISKit ACQUA. This was done in order to encourage researchers and developers interested in
    quantum algorithms to extend the QISKit ACQUA framework with their novel research contributions.
    New optimizers for quantum variational algorithms should be installed in the ``qiskit_acqua/utils/optimizers`` folder and derive from the
    ``Optimizer`` class.


Local Optimizers
----------------

This section presents the local optimizers made available in QISKit ACQUA, and meant to be used in conjunction with a quantum variational
algorithms.  Theae optimizers are based on the ``scipy.optimize.minimize`` optimization function in 
`SciPy.org <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.
They all have a common pattern for parameters. Specifically, The ``tol`` parameter, whose value
must be a ``float`` indicating *tolerance for termination*,
is from the ``scipy.optimize.minimize``  method itself, while the remaining parameters are
from the `options
dictionary <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>`__,
which may be referred to for further information.

Conjugate Gradient (CG) Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CG is an algorithm for the numerical solution of systems of linear equations whose matrices are symmetric and positive-definite.
It is an *iterative algorithm* in that it uses an initial guess to generate a sequence of improving approximate solutions for a problems, in whicheach approximation is derived from the previous ones.  It is often used to solve unconstrained optimization problems, such as energy minimization.

The following parameters are supported:

-  The maximum number of iterations to perform:

   .. code:: pyton

       maxiter : int

   An integer value is expected,  The default is ``20``.

-  A Boolean value indicating whether or not to print convergence messages:

    .. code:: python

        disp : bool

   The default value is ``False``.

-  A tolerance value that must be greater than the gradient norm before successful termination.

    .. code:: python

        gtol : float

   A number is expected here.  The default value is ``1e-05``.


-  The tolerance for termination:

    .. code::

        tol : number

   This parameter is optional.  If specified, the value of this parameter must be a number, otherwise, it is  ``Nonw``.
   The default is ``None``.

Constrained Optimization BY Linear Approximation (COBYLA)
---------------------------------------------------------
COBYLA is a numerical optimization method for constrained problems where the derivative of the objective function is not known.
COBYLA supports the following parameters:

-  The maximum number of iterations to perform:

   .. code:: pyton

       maxiter : int

   An integer value is expected,  The default is ``1000``.

-  A Boolean value indicating whether or not to print convergence messages:

    .. code:: python

        disp : bool

   The default value is ``False``.

-  Reasonable initial changes to the variable:

   .. code:: python

       rhobeg : float

   The default value is ``1.0``.

-  The tolerance for termination:

    .. code::

        tol : float

   This parameter is optional.  If specified, the value of this parameter must be of type ``float``, otherwise, it is  ``Nonw``.
   The default is ``None``.

L_BFGS_B
--------

This utilizes the
`scipy.optimize.fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`__
optimizer as its core.

The following parameters are supported:

-  ``maxfun``\ =\ *integer, defaults to 1000*

   Maximum number of function evaluations

-  ``factr``\ =\ *integer, defaults to 10*

   An iteration stopping parameter

-  ``iprint``\ =\ *integer, defaults to -1*

   Controls the frequency of printed output that shows optimizer
   workings.

Further detailed information on *factr* and *iprint* may be found at
`scipy.optimize.fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`__

NELDER_MEAD
-----------

It utilizes the scipy.optimize package:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

NELDER_MEAD algorithm: Unconstrained optimization. It will ignore bounds
or constraints Method Nelder-Mead uses the Simplex algorithm. This
algorithm is robust in many applications. However, if numerical
computation of derivative can be trusted, other algorithms using the
first and/or second derivatives information might be preferred for their
better performance in general.

The following parameters are supported:

-  ``maxiter``\ =\ *integer, optional*

   Maximum number of iterations to perform.

-  ``maxfev``\ =\ *integer, defaults to 1000*

   Maximum number of functional evaluations to perform.

-  ``disp``\ =True\|\ **False**

   Set to True to print convergence messages.

-  ``xatol``\ =\ *number, defaults to 0.0001*

   Absolute error in xopt between iterations that is acceptable for
   convergence.

-  ``tol``\ =\ *number, optional, defaults to None*

   Tolerance for termination

P_BFGS
------

This is a parallel use of `L_BFGS_B <#l_bfgs_b>`__ that can be useful
when the target hardware is Quantum Simulators running on a classical
machine. This allows the multiple processes to use simulation to
potentially reach a minimum faster. It has the same parameters as
`L_BFGS_B <#l_bfgs_b>`__ and additionally the following.

-  ``max_processes``\ =\ *integer, optional, minimum value is 1*

   By default P_BFGS will run one optimization in the current process
   and spawn additional processes up to the number of processor cores.
   An integer may be specified to limit the total number of processes
   (cores) used.

   Note: the parallel processes do not currently work for this optimizer
   on the Microsoft Windows platform. There it will just run the one
   optimization in the main process and hence the resulting behavior
   will be the same as the L_BFGS_B optimizer

POWELL
------

It utilizes the scipy.optimize package:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

POWELL algorithm: Unconstrained optimization. It will ignore bounds or
constraints Method Powell is a modification of Powell’s method which is
a conjugate direction method. It performs sequential one-dimensional
minimization along each vector of the directions, which is updated at
each iteration of the main minimization loop. The function need not be
differentiable, and no derivatives are taken.

The following parameters are supported:

-  ``maxiter``\ =\ *integer, optional*

   Maximum number of iterations to perform.

-  ``maxfev``\ =\ *integer, defaults to 1000*

   Maximum number of functional evaluations to perform.

-  ``disp``\ =True\|\ **False**

   Set to True to print convergence messages.

-  ``xtol``\ =\ *number, defaults to 0.0001*

   Relative error in solution xopt acceptable for convergence.

-  ``tol``\ =\ *number, optional, defaults to None*

   Tolerance for termination

SLSQP
-----

It utilizes the scipy.optimize package:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method SLSQP uses Sequential Least SQuares Programming to minimize a
function of several variables with any combination of bounds, equality
and inequality constraints. The method wraps the SLSQP Optimization
subroutine originally implemented by Dieter Kraft. Note that the wrapper
handles infinite values in bounds by converting them into large floating
values.

The following parameters are supported:

-  ``maxiter``\ =\ *integer, defaults to 100*

   Maximum number of iterations to perform.

-  ``disp``\ =True\|\ **False**

   Set to True to print convergence messages.

-  ``ftol``\ =\ *number, defaults to 1e-06*

   Precision goal for the value of f in the stopping criterion.

-  ``tol``\ =\ *number, optional, defaults to None*

   Tolerance for termination

SPSA
----

Simultaneous Perturbation Stochastic Approximation algorithm.

This optimizer can be used in the presence of noise, such as measurement
uncertainty on a Quantum computation, when finding a minimum. If you are
using a qasm simulator or a real device this would be an optimum choice
among the optimizers provided here.

The optimization includes a calibration that will include additional
functional evaluations to do this.

The following parameters are supported:

-  ``max_trials``\ =\ *integer, defaults to 1000*

   Maximum number of trial steps for to be taken for the optimization.
   There are two function evaluations per trial.

-  ``save_steps``\ =\ *integer, defaults to 1*

   Stores optimization outcomes each ‘save_steps’ trial steps

-  ``last_avg``\ =\ *integer, defaults to 1*

   The number of last updates of the variables to average on for the
   final objective function.

-  ``parameters``\ =\ *array of 5 numbers, optional, defaults to None*

   Control parameters for SPSA. The SPSA updates the parameters (theta)
   for objective function (J) through the following equation at
   iteration k.

      theta_{k+1} = theta_{k} + step_size \* gradient,

   -  step_size = c0 \* (k + 1 + c4)^(-c2)
   -  gradient = (J(theta_{k}+) - J(theta_{k}-)) \* delta / (2 \* c1 \*
      (k+1)^(-c3))

      -  theta_{k}+ = theta_{k} + c1 \* (k+1)^(-c3) \* delta; theta_{k}-
         = theta_{k} - c1 \* (k+1)^(-c3) \* delta

   -  J(theta): objective value of theta

   c0 to c4 are the five control parameters.

   By default, c0 are calibrated through few evaluations on the
   objective function with the initial theta. c1 to c4 are set as 0.1,
   0.602, 0.101, 0.0, respectively.

TNC
---

It utilizes the scipy.optimize package:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Method TNC uses a truncated Newton algorithm to minimize a function with
variables subject to bounds. This algorithm uses gradient information;
it is also called Newton Conjugate-Gradient. It differs from the
Newton-CG method described above as it wraps a C implementation and
allows each variable to be given upper and lower bounds.

The following parameters are supported:

-  ``maxiter``\ =\ *integer, defaults to 100*

   Maximum number of iterations to perform.

-  ``disp``\ =True\|\ **False**

   Set to True to print convergence messages.

-  ``accuracy``\ =\ *number, defaults to 0*

   Relative precision for finite difference calculations.

-  ``ftol``\ =\ *number, defaults to -1*

   Precision goal for the value of f in the stopping criterion.

-  ``xtol``\ =\ *number, defaults to -1*

   Precision goal for the value of x in the stopping criterion (after
   applying x scaling factors).

-  ``gtol``\ =\ *number, defaults to -1*

   Precision goal for the value of the projected gradient in the
   stopping criterion (after applying x scaling factors).

-  ``tol``\ =\ *number, optional, defaults to None*

   Tolerance for termination
