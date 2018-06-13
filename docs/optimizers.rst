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
algorithms.  Except for :ref:`Parallel Broyden-Fletcher-Goldfarb-Shann (P-BFGS)`, all hese optimizers are directly based on the ``scipy.optimize.minimize`` optimization function in 
`SciPy.org <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.
They all have a common pattern for parameters. Specifically, the ``tol`` parameter, whose value
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

   The default is ``20``.

-  A Boolean value indicating whether or not to print convergence messages:

   .. code:: python

        disp : bool

   The default value is ``False``.

-  A tolerance value that must be greater than the gradient norm before successful termination.

   .. code:: python

        gtol : float

   The default value is ``1e-05``.


-  The tolerance for termination:

   .. code::

        tol : float

   This parameter is optional.  If specified, the value of this parameter must be a number, otherwise, it is  ``Nonw``.
   The default is ``None``.

.. topic:: Declarative Name

   When referring to CG declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``CG``.

Constrained Optimization BY Linear Approximation (COBYLA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COBYLA is a numerical optimization method for constrained problems where the derivative of the objective function is not known.
COBYLA supports the following parameters:

-  The maximum number of iterations to perform:

   .. code:: python

       maxiter : int

   The default is ``1000``.

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

   This parameter is optional.  If specified, the value of this parameter must be of type ``float``, otherwise, it is  ``None``.
   The default is ``None``.

.. topic:: Declarative Name

   When referring to COBYLA declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``COBYLA``.

Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bound (L-BFGS-B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The target goal of L-BFGS-B is to minimize the value of a differentiable scalar function :math:`f`. 
This optimizer is a *quasi-Newton method*, meaning tha, in contrast to *Newtons's method*, it 
does not require :math:f's *Hessian* (the matrix of :math:`f`'s second derivatives)
when attempting to compute :math:`f`'s minimum value.
Like BFGS, L-BFGS is an iterative method for solving unconstrained, non-linear optimization problems, but approximates 
BFGS using a limited amount of computer memory.
L-BFGS starts with an initial estimate of the optimal value, and proceeds iteratively
to refine that estimate with a sequence of better estimates.
The derivatives of :math:`f` are used to identify the direction of steepest descent,
and also to form an estimate of the Hessian matrix (second derivative) of :math:`f`.
L-BFGS-B extends L-BFGS to handle simple, per-variable bound constraints. 

The following parameters are supported:

-  The maximum number of function evaluations:

   .. code:: python

        maxfun : int

   The default is ``1000``.

-  The maximum number of function evaluations:

   .. code:: python

        maxfun : int

   The default is ``1000``.

-  The maximum number of iterations:

   .. code:: python

        factr : int

   The default is ``10``.

-  An ``int`` value controlling the frequency of the printed output showing the  optimizer's
   operations.

   .. code:: python

       iprint : int

   The default is ``-1``.

Further detailed information on *factr* and *iprint* may be found at
`scipy.optimize.fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`__.

.. topic:: Declarative Name

   When referring to L-BFGS-B declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``L_BFGS_B``.

Nelder-Mead
~~~~~~~~~~~

The Nelder-Mead algorithm performs unnconstrained optimization; it ignores bounds
or constraints.  It is used to find the minimum or maximum of an objective function
in a multidimensional space.  It is based on the Simplex algorithm. Nelder-Mead
is robust in many applications, especially when the first and second derivativerds of the 
objective function are not known. However, if numerical
computation of the derivatives can be trusted to be accurate, other algorithms using the
first and/or second derivatives information might be preferred for their
better performance in the general case, especially in consideration of the fact that
the Nelder–Mead technique is a heuristic search method that can converge to non-stationary points.

The following parameters are supported:

-  The maximum number of iterations:

   .. code:: python

       maxiter : int

   This parameter is optional.  If specified, the value of this parameter must be of type ``int``, otherwise, it is  ``None``.
   The default is ``None``.

-  The maximum number of functional evaluations to perform:

   .. code:: python

       maxfev : int

   The default is ``1000``.

-  A ``bool`` value indicating whether or not to print convergence messages:

   .. code:: python

       disp : bool

   The default is ``False``.

-  A tolerance parameter indicating the absolute error in ``xopt`` between iterations that will be considered acceptable
   for convergence.

   .. code:: python

       xatol : float 

   The default value is ``0.0001``.

-  The tolerance for termination:

   .. code::

       tol : float

   This parameter is optional.  If specified, the value of this parameter must be of type ``float``, otherwise, it is  ``None``.
   The default is ``None``.

.. topic:: Declarative Name

   When referring to Nelder-Mead declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``NELDER_MEAD``.

Parallel Broyden-Fletcher-Goldfarb-Shann (P-BFGS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

P-BFGS is a parallellized version of  :ref:`Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bound (L-BFGS-B)`,
with which it shares the same parameters.
P-BFGS can be useful when the target hardware is a quantum simulator running on a classical
machine. This allows the multiple processes to use simulation to
potentially reach a minimum faster. The parallelization may help the optimizer avoid getting stuck
at local mimima.  In addition to the parameters of
L-BFGS-B, P-BFGS supports the following parameter:

-  The maximum numer of processes spawned by P-BFGS:

   .. code:: python

       max_processes = 1 | 2 | ...

   By default, P-BFGS runs one optimization in the current process
   and spawns additional processes up to the number of processor cores.
   An ``int`` value may be specified to limit the total number of processes
   (or cores) used.  This parameter is optional.  If specified, the value of this parameter must be of type ``int``,
   otherwise, it is ``None``.
   The default is ``None``.

.. note::
   The parallel processes do not currently work for this optimizer
   on the Microsoft Windows platform. There, P-BFGS will just run the one
   optimization in the main process, without spawning new processes.
   Therefore, the resulting behavior
   will be the same as the L-BFGS-B optimizer.

.. topic:: Declarative Name

   When referring to P-BFGS declaratively inside QISKit ACQUA,
   its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``P_BFGS``.

Powell
~~~~~~

The Powell algorithm performs unconstrained optimization; it ignores bounds or
constraints. Powell is
a *conjugate direction method*: it performs sequential one-dimensional
minimization along each directional vector, which is updated at
each iteration of the main minimization loop. The function being minimized need not be
differentiable, and no derivatives are taken.

The following parameters are supported:

-  The maximum number of iterations:

   .. code:: python

       maxiter : int

   This parameter is optional.  If specified, the value of this parameter must be of type ``int``, otherwise, it is  ``None``.
   The default is ``None``.

-  The maximum number of functional evaluations to perform:

   .. code:: python

       maxfev : int

   The default value is ``1000``.

-  A ``bool`` value indicating whether or not to print convergence messages:

   .. code:: python

      disp : bool

   The default is ``False``.

-  A tolerance parameter indicating the absolute error in ``xopt`` between iterations that will be considered acceptable
   for convergence.

   .. code:: python

       xtol : float

   The default value is ``0.0001``.

-  The tolerance for termination:

   .. code::

       tol : float

   This parameter is optional.  If specified, the value of this parameter must be of type ``float``, otherwise, it is  ``None``.
   The default is ``None``.

.. topic:: Declarative Name

   When referring to Powell declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``POWELL``.

Sequential Least SQuares Programming (SLSQP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SLSQP minimizes a
function of several variables with any combination of bounds, equality
and inequality constraints. The method wraps the SLSQP Optimization
subroutine originally implemented by Dieter Kraft.
SLSQP is ideal for  mathematical problems for which the objective function and the constraints are twice continuously differentiable.
Note that the wrapper
handles infinite values in bounds by converting them into large floating
values.

The following parameters are supported:

-  The maximum number of iterations:

   .. code:: python

       maxiter : int

   The default is ``100``.

-  A ``bool`` value indicating whether or not to print convergence messages:

   .. code:: python

       disp : bool

   The default is ``False``.

-  A tolerance value indicating precision goal for the value of the objective function in the stopping criterion.

   .. code:: python

       gtol : float

   The default value is ``1e-06``.

-  The tolerance for termination:

   .. code::

       tol : number

   This parameter is optional.  If specified, the value of this parameter must be a number, otherwise, it is  ``Nonw``.
   The default is ``None``.

.. topic:: Declarative Name

   When referring to SLSQP declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``SLSQP``.

Simultaneous Perturbation Stochastic Approximation (SPSA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPSA is an algorithmic method for optimizing systems with multiple unknown parameters.
As an optimization method, it is appropriately suited to large-scale population models, adaptive modeling,and simulation optimization. Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.
SPSA is a descent method capable of finding global minima,
sharing this property with other methods as simulated annealing.
Its main feature is the gradient approximation, which requires only two
measurements of the objective function, regardless of the dimension of the optimization problem.

.. note::
    SPSA can be used in the presence of noise, and it is therefore indicated in situations
    involving measurement uncertainty on a quantum computation when finding a minimum. If you are
    executing a variational algorithm using a Quantum ASseMbly Language (QASM) simulator or a real device,
    SPSA would be the most  recommended choice among the optimizers provided here.

The optimization process includes a calibration phase, which requires additional
functional evaluations.  Overall, the following parameters are supported:

-  Maximum number of trial steps for to be taken for the optimization.
   There are two function evaluations per trial:

   .. code:: python

        max_trials : int
   
   The default value is ``1000``.

-  An ``int`` value determining how often optimization outcomes should be stored during execution:

   .. code:: python

        save_steps : int

   SPSA will store optimization outcomes every ``save_steps`` trial steps.  The default value is ``1``.

-  The number of last updates of the variables to average on for the
   final objective function:

   .. code:: python

       last_avg : int

   The default value is ``1``.


-  Control parameters for SPSA:

   .. code:: python

       parameters = list_of_5_numbers

   This is an optional parameter, consisting of a list of 5 ``float`` elements.  The default value is ``None``. 
   SPSA updates the parameters (``theta``)
   for the objective function (``J``) through the following equation at
   iteration ``k``:

   .. code:: python
        theta_{k+1} = theta_{k} + step_size * gradient
        step_size = c0 * (k + 1 + c4)^(-c2)
        gradient = (J(theta_{k}+) - J(theta_{k}-)) * delta / (2 * c1 * (k + 1)^(-c3))
        theta_{k}+ = theta_{k} + c1 * ( k + 1)^(-c3) * delta
        theta_{k}- = theta_{k} - c1 * ( k + 1)^(-c3) * delta

   ``J(theta)`` is the  objective value of ``theta``. ``c0``, ``c1``, ``c2``, ``c3`` and ``c4`` are the five control parameters.
   By default, ``c0`` is calibrated through a few evaluations on the
   objective function with the initial ``theta``. ``c1``, ``c2``, ``c3`` and ``c4`` are set as ``0.1``,
   ``0.602``, ``0.101``, ``0.0``, respectively.

.. topic:: Declarative Name

   When referring to SPSA declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``SPSA``.


Truncated Newton (TNC)
~~~~~~~~~~~~~~~~~~~~~~
TNC uses a truncated Newton algorithm to minimize a function with
variables subject to bounds. This algorithm uses gradient information;
it is also called Newton Conjugate-Gradient. It differs from the
:ref:`Conjugate Gradient (CG) Method` method as it wraps a C implementation and
allows each variable to be given upper and lower bounds.

The following parameters are supported:

-  The maximum number of iterations:

   .. code:: python

        maxiter : int

   The default is ``100``.

-  A Boolean value indicating whether or not to print convergence messages:

   .. code:: python

        disp : bool

   The default value is ``False``.

-  Relative precision for finite difference calculations:

   .. code:: python

        accuracy : float

   The default value is ``0.0``.

-  A tolerance value indicating the precision goal for the value of the objective function ``f`` in the stopping criterion.

   .. code:: python

        ftol : float

   The default value is ``-1``.

-  A tolerance value indicating precision goal for the value of ``x`` in the stopping criterion, after applying ``x`` scaling factors.

   .. code:: python

        xtol : float

   The default value is ``-1``.

-  A tolerance value indicating precision goal for the value of the projected gradient ``g`` in the stopping criterion,
   after applying ``x`` scaling factors.

   .. code:: python

        gtol : float

   The default value is ``-1``.

-  The tolerance for termination:

   .. code::

        tol : number

   This parameter is optional.  If specified, the value of this parameter must be a number, otherwise, it is  ``Nonw``.
   The default is ``None``

.. topic:: Declarative Name

   When referring to TNC declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``TNC``.
.
