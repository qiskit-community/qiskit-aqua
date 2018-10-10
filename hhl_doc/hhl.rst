.. _hhl:

===
HHL
===

The *HHL algorithm* (after the author's surnames Harrow-Hassidim-LLoyd) is a quantum algorithm to solve systems of linear equations. `It was proposed <https://arxiv.org/abs/0811.3171>`__ in 2008 and the proposed algorithm provides an exponential speed-up against classical methods.

Basic concept

Let Ax=b be the linear system that we want to solve. Using aquas'
implementation of the Quantum Phase Estimation algorithm (QPE) :ref:`QPE` , the linear system
is transformed into diagonal form in which the matrix A is easily invertible.
The inversion is achieved by rotating an ancillar qubit by an angle
:math:`\arcsin{ \frac{C}{\theta}}` around the y-axis.
This leaves the system in a state proportional to the solution vector :math:`\ket{x}`
In many cases one is not interested in the single vector elements of :math:`\ket{x}` but only on 
certain properties. These are accessible by using problem-specific operators. Another use-case is the implementation in a larger quantum program.


IMAGE
-----
Usage
-----

.. topic:: Extending the HHL Library

    Consistent with its unique  design, Aqua has a modular and
    extensible architecture. Algorithms and their supporting objects, such as IQFTs,
    are pluggable modules in Aqua. This is done in order to encourage researchers and
    developers interested in
    :ref:`aqua-extending` to extend the Aqua framework with their novel research contributions.
    New IQFTs are typically installed in the ``qiskit_aqua/utils/iqfts``
    folder and derive from the ``IQFT`` class.  Aqua also allows for
    :ref:`aqua-dynamically-discovered-components`: new IQFTs can register themselves
    as Aqua extensions and be dynamically discovered at run time independent of their
    location in the file system.
----------
Components
----------
The HHL implementation is built in a modular fashion. The user can choose between different approaches to perform the matrix inversion. If later on another method for calculating the eigenvalues is available, it can be interchanged with the QPE.
In the following, a short description of the available methods and configurable parameters is presented.
 
^^^
Eigs
^^^
The implementation of HHL uses a slightly modified version of aquas' QPE algorithm.
It has support for negative Eigenvalues and uses a different method towards the calculation of the evolution time.

^^^^^^^^^^
Reciprocal
^^^^^^^^^^
Three different methods are available in aqua to invert the matrix A as proposed in the algorithm. They differ in accuracy, number of work qubits needed and the circuit depth.

TableLookup


.. topic:: Declarative Name

    When referring to the LookUp inversion method inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it,
    is ``LOOKUP``.




   
