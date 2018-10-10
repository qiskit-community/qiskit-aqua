.. _hhl:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
HHL algorithm for solving linear systems (HHL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The *HHL algorithm* (after the author's surnames Harrow-Hassidim-LLoyd) is a quantum algorithm to solve systems of linear equations Ax=b. 
Using the Quantum Phase Estimation algorithm (QPE) :ref:`QPE` , the linear system
is transformed into diagonal form in which the matrix A is easily invertible.
The inversion is achieved by rotating an ancillar qubit by an angle
:math:`\arcsin{ \frac{C}{\theta}}` around the y-axis. :ref:`Reciprocal`.
After uncomputing the register storing the Eigenvalues using the inverse QPE, one measures the ancillar qubit. A measurement of 1 indicates that the matrix inversion suceeded.
This leaves the system in a state proportional to the solution vector :math:`\ket{x}`
In many cases one is not interested in the single vector elements of :math:`\ket{x}` but only on 
certain properties. These are accessible by using problem-specific operators. Another use-case is the implementation in a larger quantum program.


.. seealso::

    Consult the documentation on :ref:`iqfts`,  :ref:`initial-states`, :ref:`eigs`, :ref:`reciprocals`
    for more details.

In addition to requiring QPE, matrix inversion method (:ref:`reciprocals`) and a matrix /  initial state as part of its
configuration, HHL also exposes the following parameter settings:

- The run mode:
  .. code:: python
     mode = "state_tomography" | "circuit" | "swap_test"
  These different modes allow testing different settings on a simulator or
  executing the algorithm in a real usecase. Via ``"state_tomography"``, the solution vector is reconstructed using repetetive measurements (or reading out the state vector on the simulator). The ``"swap_test"`` setting triggers a swap test in which the fidelity between the HHL result and the classical result is calculated.


.. topic:: Declarative Name

   When referring to HHL declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``HHL``.

.. topic:: Problems Supported

   In Aqua, HHL supports the ``linear_system`` problem.
