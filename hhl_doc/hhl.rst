.. _hhl:

===
HHL
===

The *HHL algorithm* (after the author's surnames Harrow-Hassidim-LLoyd) is a quantum algorithm to solve systems of linear equations.
.. topic:: Basic concept
   Let Ax=b be the linear system that we want to solve. Using aquas'
   implementation of the Quantum Phase Estimation algorithm (QPE), the linear system
   is transformed into diagonal form in which the matrix A is easily invertible.
   The inversion is achieved by rotating an ancillar qubit by an angle
   :math:`\arcsin{ \frac{C}{\theta}}` around the y-axis.
   This leaves the system in a state proportional to the solution vector :math:`\ket{x}`
   
