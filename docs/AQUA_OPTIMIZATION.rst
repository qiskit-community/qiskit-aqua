.. _aqua-optimization:

*****************
Aqua Optimization
*****************

Qiskit Aqua Optimization is a set of tools and algorithms
that enable experimenting with optimization problems via quantum computing. Aqua Optimization
is the only end-to-end software stack that translates optimization-specific problems
into inputs for one of the :ref:`quantum-algorithms` in :ref:`aqua-library`,
which in turn uses Qiskit Terra for the actual quantum computation on top a
quantum simulator or a real quantum hardware device.

Aqua Optimization allows users with different levels of experience to execute optimization
experiments and contribute to the quantum computing optimization software stack.
Users with a pure optimization background or interests can continue to configure
optimization problems without having to learn the details of quantum computing.

--------------------------------
Optimization-specific Algorithms
--------------------------------

:ref:`aqua-library` includes numerous quantum algorithms
that can be used to experiment with optimization problems,
such as :ref:`vqe`, :ref:`qaoa`, :ref:`qpe`, :ref:`iqpe` and :ref:`grover`.

Research and developers interested in :ref:`aqua-extending` with new optimization-specific
capabilities can take advantage
of the modular architecture of Aqua and easily extend Aqua with more algorithms
and algorithm components, such as new :ref:`oracles` for the :ref:`grover` algorithm,
new :ref:`optimizers` and :ref:`variational-forms` for :ref:`vqe` and :ref:`qaoa`,
and new :ref:`iqfts` for :ref:`qpe` and :ref:`iqpe`.

To produce reference values and compare and contrast results during experimentation,
the Aqua library of :ref:`classical-reference-algorithms` also includes the
:ref:`exact-eigensolver` and :ref:`cplex` classical algorithms.

--------
Examples
-------- 

The ``optimization`` folder of the `Aqua Tutorials GitHub Repository
<https://github.com/Qiskit/aqua-tutorials>`__ contains numerous
`Jupyter Notebooks <http://jupyter.org/>`__
explaining how to use Aqua Optimization.

---------------------
Optimization Problems
---------------------

Aqua Optimization can already be used to experiment with numerous well known optimization
problems, such as:

1. `Stable Set <https://github.com/Qiskit/aqua-tutorials/blob/master/optimization/stableset.ipynb>`__
2. `Maximum Cut (MaxCut) <https://github.com/Qiskit/aqua-tutorials/blob/master/optimization/maxcut.ipynb>`__
3. `Partition <https://github.com/Qiskit/aqua-tutorials/blob/master/optimization/partition.ipynb>`__
4. `3 Satisfiability (3-SAT) <https://github.com/Qiskit/aqua-tutorials/blob/master/optimization/grover.ipynb>`__