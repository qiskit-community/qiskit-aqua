.. _translators:

===========
Translators
===========

The translation layer in Aqua Chemistry maps high-level classically generated input
to a qubit operator, which becomes the input to one of Aqua's :ref:`quantum-algorithms`.
As part of this layer, Aqua Chemistry offers three qubit mapping functions.

.. _jordan-wigner:

-------------
Jordan-Wigner
-------------
The `Jordan-Wigner transformation <https://rd.springer.com/article/10.1007%2FBF01331938>`__,
maps spin operators onto fermionic creation and annihilation operators.
It was proposed by Ernst Pascual Jordan and Eugene Paul Wigner
for one-dimensional lattice models,
but now two-dimensional analogues of the transformation have also been created.
The Jordan–Wigner transformation is often used to exactly solve 1D spin-chains
by transforming the spin operators to fermionic operators and then diagonalizing
in the fermionic basis.

.. _parity:

------
Parity
------

The `parity-mapping transformation <https://arxiv.org/abs/1701.08213>`__.
optimizes encodings of fermionic many-body systems by qubits
in the presence of symmetries.
Such encodings eliminate redundant degrees of freedom in a way that preserves
a simple structure of the system Hamiltonian enabling quantum simulations with fewer qubits. 

.. _bravyi-kitaev:

-------------
Bravyi-Kitaev
-------------

Also known as *binary-tree-based qubit mapping*, the `Bravyi-Kitaev transformation
<https://www.sciencedirect.com/science/article/pii/S0003491602962548>`__
is a method of mapping the occupation state of a
fermionic system onto qubits. This transformation maps the Hamiltonian of :math:`n`
interacting fermions to an :math:`\mathcal{O}(\log n)`
local Hamiltonian of :math:`n` qubits.
This is an improvement in locality over the Jordan–Wigner transformation, which results
in an :math:`\mathcal{O}(n)` local qubit Hamiltonian.
The Bravyi–Kitaev transformation was proposed by Sergey B. Bravyi and Alexei Yu. Kitaev.
