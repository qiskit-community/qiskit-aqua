# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
r"""
===================================================
Qiskit's chemistry module (:mod:`qiskit.chemistry`)
===================================================

.. currentmodule:: qiskit.chemistry

This is Qiskit's chemistry module that provides for experimentation with chemistry
domain problems such as ground state energy and excited state energies of molecules.

Overview
========

This is an overview of the workings of the chemistry module and how it may be used. There
are different levels of exposure to its functionality, allowing for experimentation at different
abstractions. The outline below starts with the flow that provides the most control of the
process.

A classical chemistry driver is first instantiated, from the available :mod:`~.drivers`,
by means of a molecule specification, along with other configuration such as basis set and method
(RHF, ROHF, UHF). This configuration may include custom settings for the specific driver for more
custom control over the driver's behavior. When the driver is run the output is a mostly driver
independent :class:`~.QMolecule`. This contains various quantities that were
computed including one and two-body electron integrals that are used as input
:class:`~qiskit.chemistry.FermionicOperator`. Mostly driver independent means that these integrals,
for example, will be there from every driver but the values may differ due to how each underlying
chemistry library/program computes them. Also some fields in the QMolecule may not be populated by
all drivers, for instance dipole integrals are not output from the PyQuante driver, and hence the
dipole moment cannot be computed by qiskit.chemistry when using this driver. The FermionicOperator
once created can then be converted/mapped to a qubit operator for use as input to an Aqua
algorithms. The operator must be in qubit form at this stage since the execution target of the
algorithm will be a quantum device, or simulator ,comprised of qubits and the mapping is needed as
qubits behave differently than fermions. Once the algorithm is run it will compute the electronic
part of the quantity, such as the electronic ground state energy. To get the total ground state
energy this can be combined with the nuclear repulsion energy in the QMolecule.

Instead of using the FermionicOperator the :class:`.core.Hamiltonian` may be used. This
itself uses the FermionicOperator but provides a higher level of function to simplify use. For
instance the FermionicOperator supports particle-hole transformation and different mappings. And to
compute dipole moments each of the X, Y and Z dipole integrals must be prepared, as individual
FermionicOperators, in a like manner to the main electronic energy one, i.e. same transformations,
and eventually same qubit mapping. The core.Hamiltonian class does all this and more, such as
orbital reductions, frozen core and automatic symmetry reduction. When run with a QMolecule output
from a driver it produces qubit operators that can be given directly to Aqua algorithms for
the computation. Also available are several properties, such as number of particles and number of
orbitals, that are needed to correctly instantiate chemistry specific components such as
:class:`~.components.variational_forms.UCCSD` and :class:`~.components.initial_states.HartreeFock`.
Using the FermionicOperator directly requires taking the initial values for the QMolecule and then
keeping track of any changes based on any orbital elimination and/or freezing, and/or Z2Symmetry
reductions that are done. Once the output qubit operators have been used with an Aqua algorithm,
to compute the electronic result, this result can be fed back to the core.Hamiltonian, to
:meth:`~.core.Hamiltonian.process_algorithm_result` which will then compute a final total result
including a user friendly formatted text result that may be printed.

Lastly the chemistry :mod:`~.applications` may be used. These are given a chemistry driver and,
in the case of :class:`~applications.MolecularGroundStateEnergy` an optional instance of an Aqua
:class:`~qiskit.aqua.algorithms.MinimumEigensolver`, such as :class:`~qiskit.aqua.algorithms.VQE`.
Optional since components such as :class:`~.components.variational_forms.UCCSD` need certain
information that may be unknown at this point. So alternatively, when its method
:meth:`~applications.MolecularGroundStateEnergy.compute_energy` is run a callback can be provided
which will later be passed information, such as number of particles and orbitals, and allow a
complete MinimumEigensolver to be built using say UCCSD with HartreeFock, and subsequently returned
to the application and run. MinimumEigensolver itself uses the core.Hamiltonian class wrapping
it to form this high level application.

Mappings
++++++++
To map the FermionicOperator to a qubit operator the chemistry module supports the following
mappings:

Jordan Wigner
    The `Jordan-Wigner transformation <https://rd.springer.com/article/10.1007%2FBF01331938>`__,
    maps spin operators onto fermionic creation and annihilation operators.
    It was proposed by Ernst Pascual Jordan and Eugene Paul Wigner
    for one-dimensional lattice models,
    but now two-dimensional analogues of the transformation have also been created.
    The Jordan–Wigner transformation is often used to exactly solve 1D spin-chains
    by transforming the spin operators to fermionic operators and then diagonalizing
    in the fermionic basis.

Parity
    The `parity-mapping transformation <https://arxiv.org/abs/1701.08213>`__.
    optimizes encodings of fermionic many-body systems by qubits
    in the presence of symmetries.
    Such encodings eliminate redundant degrees of freedom in a way that preserves
    a simple structure of the system Hamiltonian enabling quantum simulations with fewer qubits.

Bravyi-Kitaev
    Also known as *binary-tree-based qubit mapping*, the `Bravyi-Kitaev transformation
    <https://www.sciencedirect.com/science/article/pii/S0003491602962548>`__
    is a method of mapping the occupation state of a
    fermionic system onto qubits. This transformation maps the Hamiltonian of :math:`n`
    interacting fermions to an :math:`\mathcal{O}(\log n)`
    local Hamiltonian of :math:`n` qubits.
    This is an improvement in locality over the Jordan–Wigner transformation, which results
    in an :math:`\mathcal{O}(n)` local qubit Hamiltonian.
    The Bravyi–Kitaev transformation was proposed by Sergey B. Bravyi and Alexei Yu. Kitaev.

Bravyi-Kitaev Superfast
    `Bravyi-Kitaev Superfast (BKSF) <https://aip.scitation.org/doi/10.1063/1.5019371>`__ algorithm
    is a mapping from fermionic operators to qubit operators. BKSF algorithm defines an abstract
    model where the fermionic modes are mapped to vertices of an interaction graph. The edges of
    the graph correspond to the interaction between the modes. The graph can be constructed from
    the Hamiltonian. The simulation is done by putting qubits on the edges of the graph. Each
    fermionic operator costs :math:`\mathcal{O}(d)` qubit operations, where :math:`d` is the
    degree of the interaction graph. The BKSF was proposed by Kanav Setia and James D. Whitfield.


The classes and submodules of qiskit.chemistry are now listed for reference:

Chemistry Error
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QiskitChemistryError

Chemistry Classes
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BosonicOperator
   FermionicOperator
   QMolecule
   WatsonHamiltonian
   MP2Info

Submodules
==========

.. autosummary::
   :toctree:

   applications
   algorithms
   components
   core
   drivers

"""

from .qiskit_chemistry_error import QiskitChemistryError
from .qmolecule import QMolecule
from .watson_hamiltonian import WatsonHamiltonian
from .bosonic_operator import BosonicOperator
from .fermionic_operator import FermionicOperator
from .mp2info import MP2Info
from ._logging import (get_qiskit_chemistry_logging,
                       set_qiskit_chemistry_logging)

__all__ = ['QiskitChemistryError',
           'QMolecule',
           'WatsonHamiltonian',
           'BosonicOperator',
           'FermionicOperator',
           'MP2Info',
           'get_qiskit_chemistry_logging',
           'set_qiskit_chemistry_logging']
