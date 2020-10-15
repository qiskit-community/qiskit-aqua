# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The minimum eigensolver factory for ground state calculation algorithms."""

from typing import Optional
import numpy as np

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver, VQE
from qiskit.aqua.operators import ExpectationBase
from qiskit.aqua.components.optimizers import Optimizer
from ....components.variational_forms import UCCSD
from ....transformations.fermionic_transformation import FermionicTransformation
from ....components.initial_states import HartreeFock

from .minimum_eigensolver_factory import MinimumEigensolverFactory


class VQEUCCSDFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UCCSD ansatz wavefunction."""

    def __init__(self,
                 quantum_instance: QuantumInstance,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False) -> None:
        """
        Args:
            quantum_instance: The quantum instance used in the minimum eigensolver.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
        """
        self._quantum_instance = quantum_instance
        self._optimizer = optimizer
        self._initial_point = initial_point
        self._expectation = expectation
        self._include_custom = include_custom

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Getter of the quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """Setter of the quantum instance."""
        self._quantum_instance = q_instance

    @property
    def optimizer(self) -> Optimizer:
        """Getter of the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Setter of the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> np.ndarray:
        """Getter of the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray) -> None:
        """Setter of the initial point."""
        self._initial_point = initial_point

    @property
    def expectation(self) -> ExpectationBase:
        """Getter of the expectation."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: ExpectationBase) -> None:
        """Setter of the expectation."""
        self._expectation = expectation

    @property
    def include_custom(self) -> bool:
        """Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        return self._include_custom

    @include_custom.setter
    def include_custom(self, include_custom: bool) -> None:
        """Setter of the ``include_custom`` setting for the ``expectation`` setting."""
        self._include_custom = include_custom

    def get_solver(self,  # type: ignore
                   transformation: FermionicTransformation) -> MinimumEigensolver:
        """Returns a VQE with a UCCSD wavefunction ansatz, based on ``transformation``.
        This works only with a ``FermionicTransformation``.

        Args:
            transformation: a fermionic qubit operator transformation.

        Returns:
            A VQE suitable to compute the ground state of the molecule transformed
            by ``transformation``.
        """
        num_orbitals = transformation.molecule_info['num_orbitals']
        num_particles = transformation.molecule_info['num_particles']
        qubit_mapping = transformation.qubit_mapping
        two_qubit_reduction = transformation.molecule_info['two_qubit_reduction']
        z2_symmetries = transformation.molecule_info['z2_symmetries']

        initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                    two_qubit_reduction, z2_symmetries.sq_list)
        var_form = UCCSD(num_orbitals=num_orbitals,
                         num_particles=num_particles,
                         initial_state=initial_state,
                         qubit_mapping=qubit_mapping,
                         two_qubit_reduction=two_qubit_reduction,
                         z2_symmetries=z2_symmetries)
        vqe = VQE(var_form=var_form,
                  quantum_instance=self._quantum_instance,
                  optimizer=self._optimizer,
                  initial_point=self._initial_point,
                  expectation=self._expectation,
                  include_custom=self._include_custom)
        return vqe

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()
