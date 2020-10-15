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

"""
A ground state calculation employing the Orbital-Optimized VQE (OOVQE) algorithm.
"""

from typing import Optional, List, Union, Tuple
import logging
import copy
import numpy as np
from scipy.linalg import expm
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQE, MinimumEigensolver
from qiskit.aqua.operators import LegacyBaseOperator

from .ground_state_eigensolver import GroundStateEigensolver
from .minimum_eigensolver_factories import MinimumEigensolverFactory
from ...components.variational_forms import UCCSD
from ...fermionic_operator import FermionicOperator
from ...bosonic_operator import BosonicOperator
from ...drivers.base_driver import BaseDriver
from ...drivers.fermionic_driver import FermionicDriver
from ...transformations.fermionic_transformation import FermionicTransformation
from ...results.electronic_structure_result import ElectronicStructureResult
from ...qmolecule import QMolecule

logger = logging.getLogger(__name__)


class OrbitalOptimizationVQE(GroundStateEigensolver):
    r""" A ground state calculation employing the OOVQE algorithm.
    The Variational Quantum Eigensolver (VQE) algorithm enhanced with the Orbital Optimization (OO).
    The core of the approach resides in the optimization of orbitals through the
    AO-to-MO coefficients matrix C. In the usual VQE, the latter remains constant throughout
    the simulation. Here, its elements are modified according to C=Ce^(-kappa) where kappa is
    an anti-hermitian matrix. This transformation preserves the spectrum but modifies the
    amplitudes of the ground state of given operator such that in the end a given ansatz
    can be closest to that ground state, producing larger overlap and lower eigenvalue than
    conventional VQE.  Kappa is parametrized and optimized inside the OOVQE in the same way as
    the gate angles. Therefore, at each step of OOVQE the coefficient matrix C is modified and
    the operator is recomputed, unlike usual VQE where operator remains constant.
    Iterative OO refers to optimization in two steps, first the wavefunction and then the
    orbitals. It allows for faster optimization as the operator is not recomputed when
    wavefunction is optimized. It is recommended to use the iterative method on real device/qasm
    simulator with noise to facilitate the convergence of the classical optimizer.
    For more details of this method refer to: https://aip.scitation.org/doi/10.1063/1.5141835
    """

    def __init__(self,
                 transformation: FermionicTransformation,
                 solver: Union[MinimumEigensolver, MinimumEigensolverFactory],
                 initial_point: Optional[np.ndarray] = None,
                 orbital_rotation: Optional['OrbitalRotation'] = None,
                 bounds: Optional[np.ndarray] = None,
                 iterative_oo: bool = True,
                 iterative_oo_iterations: int = 2,
                 ):
        """
        Args:
            transformation: a fermionic driver to operator transformation strategy.
            solver: a VQE instance or a factory for the VQE solver employing any custom
                variational form, such as the `VQEUCCSDFactory`. Both need to use the UCCSD
                variational form.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            orbital_rotation: instance of
                :class:`~qiskit.chemistry.ground_state_calculation.OrbitalRotation` class
                that creates the matrices that rotate the orbitals needed to produce the rotated
                MO coefficients C as C = C0 * exp(-kappa).
            bounds: bounds for variational form and orbital rotation
                 parameters given to a classical optimizer.
            iterative_oo: when ``True`` optimize first the variational form and then the
                orbitals, iteratively. Otherwise, the wavefunction ansatz and orbitals are
                optimized simultaneously.
            iterative_oo_iterations: number of iterations in the iterative procedure,
                set larger to be sure to converge to the global minimum.
        Raises:
            AquaError: if the number of orbital optimization iterations is less or equal to zero.
        """

        super().__init__(transformation, solver)
        if not isinstance(self._transformation, FermionicTransformation):
            raise AquaError('OrbitalOptimizationVQE requires a FermionicTransformation.')
        from typing import cast
        self._transformation = cast(FermionicTransformation, self._transformation)

        self.initial_point = initial_point
        self._orbital_rotation = orbital_rotation
        self._bounds = bounds
        self._iterative_oo = iterative_oo
        self._iterative_oo_iterations = iterative_oo_iterations

        # internal parameters of the algorithm
        self._driver = None  # type: Optional[FermionicDriver]
        self._qmolecule = None  # type: Optional[QMolecule]
        self._qmolecule_rotated = None  # type: Optional[QMolecule]

        self._fixed_wavefunction_params = None
        self._num_parameters_oovqe = None
        self._additional_params_initialized = False
        self.var_form_num_parameters = None
        self.var_form_bounds = None
        self._vqe = None  # type: Optional[VQE]
        self._bound_oo = None  # type: Optional[List]

    def returns_groundstate(self) -> bool:
        return True

    def _set_operator_and_vqe(self, driver: BaseDriver):
        """ Initializes the operators using provided driver of qmolecule."""

        if not isinstance(self._transformation, FermionicTransformation):
            raise AquaError('OrbitalOptimizationVQE requires a FermionicTransformation.')
        if not isinstance(driver, FermionicDriver):
            raise AquaError('OrbitalOptimizationVQE only works with Fermionic Drivers.')

        if self._qmolecule is None:
            # in future, self._transformation.transform should return also qmolecule
            # to avoid running the driver twice
            self._qmolecule = driver.run()
            operator, aux_operators = self._transformation._do_transform(self._qmolecule)
        else:
            operator, aux_operators = self._transformation._do_transform(self._qmolecule)
        if operator is None:  # type: ignore
            raise AquaError("The operator was never provided.")

        if isinstance(self.solver, MinimumEigensolverFactory):
            # this must be called after transformation.transform
            self._vqe = self.solver.get_solver(self.transformation)
        else:
            self._vqe = self.solver

        if not isinstance(self._vqe, VQE):
            raise AquaError(
                "The OrbitalOptimizationVQE algorithm requires the use of the VQE " +
                "MinimumEigensolver.")
        if not isinstance(self._vqe.var_form, UCCSD):
            raise AquaError(
                "The OrbitalOptimizationVQE algorithm requires the use of the UCCSD varform.")

        self._vqe.operator = operator
        self._vqe.aux_operators = aux_operators

    def _set_bounds(self,
                    bounds_var_form_val: tuple = (-2 * np.pi, 2 * np.pi),
                    bounds_oo_val: tuple = (-2 * np.pi, 2 * np.pi)) -> None:
        """ Initializes the array of bounds of wavefunction and OO parameters.
        Args:
            bounds_var_form_val: pair of bounds between which the optimizer confines the
                                 values of wavefunction parameters.
            bounds_oo_val: pair of bounds between which the optimizer confines the values of
                           OO parameters.
        Raises:
            AquaError: Instantiate OrbitalRotation class and provide it to the
                       orbital_rotation keyword argument
        """
        self._bounds = []
        bounds_var_form = [bounds_var_form_val for _ in range(self._vqe.var_form.num_parameters)]
        self._bound_oo = \
            [bounds_oo_val for _ in range(self._orbital_rotation.num_parameters)]
        self._bounds = bounds_var_form + self._bound_oo
        self._bounds = np.array(self._bounds)

    def _set_initial_point(self, initial_pt_scalar: float = 1e-1) -> None:
        """ Initializes the initial point for the algorithm if the user does not provide his own.
        Args:
            initial_pt_scalar: value of the initial parameters for wavefunction and orbital rotation
        """
        self.initial_point = [initial_pt_scalar for _ in range(self._num_parameters_oovqe)]

    def _initialize_additional_parameters(self, driver: BaseDriver):
        """ Initializes additional parameters of the OOVQE algorithm. """

        if not isinstance(self._transformation, FermionicTransformation):
            raise AquaError('OrbitalOptimizationVQE requires a FermionicTransformation.')

        self._set_operator_and_vqe(driver)

        if self._orbital_rotation is None:
            self._orbital_rotation = OrbitalRotation(num_qubits=self._vqe.var_form.num_qubits,
                                                     transformation=self._transformation,
                                                     qmolecule=self._qmolecule)
        self._num_parameters_oovqe = \
            self._vqe.var_form.num_parameters + self._orbital_rotation.num_parameters

        if self.initial_point is None:
            self._set_initial_point()
        else:
            if len(self.initial_point) is not self._num_parameters_oovqe:
                raise AquaError(
                    'Number of parameters of OOVQE ({}) does not match the length of the '
                    'intitial_point ({})'.format(self._num_parameters_oovqe,
                                                 len(self.initial_point)))
        if self._bounds is None:
            self._set_bounds(self._orbital_rotation.parameter_bound_value)
        if self._iterative_oo_iterations < 1:
            raise AquaError('Please set iterative_oo_iterations parameter to a positive number,'
                            ' got {} instead'.format(self._iterative_oo_iterations))

        # copies to overcome incompatibilities with error checks in VQAlgorithm class
        self.var_form_num_parameters = self._vqe.var_form.num_parameters
        self.var_form_bounds = copy.copy(self._vqe.var_form._bounds)
        self._additional_params_initialized = True

    def _energy_evaluation_oo(self, parameters: np.ndarray) -> Union[float, List[float]]:
        """ Evaluate energy at given parameters for the variational form and parameters for
        given rotation of orbitals.
        Args:
            parameters: parameters for variational form and orbital rotations.
        Returns:
            energy of the hamiltonian of each parameter.
        Raises:
            AquaError: Instantiate OrbitalRotation class and provide it to the
                       orbital_rotation keyword argument
        """

        if not isinstance(self._transformation, FermionicTransformation):
            raise AquaError('OrbitalOptimizationVQE requires a FermionicTransformation.')

        # slice parameter lists
        if self._iterative_oo:
            parameters_var_form = self._fixed_wavefunction_params
            parameters_orb_rot = parameters
        else:
            parameters_var_form = parameters[:self.var_form_num_parameters]
            parameters_orb_rot = parameters[self.var_form_num_parameters:]

        logger.info('Parameters of wavefunction are: \n%s', repr(parameters_var_form))
        logger.info('Parameters of orbital rotation are: \n%s', repr(parameters_orb_rot))

        # rotate the orbitals
        if self._orbital_rotation is None:
            raise AquaError('Instantiate OrbitalRotation class and provide it to the '
                            'orbital_rotation keyword argument')

        self._orbital_rotation.orbital_rotation_matrix(parameters_orb_rot)

        # preserve original qmolecule and create a new one with rotated orbitals
        self._qmolecule_rotated = copy.copy(self._qmolecule)
        OrbitalOptimizationVQE._rotate_orbitals_in_qmolecule(
            self._qmolecule_rotated, self._orbital_rotation)

        # construct the qubit operator
        operator, aux_operators = self._transformation._do_transform(self._qmolecule_rotated)
        if isinstance(operator, LegacyBaseOperator):
            operator = operator.to_opflow()
        self._vqe.operator = operator
        self._vqe.aux_operators = aux_operators
        logger.debug('Orbital rotation parameters of matrix U at evaluation %d returned'
                     '\n %s', self._vqe._eval_count, repr(self._orbital_rotation.matrix_a))
        self._vqe.var_form._num_parameters = self.var_form_num_parameters

        # compute the energy on given state
        mean_energy = self._vqe._energy_evaluation(parameters=parameters_var_form)

        return mean_energy

    def solve(self,
              driver: BaseDriver,
              aux_operators: Optional[Union[List[FermionicOperator],
                                            List[BosonicOperator]]] = None) \
            -> ElectronicStructureResult:

        self._initialize_additional_parameters(driver)

        if not isinstance(self._transformation, FermionicTransformation):
            raise AquaError('OrbitalOptimizationVQE requires a FermionicTransformation.')

        self._vqe._eval_count = 0

        # initial orbital rotation starting point is provided
        if self._orbital_rotation.matrix_a is not None and self._orbital_rotation.matrix_b is not \
                None:
            self._qmolecule_rotated = copy.copy(self._qmolecule)
            OrbitalOptimizationVQE._rotate_orbitals_in_qmolecule(
                self._qmolecule_rotated, self._orbital_rotation)
            operator, aux_operators = self._transformation._do_transform(self._qmolecule_rotated)
            self._vqe.operator = operator
            self._vqe.aux_operators = aux_operators

            logger.info(
                '\n\nSetting the initial value for OO matrices and rotating Hamiltonian \n')
            logger.info('Optimising  Orbital Coefficient Rotation Alpha: \n%s',
                        repr(self._orbital_rotation.matrix_a))
            logger.info('Optimising  Orbital Coefficient Rotation Beta: \n%s',
                        repr(self._orbital_rotation.matrix_b))

        # save the original number of parameters as we modify their number to bypass the
        # error checks that are not tailored to OOVQE

        # iterative method
        if self._iterative_oo:
            for _ in range(self._iterative_oo_iterations):
                # optimize wavefunction ansatz
                logger.info('OrbitalOptimizationVQE: Ansatz optimization, orbitals fixed.')
                self._vqe.var_form._num_parameters = self.var_form_num_parameters
                if isinstance(self._vqe.operator, LegacyBaseOperator):  # type: ignore
                    self._vqe.operator = self._vqe.operator.to_opflow()  # type: ignore
                self._vqe.var_form._bounds = self.var_form_bounds
                vqresult_wavefun = self._vqe.find_minimum(
                    initial_point=self.initial_point[:self.var_form_num_parameters],
                    var_form=self._vqe.var_form,
                    cost_fn=self._vqe._energy_evaluation,
                    optimizer=self._vqe.optimizer)
                self.initial_point[:self.var_form_num_parameters] = vqresult_wavefun.optimal_point

                # optimize orbitals
                logger.info('OrbitalOptimizationVQE: Orbital optimization, ansatz fixed.')
                self._vqe.var_form._bounds = self._bound_oo
                self._vqe.var_form._num_parameters = self._orbital_rotation.num_parameters
                self._fixed_wavefunction_params = vqresult_wavefun.optimal_point
                vqresult = self._vqe.find_minimum(
                    initial_point=self.initial_point[self.var_form_num_parameters:],
                    var_form=self._vqe.var_form,
                    cost_fn=self._energy_evaluation_oo,
                    optimizer=self._vqe.optimizer)
                self.initial_point[self.var_form_num_parameters:] = vqresult.optimal_point
        else:
            # simultaneous method (ansatz and orbitals are optimized at the same time)
            self._vqe.var_form._bounds = self._bounds
            self._vqe.var_form._num_parameters = len(self._bounds)
            vqresult = self._vqe.find_minimum(initial_point=self.initial_point,
                                              var_form=self._vqe.var_form,
                                              cost_fn=self._energy_evaluation_oo,
                                              optimizer=self._vqe.optimizer)

        # write original number of parameters to avoid errors due to parameter number mismatch
        self._vqe.var_form._num_parameters = self.var_form_num_parameters

        # extend VQE returned information with additional outputs
        result = OOVQEResult()
        result.computed_electronic_energy = vqresult.optimal_value
        result.num_optimizer_evals = vqresult.optimizer_evals
        result.optimal_point = vqresult.optimal_point
        if self._iterative_oo:
            result.optimal_point_ansatz = self.initial_point[self.var_form_num_parameters:]
            result.optimal_point_orbitals = self.initial_point[:self.var_form_num_parameters]
        else:
            result.optimal_point_ansatz = vqresult.optimal_point[:self.var_form_num_parameters]
            result.optimal_point_orbitals = vqresult.optimal_point[self.var_form_num_parameters:]
        result.eigenenergies = [vqresult.optimal_value + 0j]

        #  copy parameters bypass the error checks that are not tailored to OOVQE
        _ret_temp_params = copy.copy(vqresult.optimal_point)
        self._vqe._ret = {}
        self._vqe._ret['opt_params'] = vqresult.optimal_point[:self.var_form_num_parameters]
        if self._iterative_oo:
            self._vqe._ret['opt_params'] = vqresult_wavefun.optimal_point
        result.eigenstates = [self._vqe.get_optimal_vector()]
        if not self._iterative_oo:
            self._vqe._ret['opt_params'] = _ret_temp_params

        if self._vqe.aux_operators is not None:
            #  copy parameters bypass the error checks that are not tailored to OOVQE
            self._vqe._ret['opt_params'] = vqresult.optimal_point[:self.var_form_num_parameters]
            if self._iterative_oo:
                self._vqe._ret['opt_params'] = vqresult_wavefun.optimal_point
            self._vqe._eval_aux_ops()
            result.aux_operator_eigenvalues = self._vqe._ret['aux_ops'][0]
            if not self._iterative_oo:
                self._vqe._ret['opt_params'] = _ret_temp_params

        result.cost_function_evals = self._vqe._eval_count
        self.transformation.interpret(result)

        return result

    @staticmethod
    def _rotate_orbitals_in_qmolecule(qmolecule: QMolecule,
                                      orbital_rotation: 'OrbitalRotation') -> None:
        """ Rotates the orbitals by applying a modified a anti-hermitian matrix
        (orbital_rotation.matrix_a) onto the MO coefficients matrix and recomputes all the
        quantities dependent on the MO coefficients. Be aware that qmolecule is modified
        when this executes.
        Args:
            qmolecule: instance of QMolecule class
            orbital_rotation: instance of OrbitalRotation class
        """

        # 1 and 2 electron integrals (required) from AO to MO basis
        qmolecule.mo_coeff = np.matmul(qmolecule.mo_coeff,
                                       orbital_rotation.matrix_a)
        qmolecule.mo_onee_ints = qmolecule.oneeints2mo(qmolecule.hcore,
                                                       qmolecule.mo_coeff)
        # support for unrestricted spins
        if qmolecule.mo_coeff_b is not None:
            qmolecule.mo_coeff_b = np.matmul(qmolecule.mo_coeff_b,
                                             orbital_rotation.matrix_b)
            qmolecule.mo_onee_ints_b = qmolecule.oneeints2mo(qmolecule.hcore,
                                                             qmolecule.mo_coeff)

        qmolecule.mo_eri_ints = qmolecule.twoeints2mo(qmolecule.eri,
                                                      qmolecule.mo_coeff)
        if qmolecule.mo_coeff_b is not None:
            mo_eri_b = qmolecule.twoeints2mo(qmolecule.eri,
                                             qmolecule.mo_coeff_b)
            norbs = qmolecule.mo_coeff.shape[0]
            qmolecule.mo_eri_ints_bb = mo_eri_b.reshape(norbs, norbs, norbs, norbs)
            qmolecule.mo_eri_ints_ba = qmolecule.twoeints2mo_general(
                qmolecule.eri, qmolecule.mo_coeff_b, qmolecule.mo_coeff_b, qmolecule.mo_coeff,
                qmolecule.mo_coeff)
            qmolecule.mo_eri_ints_ba = qmolecule.mo_eri_ints_ba.reshape(norbs, norbs,
                                                                        norbs, norbs)
        # dipole integrals (if available) from AO to MO
        if qmolecule.x_dip_ints is not None:
            qmolecule.x_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.x_dip_ints,
                                                            qmolecule.mo_coeff)
            qmolecule.y_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.y_dip_ints,
                                                            qmolecule.mo_coeff)
            qmolecule.z_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.z_dip_ints,
                                                            qmolecule.mo_coeff)
        # support for unrestricted spins
        if qmolecule.mo_coeff_b is not None and qmolecule.x_dip_ints is not None:
            qmolecule.x_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.x_dip_ints,
                                                              qmolecule.mo_coeff_b)
            qmolecule.y_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.y_dip_ints,
                                                              qmolecule.mo_coeff_b)
            qmolecule.z_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.z_dip_ints,
                                                              qmolecule.mo_coeff_b)


class OrbitalRotation:
    r""" Class that regroups methods for creation of matrices that rotate the MOs.
    It allows to create the unitary matrix U = exp(-kappa) that is parameterized with kappa's
    elements. The parameters are the off-diagonal elements of the anti-hermitian matrix kappa.
    """

    def __init__(self,
                 num_qubits: int,
                 transformation: FermionicTransformation,
                 qmolecule: Optional[QMolecule] = None,
                 orbital_rotations: list = None,
                 orbital_rotations_beta: list = None,
                 parameters: list = None,
                 parameter_bounds: list = None,
                 parameter_initial_value: float = 0.1,
                 parameter_bound_value: Tuple[float, float] = (-2 * np.pi, 2 * np.pi)) -> None:
        """
        Args:
            num_qubits: number of qubits necessary to simulate a particular system.
            transformation: a fermionic driver to operator transformation strategy.
            qmolecule: instance of the :class:`~qiskit.chemistry.QMolecule` class which has methods
                needed to recompute one-/two-electron/dipole integrals after orbital rotation
                (C = C0 * exp(-kappa)). It is not required but can be used if user wished to
                provide custom integrals for instance.
            orbital_rotations: list of alpha orbitals that are rotated (i.e. [[0,1], ...] the
                0-th orbital is rotated with 1-st, which corresponds to non-zero entry 01 of
                the matrix kappa).
            orbital_rotations_beta: list of beta orbitals that are rotated.
            parameters: orbital rotation parameter list of matrix elements that rotate the MOs,
                each associated to a pair of orbitals that are rotated
                (non-zero elements in matrix kappa), or elements in the orbital_rotation(_beta)
                lists.
            parameter_bounds: parameter bounds
            parameter_initial_value: initial value for all the parameters.
            parameter_bound_value: value for the bounds on all the parameters
        """

        self._num_qubits = num_qubits
        self._transformation = transformation
        self._qmolecule = qmolecule

        self._orbital_rotations = orbital_rotations
        self._orbital_rotations_beta = orbital_rotations_beta
        self._parameter_initial_value = parameter_initial_value
        self._parameter_bound_value = parameter_bound_value
        self._parameters = parameters
        if self._parameters is None:
            self._create_parameter_list_for_orbital_rotations()

        self._num_parameters = len(self._parameters)
        self._parameter_bounds = parameter_bounds
        if self._parameter_bounds is None:
            self._create_parameter_bounds()

        self._freeze_core = self._transformation._freeze_core
        self._core_list = self._qmolecule.core_orbitals if self._freeze_core else None

        if self._transformation._two_qubit_reduction is True:
            self._dim_kappa_matrix = int((self._num_qubits + 2) / 2)
        else:
            self._dim_kappa_matrix = int(self._num_qubits / 2)

        self._check_for_errors()
        self._matrix_a = None
        self._matrix_b = None

    def _check_for_errors(self) -> None:
        """ Checks for errors such as incorrect number of parameters and indices of orbitals. """

        # number of parameters check
        if self._orbital_rotations_beta is None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) != len(self._parameters):
                raise AquaError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        elif self._orbital_rotations_beta is not None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) + len(self._orbital_rotations_beta) != len(
                    self._parameters):
                raise AquaError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        # indices of rotated orbitals check
        for exc in self._orbital_rotations:
            if exc[0] > (self._dim_kappa_matrix - 1):
                raise AquaError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}, '.format(exc[0]))
            if exc[1] > (self._dim_kappa_matrix - 1):
                raise AquaError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}'.format(exc[1]))
        if self._orbital_rotations_beta is not None:
            for exc in self._orbital_rotations_beta:
                if exc[0] > (self._dim_kappa_matrix - 1):
                    raise AquaError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[0]))
                if exc[1] > (self._dim_kappa_matrix - 1):
                    raise AquaError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[1]))

    def _create_orbital_rotation_list(self) -> None:
        """ Creates a list of indices of matrix kappa that denote the pairs of orbitals that
        will be rotated. For instance, a list of pairs of orbital such as [[0,1], [0,2]]. """

        if self._transformation._two_qubit_reduction:
            half_as = int((self._num_qubits + 2) / 2)
        else:
            half_as = int(self._num_qubits / 2)

        self._orbital_rotations = []

        for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    self._orbital_rotations.append([i, j])

    def _create_parameter_list_for_orbital_rotations(self) -> None:
        """ Initializes the initial values of orbital rotation matrix kappa. """

        # creates the indices of matrix kappa and prevent user from trying to rotate only betas
        if self._orbital_rotations is None:
            self._create_orbital_rotation_list()
        elif self._orbital_rotations is None and self._orbital_rotations_beta is not None:
            raise AquaError('Only beta orbitals labels (orbital_rotations_beta) have been provided.'
                            'Please also specify the alpha orbitals (orbital_rotations) '
                            'that are rotated as well. Do not specify anything to have by default '
                            'all orbitals rotated.')

        if self._orbital_rotations_beta is not None:
            num_parameters = len(self._orbital_rotations + self._orbital_rotations_beta)
        else:
            num_parameters = len(self._orbital_rotations)
        self._parameters = [self._parameter_initial_value for _ in range(num_parameters)]

    def _create_parameter_bounds(self) -> None:
        """ Create bounds for parameters. """
        self._parameter_bounds = [self._parameter_bound_value for _ in range(self._num_parameters)]

    def orbital_rotation_matrix(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Creates 2 matrices K_alpha, K_beta that rotate the orbitals through MO coefficient
        C_alpha = C_RHF * U_alpha where U = e^(K_alpha), similarly for beta orbitals. """

        self._parameters = parameters
        k_matrix_alpha = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))
        k_matrix_beta = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))

        # allows to selectively rotate pairs of orbitals
        if self._orbital_rotations_beta is None:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[i]
        else:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]

            for j, exc in enumerate(self._orbital_rotations_beta):
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[j + len(self._orbital_rotations)]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[j + len(self._orbital_rotations)]

        if self._freeze_core:
            half_as = int(self._dim_kappa_matrix + len(self._core_list))
            k_matrix_alpha_full = np.zeros((half_as, half_as))
            k_matrix_beta_full = np.zeros((half_as, half_as))
            # rotating only non-frozen part of orbitals
            dim_full_k = k_matrix_alpha_full.shape[0]  # pylint: disable=unsubscriptable-object

            if self._core_list is None:
                raise AquaError('Give _core_list, the list of molecular spatial orbitals that are '
                                'frozen (e.g. [0] for the 1s or [0,1] for respectively Li2 or N2 '
                                'for example).')
            lower = len(self._core_list)
            upper = dim_full_k
            k_matrix_alpha_full[lower:upper, lower:upper] = k_matrix_alpha
            k_matrix_beta_full[lower:upper, lower:upper] = k_matrix_beta
            self._matrix_a = expm(k_matrix_alpha_full)
            self._matrix_b = expm(k_matrix_beta_full)
        else:
            self._matrix_a = expm(k_matrix_alpha)
            self._matrix_b = expm(k_matrix_beta)

        return self._matrix_a, self._matrix_b

    @property
    def matrix_a(self) -> np.ndarray:
        """Returns matrix A."""
        return self._matrix_a

    @property
    def matrix_b(self) -> np.ndarray:
        """Returns matrix B. """
        return self._matrix_b

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters."""
        return self._num_parameters

    @property
    def parameter_bound_value(self) -> Tuple[float, float]:
        """Returns a value for the bounds on all the parameters."""
        return self._parameter_bound_value


class OOVQEResult(ElectronicStructureResult):
    r""" OOVQE Result. """

    @property
    def computed_electronic_energy(self) -> float:
        """ Returns the ground state energy. """
        return self.get('computed_electronic_energy')

    @computed_electronic_energy.setter
    def computed_electronic_energy(self, value: float) -> None:
        """ Sets the ground state energy. """
        self.data['computed_electronic_energy'] = value

    @property
    def cost_function_evals(self) -> int:
        """ Returns number of cost function evaluations. """
        return self.get('cost_function_evals')

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """ Sets the number of cost function evaluations. """
        self.data['cost_function_evals'] = value

    @property
    def num_optimizer_evals(self) -> int:
        """ Returns the number of cost function evaluations in the optimizer """
        return self.get('num_optimizer_evals')

    @num_optimizer_evals.setter
    def num_optimizer_evals(self, value: float) -> None:
        """ Sets the number of cost function evaluations in the optimizer """
        self.data['num_optimizer_evals'] = value

    @property
    def optimal_point(self) -> list:
        """ Returns the optimal parameters. """
        return self.get('optimal_point')

    @optimal_point.setter
    def optimal_point(self, value: list) -> None:
        """ Sets the optimal parameters. """
        self.data['optimal_point'] = value

    @property
    def optimal_point_ansatz(self) -> list:
        """ Returns the optimal parameters for the . """
        return self.get('optimal_point_ansatz')

    @optimal_point_ansatz.setter
    def optimal_point_ansatz(self, value: list) -> None:
        """ Sets the optimal parameters for the ansatz. """
        self.data['optimal_point_ansatz'] = value

    @property
    def optimal_point_orbitals(self) -> list:
        """ Returns the optimal parameters of the orbitals. """
        return self.get('optimal_point_orbitals')

    @optimal_point_orbitals.setter
    def optimal_point_orbitals(self, value: list) -> None:
        """ Sets the optimal parameters of the orbitals. """
        self.data['optimal_point_orbitals'] = value
