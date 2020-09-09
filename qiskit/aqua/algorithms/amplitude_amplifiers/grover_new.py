# -*- coding: utf-8 -*-

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

"""Grover's Search algorithm."""

from typing import Optional, Union, Dict, List, Any
import warnings
import logging
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers import BaseBackend
from qiskit.quantum_info import Statevector, DensityMatrix, Operator

from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.oracles import Oracle, TruthTableOracle
from qiskit.aqua.components.initial_states import InitialState

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class Grover_new(QuantumAlgorithm):
    r"""Grover's Search algorithm.

    Grover’s Search is a well known quantum algorithm for searching through
    unstructured collections of records for particular targets with quadratic
    speedup compared to classical algorithms.

    Given a set :math:`X` of :math:`N` elements :math:`X=\{x_1,x_2,\ldots,x_N\}`
    and a boolean function :math:`f : X \rightarrow \{0,1\}`, the goal of an
    unstructured-search problem is to find an element :math:`x^* \in X` such
    that :math:`f(x^*)=1`.

    Unstructured search is often alternatively formulated as a database search
    problem, in which, given a database, the goal is to find in it an item that
    meets some specification.

    The search is called *unstructured* because there are no guarantees as to how
    the database is ordered.  On a sorted database, for instance, one could perform
    binary search to find an element in :math:`\mathbb{O}(\log N)` worst-case time.
    Instead, in an unstructured-search problem, there is no prior knowledge about
    the contents of the database. With classical circuits, there is no alternative
    but to perform a linear number of queries to find the target element.
    Conversely, Grover's Search algorithm allows to solve the unstructured-search
    problem on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries.

    All that is needed for carrying out a search is an oracle from Aqua's
    :mod:`~qiskit.aqua.components.oracles` module for specifying the search criterion,
    which basically indicates a hit or miss for any given record.  More formally, an
    oracle :math:`O_f` is an object implementing a boolean function
    :math:`f` as specified above.  Given an input :math:`x \in X`,
    :math:`O_f` implements :math:`f(x)`.  The details of how :math:`O_f` works are
    unimportant; Grover's search algorithm treats the oracle as a black box.

    For example the :class:`~qiskit.aqua.components.oracles.LogicalExpressionOracle`
    can take as input a SAT problem in
    `DIMACS CNF format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__
    and be used with Grover algorithm to find a satisfiable assignment.

    Signature: 

    Q = A S_0 A_dg S_f 
    Grover(bad_state_reflection, (or marker, oracle? should be the same name as the circuit library)
    is_good_state: callable | List[int] | Statevector),
    num_iterations: None,
    num_solutions: None / incremental = False,
    lam = 1.34,
    grover_operator=None (just pass custom grover operator)
    )

    Should internally use Grover operator to construct Q, then "applying j iterations of Grover"
    only means apply Q j-times where, Q is the grover op)


    num_solutions: is used to decide num_iterations
    is_good_state: using for what? Is this the answer of the problem? If so, we don't need to run
    the Grover algorithm.
    """

    # Constructor of the Grover operator. For the reference.
    # def __init__(self, oracle: Union[QuantumCircuit, Statevector],
    #               state_in: Optional[QuantumCircuit] = None,
    #               zero_reflection: Optional[Union[QuantumCircuit, DensityMatrix, Operator]] = None
    #               reflection_qubits: Optional[List[int]] = None,
    #               insert_barriers: bool = False,
    #               mcx: str = 'noancilla',
    #               name: str = 'Q') -> None:

    # The new constructor for Grover class
    def __init__(self, oracle: Union[Oracle, QuantumCircuit, Statevector],
                 init_state: Optional[Union[InitialState, QuantumCircuit]] = None,
                 incremental: bool = False,
                 num_iterations: int = 1,
                 lam: float = 1.34,
                 rotation_counts: Optional[list] = None,
                 num_solutions: Optional[int] = None,
                 mct_mode: str = 'noancilla',
                 insert_barriers: bool = False,
                 zero_reflection: Optional[Union[QuantumCircuit, DensityMatrix, Operator]] = None,
                 is_good_state: Union[callable, List[int], Statevector] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 name: str = 'Q',
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:

        # The original constructor for Grover class
        # def __init__(self,
        #              oracle: Oracle, init_state: Optional[InitialState = None],
        #              incremental: bool = False,
        #              num_iterations: int = 1,
        #              lam: float = 1.34,
        #              rotation_counts: Optional[list] = None,
        #              mct_mode: str = 'basic',
        #              quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None
        #              ) -> None:
        # pylint: disable=line-too-long
        r"""
        Args:
            oracle: The oracle component
            init_state: An optional initial quantum state. If None (default) then Grover's Search
                 by default uses uniform superposition to initialize its quantum state. However,
                 an initial state may be supplied, if useful, for example, if the user has some
                 prior knowledge regarding where the search target(s) might be located.
            incremental: Whether to use incremental search mode (True) or not (False).
                 Supplied *num_iterations* is ignored when True and instead the search task will
                 be carried out in successive rounds, using circuits built with incrementally
                 higher number of iterations for the repetition of the amplitude amplification
                 until a target is found or the maximal number :math:`\log N` (:math:`N` being the
                 total number of elements in the set from the oracle used) of iterations is
                 reached. The implementation follows Section 4 of [2].
            lam: For incremental search mode, the maximum number of repetition of amplitude
                 amplification increases by factor lam in every round,
                 :math:`R_{i+1} = lam \times R_{i}`. If this parameter is not set, the default
                 value lam = 1.34 is used, which is proved to be optimal [1].
            rotation_counts: For incremental mode, if rotation_counts is defined, parameter *lam*
                is ignored. rotation_counts is the list of integers that defines the number of
                repetition of amplitude amplification for each round.
            num_iterations: How many times the marking and reflection phase sub-circuit is
                repeated to amplify the amplitude(s) of the target(s). Has a minimum value of 1.
            mct_mode: Multi-Control Toffoli mode ('basic' | 'basic-dirty-ancilla' |
                'advanced' | 'noancilla')
            quantum_instance: Quantum Instance or Backend
            grover_operator: TODO
            insert_barriers: TODO
            is_good_state: TODO
            name: TODO
            num_solutions: TODO
            zero_reflection: TODO

        Raises:
            AquaError: evaluate_classically() missing from the input oracle
            ValueError: If ``oracle`` is of unsupported type.
            ValueError: If ``init_state`` is of unsupported type.

        References:
            [1]: Baritompa et al., Grover's Quantum Algorithm Applied to Global Optimization
                 `<https://www.researchgate.net/publication/220133694_Grover%27s_Quantum_Algorithm_Applied_to_Global_Optimization>`_
            [2]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        validate_min('num_iterations', num_iterations, 1)
        validate_in_set('mct_mode', mct_mode,
                        {'basic', 'basic-dirty-ancilla',
                         'advanced', 'noancilla'})
        super().__init__(quantum_instance)

        if not callable(getattr(oracle, "evaluate_classically", None)):
            raise AquaError(
                'Missing the evaluate_classically() method from the provided oracle instance.'
            )

        # TODO
        # Call GroverOperator with the information of arguments. Construct circuits for state_in,
        # oracle, and diffusion.
        # If arguments are not circuits, output deprecated messages.

        # self._oracle = oracle
        # self._mct_mode = mct_mode
        # self._init_state = \
        #     init_state if init_state else Custom(len(oracle.variable_register), state='uniform')
        # self._init_state_circuit = \
        #     self._init_state.construct_circuit(mode='circuit', register=oracle.variable_register)
        # self._init_state_circuit_inverse = self._init_state_circuit.inverse()

        # self._diffusion_circuit = self._construct_diffusion_circuit()

        if mct_mode is not None:
            warnings.warn('The mct_mode argument is deprecated as of 0.8.0, and will be removed no '
                          'earlier than 3 months after the release date. You should use the '
                          'mcx_mode argument instead.', DeprecationWarning, stacklevel=2)
            mcx_mode = mct_mode
        self._mcx_mode = mcx_mode

        # Construct GroverOperator circuit
        if grover_operator is None:
            # Grover operatorがNoneなら自分で作る。以下はif GroverOperator is Noneのelseに書くべきかも
            if isinstance(oracle, (QuantumCircuit, Statevector)):
                _oracle = oracle
            elif isinstance(oracle, Oracle):
                warnings.warn('Passing an qiskit.aqua.components.oracles.Oracle object is '
                              'deprecated as of 0.8.0, and the support will be removed no '
                              'earlier than 3 months after the release date. You should pass a '
                              'QuantumCircuit or Statevector argument instead. See also the '
                              'qiskit.circuit.library.GroverOperator for more information.',
                              DeprecationWarning, stacklevel=2)
                _oracle = QuantumCircuit(oracle.circuit.num_qubits)

                if isinstance(oracle, TruthTableOracle):
                    index = 0
                    for qreg in oracle.circuit.qregs:
                        if qreg.name == "o":
                            break
                        index += qreg.size
                    _output_register = index
                else:
                    _output_register = [i for i, qubit in enumerate(oracle.circuit.qubits)
                                        if qubit in oracle.output_register[:]]

                _oracle.x(_output_register)
                _oracle.h(_output_register)
                _oracle.compose(oracle.circuit, list(range(oracle.circuit.num_qubits)),
                                inplace=True)
                _oracle.h(_output_register)
                _oracle.x(_output_register)

                _reflection_qubits = [i for i, qubit in enumerate(oracle.circuit.qubits)
                                      if qubit in oracle.variable_register[:]]

                self._is_good_state = oracle.evaluate_classically
            else:
                raise ValueError('Unsupported type "{}" of oracle'.format(type(oracle)))

            # need a deprecation message for initial_state, if we change the name of the argument to
            # state_in should we cover the case of isinstance(state_in, InitialState) and
            # isinstance(oracle, QuantumCircuit)?
            if isinstance(init_state, QuantumCircuit):
                _state_preparation = init_state
            elif isinstance(init_state, InitialState) and isinstance(oracle, Oracle):
                # add deprecation message, please pass circuit
                _state_preparation = init_state.construct_circuit(
                    mode='circuit', register=oracle.variable_register
                    )
            elif init_state is None:
                _state_preparation = init_state
            else:
                raise ValueError('Unsupported type "{}" of init_state'.format(type(init_state)))

            self._grover_operator = GroverOperator(
                oracle=_oracle, state_preparation=_state_preparation,
                reflection_qubits=_reflection_qubits, mcx_mode=self._mcx_mode)
        else:
            self._grover_operator = grover_operator

        self._max_num_iterations = np.ceil(2 ** (len(oracle.variable_register) / 2))
        self._incremental = incremental
        self._lam = lam
        self._rotation_counts = rotation_counts
        self._num_iterations = num_iterations if not incremental else 1
        if incremental:
            logger.debug('Incremental mode specified, ignoring "num_iterations".')
        else:
            if num_iterations > self._max_num_iterations:
                logger.warning('The specified value %s for "num_iterations" '
                               'might be too high.', num_iterations)
        self._ret = {}  # type: Dict[str, Any]
        self._qc_aa_iteration = None
        self._qc_amplitude_amplification = None
        self._qc_measurement = None

        # TODO
        # We don't need _construct_diffusion_circuit() method since GroverOperator creates diffusion
        # circuit for us.

    # def _construct_diffusion_circuit(self):
    #     qc = QuantumCircuit(self._oracle.variable_register)
    #     num_variable_qubits = len(self._oracle.variable_register)
    #     num_ancillae_needed = 0
    #     if self._mct_mode == 'basic' or self._mct_mode == 'basic-dirty-ancilla':
    #         num_ancillae_needed = max(0, num_variable_qubits - 2)
    #     elif self._mct_mode == 'advanced' and num_variable_qubits >= 5:
    #         num_ancillae_needed = 1

    #     # check oracle's existing ancilla and add more if necessary
    #     num_oracle_ancillae = \
    #         len(self._oracle.ancillary_register) if self._oracle.ancillary_register else 0
    #     num_additional_ancillae = num_ancillae_needed - num_oracle_ancillae
    #     if num_additional_ancillae > 0:
    #         extra_ancillae = QuantumRegister(num_additional_ancillae, name='a_e')
    #         qc.add_register(extra_ancillae)
    #         ancilla = list(extra_ancillae)
    #         if num_oracle_ancillae > 0:
    #             ancilla += list(self._oracle.ancillary_register)
    #     else:
    #         ancilla = self._oracle.ancillary_register

    #     if self._oracle.ancillary_register:
    #         qc.add_register(self._oracle.ancillary_register)
    #     qc.barrier(self._oracle.variable_register)
    #     qc += self._init_state_circuit_inverse
    #     qc.u3(pi, 0, pi, self._oracle.variable_register)
    #     qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
    #     qc.mct(
    #         self._oracle.variable_register[0:num_variable_qubits - 1],
    #         self._oracle.variable_register[num_variable_qubits - 1],
    #         ancilla,
    #         mode=self._mct_mode
    #     )
    #     qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
    #     qc.u3(pi, 0, pi, self._oracle.variable_register)
    #     qc += self._init_state_circuit
    #     qc.barrier(self._oracle.variable_register)
    #     return qc

    # TODO
    # We don't need qc_amplitude_amplification_iteration() too. We can just return self.grover_op
    # for the grover operator circuit

    @property
    def qc_amplitude_amplification_iteration(self):
        """ qc amplitude amplification iteration """
#        if self._qc_aa_iteration is None:
        # self._qc_aa_iteration = QuantumCircuit()
        # self._qc_aa_iteration += self._oracle.circuit
        # self._qc_aa_iteration += self._diffusion_circuit

        # return self._qc_aa_iteration
        return self._grover_operator

        # TODO
        # This method is to run the Grover circuit that is already created in construct_circuit()
        # method.
        # The circuit contains the Grover operator circuit with the specified number of iterations.
        # So we can just run the circuit.

    def _run_with_existing_iterations(self):
        if self._quantum_instance.is_statevector:

            # TODO
            # For the statevector simulator, I need to read the following code to see what's
            # happening. Probably, returning the result with the highest probability?

            qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            # variable_register_density_matrix = get_subsystem_density_matrix(
            #     complete_state_vec,
            #     range(len(self._oracle.variable_register), qc.width())
            # )
            variable_register_density_matrix = get_subsystem_density_matrix(
                complete_state_vec,
                range(len(self._grover_operator.reflection_qubits), qc.width())
            )
            variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
            max_amplitude = max(
                variable_register_density_matrix_diag.min(),
                variable_register_density_matrix_diag.max(),
                key=abs
            )
            max_amplitude_idx = \
                np.where(variable_register_density_matrix_diag == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, len(
                self._grover_operator.reflection_qubits))
        else:

            # TODO When it's not the statevector simulator. We can just run the circuit.

            qc = self.construct_circuit(measurement=True)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement
        # oracle_evaluation, assignment = self._oracle.evaluate_classically(top_measurement)
        if callable(self._is_good_state):
            oracle_evaluation, assignment = self._is_good_state(top_measurement)
        else:
            raise ValueError('Unsupported type "{}" of is_good_state'.format(
                type(self._is_good_state)))

        return assignment, oracle_evaluation

    def construct_circuit(self, measurement=False):
        """Construct the quantum circuit.

        Args:
            measurement (bool): Boolean flag to indicate if
                measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """

        # TODO I need to check how the incremental version of Grover algorithm works.
        # It gradually increases the number of iterations, but when it will stop?

        if self._incremental:
            if self._qc_amplitude_amplification is None:
                # self._qc_amplitude_amplification = \
                #     QuantumCircuit() + self.qc_amplitude_amplification_iteration
                self._qc_amplitude_amplification = QuantumCircuit(self._grover_operator.num_qubits)
                self._qc_amplitude_amplification.compose(
                    self.qc_amplitude_amplification_iteration,
                    list(range(self._grover_operator.num_qubits)), inplace=True
                    )
        else:

            # TODO Call self._qc_amplitude_amplification.combine(self.grover_op) in the iteration
            # to create a whole circuit for the grover algorithm with the specified number of
            # iterations.

            # self._qc_amplitude_amplification = QuantumCircuit()
            self._qc_amplitude_amplification = QuantumCircuit(self._grover_operator.num_qubits)

            for _ in range(self._num_iterations):
                # self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
                self._qc_amplitude_amplification.compose(self._grover_operator, list(
                    range(self._grover_operator.num_qubits)), inplace=True)

        # Create Initial state. This process should include initializeing oracle as |-> state as
        # well.

        # qc = QuantumCircuit(self._oracle.variable_register, self._oracle.output_register)
        qc = QuantumCircuit(self._grover_operator.num_qubits, name='Grover')

        # ----------------------------------------------
        # qc.u3(pi, 0, pi, self._oracle.output_register)  # x
        # qc.u2(0, pi, self._oracle.output_register)  # h
        # ----------------------------------------------

        # Create super positions for initial state.
        qc.compose(self._grover_operator.state_preparation, list(
            range(self._grover_operator.num_qubits)), inplace=True)

#        qc += self._init_state_circuit
#        qc += self._qc_amplitude_amplification

        qc.compose(self._qc_amplitude_amplification, list(
            range(self._grover_operator.num_qubits)), inplace=True)

        if measurement:
            # measure output_register as well?
            # measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            # qc.add_register(measurement_cr)
            # qc.measure(self._oracle.variable_register, measurement_cr
            measurement_cr = ClassicalRegister(
                len(self._grover_operator.reflection_qubits), name='m')
            qc.add_register(measurement_cr)
            qc.measure(self._grover_operator.reflection_qubits, measurement_cr)

        self._ret['circuit'] = qc
        return qc

    def _run(self):
        # TODO I need to check how the incremental version of Grover algorithm works

        if self._incremental:

            def _try_target_num_iterations():
                # Create a grover circuit with the specified number of iterations
                # (target_num_iterations) for the incremental version,
                # and then execute it.

                self._qc_amplitude_amplification = QuantumCircuit(self._grover_operator.num_qubits)
                for _ in range(int(target_num_iterations)):
                    # self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
                    self._qc_amplitude_amplification.compose(self._grover_operator, list(
                        range(self._grover_operator.num_qubits)), inplace=True)
                return self._run_with_existing_iterations()

            # TODO Rotation counts?

            if self._rotation_counts:
                for target_num_iterations in self._rotation_counts:
                    assignment, oracle_evaluation = _try_target_num_iterations()
                    if oracle_evaluation:
                        break
                    if target_num_iterations > self._max_num_iterations:
                        break
            else:
                current_max_num_iterations = 1
                while current_max_num_iterations < self._max_num_iterations:
                    target_num_iterations = self.random.integers(current_max_num_iterations) + 1
                    assignment, oracle_evaluation = _try_target_num_iterations()
                    if oracle_evaluation:
                        break
                    current_max_num_iterations = \
                        min(self._lam * current_max_num_iterations, self._max_num_iterations)

        else:
            # For the usual version (not the incremental version) we just call
            # self._run_with_existing_iterations()
            # which run the grover circuit with the specified number of iterations
            self._qc_amplitude_amplification = QuantumCircuit()
            assignment, oracle_evaluation = self._run_with_existing_iterations()

        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation
        return self._ret
