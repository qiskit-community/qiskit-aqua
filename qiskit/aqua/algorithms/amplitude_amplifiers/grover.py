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

"""Grover's search algorithm."""

from typing import Optional, Union, Dict, List, Any, Callable
import logging
import warnings
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers import BaseBackend
from qiskit.quantum_info import Statevector

from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils import get_subsystem_density_matrix, name_args
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.algorithms import QuantumAlgorithm, AlgorithmResult
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.oracles import Oracle, TruthTableOracle


logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
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

    Should internally use Grover operator to construct Q, then "applying j iterations of Grover"
    only means apply Q j-times where, Q is the grover op)
    """

    @name_args([
        ('oracle', ),
        ('good_state', {InitialState: 'init_state'}),
        ('state_preparation', {bool: 'incremental'}),
        ('iterations', ),
        ('post_processing', {float: 'lam'}),
        ('grover_operator', {list: 'rotation_counts'}),
        ('quantum_instance', {str: 'mct_mode'}),
        ('incremental', {(BaseBackend, QuantumInstance): 'quantum_instance'})
    ], skip=1)  # skip the argument 'self'
    def __init__(self,
                 oracle: Union[Oracle, QuantumCircuit, Statevector],
                 good_state: Union[Callable[[str], bool], List[int], Statevector] = None,
                 state_preparation: Optional[Union[QuantumCircuit, bool]] = None,
                 iterations: Union[int, List[int]] = 1,
                 post_processing: Callable[[List[int]], List[int]] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
                 init_state: Optional[InitialState] = None,
                 incremental: bool = False,
                 num_iterations: Optional[int] = None,
                 lam: Optional[float] = None,
                 rotation_counts: Optional[List[int]] = None,
                 mct_mode: Optional[str] = None,
                 ) -> None:
        # pylint: disable=line-too-long
        r"""
        Args:
            oracle: The oracle component
            state_preparation: An optional initial quantum state. If None (default) then Grover's
                 Search by default uses uniform superposition to initialize its quantum state.
                 However, an initial state may be supplied, if useful, for example, if the user has
                 some prior knowledge regarding where the search target(s) might be located.
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
            mct_mode: Multi-Control Toffoli mode ('basic' | 'basic-dirty-ancilla' |
                'advanced' | 'noancilla')
            quantum_instance: Quantum Instance or Backend
            grover_operator: A GroverOperator for the Grover's algorithm can be set directly.
            good_state: Answers the Grover's algorithm is looking for.
                It is used to check whether the result is correct or not.
            iterations: TODO
            post_processing: TODO
            init_state: An optional initial quantum state. If None (default) then Grover's Search
                 by default uses uniform superposition to initialize its quantum state. However,
                 an initial state may be supplied, if useful, for example, if the user has some
                 prior knowledge regarding where the search target(s) might be located.
            num_iterations: How many times the marking and reflection phase sub-circuit is
                repeated to amplify the amplitude(s) of the target(s). Has a minimum value of 1.

        Raises:
            TypeError: If ``init_state`` is of unsupported type or is of type ``InitialState` but
                the oracle is not of type ``Oracle``.
            AquaError: evaluate_classically() missing from the input oracle
            TypeError: If ``oracle`` is of unsupported type.


        References:
            [1]: Baritompa et al., Grover's Quantum Algorithm Applied to Global Optimization
                 `<https://www.researchgate.net/publication/220133694_Grover%27s_Quantum_Algorithm_Applied_to_Global_Optimization>`_
            [2]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        super().__init__(quantum_instance)

        # init_state has been renamed to state_preparation
        if init_state is not None:
            warnings.warn('The init_state argument is deprecated as of 0.8.0, and will be removed '
                          'no earlier than 3 months after the release date. You should use the '
                          'state_preparation argument instead and pass a QuantumCircuit or '
                          'Statevector instead of an InitialState.',
                          DeprecationWarning, stacklevel=3)
            state_preparation = init_state

        if mct_mode is not None:
            validate_in_set('mct_mode', mct_mode,
                            {'basic', 'basic-dirty-ancilla',
                             'advanced', 'noancilla'})
            warnings.warn('The mct_mode argument is deprecated as of 0.8.0, and will be removed no '
                          'earlier than 3 months after the release date. If you want to use a '
                          'special MCX mode you should use the GroverOperator in '
                          'qiskit.circuit.library directly and pass it to the grover_operator '
                          'keyword argument.', DeprecationWarning, stacklevel=3)
        else:
            mct_mode = 'noancilla'

        if rotation_counts is not None:
            warnings.warn('The rotation_counts argument is deprecated as of 0.8.0, and will be '
                          'removed no earlier than 3 months after the release date. '
                          'If you want to use the incremental mode with the rotation_counts '
                          'argument or you should use the iterations argument instead and pass '
                          'a list of integers',
                          DeprecationWarning, stacklevel=3)

        if lam is not None:
            warnings.warn('The lam argument is deprecated as of 0.8.0, and will be '
                          'removed no earlier than 3 months after the release date. '
                          'If you want to use the incremental mode with the lam argument, '
                          'you should use the iterations argument instead and pass '
                          'a list of integers calculated with the lam argument.',
                          DeprecationWarning, stacklevel=3)
        else:
            lam = 1.34

        if num_iterations is not None:
            validate_min('num_iterations', num_iterations, 1)
            warnings.warn('The num_iterations argument is deprecated as of 0.8.0, and will be '
                          'removed no earlier than 3 months after the release date. '
                          'If you want to use the num_iterations argument '
                          'you should use the iterations argument instead and pass an integer '
                          'for the number of iterations.',
                          DeprecationWarning, stacklevel=3)

        self._oracle = oracle
        # Construct GroverOperator circuit
        if grover_operator is not None:
            self._grover_operator = grover_operator
        else:
            # check the type of state_preparation
            if isinstance(state_preparation, InitialState):
                warnings.warn('Passing an InitialState component is deprecated as of 0.8.0, and '
                              'will be removed no earlier than 3 months after the release date. '
                              'You should pass a QuantumCircuit instead.',
                              DeprecationWarning, stacklevel=3)
                if isinstance(oracle, Oracle):
                    state_preparation = init_state.construct_circuit(
                        mode='circuit', register=oracle.variable_register
                        )
                else:
                    raise TypeError('If init_state is of type InitialState, oracle must be of type '
                                    'Oracle')
            elif not (isinstance(state_preparation, QuantumCircuit) or state_preparation is None):
                raise TypeError('Unsupported type "{}" of state_preparation'.format(
                    type(state_preparation)))

            # check to oracle type and if necessary convert the deprecated Oracle component to
            # a circuit
            reflection_qubits = None
            if isinstance(oracle, Oracle):
                if not callable(getattr(oracle, "evaluate_classically", None)):
                    raise AquaError(
                        'Missing the evaluate_classically() method \
                            from the provided oracle instance.'
                    )

                oracle, reflection_qubits, good_state = _oracle_component_to_circuit(oracle)
            elif not isinstance(oracle, (QuantumCircuit, Statevector)):
                raise TypeError('Unsupported type "{}" of oracle'.format(type(oracle)))

            self._grover_operator = GroverOperator(oracle=oracle,
                                                   state_preparation=state_preparation,
                                                   reflection_qubits=reflection_qubits,
                                                   mcx_mode=mct_mode)

        self._is_good_state = good_state
        self._post_processing = post_processing
        self._incremental = incremental
        self._lam = lam
        self._rotation_counts = rotation_counts
        self._max_num_iterations = np.ceil(2 ** (len(self._grover_operator.reflection_qubits) / 2))

        if incremental:
            if rotation_counts is not None:
                self._iterations = rotation_counts
            else:
                self._iterations = []
                current_max_num_iterations = 1.0
                while current_max_num_iterations < self._max_num_iterations:
                    self._iterations.append(current_max_num_iterations)
                    current_max_num_iterations = self._lam * current_max_num_iterations
        elif num_iterations is not None:
            self._iterations = [num_iterations]
            self._num_iterations = num_iterations
        elif isinstance(iterations, list):
            self._iterations = iterations
        else:
            validate_min('num_iterations', iterations, 1)
            self._iterations = [iterations]
            self._num_iterations = iterations

        if incremental or (isinstance(iterations, list) and len(iterations) > 1):
            logger.debug('Incremental mode specified, \
                ignoring "num_iterations" and "num_solutions".')
        elif self._max_num_iterations is not None:
            if self._num_iterations > self._max_num_iterations:
                logger.warning('The specified value %s for "num_iterations" '
                               'might be too high.', self._num_iterations)
        self._ret = {}  # type: Dict[str, Any]

    @staticmethod
    def optimal_num_iterations(num_solutions, num_qubits):
        """Return the optimal number of iterations, if the number of solutions is known.

        Args:
            num_solutions: The number of solutions.
            num_qubits: The number of qubits used to encode the states.

        Returns:
            The optimal number of iterations for Grover's algorithm to succeed.
        """
        return round((np.pi * np.sqrt(2 ** num_qubits) / num_solutions) / 4)

    def _run_experiment(self, power):
        """Run a grover experiment for a given power of the Grover operator."""
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(power, measurement=False)
            result = self._quantum_instance.execute(qc)
            statevector = result.get_statevector(qc)
            num_bits = len(self._grover_operator.reflection_qubits)
            # trace out work qubits
            if qc.width() != num_bits:
                rho = get_subsystem_density_matrix(
                    statevector,
                    range(num_bits, qc.width())
                )
                statevector = np.diag(rho)
            max_amplitude = max(statevector.max(), statevector.min(), key=abs)
            max_amplitude_idx = np.where(statevector == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, num_bits)

        else:
            qc = self.construct_circuit(power, measurement=True)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement

        return self.post_processing(top_measurement), self.is_good_state(top_measurement)

    def is_good_state(self, bitstr: str) -> bool:
        """Check whether a provided bitstring is a good state or not.

        Args:
            bitstr: The measurement as bitstring.

        Raises:
            NotImplementedError: If self._is_good_state couldn't be used to determine whether
                the bitstring is a good state.

        Returns:
            True if the measurement is a good state, False otherwise.
        """
        if callable(self._is_good_state):
            oracle_evaluation, _ = self._is_good_state(bitstr)
            return oracle_evaluation
        elif isinstance(self._is_good_state, list):
            return bitstr in self._is_good_state
        elif isinstance(self._is_good_state, Statevector):
            return bitstr in self._is_good_state.probabilities_dict()
        else:
            raise NotImplementedError('Conversion to callable not implemented for {}'.format(
                type(self._is_good_state)))

    def post_processing(self, bitstr: str) -> str:
        """Do the post-processing to the measurement result

        Args:
            bitstr: The measurement as bitstring.

        Returns:
            Do the post-processing based on the post_processing argument.
            If the post_processing argument is None and the Oracle class is used as its oracle,
            oracle.evaluate_classically is used as the post_processing.
            Otherwise, just return the input bitstr
        """
        if self._post_processing is not None:
            return self._post_processing(bitstr)

        if isinstance(self._oracle, Oracle):
            return self._oracle.evaluate_classically(bitstr)[1]

        return bitstr

    def construct_circuit(self, power: Optional[int] = None,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators.

        Args:
            power: The number of times the Grover operator is repeated. If None, this argument
                is set to ``num_iterations``.
            measurement: Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        if power is None:
            power = self._num_iterations

        qc = QuantumCircuit(self._grover_operator.num_qubits, name='Grover circuit')
        qc.compose(self._grover_operator.state_preparation, inplace=True)
        if power > 0:
            qc.compose(self._grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(self._grover_operator.reflection_qubits))
            qc.add_register(measurement_cr)
            qc.measure(self._grover_operator.reflection_qubits, measurement_cr)

        self._ret['circuit'] = qc
        return qc

    def _run(self) -> 'GroverResult':
        # If ``rotation_counts`` is specified, run Grover's circuit for the powers specified
        # in ``rotation_counts``. Once a good state is found (oracle_evaluation is True), stop.
        if not (self._incremental and self._rotation_counts is None):
            for target_num_iterations in self._iterations:
                assignment, oracle_evaluation = self._run_experiment(target_num_iterations)
                if oracle_evaluation:
                    break
                if target_num_iterations > self._max_num_iterations:
                    break
        else:
            for current_max_num_iterations in self._iterations:
                target_num_iterations = self.random.integers(current_max_num_iterations) + 1
                assignment, oracle_evaluation = self._run_experiment(target_num_iterations)
                if oracle_evaluation:
                    break

        # TODO remove all former dictionary logic
        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation

        result = GroverResult()
        if 'measurement' in self._ret:
            result.measurement = dict(self._ret['measurement'])
        if 'top_measurement' in self._ret:
            result.top_measurement = self._ret['top_measurement']
        if 'circuit' in self._ret:
            result.circuit = self._ret['circuit']
        result.assignment = self._ret['result']
        result.oracle_evaluation = self._ret['oracle_evaluation']
        return result


def _oracle_component_to_circuit(oracle: Oracle):
    """Convert an Oracle to a QuantumCircuit."""
    circuit = QuantumCircuit(oracle.circuit.num_qubits)

    if isinstance(oracle, TruthTableOracle):
        index = 0
        for qreg in oracle.circuit.qregs:
            if qreg.name == "o":
                break
            index += qreg.size
        _output_register = [index]
    else:
        _output_register = [i for i, qubit in enumerate(oracle.circuit.qubits)
                            if qubit in oracle.output_register[:]]

    circuit.x(_output_register)
    circuit.h(_output_register)
    circuit.compose(oracle.circuit, list(range(oracle.circuit.num_qubits)),
                    inplace=True)
    circuit.h(_output_register)
    circuit.x(_output_register)

    reflection_qubits = [i for i, qubit in enumerate(oracle.circuit.qubits)
                         if qubit in oracle.variable_register[:]]

    is_good_state = oracle.evaluate_classically

    return circuit, reflection_qubits, is_good_state


class GroverResult(AlgorithmResult):
    """Grover Result."""

    @property
    def measurement(self) -> Optional[Dict[str, int]]:
        """ returns measurement """
        return self.get('measurement')

    @measurement.setter
    def measurement(self, value: Dict[str, int]) -> None:
        """ set measurement """
        self.data['measurement'] = value

    @property
    def top_measurement(self) -> Optional[str]:
        """ return top measurement """
        return self.get('top_measurement')

    @top_measurement.setter
    def top_measurement(self, value: str) -> None:
        """ set top measurement """
        self.data['top_measurement'] = value

    @property
    def circuit(self) -> Optional[QuantumCircuit]:
        """ return circuit """
        return self.get('circuit')

    @circuit.setter
    def circuit(self, value: QuantumCircuit) -> None:
        """ set circuit """
        self.data['circuit'] = value

    @property
    def assignment(self) -> List[int]:
        """ return assignment """
        return self.get('assignment')

    @assignment.setter
    def assignment(self, value: List[int]) -> None:
        """ set assignment """
        self.data['assignment'] = value

    @property
    def oracle_evaluation(self) -> bool:
        """ return oracle evaluation """
        return self.get('oracle_evaluation')

    @oracle_evaluation.setter
    def oracle_evaluation(self, value: bool) -> None:
        """ set oracle evaluation """
        self.data['oracle_evaluation'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'GroverResult':
        """ create new object from a dictionary """
        return GroverResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'result':
            warnings.warn('result deprecated, use assignment property.', DeprecationWarning)
            return super().__getitem__('assignment')

        return super().__getitem__(key)
