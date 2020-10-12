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

"""Result of running PhaseEstimator"""

from typing import Optional, Dict
import numpy
from qiskit.result import Result

# Maybe we want to use this abstract class
from qiskit.aqua.algorithms import AlgorithmResult


class PhaseEstimatorResult(AlgorithmResult):
    """Store and manipulate results from running `PhaseEstimator`.

    This class is instantiated by the `PhaseEstimator` class, not via user code.
    The `PhaseEstimator` class generates a list of phases and corresponding weights. Upon completion
    it returns the results as an instance of this class. The main method for accessing the results
    is `filter_phases`.
    """

    def __init__(self, num_evaluation_qubits: int,
                 circuit_result: Result,
                 phase_array: Optional[numpy.ndarray] = None,
                 phase_dict: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            num_evaluation_qubits: number of qubits in phase-readout register.
            circuit_result: result object returned by method running circuit.
            phase_array: ndarray of phases and frequencies determined by QPE.
            phase_dict: dict of phases and counts determined by QPE.

        Note:
            Only one of `phase_array` and `phase_dict` is not `None`. `phase_array`
            is not `None` if the QPE circuit was simulated by a statevector simulator.
        """
        # int: number of qubits in phase-readout register
        self._num_evaluation_qubits = num_evaluation_qubits

        # ndarray: weights of phases computed by QPE
        self._phase_array = phase_array

        # dict: (or subclass) weights of phases computed by QPE
        self._phase_dict = phase_dict

        # result of running the circuit (on hardware or simulator)
        self._circuit_result = circuit_result

        super().__init__()

    @property
    def phase_array(self) -> numpy.ndarray:
        """Return all phases and their frequencies computed by QPE.

        This is an array whose values correspond to weights on bit strings. Only one of
        `phase_array` and `phase_dict` is not `None`.
        """
        return self._phase_array

    @property
    def phase_dict(self) -> dict:
        """Return all phases and their frequencies computed by QPE.

        This is a dict whose keys are bit strings and values are weights on bit strings. Only one of
        `phase_array` and `phase_dict` is not `None`.
        """
        return self._phase_dict

    @property
    def circuit_result(self) -> Result:
        """Return the result object returned by running the QPE circuit (on hardware or simulator).

        This is useful for inspecting and troubleshooting the QPE algorithm.
        """
        return self._circuit_result

    def single_phase(self) -> float:
        r"""Return the estimated phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. It is assumed that the input vector is an
        eigenvector of the unitary so that the peak of the probability density occurs at the bit
        string that most closely approximates the true phase.
        """
        if self._phase_dict is not None:
            binary_phase_string = max(self._phase_dict, key=self._phase_dict.get)
        else:
            # numpy.argmax ignores complex part of number. But, we take abs anyway
            idx = numpy.argmax(abs(self._phase_array))
            binary_phase_string = numpy.binary_repr(idx, self._num_evaluation_qubits)[::-1]
        phase = _bit_string_to_phase(binary_phase_string)
        return phase

    def filter_phases(self, cutoff: float = 0.0,
                      as_float: bool = True) -> Dict:
        """Return a filtered dict of phases (keys) and frequencies (values).

        Only phases with frequencies (counts) larger than `cutoff` are included.
        It is assumed that the `run` method has been called so that the phases have been computed.
        When using a noiseless, shot-based simulator to read a single phase that can
        be represented exactly by `num_evaluation_qubits`, all the weight will
        be concentrated on a single phase. In all other cases, many, or all, bit
        strings will have non-zero weight. This method is useful for filtering
        out these uninteresting bit strings.

        Args:
            cutoff: Minimum weight of number of counts required to keep a bit string.
                    The default value is `0.0`.
            as_float: If `True`, returned keys are floats in :math:`[0.0, 1.0)`. If `False`
                      returned keys are bit strings.

        Returns:
            A filtered dict of phases (keys) and frequencies (values).
        """
        if self._phase_dict is not None:
            counts = self._phase_dict
            if as_float:
                phases = {_bit_string_to_phase(k): counts[k]
                          for k in counts.keys() if counts[k] > cutoff}
            else:
                phases = {k: counts[k] for k in counts.keys() if counts[k] > cutoff}  # type: ignore

        else:
            phases = {}
            for idx, amplitude in enumerate(self._phase_array):
                if amplitude > cutoff:
                    # Each index corresponds to a computational basis state with the LSB rightmost.
                    # But, we chose to apply the unitaries such that the phase is recorded
                    # in reverse order. So, we reverse the bitstrings here.
                    binary_phase_string = numpy.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                    if as_float:
                        _key = _bit_string_to_phase(binary_phase_string)
                    else:
                        _key = binary_phase_string
                    phases[_key] = amplitude

            phases = _sort_phases(phases)

        return phases


def _bit_string_to_phase(binary_string: str) -> float:
    """Convert bit string to a normalized phase in :math:`[0,1)`.

    It is assumed that the bit string is correctly padded and that the order of
    the bits has been reversed relative to their order when the counts
    were recorded. The LSB is the right most when interpreting the bitstring as
    a phase.

    Args:
        binary_string: A string of characters '0' and '1'.

    Returns:
        a phase scaled to :math:`[0,1)`.
    """
    n_qubits = len(binary_string)
    return int(binary_string, 2) / (2 ** n_qubits)


def _sort_phases(phases: Dict) -> Dict:
    """Sort a dict of bit strings representing phases (keys) and frequencies (values) by bit string.

    The bit strings are sorted according to increasing phase. This relies on Python
    preserving insertion order when building dicts.
    """
    pkeys = list(phases.keys())
    pkeys.sort(reverse=False)  # Sorts in order of the integer encoded by binary string
    phases = {k: phases[k] for k in pkeys}
    return phases
