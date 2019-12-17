# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Univariate Piecewise Linear Objective Function, applies controlled Y-rotation to target qubit.
Control qubits represent integer value, and rotation approximates a piecewise
linear function of the amplitude f:
    |x>|0> --> |x>( sqrt(1 - f(x))|0> + sqrt(f(x))|1> )
"""

import numpy as np
from qiskit.aqua.circuits.piecewise_linear_rotation import PiecewiseLinearRotation as PwlRot
from qiskit.aqua.utils import CircuitFactory

# pylint: disable=invalid-name


class UnivariatePiecewiseLinearObjective(CircuitFactory):

    """
    Univariate Piecewise Linear Objective Function, applies controlled Y-rotation to target qubit.
    Control qubits represent integer value, and rotation approximates a piecewise
    linear function of the amplitude f:
        |x>|0> --> |x>( sqrt(1 - f(x))|0> + sqrt(f(x))|1> )
    """

    def __init__(self, num_state_qubits, min_state_value, max_state_value,
                 breakpoints, slopes, offsets, f_min, f_max, c_approx,
                 i_state=None, i_objective=None):
        """
        Constructor.

        Args:
            num_state_qubits (int): number of qubits to represent the state
            min_state_value (float): lower bound of values to be represented by state qubits
            max_state_value (float): upper bound of values to be represented by state qubits
            breakpoints (Union(list, numpy.ndarray)): breakpoints of piecewise linear function
            slopes (Union(list, numpy.ndarray)): slopes of linear segments
            offsets (Union(list, numpy.ndarray)): offset of linear segments
            f_min (float): minimal value of resulting function
                            (required for normalization of amplitude)
            f_max (float): maximal value of resulting function
                            (required for normalization of amplitude)
            c_approx (float): approximating factor (linear segments are approximated by
                                             contracting rotation
                                            around pi/4, where sin^2() is locally linear)
            i_state (int): indices of qubits that represent the state
            i_objective (int): index of target qubit to apply the rotation to
        """
        super().__init__(num_state_qubits + 1)

        self.num_state_qubits = num_state_qubits
        self.min_state_value = min_state_value
        self.max_state_value = max_state_value

        # sort breakpoints
        i_sort = np.argsort(breakpoints)
        breakpoints = np.array(breakpoints)[i_sort]
        slopes = np.array(slopes)[i_sort]
        offsets = np.array(offsets)[i_sort]

        # drop breakpoints and corresponding values below min_state_value or above max_state_value
        for i in reversed(range(len(breakpoints))):
            if breakpoints[i] <= (self.min_state_value - 1e-6) or \
                    breakpoints[i] >= (self.max_state_value + 1e-6):
                breakpoints = np.delete(breakpoints, i)
                slopes = np.delete(slopes, i)
                offsets = np.delete(offsets, i)

        # make sure the minimal value is included in the breakpoints
        min_value_included = False
        for bp in breakpoints:
            if np.isclose(bp, min_state_value):
                min_value_included = True
                break
        if not min_value_included:
            breakpoints = np.append(min_state_value, breakpoints)
            slopes = np.append(0, slopes)
            offsets = np.append(0, offsets)

        # store parameters
        self._breakpoints = breakpoints
        self._slopes = slopes
        self._offsets = offsets
        self._f_min = f_min
        self._f_max = f_max
        self._c_approx = c_approx

        # get and store qubit indices
        self.i_state = None
        if i_state is not None:
            self.i_state = i_state
        else:
            self.i_state = list(range(num_state_qubits))

        self.i_objective = None
        if i_objective is not None:
            self.i_objective = i_objective
        else:
            self.i_objective = num_state_qubits

        # map breakpoints, slopes, and offsets such that they fit {0, ..., 2^n-1}
        lb = min_state_value
        ub = max_state_value
        self._mapped_breakpoints = []
        self._mapped_slopes = []
        self._mapped_offsets = []
        for i, _ in enumerate(breakpoints):
            mapped_breakpoint = (breakpoints[i] - lb) / (ub - lb) * (2**num_state_qubits - 1)
            if mapped_breakpoint <= 2**num_state_qubits - 1:
                self._mapped_breakpoints += [mapped_breakpoint]

                # factor (ub - lb) / (2^n - 1) is for the scaling of x to [l,u]
                # note that the +l for mapping to [l,u] is already included in
                # the offsets given as parameters
                self._mapped_slopes += [slopes[i] * (ub - lb) / (2**num_state_qubits - 1)]
                self._mapped_offsets += [offsets[i]]
        self._mapped_breakpoints = np.array(self._mapped_breakpoints)
        self._mapped_slopes = np.array(self._mapped_slopes)
        self._mapped_offsets = np.array(self._mapped_offsets)

        # approximate linear behavior by scaling and contracting around pi/4
        if len(self._mapped_breakpoints):  # pylint: disable=len-as-condition
            self._slope_angles = np.zeros(len(breakpoints))
            self._offset_angles = np.pi / 4 * (1 - c_approx) * np.ones(len(breakpoints))
            for i in range(len(breakpoints)):
                self._slope_angles[i] = \
                    np.pi * c_approx * self._mapped_slopes[i] / 2 / (f_max - f_min)
                self._offset_angles[i] += \
                    np.pi * c_approx * (self._mapped_offsets[i] - f_min) / 2 / (f_max - f_min)

            # multiply by 2 since Y-rotation uses theta/2 as angle
            self._slope_angles = 2 * self._slope_angles
            self._offset_angles = 2 * self._offset_angles

            # create piecewise linear Y rotation
            self._pwl_ry = PwlRot(
                self._mapped_breakpoints,
                self._slope_angles,
                self._offset_angles,
                num_state_qubits,
                i_state=i_state,
                i_target=i_objective
            )

        else:
            self.offset_angle = 0
            self.slope_angle = 0

            # create piecewise linear Y rotation
            self._pwl_ry = None

    def value_to_estimation(self, value):
        """ value to estimation """
        if self._c_approx < 1:
            # map normalized value back to estimation
            estimator = value - 1 / 2 + np.pi / 4 * self._c_approx
            estimator *= 2 / np.pi / self._c_approx
            estimator *= self._f_max - self._f_min
            estimator += self._f_min
            return estimator
        else:
            return value

    def required_ancillas(self):
        """ requires ancillas """
        return self._pwl_ry.required_ancillas()

    def build(self, qc, q, q_ancillas=None, params=None):
        """ build """
        q_state = [q[i] for i in self.i_state]
        q_objective = q[self.i_objective]

        # apply piecewise linear rotation
        self._pwl_ry.build(qc, q_state + [q_objective], q_ancillas)
