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

""" An Object to represent State Functions constructed from Operators """


import numpy as np
import re
from functools import reduce

from qiskit.quantum_info import Statevector

from qiskit.aqua.operators.operator_base import OperatorBase


class StateFn(OperatorBase):
    """ A class for representing state functions over binary strings, which are equally defined to be
    1) A complex function over a single binary string (as compared to an operator, which is defined as a function
    over two binary strings).
    2) An Operator with one parameter of its evaluation function always fixed. For example, if we fix one parameter in
    the eval function of a Matrix-defined operator to be '000..0', the state function is defined by the vector which
    is the first column of the matrix (or rather, an index function over this vector). A circuit-based operator with
    one parameter fixed at '000...0' can be interpreted simply as the quantum state prepared by composing the
    circuit with the |000...0⟩ state.

    NOTE: This state function is not restricted to wave functions, as there is no requirement of normalization.

    This object is essentially defined by the operators it holds in the primitive property.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # NOTE: We call this density but we don't enforce normalization!!
    # TODO allow normalization somehow?
    def __init__(self, density_primitive, coeff=1.0):
        # TODO change name from primitive to something else
        """
        Args:
            density_primitive(str, dict, OperatorBase, np.ndarray, list)
            coeff(int, float, complex): A coefficient by which to multiply the state
        """
        # If the initial density is a string, treat this as a density dict with only a single basis state.
        if isinstance(density_primitive, str):
            self._density_primitive = {density_primitive: 1}

        # If the initial density is set to a counts dict, Statevector, or an operator, treat it as a density operator,
        # where the eval function is equal to eval(my_str, my_str), e.g. a lookup along the diagonal.
        elif isinstance(density_primitive, (dict, OperatorBase, Statevector)):
            self._density_primitive = density_primitive

        # TODO accept Qiskit results object (to extract counts or sv), reverse if necessary

        # TODO: Should we only allow correctly shaped vectors, e.g. vertical? Should we reshape to make contrast with
        #  measurement more accurate?
        elif isinstance(density_primitive, (np.ndarray, list)):
            self._density_primitive = Statevector(density_primitive)

        # TODO figure out custom callable later
        # if isinstance(self.density_primitive, callable):
        #     self._fixed_param = '0'

        self._coeff = coeff

    @property
    def density_primitive(self):
        return self._density_primitive

    @property
    def num_qubits(self):
        # Basis state
        if isinstance(self.density_primitive, str):
            return 0 if val1 == self.density_primitive else 1

        # If the primitive is lookup of bitstrings, we define all missing strings to have a function value of zero.
        elif isinstance(self.density_primitive, dict):
            return len(list(self.density_primitive.keys())[0])

        elif isinstance(self.density_primitive, callable):
            return self.density_primitive(val1)

        elif isinstance(self.density_primitive, OperatorBase):
            return self._density_primitive.num_qubits

    def eval(self, val1=None, val2=None):
        # Validate bitstring: re.fullmatch(rf'[01]{{{0}}}', val1)
        if val2:
            raise ValueError('Second parameter of statefuntion eval is fixed and cannot be passed into eval().')

        # Basis state
        if isinstance(self.density_primitive, str):
            return self.coeff if val1 == self.density_primitive else 0

        # If the primitive is lookup of bitstrings, we define all missing strings to have a function value of zero.
        elif isinstance(self.density_primitive, dict):
            return self.density_primitive.get(val1, 0) * self.coeff

        elif isinstance(self.density_primitive, OperatorBase):
            return self.density_primitive.eval(val1=val1, val2=val1) * self.coeff

        elif isinstance(self.density_primitive, Statevector):
            index1 = int(val1, 2)
            return self.density_primitive.data[index1] * self.coeff

        elif hasattr(self.density_primitive, 'eval'):
            if self._fixed_param == 'diag':
                return self.density_primitive.eval(val1=val1, val2=val1)

    # TODO use to_matrix() instead to be consistent with Operator?
    def to_vector(self):
        pass

    def to_matrix(self):
        pass

    def adjoint(self):
        # return Measurement(self)
        pass

    def sample(self, shots):
        pass
