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

""" An Object to represent State Functions """


import numpy as np
from functools import reduce
from abc import abstractmethod

from qiskit.aqua.operators.operator_base import OperatorBase


class StateFunctionBase(OperatorBase):
    """ A class for representing state functions over binary strings, which are equally defined to be
    1) A complex function over a single binary string (as compared to an operator, which is defined as a function
    over two binary strings).
    2) An Operator with one parameter of its evaluation function always fixed. For example, if we fix one parameter in
    the eval function of a Matrix-defined operator to be '000..0', the state function is defined by the vector which
    is the first column of the matrix (or rather, an index function over this vector). A circuit-based operator with
    one parameter fixed at '000...0' can be interpreted simply as the quantum state prepared by composing the
    circuit with the |000...0⟩ state.

    NOTE: This state function is not restricted to wave functions, as there is no requirement of normalization.

    This object is essentially defined by the operators it holds.
    """

    @abstractmethod
    def sample(self, shots):
        """ Sample the statefunction as a normalized probability distribution."""
        raise NotImplementedError