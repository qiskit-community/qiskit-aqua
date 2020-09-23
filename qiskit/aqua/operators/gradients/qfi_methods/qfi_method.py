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

""" QFIMethod Class """

import logging
from abc import abstractmethod
from typing import List, Union, Optional

from qiskit.aqua.operators.converters.converter_base import ConverterBase
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.circuit import Parameter, ParameterVector

logger = logging.getLogger(__name__)


class QFIMethod(ConverterBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields Quantum Fisher Information metric tensor
    with respect to the circuit parameters

    This is distinct from DerivativeBase converters which take gradients of composite
    operators and handle things like differentiating combo_fn's and enforcing prodct rules
    when operator coeficients are parameterized. 

    QFIMethod - uses quantum techniques to get the QFI of circuits
    DerivativeBase   - uses classical techniques to differentiate opflow data strctures
    """

    # pylint: disable=arguments-differ
    @abstractmethod
    def convert(self,
                operator: OperatorBase,
                params: Union[ParameterVector, Parameter, List[Parameter]],
                approx: Optional[str] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are getting the QFI of
            params: The parameters we are computing the QFI with respect to..
            approx: The type of approximation to use when taking the gradient

        Returns:
            An operator whose evaluation yields the QFI metric tensor.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        raise NotImplementedError
