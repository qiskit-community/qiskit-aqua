# -*- coding: utf-8 -*-

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

""" Circuit Sampler Base """

from typing import List, Dict, Optional
import logging
from abc import abstractmethod

from ..operator_base import OperatorBase
from ..converters import ConverterBase

logger = logging.getLogger(__name__)


class CircuitSamplerBase(ConverterBase):
    """ A base for Circuit Samplers. A circuit sampler is a converter for replacing
    CircuitStateFns with DictSateFns representing samples of the StateFn.

    """

    # pylint: disable=arguments-differ
    @abstractmethod
    def convert(self,
                operator: OperatorBase,
                params: dict = None) -> OperatorBase:
        """ Accept the Operator and return the converted Operator """
        raise NotImplementedError

    @abstractmethod
    def sample_circuits(self,
                        op_circuits: Optional[List] = None,
                        param_bindings: Optional[List] = None) -> Dict:
        """ Accept a list of op_circuits and return a list of count dictionaries for each."""
        raise NotImplementedError
