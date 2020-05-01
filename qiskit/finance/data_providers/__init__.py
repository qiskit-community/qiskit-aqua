# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Data Providers Packages """

from ._base_data_provider import BaseDataProvider, DataType, StockMarket, QiskitFinanceError
from .data_on_demand_provider import DataOnDemandProvider
from .exchange_data_provider import ExchangeDataProvider
from .wikipedia_data_provider import WikipediaDataProvider
from .random_data_provider import RandomDataProvider

__all__ = [
    'BaseDataProvider', 'DataType', 'QiskitFinanceError', 'StockMarket', 'RandomDataProvider',
    'DataOnDemandProvider', 'ExchangeDataProvider', 'WikipediaDataProvider'
]
