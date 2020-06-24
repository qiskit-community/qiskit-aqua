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

"""
Data Providers (:mod:`qiskit.finance.data_providers`)
=====================================================

.. currentmodule:: qiskit.finance.data_providers

A selection of providers for financial data. These may be backed by
an external service that sources the actual data; please refer to the
specific provider class below, for more information in that regard.

Data Provider Base Class
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseDataProvider

Data Provider Types
===================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    StockMarket

Data Providers
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DataOnDemandProvider
   ExchangeDataProvider
   WikipediaDataProvider
   YahooDataProvider
   RandomDataProvider

"""

from ._base_data_provider import BaseDataProvider, StockMarket
from .data_on_demand_provider import DataOnDemandProvider
from .exchange_data_provider import ExchangeDataProvider
from .wikipedia_data_provider import WikipediaDataProvider
from .yahoo_data_provider import YahooDataProvider
from .random_data_provider import RandomDataProvider

__all__ = [
    'BaseDataProvider', 'StockMarket', 'RandomDataProvider',
    'DataOnDemandProvider', 'ExchangeDataProvider', 'WikipediaDataProvider',
    'YahooDataProvider'
]
