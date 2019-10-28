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
Python implementation of provider of mock stock-market data, which are generated pseudo-randomly.
"""

import datetime
import logging
import random

import numpy as np
import pandas as pd

from qiskit.finance.data_providers import (BaseDataProvider,
                                           DataType,
                                           StockMarket,
                                           QiskitFinanceError)

logger = logging.getLogger(__name__)


class RandomDataProvider(BaseDataProvider):
    """
    Python implementation of provider of mock stock-market data,
    which are generated pseudo-randomly.
    """

    CONFIGURATION = {
        "name": "RND",
        "description": "Pseudo-Random Data Provider",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "id": "rnd_schema",
            "type": "object",
            "properties": {
                "stockmarket": {
                    "type": "string",
                    "default": "RANDOM"
                },
                "datatype": {
                    "type": "string",
                    "default": DataType.DAILYADJUSTED.value,
                    "enum": [DataType.DAILYADJUSTED.value]
                },
            },
        }
    }

    def __init__(self,
                 tickers=None,
                 stockmarket=StockMarket.RANDOM,
                 start=datetime.datetime(2016, 1, 1),
                 end=datetime.datetime(2016, 1, 30),
                 seed=None):
        """
        Initializer
        Args:
            tickers (str or list): tickers
            stockmarket (StockMarket): RANDOM
            start (datetime): first data point
            end (datetime): last data point precedes this date
            seed (None or int): shall a seed be used?
        Raises:
            QiskitFinanceError: provider doesn't support stock market value
        """
        super().__init__()
        tickers = tickers if tickers is not None else ["TICKER1", "TICKER2"]
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        if stockmarket not in [StockMarket.RANDOM]:
            msg = "RandomDataProvider does not support "
            msg += stockmarket.value
            msg += " as a stock market. Please use Stockmarket.RANDOM."
            raise QiskitFinanceError(msg)

        # This is to aid serialization; string is ok to serialize
        self._stockmarket = str(stockmarket.value)

        self._start = start
        self._end = end
        self._seed = seed

        # self.validate(locals())

    @staticmethod
    def check_provider_valid():
        """ check provider valid """
        return

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            RandomDataProvider: Driver object
        Raises:
            QiskitFinanceError: invalid section
        """
        if section is None or not isinstance(section, dict):
            raise QiskitFinanceError(
                'Invalid or missing section {}'.format(section))

        # params = section
        kwargs = {}
        # for k, v in params.items():
        #    if k == ExchangeDataDriver. ...: v = UnitsType(v)
        #    kwargs[k] = v
        logger.debug('init_from_input: %s', kwargs)
        return cls(**kwargs)

    def run(self):
        """
        Generates data pseudo-randomly, thus enabling get_similarity_matrix
        and get_covariance_matrix methods in the base class.
        """
        self.check_provider_valid()

        length = (self._end - self._start).days
        if self._seed:
            random.seed(self._seed)
            np.random.seed(self._seed)

        self._data = []
        for _ in self._tickers:
            d_f = pd.DataFrame(
                np.random.randn(length)).cumsum() + random.randint(1, 101)
            trimmed = np.maximum(d_f[0].values, np.zeros(len(d_f[0].values)))
            # pylint: disable=no-member
            self._data.append(trimmed.tolist())
