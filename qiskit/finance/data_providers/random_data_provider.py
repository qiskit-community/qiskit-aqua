# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Pseudo-randomly generated mock stock-market data provider """

from typing import Optional, Union, List
import datetime
import logging
import random

import numpy as np
import pandas as pd

from ._base_data_provider import BaseDataProvider, StockMarket
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class RandomDataProvider(BaseDataProvider):
    """Pseudo-randomly generated mock stock-market data provider.
    """

    def __init__(self,
                 tickers: Optional[Union[str, List[str]]] = None,
                 stockmarket: StockMarket = StockMarket.RANDOM,
                 start: datetime = datetime.datetime(2016, 1, 1),
                 end: datetime = datetime.datetime(2016, 1, 30),
                 seed: Optional[int] = None) -> None:
        """
        Initializer
        Args:
            tickers: tickers
            stockmarket: RANDOM
            start: first data point
            end: last data point precedes this date
            seed: shall a seed be used?
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

    def run(self):
        """
        Generates data pseudo-randomly, thus enabling get_similarity_matrix
        and get_covariance_matrix methods in the base class.
        """

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
