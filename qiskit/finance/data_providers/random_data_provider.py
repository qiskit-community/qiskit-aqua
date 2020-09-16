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

import numpy as np

from qiskit.aqua import MissingOptionalLibraryError
from ._base_data_provider import BaseDataProvider

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

logger = logging.getLogger(__name__)


class RandomDataProvider(BaseDataProvider):
    """Pseudo-randomly generated mock stock-market data provider.
    """

    def __init__(self,
                 tickers: Optional[Union[str, List[str]]] = None,
                 start: datetime.datetime = datetime.datetime(2016, 1, 1),
                 end: datetime.datetime = datetime.datetime(2016, 1, 30),
                 seed: Optional[int] = None) -> None:
        """
        Initializer
        Args:
            tickers: tickers
            start: first data point
            end: last data point precedes this date
            seed: shall a seed be used?
        Raises:
            MissingOptionalLibraryError: Pandas not installed
        """
        super().__init__()
        if not _HAS_PANDAS:
            raise MissingOptionalLibraryError(
                libname='Pandas',
                name='RandomDataProvider',
                pip_install='pip install pandas')
        tickers = tickers if tickers is not None else ["TICKER1", "TICKER2"]
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self._start = start
        self._end = end
        self._seed = seed

    def run(self) -> None:
        """
        Generates data pseudo-randomly, thus enabling get_similarity_matrix
        and get_covariance_matrix methods in the base class.
        """

        length = (self._end - self._start).days
        generator = np.random.default_rng(self._seed)
        self._data = []
        for _ in self._tickers:
            d_f = pd.DataFrame(
                generator.standard_normal(length)).cumsum() + generator.integers(1, 101)
            trimmed = np.maximum(d_f[0].values, np.zeros(len(d_f[0].values)))
            trimmed_list = trimmed.tolist()
            # find index of first 0 element
            zero_idx = next((idx for idx, val in enumerate(trimmed_list) if val == 0), -1)
            if zero_idx >= 0:
                # set to 0 all values after first 0
                trimmed_list = \
                    [val if idx < zero_idx else 0 for idx, val in enumerate(trimmed_list)]
            self._data.append(trimmed_list)
