# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from enum import Enum
import datetime
import logging
import random

import numpy as np
import pandas as pd

from qiskit.aqua.translators.data_providers import BaseDataProvider, DataType, QiskitFinanceError

logger = logging.getLogger(__name__)


class StockMarket(Enum):
    RANDOM = 'RANDOM'


class RandomDataProvider(BaseDataProvider):
    """Python implementation of provider of mock stock-market data, which are generated pseudo-randomly.
    """

    CONFIGURATION = {
        "name": "RND",
        "description": "Pseudo-Random Data Provider",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
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
                    "oneOf": [{
                        "enum": [DataType.DAILYADJUSTED.value]
                    }]
                },
            },
        }
    }

    def __init__(self,
                 tickers=["TICKER1", "TICKER2"],
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
        """
        super().__init__()

        #if not isinstance(atoms, list) and not isinstance(atoms, str):
        #    raise QiskitFinanceError("Invalid atom input for RANDOM data provider '{}'".format(atoms))

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self._stockmarket = stockmarket.value
        self._start = start
        self._end = end
        self._seed = seed

        #self.validate(locals())

    @staticmethod
    def check_provider_valid():
        return

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        if section is None or not isinstance(section, dict):
            raise QiskitFinanceError(
                'Invalid or missing section {}'.format(section))

        params = section
        kwargs = {}
        #for k, v in params.items():
        #    if k == ExchangeDataDriver. ...: v = UnitsType(v)
        #    kwargs[k] = v
        logger.debug('init_from_input: {}'.format(kwargs))
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
        for ticker in self._tickers:
            df = pd.DataFrame(
                np.random.randn(length)).cumsum() + random.randint(1, 101)
            trimmed = np.maximum(df[0].values, np.zeros(len(df[0].values)))
            self._data.append(trimmed.tolist())
