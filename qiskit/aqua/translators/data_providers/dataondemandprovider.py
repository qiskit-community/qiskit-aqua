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

from qiskit.aqua.translators.data_providers import BaseDataProvider, DataType, QiskitFinanceError
import importlib
from enum import Enum
import logging
import datetime

logger = logging.getLogger(__name__)


class StockMarket(Enum):
    NASDAQ = 'NASDAQ'
    NYSE = 'NYSE'
    
class DataOnDemandProvider(BaseDataProvider):
    """Python implementation of an NASDAQ Data on Demand data provider.
    Please see:
    https://github.com/Qiskit/qiskit-tutorials/qiskit/finance/data_providers/time_series.ipynb
    for instructions on use, which involve obtaining a NASDAQ DOD access token.
    """

    CONFIGURATION = {
        "name": "DOD",
        "description": "NASDAQ Data on Demand Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "dod_schema",
            "type": "object",
            "properties": {
                "stockmarket": {
                    "type": "string",
                    "default": StockMarket.NASDAQ.value,
                    "oneOf": [
                         {"enum": [
                            StockMarket.NASDAQ.value,
                            StockMarket.NYSE.value,
                         ]}
                    ]
                },
                "datatype": {
                    "type": "string",
                    "default": DataType.DAILYADJUSTED.value,
                    "oneOf": [
                         {"enum": [
                            DataType.DAILYADJUSTED.value,
                            DataType.DAILY.value,
                            DataType.BID.value,
                            DataType.ASK.value,
                         ]}
                    ]
                },    
            },
        }
    }

    def __init__(self,
                 token,
                 tickers,
                 stockmarket = StockMarket.NASDAQ,
                 start = datetime.datetime(2016,1,1),
                 end = datetime.datetime(2016,1,30)):
        """
        Initializer
        Args:
            token (str): quandl access token
            tickers (str or list): tickers
            stockmarket (StockMarket): LONDON, EURONEXT, or SINGAPORE
        """
        #if not isinstance(atoms, list) and not isinstance(atoms, str):
        #    raise QiskitFinanceError("Invalid atom input for DOD Driver '{}'".format(atoms))

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self.validate(locals())
        super().__init__()
        self._stockmarket = stockmarket # .value?
        self._token = token
        self._start = start
        self._end = end

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
            raise QiskitFinanceError('Invalid or missing section {}'.format(section))

        params = section
        kwargs = {}
        #for k, v in params.items():
        #    if k == ExchangeDataDriver. ...: v = UnitsType(v)
        #    kwargs[k] = v
        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        self.check_provider_valid()
        import re
        import urllib3
        from urllib.parse import urlencode
        http = urllib3.PoolManager()
        import json
        URL = 'https://dataondemand.nasdaq.com/api/v1/quotes?'
        self._data = []
        for ticker in self._tickers:
          values = {'_Token' : self._token,
          'symbols' : [ticker],
          'start' : self._start.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"), 
          'end' : self._end.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"), 
          'next_cursor': 0
          }
          encoded = URL + urlencode(values)
          try: 
            response = http.request('POST', encoded)
            if response.status != 200:
              msg = "Accessing NASDAQ Data on Demand with parameters {} encoded into ".format(values)
              msg += encoded
              msg += " failed. Hint: Check the _Token. Check the spelling of tickers."
              raise QiskitFinanceError(msg)
            quotes = json.loads(response.data.decode('utf-8'))["quotes"]
            priceEvolution = []
            for q in quotes: priceEvolution.append(q["ask_price"])
            self._data.append(priceEvolution)
          except Exception as e:
            raise QiskitFinanceError('Accessing NASDAQ Data on Demand failed.') from e