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
    
class WikipediaDataProvider(BaseDataProvider):
    """Python implementation of a Wikipedia data provider.
    Please see:
    https://github.com/Qiskit/qiskit-tutorials/qiskit/finance/data_providers/time_series.ipynb
    for instructions on use."""

    CONFIGURATION = {
        "name": "WIKI",
        "description": "Wikipedia Data Provider",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "edi_schema",
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
                         ]}
                    ]
                },    
            },
        }
    }

    def __init__(self,
                 token = "",
                 tickers = [],
                 stockmarket = StockMarket.NASDAQ.value,
                 start = datetime.datetime(2016,1,1),
                 end = datetime.datetime(2016,1,30)):
        """
        Initializer
        Args:
            token (str): quandl access token, which is not needed, strictly speaking
            tickers (str or list): tickers
            stockmarket (StockMarket): NASDAQ, NYSE
        """
        #if not isinstance(atoms, list) and not isinstance(atoms, str):
        #    raise QiskitFinanceError("Invalid atom input for Wikipedia Driver '{}'".format(atoms))

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self.validate(locals())
        super().__init__()
        self._stockmarket = stockmarket # .value?
        self._token = token
        self._tickers = tickers
        self._start = start
        self._end = end
        self._data = []

    @staticmethod
    def check_provider_valid():
        err_msg = 'quandl is not installed.'
        try:
            spec = importlib.util.find_spec('quandl')
            if spec is not None:
                return
        except Exception as e:
            logger.debug('quandl check error {}'.format(str(e)))
            raise QiskitFinanceError(err_msg) from e

        raise QiskitFinanceError(err_msg)

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            DataProvider object
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
        import quandl
        quandl.ApiConfig.api_key = self._token
        quandl.ApiConfig.api_version = '2015-04-09'
        self._data = []
        for (cnt, s) in enumerate(self._tickers):
          try:
            d = quandl.get("WIKI/" + s, start_date=self._start, end_date=self._end)
          except Exception as e: # The exception will be urllib3 NewConnectionError, but it can get dressed by quandl
            raise QiskitFinanceError("Cannot retrieve Wikipedia data.") from e
          try:
            self._data.append(d["Adj. Close"])
          except KeyError as e:
            raise QiskitFinanceError("Cannot parse quandl output.") from e