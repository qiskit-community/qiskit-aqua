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

import datetime
import importlib
import logging

from qiskit.aqua.translators.data_providers import BaseDataProvider, DataType, StockMarket, QiskitFinanceError

logger = logging.getLogger(__name__)


class ExchangeDataProvider(BaseDataProvider):
    """Python implementation of an Exchange Data provider.
    Please see:
    https://github.com/Qiskit/qiskit-tutorials/qiskit/finance/data_providers/time_series.ipynb
    for instructions on use, which involve obtaining a Quandl access token.
    """

    CONFIGURATION = {
        "name": "EDI",
        "description": "Exchange Data International Data Provider",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "edi_schema",
            "type": "object",
            "properties": {
                "stockmarket": {
                    "type":
                    "string",
                    "default":
                    StockMarket.LONDON.value,
                    "oneOf": [{
                        "enum": [
                            StockMarket.LONDON.value,
                            StockMarket.EURONEXT.value,
                            StockMarket.SINGAPORE.value,
                        ]
                    }]
                },
                "datatype": {
                    "type":
                    "string",
                    "default":
                    DataType.DAILYADJUSTED.value,
                    "oneOf": [{
                        "enum": [
                            DataType.DAILYADJUSTED.value,
                            DataType.DAILY.value,
                        ]
                    }]
                },
            },
        }
    }

    def __init__(self,
                 token,
                 tickers,
                 stockmarket=StockMarket.LONDON,
                 start=datetime.datetime(2016, 1, 1),
                 end=datetime.datetime(2016, 1, 30)):
        """
        Initializer
        Args:
            token (str): quandl access token
            tickers (str or list): tickers
            stockmarket (StockMarket): LONDON, EURONEXT, or SINGAPORE
        """

        super().__init__()

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        if not (stockmarket in [
                StockMarket.LONDON, StockMarket.EURONEXT, StockMarket.SINGAPORE
        ]):
            msg = "ExchangeDataProvider does not support "
            msg += stockmarket.value
            msg += " as a stock market."
            raise QiskitFinanceError(msg)

        # This is to aid serialisation; string is ok to serialise
        self._stockmarket = str(stockmarket.value)

        self._token = token
        self._tickers = tickers
        self._start = start
        self._end = end

        # self.validate(locals())

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
            raise QiskitFinanceError(
                'Invalid or missing section {}'.format(section))

        params = section
        kwargs = {}
        #for k, v in params.items():
        #    if k == ExchangeDataProvider. ...: v = UnitsType(v)
        #    kwargs[k] = v
        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        """ Loads data, thus enabling get_similarity_matrix and get_covariance_matrix methods in the base class. """
        self.check_provider_valid()
        import quandl
        self._data = []
        quandl.ApiConfig.api_key = self._token
        quandl.ApiConfig.api_version = '2015-04-09'
        for (cnt, s) in enumerate(self._tickers):
            try:
                d = quandl.get(self._stockmarket + "/" + s,
                               start_date=self._start,
                               end_date=self._end)
            except Exception as e:  # The exception will be AuthenticationError, if the token is wrong
                raise QiskitFinanceError(
                    "Cannot retrieve Exchange Data data.") from e
            try:
                self._data.append(d["Adj. Close"])
            except KeyError as e:
                raise QiskitFinanceError("Cannot parse quandl output.") from e
