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

""" wikipedia data provider """

import datetime
import importlib
import logging

import quandl
from quandl.errors.quandl_error import NotFoundError

from qiskit.finance.data_providers import (BaseDataProvider, DataType,
                                           StockMarket, QiskitFinanceError)

logger = logging.getLogger(__name__)


class WikipediaDataProvider(BaseDataProvider):
    """Python implementation of a Wikipedia data provider.
    Please see:
    https://github.com/Qiskit/qiskit-tutorials/qiskit/finance/data_providers/time_series.ipynb
    for instructions on use."""

    CONFIGURATION = {
        "name": "WIKI",
        "description": "Wikipedia Data Provider",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "id": "edi_schema",
            "type": "object",
            "properties": {
                "stockmarket": {
                    "type":
                    "string",
                    "default": StockMarket.NASDAQ.value,
                    "enum": [
                        StockMarket.NASDAQ.value,
                        StockMarket.NYSE.value,
                    ]
                },
                "datatype": {
                    "type":
                    "string",
                    "default": DataType.DAILYADJUSTED.value,
                    "enum": [
                        DataType.DAILYADJUSTED.value,
                        DataType.DAILY.value,
                    ]
                },
            },
        }
    }

    def __init__(self,
                 token=None,
                 tickers=None,
                 stockmarket=StockMarket.NASDAQ,
                 start=datetime.datetime(2016, 1, 1),
                 end=datetime.datetime(2016, 1, 30)):
        """
        Initializer
        Args:
            token (str): quandl access token, which is not needed, strictly speaking
            tickers (str or list): tickers
            stockmarket (StockMarket): NASDAQ, NYSE
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        Raises:
            QiskitFinanceError: provider doesn't support stock market input
        """
        # if not isinstance(atoms, list) and not isinstance(atoms, str):
        #    raise QiskitFinanceError("Invalid atom input for Wikipedia Driver '{}'".format(atoms))
        super().__init__()
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        if stockmarket not in [StockMarket.NASDAQ, StockMarket.NYSE]:
            msg = "WikipediaDataProvider does not support "
            msg += stockmarket.value
            msg += " as a stock market."
            raise QiskitFinanceError(msg)

        # This is to aid serialization; string is ok to serialize
        self._stockmarket = str(stockmarket.value)

        self._token = token
        self._tickers = tickers
        self._start = start
        self._end = end
        self._data = []

        # self.validate(locals())

    @staticmethod
    def check_provider_valid():
        """ checks if provider is valid """
        err_msg = 'quandl is not installed.'
        try:
            spec = importlib.util.find_spec('quandl')
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug('quandl check error %s', str(ex))
            raise QiskitFinanceError(err_msg) from ex

        raise QiskitFinanceError(err_msg)

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            WikipediaDataProvider: data provider object
        Raises:
            QiskitFinanceError: Invalid section
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
        Loads data, thus enabling get_similarity_matrix and
        get_covariance_matrix methods in the base class.
        """
        self.check_provider_valid()
        if self._token:
            quandl.ApiConfig.api_key = self._token
        quandl.ApiConfig.api_version = '2015-04-09'
        self._data = []
        for _, __s in enumerate(self._tickers):
            try:
                __d = quandl.get("WIKI/" + __s,
                                 start_date=self._start,
                                 end_date=self._end)
            except NotFoundError as ex:
                raise QiskitFinanceError(
                    "Cannot retrieve Wikipedia data due to an invalid token."
                ) from ex
            # The exception will be urllib3 NewConnectionError, but it can get dressed by quandl
            except Exception as ex:  # pylint: disable=broad-except
                raise QiskitFinanceError(
                    "Cannot retrieve Wikipedia data.") from ex
            try:
                self._data.append(__d["Adj. Close"])
            except KeyError as ex:
                raise QiskitFinanceError("Cannot parse quandl output.") from ex
