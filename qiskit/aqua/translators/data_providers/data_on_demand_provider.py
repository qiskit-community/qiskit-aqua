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
import certifi
import json
import logging
import urllib3
from urllib.parse import urlencode

from qiskit.aqua.translators.data_providers import BaseDataProvider, DataType, StockMarket, QiskitFinanceError

logger = logging.getLogger(__name__)


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
                    "type":
                    "string",
                    "default":
                    StockMarket.NASDAQ.value,
                    "oneOf": [{
                        "enum": [
                            StockMarket.NASDAQ.value,
                            StockMarket.NYSE.value,
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
                            DataType.BID.value,
                            DataType.ASK.value,
                        ]
                    }]
                },
            },
        }
    }

    def __init__(self,
                 token,
                 tickers,
                 stockmarket=StockMarket.NASDAQ,
                 start=datetime.datetime(2016, 1, 1),
                 end=datetime.datetime(2016, 1, 30),
                 verify=None):
        """
        Initializer
        Args:
            token (str): quandl access token
            tickers (str or list): tickers
            stockmarket (StockMarket): NYSE or NASDAQ
            start (datetime): first data point
            end (datetime): last data point precedes this date
            verify (None or str or boolean): if verify is None, certifi certificates will be used (default);
                if this is False, no certificates will be checked; if this is a string, it should be pointing
                to a cerfificate for the HTTPS connection to NASDAQ (dataondemand.nasdaq.com), either in the
                form of a CA_BUNDLE file or a directory wherein to look.
        """
        # if not isinstance(atoms, list) and not isinstance(atoms, str):
        #    raise QiskitFinanceError("Invalid atom input for DOD Driver '{}'".format(atoms))

        super().__init__()

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        if not (stockmarket in [StockMarket.NASDAQ, StockMarket.NYSE]):
            msg = "NASDAQ Data on Demand does not support "
            msg += stockmarket.value
            msg += " as a stock market."
            raise QiskitFinanceError(msg)

        # This is to aid serialisation; string is ok to serialise
        self._stockmarket = str(stockmarket.value)

        self._token = token
        self._start = start
        self._end = end
        self._verify = verify

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
        """ Loads data, thus enabling get_similarity_matrix and get_covariance_matrix methods in the base class. """
        self.check_provider_valid()
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                   ca_certs=certifi.where())
        URL = 'https://dataondemand.nasdaq.com/api/v1/quotes?'
        self._data = []
        for ticker in self._tickers:
            values = {
                '_Token': self._token,
                'symbols': [ticker],
                'start': self._start.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                'end': self._end.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                'next_cursor': 0
            }
            encoded = URL + urlencode(values)
            try:
                if self._verify is None:
                    response = http.request(
                        'POST', encoded
                    )  # this runs certifi verification, as per the set-up of the urllib3
                else:
                    response = http.request(
                        'POST', encoded, verify=self._verify
                    )  # this disables certifi verification (False) or forces the certificate path (str)
                if response.status != 200:
                    msg = "Accessing NASDAQ Data on Demand with parameters {} encoded into ".format(
                        values)
                    msg += encoded
                    msg += " failed. Hint: Check the _Token. Check the spelling of tickers."
                    raise QiskitFinanceError(msg)
                quotes = json.loads(response.data.decode('utf-8'))["quotes"]
                priceEvolution = []
                for q in quotes:
                    priceEvolution.append(q["ask_price"])
                self._data.append(priceEvolution)
            except Exception as e:
                raise QiskitFinanceError(
                    'Accessing NASDAQ Data on Demand failed.') from e
            http.clear()
