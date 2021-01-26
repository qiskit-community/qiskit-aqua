# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" NASDAQ Data on demand data provider. """

from typing import Optional, Union, List
import datetime
from urllib.parse import urlencode
import logging
import json
import certifi
import urllib3

from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class DataOnDemandProvider(BaseDataProvider):
    """NASDAQ Data on Demand data provider.

    Please see:
    https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/finance/11_time_series.ipynb
    for instructions on use, which involve obtaining a NASDAQ DOD access token.
    """

    def __init__(self,
                 token: str,
                 tickers: Union[str, List[str]],
                 start: datetime.datetime = datetime.datetime(2016, 1, 1),
                 end: datetime.datetime = datetime.datetime(2016, 1, 30),
                 verify: Optional[Union[str, bool]] = None) -> None:
        """
        Args:
            token: data on demand access token
            tickers: tickers
            start: first data point
            end: last data point precedes this date
            verify: if verify is None, certify certificates
                will be used (default);
                if this is False, no certificates will be checked; if this is a string,
                it should be pointing
                to a certificate for the HTTPS connection to NASDAQ (dataondemand.nasdaq.com),
                either in the
                form of a CA_BUNDLE file or a directory wherein to look.
        """
        super().__init__()

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self._token = token
        self._start = start
        self._end = end
        self._verify = verify

    def run(self) -> None:
        """
        Loads data, thus enabling get_similarity_matrix and get_covariance_matrix
        methods in the base class.
        """

        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                   ca_certs=certifi.where())
        url = 'https://dataondemand.nasdaq.com/api/v1/quotes?'
        self._data = []
        for ticker in self._tickers:
            values = {
                '_Token': self._token,
                'symbols': [ticker],
                'start': self._start.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                'end': self._end.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                'next_cursor': 0
            }
            encoded = url + urlencode(values)
            try:
                if self._verify is None:
                    response = http.request(
                        'POST', encoded
                    )  # this runs certificate verification, as per the set-up of the urllib3
                else:
                    # this disables certificate verification (False)
                    # or forces the certificate path (str)
                    response = http.request(
                        'POST', encoded, verify=self._verify
                    )
                if response.status != 200:
                    msg = "Accessing NASDAQ Data on Demand with parameters {} encoded into ".format(
                        values)
                    msg += encoded
                    msg += " failed. Hint: Check the _Token. Check the spelling of tickers."
                    raise QiskitFinanceError(msg)
                quotes = json.loads(response.data.decode('utf-8'))["quotes"]
                price_evolution = []
                for q in quotes:
                    price_evolution.append(q["ask_price"])
                self._data.append(price_evolution)
            except Exception as ex:  # pylint: disable=broad-except
                raise QiskitFinanceError(
                    'Accessing NASDAQ Data on Demand failed.') from ex
            http.clear()
