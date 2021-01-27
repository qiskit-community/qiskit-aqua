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

""" Wikipedia data provider. """

from typing import Optional, Union, List
import datetime
import logging

from qiskit.aqua import MissingOptionalLibraryError
from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

try:
    import quandl
    _HAS_QUANDL = True
except ImportError:
    _HAS_QUANDL = False

logger = logging.getLogger(__name__)


class WikipediaDataProvider(BaseDataProvider):
    """Wikipedia data provider.

    Please see:
    https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/finance/11_time_series.ipynb
    for instructions on use.
    """

    def __init__(self,
                 token: Optional[str] = None,
                 tickers: Optional[Union[str, List[str]]] = None,
                 start: datetime.datetime = datetime.datetime(2016, 1, 1),
                 end: datetime.datetime = datetime.datetime(2016, 1, 30)) -> None:
        """
        Initializer
        Args:
            token: quandl access token, which is not needed, strictly speaking
            tickers: tickers
            start: start time
            end: end time
         Raises:
            MissingOptionalLibraryError: Quandl not installed
        """
        super().__init__()
        if not _HAS_QUANDL:
            raise MissingOptionalLibraryError(
                libname='Quandl',
                name='WikipediaDataProvider',
                pip_install='pip install quandl')
        self._tickers = None  # type: Optional[Union[str, List[str]]]
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace('\n', ';').split(";")
        self._n = len(self._tickers)

        self._token = token
        self._tickers = tickers
        self._start = start.strftime('%Y-%m-%d')
        self._end = end.strftime('%Y-%m-%d')
        self._data = []

    def run(self) -> None:
        """
        Loads data, thus enabling get_similarity_matrix and
        get_covariance_matrix methods in the base class.
        """
        quandl.ApiConfig.api_key = self._token
        quandl.ApiConfig.api_version = '2015-04-09'
        self._data = []
        stocks_notfound = []
        for _, ticker_name in enumerate(self._tickers):
            stock_data = None
            name = 'WIKI' + "/" + ticker_name
            try:
                stock_data = quandl.get(name,
                                        start_date=self._start,
                                        end_date=self._end)
            except quandl.AuthenticationError as ex:
                raise QiskitFinanceError("Quandl invalid token.") from ex
            except quandl.NotFoundError as ex:
                stocks_notfound.append(name)
                continue
            except quandl.QuandlError as ex:
                raise QiskitFinanceError("Quandl Error for '{}'.".format(name)) from ex

            try:
                self._data.append(stock_data["Adj. Close"])
            except KeyError as ex:
                raise QiskitFinanceError("Cannot parse quandl output.") from ex

        if stocks_notfound:
            raise QiskitFinanceError('Stocks not found: {}. '.format(stocks_notfound))
