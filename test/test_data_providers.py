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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua.translators.data_providers import *
from qiskit.aqua.translators.data_providers import QiskitFinanceError
import datetime

class TestDataProviders(QiskitAquaTestCase):
    """Tests data providers for the Portfolio Optimization and Diversification."""

    def setUp(self):
        super().setUp()

    def test_wikipedia(self):
        from qiskit.aqua.translators.data_providers.wikipediadataprovider import StockMarket
        wiki = WikipediaDataProvider(token = "",
                         tickers = ["GOOG", "AAPL"],
                         stockmarket = StockMarket.NASDAQ.value,
                         start = datetime.datetime(2016,1,1),
                         end = datetime.datetime(2016,1,30))
        # can throw QiskitFinanceError
        try:
            wiki.run()
            similarity = np.array([[1.00000000e+00, 8.44268222e-05],
                                   [8.44268222e-05, 1.00000000e+00]])
            covariance = np.array([[269.60118129, 25.42252332], 
                                   [ 25.42252332, 7.86304499]])
            self.get_similarity_matrix()
            self.get_covariance()
            numpy.testing.assert_array_almost_equal(self.rho, similarity, decimal = 3) 
            numpy.testing.assert_array_almost_equal(self.cov, covariance, decimal = 3)
        except QiskitFinanceError:
            print("Test of WikipediaDataProvider skipped due to the per-day usage limits.")
            # The trouble for automating testing is that after 50 tries from one IP address within a day
            # Quandl complains about the free usage tier limits:
            # quandl.errors.quandl_error.LimitExceededError: (Status 429) (Quandl Error QELx01) 
            # You have exceeded the anonymous user limit of 50 calls per day. To make more calls 
            # today, please register for a free Quandl account and then include your API key with your requests.
            # This gets "dressed" as QiskitFinanceError.
            # This also introduces a couple of seconds of a delay.
        
    def test_nasdaq(self):
        from qiskit.aqua.translators.data_providers.dataondemandprovider import StockMarket
        nasdaq = DataOnDemandProvider(token = "REPLACE-ME",
                         tickers = ["GOOG", "AAPL"],
                         stockmarket = StockMarket.NASDAQ.value,
                         start = datetime.datetime(2016,1,1),
                         end = datetime.datetime(2016,1,2))
        with self.assertRaises(QiskitFinanceError):
          nasdaq.run()
        # will throw QiskitFinanceError, because there is no valid token; otherwise, we could continue as:
        """
        similarity = np.array([[1.00000000e+00, 8.44268222e-05],
         [8.44268222e-05, 1.00000000e+00]])
        covariance = np.array([[269.60118129, 25.42252332], 
         [ 25.42252332, 7.86304499]])
        self.get_similarity_matrix()
        self.get_covariance()
        numpy.testing.assert_array_almost_equal(self.rho, similarity, decimal = 3) 
        numpy.testing.assert_array_almost_equal(self.cov, covariance, decimal = 3)
        """
        
    def test_exchangedata(self):
        from qiskit.aqua.translators.data_providers.exchangedataprovider import StockMarket
        lse = ExchangeDataProvider(token = "REPLACE-ME",
                         tickers = ["AIBGl", "AVSTl"],
                         stockmarket = StockMarket.LONDON.value,
                         start = datetime.datetime(2019,1,1),
                         end = datetime.datetime(2019,1,30))
        with self.assertRaises(QiskitFinanceError):
          lse.run()