# -*- coding: utf-8 -*-

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

""" Test Data Providers """

import unittest
import datetime
from test.finance import QiskitFinanceTestCase
import warnings
import numpy as np
from qiskit.finance import QiskitFinanceError
from qiskit.finance.data_providers import (RandomDataProvider,
                                           WikipediaDataProvider,
                                           StockMarket,
                                           DataOnDemandProvider,
                                           ExchangeDataProvider)


# This can be run as python -m unittest test.test_data_providers.TestDataProviders

class TestDataProviders(QiskitFinanceTestCase):
    """Tests data providers for the Portfolio Optimization and Diversification."""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", message="unclosed", category=ResourceWarning)

    def test_wrong_use(self):
        """ wrong use test """
        rnd = RandomDataProvider(seed=1)
        # Now, the .run() method is expected, which does the actual data loading
        # (and can take seconds or minutes,
        # depending on the data volumes, hence not ok in the constructor)
        self.assertRaises(QiskitFinanceError, rnd.get_covariance_matrix)
        self.assertRaises(QiskitFinanceError, rnd.get_similarity_matrix)
        wiki = WikipediaDataProvider(
            token="",
            tickers=["GOOG", "AAPL"],
            stockmarket=StockMarket.NASDAQ,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30)
        )
        # Now, the .run() method is expected, which does the actual data loading
        self.assertRaises(QiskitFinanceError, wiki.get_covariance_matrix)
        self.assertRaises(QiskitFinanceError, wiki.get_similarity_matrix)

    def test_random(self):
        """ random test """
        rnd = RandomDataProvider(seed=1)
        rnd.run()
        similarity = np.array([[1.00000000e+00, 6.2284804e-04], [6.2284804e-04, 1.00000000e+00]])
        covariance = np.array([[1.75870991, -0.32842528], [-0.32842528, 2.31429182]])
        np.testing.assert_array_almost_equal(rnd.get_covariance_matrix(), covariance, decimal=3)
        np.testing.assert_array_almost_equal(rnd.get_similarity_matrix(), similarity, decimal=3)

    def test_wikipedia(self):
        """ wikipedia test """
        wiki = WikipediaDataProvider(
            token="",
            tickers=["GOOG", "AAPL"],
            stockmarket=StockMarket.NASDAQ,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30)
        )
        # can throw QiskitFinanceError
        try:
            wiki.run()
            similarity = np.array([
                [1.00000000e+00, 8.44268222e-05],
                [8.44268222e-05, 1.00000000e+00]
            ])
            covariance = np.array([
                [269.60118129, 25.42252332],
                [25.42252332, 7.86304499]
            ])
            np.testing.assert_array_almost_equal(wiki.get_covariance_matrix(),
                                                 covariance, decimal=3)
            np.testing.assert_array_almost_equal(wiki.get_similarity_matrix(),
                                                 similarity, decimal=3)
        except QiskitFinanceError:
            self.skipTest("Test of WikipediaDataProvider skipped due to the per-day usage limits.")
            # The trouble for automating testing is that after 50 tries
            # from one IP address within a day
            # Quandl complains about the free usage tier limits:
            # quandl.errors.quandl_error.LimitExceededError: (Status 429) (Quandl Error QELx01)
            # You have exceeded the anonymous user limit of 50 calls per day. To make more calls
            # today, please register for a free Quandl account and then include your API
            # key with your requests.
            # This gets "dressed" as QiskitFinanceError.
            # This also introduces a couple of seconds of a delay.

    def test_nasdaq(self):
        """ nasdaq test """
        nasdaq = DataOnDemandProvider(
            token="REPLACE-ME",
            tickers=["GOOG", "AAPL"],
            stockmarket=StockMarket.NASDAQ,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 2)
        )
        try:
            nasdaq.run()
            self.fail("Test of DataOnDemandProvider should have failed due to the lack of a token.")
        except QiskitFinanceError:
            self.skipTest("Test of DataOnDemandProvider skipped due to the lack of a token.")

    def test_exchangedata(self):
        """ exchange data test """
        lse = ExchangeDataProvider(
            token="REPLACE-ME",
            tickers=["AIBGl", "AVSTl"],
            stockmarket=StockMarket.LONDON,
            start=datetime.datetime(2019, 1, 1),
            end=datetime.datetime(2019, 1, 30)
        )
        try:
            lse.run()
            self.fail("Test of DataOnDemandProvider should have failed due to the lack of a token.")
        except QiskitFinanceError:
            self.skipTest("Test of DataOnDemandProvider skipped due to the lack of a token.")


if __name__ == '__main__':
    unittest.main()
