# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
CPLEX Installation
==================
The :class:`ClassicalCPLEX` algorithm utilizes CPLEX from the `IBM ILOG CPLEX Optimization Studio
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/COS_KC_home.html>`__
and this should be `installed
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/setup_synopsis.html>`__
along with the `Python API
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html>`__
to CPLEX for the algorithm to be operational.

.. note::

    The above links are to the latest version of IBM ILOG CPLEX at the time of writing.
    Information links for other versions can be found on the above linked pages under the
    `Change version or product` drop-down list.

"""

from .classical_cplex import ClassicalCPLEX, CPLEX_Ising

__all__ = ['ClassicalCPLEX',
           'CPLEX_Ising']
