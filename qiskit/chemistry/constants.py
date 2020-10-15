# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Defines some constants used in chemical calculations.
"""

# multiplicative conversions
N_A = 6.02214129E23  # particles per mol

KCAL_PER_MOL_TO_J_PER_MOLECULE = 6.947695E-21
HARTREE_TO_KCAL_PER_MOL = 627.509474
HARTREE_TO_J_PER_MOL = 2625499.63922
HARTREE_TO_KJ_PER_MOL = 2625.49963922
HARTREE_TO_PER_CM = 219474.63
J_PER_MOL_TO_PER_CM = 0.08359347178
CAL_TO_J = 4.184
HARTREE_TO_J = 4.3597443380807824e-18  # HARTREE_TO_J_PER_MOL / N_A
J_TO_HARTREE = 2.293712480489655e+17  # 1.0 / HARTREE_TO_J
M_TO_ANGSTROM = 1E10
ANGSTROM_TO_M = 1E-10


# physical constants
C_CM_PER_S = 2.9979245800E10
C_M_PER_S = 2.9979245800E8
HBAR_J_S = 1.054571800E-34  # note this is h/2Pi
H_J_S = 6.62607015E-34
KB_J_PER_K = 1.3806488E-23
