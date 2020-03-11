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
Components (:mod:`qiskit.aqua.components`)
==========================================
Components were designed to be swappable sub-parts of an algorithm. Different implementations
of a component type can thereby be exchanged to potentially alter the behavior and outcome of
the algorithm. For example :class:`~qiskit.aqua.algorithms.VQE` takes an
:class:`~qiskit.aqua.components.optimizers.Optimizer` and a
:class:`~qiskit.aqua.components.variational_forms.VariationalForm` components. There are a
selection of both different :mod:`~qiskit.aqua.components.optimizers` and
:mod:`~qiskit.aqua.components.variational_forms` that can be chosen from according the nature of
the problem. Some optimizers use gradients, others have alternative techniques to finding a
minimum. Variational forms include heuristic ansatzes such as
:class:`~qiskit.aqua.components.variational_forms.RYRZ` and types designed for specific problems
such as :class:`~qiskit.chemistry.components.variational_forms.UCCSD` for chemistry and ground
state energy computation.

Components may also be used in other components. For example the
:class:`~qiskit.aqua.components.uncertainty_models.UnivariateVariationalDistribution` takes a
:class:`~qiskit.aqua.components.variational_forms.VariationalForm`.

Each type of component has a base class that can be extended to provide a new implementation. For
example the base class for :mod:`~qiskit.aqua.components.variational_forms` is
:class:`~qiskit.aqua.components.variational_forms.VariationalForm`. For more information refer
to the component type of interest below.

.. currentmodule:: qiskit.aqua.components

Submodules
==========

.. autosummary::
   :toctree:

   eigs
   feature_maps
   initial_states
   iqfts
   multiclass_extensions
   neural_networks
   optimizers
   oracles
   qfts
   reciprocals
   uncertainty_models
   uncertainty_problems
   variational_forms

"""
