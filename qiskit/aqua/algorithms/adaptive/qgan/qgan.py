#!/usr/bin/env python3
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

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.stats import entropy

import csv

import logging
import sys
if sys.version_info < (3, 5):
    raise Exception('Please use Python version 3.5 or greater.')
sys.path.append('..')

from copy import deepcopy

import torch


from qiskit.providers.aer import Aer

from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm


from qiskit.aqua import aqua_globals, QuantumInstance

from .generator import Generator
from .discriminator import Discriminator

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


logger = logging.getLogger(__name__)

class QGAN(QuantumAlgorithm):
    """
    Quantum Generative Adversarial Network.

    """
    CONFIGURATION = {
        'name': 'Qgan',
        'description': 'Quantum Generative Adversarial Network',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Qgan_schema',
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'array',
                    'default': None
                },
                'bounds': {
                    'type': 'array',
                    'default': None
                },
                'batch_size': {
                    'type': 'integer',
                    'default': 500
                },
                'num_epochs': {
                    'type': 'integer',
                    'default': 3000
                },
                'snapshot_dir': {
                    'type': ['string', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
    }

    def __init__(self, data, bounds=None, num_qubits=None, batch_size=500, num_epochs=3000, snapshot_dir=''):
        """
        Initialize qGAN.
        :param data: training data of dimension k
        :param bounds: list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
                        if univariate data: [min_0,max_0]
        :param num_qubits: list/numpy array, list of k numbers of qubits to determine representation resolution,
        i.e. n qubits enable the representation of 2**n values [num_qubits_0,..., num_qubits_k-1]
        :param batch_size: batch size
        :param num_epochs: number of training epochs
        :param snapshot_dir: if path given store cvs file with parameters to the directory
        """

        self.validate(locals())
        super().__init__()
        if data is None:
            raise AquaError('Training data not given.')
        self._data = np.array(data)
        if bounds is None:
            bounds_min = np.percentile(self._data, 5, axis=0)
            bounds_max = np.percentile(self._data, 95, axis=0)
            bounds = []
            for i in range(len(bounds_min)):
                bounds.append([bounds_min[i], bounds_max[i]])
        if np.ndim(data)>1:
            if len(bounds) != (len(num_qubits) or len(data[0])):
                raise AquaError('Dimensions of the data, the length of the data bounds and the numbers of qubits per '
                                'dimension are incompatible.')
        else:
            if (np.ndim(bounds) or len(num_qubits)) != 1:
                raise AquaError('Dimensions of the data, the length of the data bounds and the numbers of qubits per '
                                'dimension are incompatible.')
        self._bounds = np.array(bounds)
        self._num_qubits = num_qubits
        if np.ndim(data)>1:
            if self._num_qubits is None:
                self._num_qubits = np.ones[len(data[0])]*3
            self._prob_data = np.zeros(int(np.prod(np.power(np.ones(len(self._data[0]))*2, self._num_qubits))))
        else:
            if self._num_qubits is None:
                self._num_qubits = np.array([3])
            self._prob_data = np.zeros(int(np.prod(np.power(np.array([2]), self._num_qubits))))
        self._data_grid = []
        self._grid_elements = None
        self._prepare_data()
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self.generator = None
        self.discriminator = None
        self._snapshot_dir = snapshot_dir
        self._quantum_instance = None
        self.g_loss = []
        self.d_loss = []
        self.rel_entr = []
        self._tol_rel_ent = None
        self.random_seed = 7
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        aqua_globals.random_seed = self.random_seed




    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize qGAN via parameters dictionary and algorithm input instance.
        Args:
            params: parameters dictionary
            algo_input: Input instance
        Returns:
            QGAN: qgan object
        """

        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: Input instance
        """
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        qgan_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        data = qgan_params.get('data')
        bounds = qgan_params.get('bounds')
        batch_size = qgan_params.get('batch_size')
        num_epochs = qgan_params.get('num_epochs')
        snapshot_dir = qgan_params.get('snapshot_dir')

        return cls(data, bounds, batch_size, num_epochs, snapshot_dir)

    def set_seed(self, seed):
        """
        Set a custom seed.
        :param seed: int, seed to be set
        """
        self.random_seed = seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        aqua_globals.random_seed = self.random_seed


    def _prepare_data(self):
        """
        Discretize and truncate the input data such that it is compatible wih the chosen data resolution.
        """
        # Truncate the data
        if np.ndim(self._bounds) == 1:
            bounds = np.reshape(self._bounds, (1,len(self._bounds)))
        else:
            bounds = self._bounds
        self._data = self._data.reshape((len(self._data), len(self._num_qubits)))
        temp = []
        for i, data_sample in enumerate(self._data):
            append = True
            for j, entry in enumerate(data_sample):
                if entry < bounds[j,0]:
                    append = False
                if entry > bounds[j,1]:
                    append = False
            if append:
                temp.append(list(data_sample))
        self._data = np.array(temp)

        # Fit the data to the data resolution. i.e. grid
        for j, prec in enumerate(self._num_qubits):
            data_row = self._data[:, j] #dim j of all data samples
            grid = np.linspace(bounds[j, 0], bounds[j, 1], (2 ** prec)) #prepare data grid for dim j
            index_grid = np.searchsorted(grid, data_row-(grid[1]-grid[0])*0.5) #find index for data sample in grid
            for k, index in enumerate(index_grid):
                self._data[k, j] = grid[index]
            if j == 0:
                if len(self._num_qubits) > 1:
                    self._data_grid = [grid]
                else:
                    self._data_grid = grid
                self._grid_elements = grid
            elif j==1:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = [g_e]
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = temp
            else:
                self._data_grid.append(grid)
                temp = []
                for g_e in self._grid_elements:
                    for g in grid:
                        temp0 = deepcopy(g_e)
                        temp0.append(g)
                        temp.append(temp0)
                self._grid_elements = deepcopy(temp)
        self._data_grid = np.array(self._data_grid)
        self._data = np.reshape(self._data, (len(self._data), len(self._data[0])))
        for data in self._data:
            for i, element in enumerate(self._grid_elements):
                if all(data == element):
                    self._prob_data[i] += 1 / len(self._data)
        self._prob_data = [1e-10 if x == 0 else x for x in self._prob_data]
        return

    def get_rel_entr(self):
        samples_gen, prob_gen = self.generator.get_samples(self._quantum_instance, shots=10000)
        temp = np.zeros(len(self._grid_elements))
        for j, sample in enumerate(samples_gen):
            for i, element in enumerate(self._grid_elements):
                if all(sample == element):
                    temp[i] += prob_gen[j]
        prob_gen = temp
        prob_gen = [1e-8 if x == 0 else x for x in prob_gen]
        rel_entr = entropy(prob_gen, self._prob_data)
        return rel_entr

    def _store_params(self, e, d_loss, g_loss, rel_entr):
        with open(self._snapshot_dir + 'output.csv', mode='a') as csv_file:
            fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator', 'rel_entropy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'epoch': e, 'loss_discriminator': np.average(d_loss),
                             'loss_generator': np.average(g_loss), 'params_generator':
                                 self.generator.generator_circuit.params, 'rel_entropy': rel_entr})
        #Store torch discriminator model
        torch.save(self.discriminator.discriminator, self._snapshot_dir + 'discriminator.pt')

    def set_quantum_instance(self, quantum_instance=None):

        """
        Run training.
        :param quantum_instance: QuantumInstance, Used for running the quantum circuit -
                                 Note that the supported backends are: statevector_simulator, qasm_simulator and actual
                                 quantum hardware
        """
        if quantum_instance is None:
            self._quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
                                                     shots=self._batch_size, seed=self._random_seed,
                                                     seed_mapper=self._random_seed)
        else:
            if quantum_instance.backend_name == ('unitary_simulator' or 'clifford_simulator'):
                raise AquaError(
                    'Chosen backend not supported - Set backend either to statevector_simulator, qasm_simulator'
                    ' or actual quantum hardware')
            else:
                self._quantum_instance = quantum_instance
        return

    def set_generator(self, generator_circuit=None, generator_init_params=None, generator_optimizer=None):
        """
        Initialize generator.
        :param generator_circuit: VariationalForm, parametrized quantum circuit which sets the structure of the quantum generator
        :param generator_init_params: array, initial parameters for the generator circuit
        :param generator_optimizer: Optimizer, optimizer to be used for the training of the generator
        """

        self.generator = Generator(self._bounds, self._num_qubits, self._data_grid, generator_circuit,
                                   generator_init_params, generator_optimizer)
        return

    def set_discriminator(self, discriminator_net=None, discriminator_optimizer=None):
        """
        Initialize discriminator.
        :param discriminator_net: torch.nn.Module or None, Discriminator network.
        :param discriminator_optimizer: torch.optim.Optimizer or None, Optimizer initialized w.r.t discriminator network parameters.

        """
        self.discriminator = Discriminator(len(self._num_qubits), discriminator_net, discriminator_optimizer)
        return


    def train(self):
        """
        Train the qGAN.

        """
        if self._snapshot_dir is not None:
            with open(self._snapshot_dir + '.csv', mode='w') as csv_file:
                fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator',
                              'rel_entropy']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

        for e in range(self._num_epochs):
            print("Epoch {}/{}...".format(e + 1, self._num_epochs))
            np.random.shuffle(self._data)
            index=0
            while (index+self._batch_size)<=len(self._data):
                real_batch = self._data[index: index+self._batch_size]
                index += self._batch_size
                generated_batch, generated_prob = self.generator.get_samples(self._quantum_instance)

                # 1. Train Discriminator
                d_loss_min = self.discriminator.train(real_batch, generated_batch, generated_prob, penalty=True)


                # 2. Train Generator
                g_loss_min = self.generator.train(self.discriminator, self._quantum_instance, self._batch_size)[0]

            self.d_loss.append(d_loss_min.detach().numpy())
            self.g_loss.append(g_loss_min)

            rel_entr = self.get_rel_entr()
            self.rel_entr.append(rel_entr)

            if self._snapshot_dir is not None:
                self._store_params(e, np.around(d_loss_min.detach().numpy(),4), np.around(g_loss_min,4), np.around(rel_entr,4))
                    
            if self._snapshot_dir is None:
                print('Loss Discriminator: ', np.around(d_loss_min.detach().numpy(),4))
                print('Loss Generator: ', np.around(g_loss_min,4))
                print('Relative Entropy: ', np.around(rel_entr,4))
            if self._tol_rel_ent is not None:
                if rel_entr <= self._tol_rel_ent:
                    break

    def run(self, quantum_instance=None, tol_rel_ent=None):
        """
        Run qGAN training
        :param quantum_instance: QuantumInstance, Used for running the quantum circuit -
                                 Note that the supported backends are: statevector_simulator, qasm_simulator and actual
                                 quantum hardware
        :param tol_rel_ent: float or None, Set tolerance level for relative entropy. If the training achieves relative entropy
            equal or lower than tolerance it finishes.
        :return: funct, Run the training of the qGAN
        """
        if self._quantum_instance is None:
            self.set_quantum_instance(quantum_instance)
        if self.generator is None:
            self.set_generator()
        if self.discriminator is None:
            self.set_discriminator()
        if tol_rel_ent is not None:
            self._tol_rel_ent = tol_rel_ent
        return self._run()

    def _run(self):
        """
        Run qGAN training
        :return: funct, Train quantum generator and classical discriminator
        """
        self.train()

        return
