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

from copy import deepcopy
import csv
import os
import logging

import numpy as np
from scipy.stats import entropy

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua import Pluggable, get_pluggable_class, PluggableType
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.neural_networks.quantum_generator import QuantumGenerator


logger = logging.getLogger(__name__)


class QGAN(QuantumAlgorithm):
    """
    Quantum Generative Adversarial Network.

    """
    CONFIGURATION = {
        'name': 'QGAN',
        'description': 'Quantum Generative Adversarial Network',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Qgan_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': ['array', 'null'],
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
                'seed': {
                    'type': ['integer'],
                    'default': 7
                },
                'tol_rel_ent': {
                    'type': ['number', 'null'],
                    'default': None
                },
                'snapshot_dir': {
                    'type': ['string', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['distribution_learning_loading'],
        'depends': [
            {
                'pluggable_type': 'generative_network',
                'default': {
                    'name': 'QuantumGenerator'
                }
            },
            {
                'pluggable_type': 'discriminative_network',
                'default': {
                    'name': 'ClassicalDiscriminator'
                }
            },
        ],
    }

    def __init__(self, data, bounds=None, num_qubits=None, batch_size=500, num_epochs=3000, seed=7,
                 discriminator=None, generator=None, tol_rel_ent=None, snapshot_dir=None):
        """
        Initialize qGAN.
        Args:
            data: array, training data of dimension k
            bounds: array, k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
                        if univariate data: [min_0,max_0]
            num_qubits: array, k numbers of qubits to determine representation resolution,
        i.e. n qubits enable the representation of 2**n values [num_qubits_0,..., num_qubits_k-1]
            batch_size: int, batch size
            num_epochs: int, number of training epochs
            tol_rel_ent: float or None, Set tolerance level for relative entropy. If the training achieves relative
            entropy equal or lower than tolerance it finishes.
            discriminator: NeuralNetwork, discriminates between real and fake data samples
            generator: NeuralNetwork, generates 'fake' data samples
            snapshot_dir: path or None, if path given store cvs file with parameters to the directory
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
        if np.ndim(data) > 1:
            if len(bounds) != (len(num_qubits) or len(data[0])):
                raise AquaError('Dimensions of the data, the length of the data bounds and the numbers of qubits per '
                                'dimension are incompatible.')
        else:
            if (np.ndim(bounds) or len(num_qubits)) != 1:
                raise AquaError('Dimensions of the data, the length of the data bounds and the numbers of qubits per '
                                'dimension are incompatible.')
        self._bounds = np.array(bounds)
        self._num_qubits = num_qubits
        if np.ndim(data) > 1:
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
        self._snapshot_dir = snapshot_dir
        self._g_loss = []
        self._d_loss = []
        self._rel_entr = []
        self._tol_rel_ent = tol_rel_ent

        self._random_seed = seed

        if generator is None:
            self.set_generator()
        else:
            self._generator = generator
        if discriminator is None:
            self.set_discriminator()
        else:
            self._discriminator = discriminator

        self.seed = self._random_seed

        self._ret = {}

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

        if algo_input is None:
            raise AquaError("Input instance not supported.")

        qgan_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_qubits = qgan_params.get('num_qubits')
        batch_size = qgan_params.get('batch_size')
        num_epochs = qgan_params.get('num_epochs')
        seed = qgan_params.get('seed')
        tol_rel_ent = qgan_params.get('tol_rel_ent')
        snapshot_dir = qgan_params.get('snapshot_dir')

        discriminator_params = params.get(Pluggable.SECTION_KEY_DISCRIMINATIVE_NETWORK)
        generator_params = params.get(Pluggable.SECTION_KEY_GENERATIVE_NETWORK)
        generator_params['num_qubits'] = num_qubits

        discriminator = get_pluggable_class(PluggableType.DISCRIMINATIVE_NETWORK,
                                            discriminator_params['name']).init_params(params)
        generator = get_pluggable_class(PluggableType.GENERATIVE_NETWORK,
                                        generator_params['name']).init_params(params)

        return cls(algo_input.data, algo_input.bounds, num_qubits, batch_size, num_epochs, seed, discriminator,
                   generator, tol_rel_ent, snapshot_dir)

    @property
    def seed(self):
        return self._random_seed

    @seed.setter
    def seed(self, s):
        """
        Args:
            s: int, random seed

        Returns:

        """
        self._random_seed = s
        aqua_globals.random_seed = self._random_seed
        self._discriminator.set_seed(self._random_seed)

    @property
    def tol_rel_ent(self):
        return self._tol_rel_ent

    @tol_rel_ent.setter
    def tol_rel_ent(self, t):
        """
        Set tolerance for relative entropy
        Args:
            t: float or None, Set tolerance level for relative entropy. If the training achieves relative
               entropy equal or lower than tolerance it finishes.
        Returns:
        """
        self._tol_rel_ent = t

    @property
    def generator(self):
        return self._generator

    def set_generator(self, generator_circuit=None, generator_init_params=None, generator_optimizer=None):
        """
        Initialize generator.
        Args:
            generator_circuit: VariationalForm, parametrized quantum circuit which sets the structure of the quantum
                               generator
            generator_init_params: array, initial parameters for the generator circuit
            generator_optimizer: Optimizer, optimizer to be used for the training of the generator

        Returns:

        """
        self._generator = QuantumGenerator(self._bounds, self._num_qubits, generator_circuit, generator_init_params,
                                           self._snapshot_dir)
        return

    @property
    def discriminator(self):
        return self._discriminator

    def set_discriminator(self):
        """
        Initialize discriminator.

        Returns:

        """
        from qiskit.aqua.components.neural_networks.classical_discriminator import ClassicalDiscriminator
        self._discriminator = ClassicalDiscriminator(len(self._num_qubits))
        self._discriminator.set_seed(self._random_seed)
        return

    @property
    def g_loss(self):
        return self._g_loss

    @property
    def d_loss(self):
        return self._d_loss

    @property
    def rel_entr(self):
        return self._rel_entr

    def _prepare_data(self):
        """
        Discretize and truncate the input data such that it is compatible wih the chosen data resolution.
        """
        # Truncate the data
        if np.ndim(self._bounds) == 1:
            bounds = np.reshape(self._bounds, (1, len(self._bounds)))
        else:
            bounds = self._bounds
        self._data = self._data.reshape((len(self._data), len(self._num_qubits)))
        temp = []
        for i, data_sample in enumerate(self._data):
            append = True
            for j, entry in enumerate(data_sample):
                if entry < bounds[j, 0]:
                    append = False
                if entry > bounds[j, 1]:
                    append = False
            if append:
                temp.append(list(data_sample))
        self._data = np.array(temp)

        # Fit the data to the data resolution. i.e. grid
        for j, prec in enumerate(self._num_qubits):
            data_row = self._data[:, j]  # dim j of all data samples
            grid = np.linspace(bounds[j, 0], bounds[j, 1], (2 ** prec))  # prepare data grid for dim j
            index_grid = np.searchsorted(grid, data_row-(grid[1]-grid[0])*0.5)  # find index for data sample in grid
            for k, index in enumerate(index_grid):
                self._data[k, j] = grid[index]
            if j == 0:
                if len(self._num_qubits) > 1:
                    self._data_grid = [grid]
                else:
                    self._data_grid = grid
                self._grid_elements = grid
            elif j == 1:
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
        samples_gen, prob_gen = self._generator.get_output(self._quantum_instance, shots=1000)
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
        with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='a') as csv_file:
            fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator', 'rel_entropy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'epoch': e, 'loss_discriminator': np.average(d_loss),
                             'loss_generator': np.average(g_loss), 'params_generator':
                                 self._generator.generator_circuit.params, 'rel_entropy': rel_entr})
        self._discriminator.save_model(self._snapshot_dir)  # Store discriminator model

    def train(self):
        """
        Train the qGAN
        """
        if self._snapshot_dir is not None:
            with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='w') as csv_file:
                fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator',
                              'rel_entropy']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

        for e in range(self._num_epochs):
            aqua_globals.random.shuffle(self._data)
            index = 0
            while (index+self._batch_size) <= len(self._data):
                real_batch = self._data[index: index+self._batch_size]
                index += self._batch_size
                generated_batch, generated_prob = self._generator.get_output(self._quantum_instance,
                                                                             shots=self._batch_size)

                # 1. Train Discriminator
                ret_d = self._discriminator.train([real_batch, generated_batch],
                                                  [np.ones(len(real_batch))/len(real_batch), generated_prob],
                                                  penalty=True)
                d_loss_min = ret_d['loss'].detach().numpy()

                # 2. Train Generator
                self._generator.set_discriminator(self._discriminator)
                ret_g = self._generator.train(self._quantum_instance, shots=self._batch_size)
                g_loss_min = ret_g['loss']

            self._d_loss.append(d_loss_min)
            self._g_loss.append(g_loss_min)

            rel_entr = self.get_rel_entr()
            self._rel_entr.append(rel_entr)
            self._ret['params_d'] = ret_d['params']
            self._ret['params_g'] = ret_g['params']
            self._ret['loss_d'] = np.around(d_loss_min, 4)
            self._ret['loss_g'] = np.around(g_loss_min, 4)
            self._ret['rel_entr'] = np.around(rel_entr, 4)

            if self._snapshot_dir is not None:
                self._store_params(e, np.around(d_loss_min.detach().numpy(), 4),
                                   np.around(g_loss_min, 4), np.around(rel_entr, 4))
                    
            logger.debug('Epoch {}/{}...'.format(e + 1, self._num_epochs))
            logger.debug('Loss Discriminator: ', np.around(d_loss_min, 4))
            logger.debug('Loss Generator: ', np.around(g_loss_min, 4))
            logger.debug('Relative Entropy: ', np.around(rel_entr, 4))

            if self._tol_rel_ent is not None:
                if rel_entr <= self._tol_rel_ent:
                    break

    def _run(self):
        """
        Run qGAN training
        Returns: dict, with generator(discriminator) parameters & loss, relative entropy

        """
        if self._quantum_instance.backend_name == ('unitary_simulator' or 'clifford_simulator'):
            raise AquaError(
                'Chosen backend not supported - Set backend either to statevector_simulator, qasm_simulator'
                ' or actual quantum hardware')

        # The number of shots must be the batch size
        self._quantum_instance.set_config(shots=self._batch_size)
        self.train()

        return self._ret
