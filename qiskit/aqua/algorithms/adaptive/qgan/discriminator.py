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

from qiskit.aqua import AquaError

import sys
if sys.version_info < (3, 5):
    raise Exception('Please use Python version 3.5 or greater.')
try:
    import torch
    from torch import nn
    from torch.autograd.variable import Variable
    from torch import optim
    torch_loaded = True
except:
    torch_loaded = False
    # raise Exception('Please install PyTorch')



class DiscriminatorNet(torch.nn.Module):
    """
    Discriminator
    """

    def __init__(self, n_features=1):
        """
        Initialize the discriminator network.
        :param n_features: int, Dimension of input data samples.
        """
        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            :param x: torch.Tensor, Discriminator input, i.e. data sample.
            :return: torch.Tensor, Discriminator output, i.e. data label.
            """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)

        return x


class Discriminator():

    def __init__(self, n_features=1, discriminator_net = None, optimizer=None):
        """
        Initialize the discriminator.

        Arguments
        ---------
        :param n_features: int, Dimension of input data vector.
        :param discriminator_net: torch.nn.Module or None, Discriminator network.
        :param optimizer: torch.optim.Optimizer or None, Optimizer initialized w.r.t discriminator network parameters.

        """
        if not torch_loaded:
            raise AquaError('Please install PyTorch.')

        self.n_features = n_features
        if isinstance(optimizer, optim.Optimizer):
            self._optimizer = optimizer
            if isinstance(discriminator_net, torch.nn.Module):
                self.discriminator = discriminator_net
            else:
                self.discriminator = DiscriminatorNet(self.n_features)
                self._optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-5, amsgrad=True)
        else:
            if isinstance(discriminator_net, torch.nn.Module):
                self.discriminator = discriminator_net
            else:
                self.discriminator = DiscriminatorNet(self.n_features)
            self._optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-5, amsgrad=True)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        return

    def save_model(self, snapshot_dir):
        torch.save(self.discriminator, snapshot_dir + 'discriminator.pt')

    def get_labels(self, x):
        """
        Get data sample labels, i.e. true or fake
        :param x: numpy array, Discriminator input, i.e. data sample.
        :return: torch.Tensor, Discriminator output, i.e. data label.
        """
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)

        return self.discriminator.forward(x)

    def loss_real(self, x, y):
        """
        Loss function for real data samples

        :param x: torch.Tensor, Discriminator output.
        :param y: torch.Tensor, Label of the data point
        :return: torch.Tensor, Loss w.r.t to the real data points.
        """
        loss_funct = nn.BCELoss()
        return loss_funct(x, y)

    def loss_fake(self, x, y, weights):  # x: out, y:labels
        """
        Loss function for fake data samples

        :param x: torch.Tensor, Discriminator output.
        :param y: torch.Tensor, Label of the data point
        :param weights: torch.Tensor, Data weights.
        :return: torch.Tensor, Loss w.r.t to the generated data points.
        """
        loss_funct = nn.BCELoss(weight=weights, reduction='sum')

        return loss_funct(x, y)

    def gradient_penalty(self, x, lambda_=5., k=0.01, c=1.):
        """
        Compute gradient penalty for discriminator optimization

        :param x: numpy array, Generated data sample.
        :param lambda_: float, Gradient penalty coefficient 1.
        :param k: float, Gradient penalty coefficient 2.
        :param c: float, Gradient penalty coefficient 3.
        :return: torch.Tensor, Gradient penalty.
        """
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(x)
        delta_ = torch.rand(x.size()) * c
        z = Variable(x+delta_, requires_grad = True)
        o = self.get_labels(z)
        d = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)

        return lambda_ * ((d.norm(p=2,dim=1) - k)**2).mean()

    def train(self, real_batch, generated_batch, generated_prob, penalty=False):
        """
        Perform one training step w.r.t to the discriminator's parameters

        :param real_batch: torch.Tensor, Training data batch.
        :param generated_batch: numpy array, Generated data batch.
        :param generated_prob: numpy array, Weights of the generated data samples, i.e. measurement frequency for
        qasm/hardware backends resp. measurement probability for statevector backend.
        :param penalty: Boolean, Indicate whether or not penalty function is applied to the loss function.
        :return: torch.Tensor, Loss function w.r.t the updated discriminator parameters.
        """

        # Reset gradients
        self._optimizer.zero_grad()

        real_batch = torch.tensor(real_batch, dtype=torch.float32)
        real_batch = Variable(real_batch)

        # Train on Real Data
        prediction_real = self.get_labels(real_batch)

        error_real = self.loss_real(prediction_real, torch.ones(len(prediction_real), 1))
        error_real.backward()  # x.grad += dloss/dx

        # Train on Generated Data
        generated_batch = np.reshape(generated_batch,(len(generated_batch), self.n_features))
        generated_prob = np.reshape(generated_prob, (len(generated_prob), 1))
        generated_prob = torch.tensor(generated_prob, dtype=torch.float32)
        prediction_fake = self.get_labels(generated_batch)

        # Calculate error and backpropagate
        error_fake = self.loss_fake(prediction_fake, torch.zeros(len(prediction_fake),1), generated_prob)
        error_fake.backward()

        if penalty:
            self.gradient_penalty(real_batch).backward()

        # Update weights with gradients
        self._optimizer.step()  # x += -lr * x.grad

        # Return error and predictions for real and fake inputs

        return 0.5*(error_real + error_fake)
