import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from hyperparameters import HyperParameters
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hps = HyperParameters()
        # Input data is 5 dimensional because data is in 5 stroke format
        self.lstm = nn.LSTM(5, self.hps.enc_hidden_size, dropout=self.hps.dropout, bidirectional=True)
        # nn.Linear(input dimension, output dimension)
        self.mu_func = nn.Linear(2 * self.hps.enc_hidden_size, self.hps.latent_vector_length)
        self.sigma_func = nn.Linear(2 * self.hps.enc_hidden_size, self.hps.latent_vector_length)
        self.fully_connect = nn.Linear(self.hps.latent_vector_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            h0 = Variable(torch.zeros(2, batch_size, self.hps.enc_hidden_size).cuda())
            c0 = Variable(torch.zeros(2, batch_size, self.hps.enc_hidden_size).cuda())
            hidden_cell = (h0, c0)
        output, (hn, cn) = self.lstm(inputs, hidden_cell)
        hidden_forward, hidden_backward = torch.split(hn, 1, 0)
        hidden_final = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)

        mu = self.mu_func(hidden_final)
        sigma_hat = self.sigma_func(hidden_final)
        sigma = torch.exp(sigma_hat / 2.)

        N = Variable(
            torch.normal(torch.zeros(self.hps.latent_vector_length), torch.ones(self.hps.latent_vector_length)).cuda())
        z = mu + sigma * N
        fc = self.fully_connect(z)
        decision = self.sigmoid(fc)
        return decision.squeeze()
