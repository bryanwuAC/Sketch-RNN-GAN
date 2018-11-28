import numpy as np
import torch
import torch.nn as nn
from hyperparameters import HyperParameters
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hps = HyperParameters()
        # Linear layer of z
        self.z_func = nn.Linear(self.hps.latent_vector_length, 2 * self.hps.dec_hidden_size)
        # Each state of LSTM takes concatenation of S_{i-1} and latent vector z
        self.lstm = nn.LSTM(self.hps.latent_vector_length + 5, self.hps.dec_hidden_size, dropout=self.hps.dropout)
        # Linear layer for yi
        self.y_func = nn.Linear(self.hps.dec_hidden_size, 6 * self.hps.num_mixture + 3)

    # In training phase, len_out must be specified to N_max + 1
    def forward(self, inputs, z, hidden_cell=None, len_out = 1):
        if hidden_cell is None:
            h0, c0 = torch.split(torch.tanh(self.z_func(z)), self.hps.dec_hidden_size, 1)
            hidden_cell = (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous())
        output, (hn, cn) = self.lstm(inputs, hidden_cell)
        # .view is the same as np.reshape
        # len_out != 1 is training
        if len_out != 1:
            y = self.y_func(output.view(-1, self.hps.dec_hidden_size))
        else:
            y = self.y_func(hn.view(-1, self.hps.dec_hidden_size))

        # See formula 5 in original paper
        params = torch.split(y, 6, 1)
        # The last element is one hot vector or current pen state
        params_pen = params[-1]
        params_GMM = torch.stack(params[:-1])

        Pi_hat, mu_x, mu_y, sigma_hat_x, sigma_hat_y, rho_hat_xy = torch.split(params_GMM, 1, 2)

        Pi = F.softmax(Pi_hat.transpose(0,1).squeeze()).view(len_out, -1, self.hps.num_mixture)

        sigma_x = torch.exp(sigma_hat_x.transpose(0,1).squeeze()).view(len_out, -1, self.hps.num_mixture)
        sigma_y = torch.exp(sigma_hat_y.transpose(0,1).squeeze()).view(len_out, -1, self.hps.num_mixture)
        rho_xy = torch.tanh(rho_hat_xy.transpose(0,1).squeeze()).view(len_out, -1, self.hps.num_mixture)

        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out, -1, self.hps.num_mixture)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out, -1, self.hps.num_mixture)


        q = F.softmax(params_pen).view(len_out, -1, 3)

        return Pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hn, cn

