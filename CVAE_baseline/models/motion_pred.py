import numpy as np
import pdb
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F

from utils.torch import *
from models.mlp import MLP
from models.rnn import RNN

class VAE(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(VAE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn + nz, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')
        
        self.alignment_layer = MLP(4096, [2048, 512, 128])
        

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)[-1]
        return h_y

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, text_embd):
        
        h_x = self.encode_x(x)
        h_text = self.alignment_layer(text_embd)

        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, h_text, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y, text_embd):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z, text_embd), mu, logvar

    def sample_prior(self, x, text_embd):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z, text_embd)


def get_vae_model(cfg, traj_dim):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', 'VAEv1')
    if model_name == 'VAEv1':
        return VAE(traj_dim, traj_dim, cfg.nz, cfg.t_pred, specs)

