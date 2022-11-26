import torch
import torch.nn as nn
import torch.nn.functional as F


class MFMLP(nn.Module):
    """
    Multi-class Field Multi Layer Perceptron
    """

    def __init__(self, d_in, c_in, dropout, n_c):
        super(MFMLP, self).__init__()
        self.d_in = d_in
        self.c_in = c_in

        self.seq1 = nn.Sequential(
            nn.Linear(self.d_in, 5),
            nn.Linear(5, 10),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.c_in * 10, 20),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(20),
            nn.Linear(20, n_c),
        )
        self.gradients = None

    def hooker(self, grad):
        self.gradients = grad

    def forward(self, x):
        b, c, d = x.shape
        assert c == self.c_in
        assert d == self.d_in
        device = x.device
        out = torch.empty(b, 10*c).to(device)
        for i in range(c):
            _x = x[:, i, :]
            _x = self.seq1(_x)
            out[:, i*10: (i+1)*10] = _x

        # x.register_hook(self.hooker)
        self.register_buffer('latent_base', out, persistent=False)

        out = self.seq2(out)
        return out


class CMFMLP(nn.Module):
    """
    Conditional Multi-class Field Multi Layer Perceptron
    """

    def __init__(self, d_cond, d_in, c_in, dropout, n_c):
        super(CMFMLP, self).__init__()
        self.d_cond = d_cond
        self.d_in = d_in
        self.c_in = c_in

        self.seq1 = nn.Sequential(
            nn.Linear(self.d_in, 5),
            nn.Linear(5, 10),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(10),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.c_in * 10 + self.d_cond, 20),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(20),
            nn.Linear(20, n_c),
        )
        self.gradients = None

    def hooker(self, grad):
        self.gradients = grad

    def forward(self, x, cond):
        b, c, d = x.shape
        assert c == self.c_in
        assert d == self.d_in
        device = x.device
        out = torch.empty(b, 10*c).to(device)
        for i in range(c):
            _x = x[:, i, :]
            _x = self.seq1(_x)
            out[:, i*10: (i+1)*10] = _x

        # x.register_hook(self.hooker)
        self.register_buffer('latent_base', out, persistent=False)
        out = torch.cat((out, cond), dim=1)
        out = self.seq2(out)
        return out
