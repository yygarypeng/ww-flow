import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        BN(in_dim) -> GELU -> Linear(in_dim -> out_dim) -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.act = nn.GELU()
        self.fc = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.act(y)
        y = self.fc(y)
        y = self.drop(y)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()

        # projection only when needed
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        # two pre-activation dense blocks
        self.block1 = DenseDropoutBlock(in_dim, out_dim, dropout)
        self.block2 = DenseDropoutBlock(out_dim, out_dim, dropout)

    def forward(self, x):
        identity = self.proj(x)
        y = self.block1(x)
        y = self.block2(y)
        return identity + y

class SNet(nn.Module):
    def __init__(self, half, hidden_dim):
        super().__init__()

        self.input_layer = nn.Linear(half, hidden_dim // 2)
        self.res1 = ResidualBlock(hidden_dim // 2, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, half),
            nn.Tanh()   # keep scale bounded
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res2(x)
        x = self.proj(x)
        return x # scale output to [-1, 1]

class TNet(nn.Module):
    def __init__(self, half, hidden_dim):
        super().__init__()

        self.input_layer = nn.Linear(half, hidden_dim // 2)
        self.res1 = ResidualBlock(hidden_dim // 2, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, half),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res2(x)
        x = self.proj(x)
        return x

class AffineCoupling(nn.Module):
    # single affine coupling: transforms second half conditioned on first half (or vice versa)
    def __init__(self, dim, hidden_dim, mask='left'):
        super().__init__()
        self.mask = mask  # 'left' or 'right'
        half = dim // 2
        # scale and translate nets: input is conditioning half -> output is half-dim
        self.s_net = SNet(half, hidden_dim)
        self.t_net = TNet(half, hidden_dim)

    def forward(self, x, reverse=False):
        # split
        d = x.shape[1] # dim of input (W,W)
        h = d // 2
        if self.mask == 'left':
            x1, x2 = x[:, :h], x[:, h:]
            cond, transform = x1, x2
        else:
            x1, x2 = x[:, :h], x[:, h:]
            cond, transform = x2, x1

        s = self.s_net(cond)
        t = self.t_net(cond)
        if not reverse:
            y2 = transform * torch.exp(s) + t
            logdet = s.sum(dim=1)
        else:
            # inverse
            y2 = (transform - t) * torch.exp(-s)
            logdet = (-s).sum(dim=1)

        if self.mask == 'left':
            out = torch.cat([cond, y2], dim=1)
        else:
            out = torch.cat([y2, cond], dim=1)

        return out, logdet
    
class ReversibleBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.coup1 = AffineCoupling(dim, hidden_dim, mask='left')
        self.coup2 = AffineCoupling(dim, hidden_dim, mask='right')

    def forward(self, x, reverse=False):
        logdet = torch.zeros(x.shape[0], device=x.device)
        if not reverse:
            x, ld = self.coup1(x, reverse=False)
            logdet = logdet + ld
            x, ld = self.coup2(x, reverse=False)
            logdet = logdet + ld
        else:
            x, ld = self.coup2(x, reverse=True)
            logdet = logdet + ld
            x, ld = self.coup1(x, reverse=True)
            logdet = logdet + ld
        return x, logdet

class Permutation(nn.Module):
    def __init__(self, dim, seed=114):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(dim, generator=g)
        self.inv_perm = self.perm.argsort()

    def forward(self, x, reverse=False):
        if reverse:
            return x[:, self.inv_perm], torch.zeros(x.shape[0], device=x.device)
        else:
            return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

