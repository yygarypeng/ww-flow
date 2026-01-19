import torch
import torch.nn as nn

class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        BN(in_dim) -> SiLU -> Linear(in_dim -> out_dim) -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.bn = nn.LayerNorm(in_dim)
        self.act = nn.SiLU()
        self.fc = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.bn(x)
        y = self.act(y)
        y = self.fc(y)
        y = self.drop(y)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
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
            nn.SiLU(),
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
            nn.SiLU(),
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


class WtoNeutrinoBlock(nn.Module):
    """
    Physics-based reversible block:
        Forward : (leptons, W bosons) -> neutrinos 4-vectors
        Reverse : (leptons, neutrinos) -> W bosons
    """

    def __init__(self, scaler_ww, scaler_lvlv):
        super().__init__()
        self.scaler_ww = scaler_ww
        self.scaler_lvlv = scaler_lvlv

    def forward(self, input, output, reverse=False):
        logdet = torch.zeros(input.size(0), device=input.device, dtype=input.dtype)

        if not reverse:
            # ----- Forward path -----
            W_norm = input    # normalized W bosons
            leptons_norm = output  # normalized leptons
        else:
            # ----- Reverse path -----
            leptons_norm = input
            nu_norm = output  # normalized neutrinos (pass from INN nets)

        # Convert scalers to tensors
        mean_ww = torch.tensor(self.scaler_ww.mean_[:8],  device=input.device, dtype=input.dtype)
        std_ww  = torch.tensor(self.scaler_ww.scale_[:8], device=input.device, dtype=input.dtype)
        mean_nu = torch.tensor(self.scaler_ww.mean_[8:], device=input.device, dtype=input.dtype)
        std_nu = torch.tensor(self.scaler_ww.scale_[8:], device=input.device, dtype=input.dtype)
        mean_lvlv = torch.tensor(self.scaler_lvlv.mean_[:8], device=input.device, dtype=input.dtype)
        std_lvlv = torch.tensor(self.scaler_lvlv.scale_[:8], device=input.device, dtype=input.dtype)
        # print("nu scaler shapes:", std_nu.shape, mean_nu.shape)
        # print("Scaler shapes:", std_ww.shape, mean_ww.shape, std_lvlv.shape, mean_lvlv.shape)

        # Denormalize leptons
        leptons_phys = leptons_norm * std_lvlv + mean_lvlv
        l0_px, l0_py, l0_pz, l0_e = leptons_phys[:,0], leptons_phys[:,1], leptons_phys[:,2], leptons_phys[:,3]
        l1_px, l1_py, l1_pz, l1_e = leptons_phys[:,4], leptons_phys[:,5], leptons_phys[:,6], leptons_phys[:,7]

        if not reverse:
            # ----- Forward path -----
            W_phys = W_norm * std_ww + mean_ww
            W0_px, W0_py, W0_pz = W_phys[:,0], W_phys[:,1], W_phys[:,2]
            W1_px, W1_py, W1_pz = W_phys[:,3], W_phys[:,4], W_phys[:,5]

            # neutrino 4-vectors (massless)
            nu0_px, nu0_py, nu0_pz = W0_px - l0_px, W0_py - l0_py, W0_pz - l0_pz
            nu1_px, nu1_py, nu1_pz = W1_px - l1_px, W1_py - l1_py, W1_pz - l1_pz
            nu0_m = torch.zeros_like(nu0_px)
            nu1_m = torch.zeros_like(nu1_px)

            nu_phys = torch.stack([
                nu0_px, nu0_py, nu0_pz,
                nu1_px, nu1_py, nu1_pz,
                nu0_m, nu1_m # preserve shape 
            ], dim=1)

            # Normalize neutrinos
            out = (nu_phys - mean_nu) / std_nu
            
            # jacobian: sum of log(std_in / std_out) for the 6 independent 3-momenta
            # Note: mass is a deterministic function, so it doesn't add to logdet
            logdet = torch.sum(torch.log(std_ww[:6]) - torch.log(std_nu[:6]))

        else:
            # ----- Reverse path -----
            nu_phys = nu_norm * std_nu + mean_nu
            nu0_px, nu0_py, nu0_pz = nu_phys[:,0], nu_phys[:,1], nu_phys[:,2]
            nu1_px, nu1_py, nu1_pz = nu_phys[:,3], nu_phys[:,4], nu_phys[:,5]
            nu0_e = torch.sqrt(nu0_px**2 + nu0_py**2 + nu0_pz**2 + 1e-16) # force massless
            nu1_e = torch.sqrt(nu1_px**2 + nu1_py**2 + nu1_pz**2 + 1e-16)

            # Reconstruct W bosons
            W0_px, W0_py, W0_pz = l0_px + nu0_px, l0_py + nu0_py, l0_pz + nu0_pz
            W1_px, W1_py, W1_pz = l1_px + nu1_px, l1_py + nu1_py, l1_pz + nu1_pz
            W0_e, W1_e = l0_e + nu0_e, l1_e + nu1_e
            W0_m = torch.sqrt(torch.clamp(W0_e**2 - (W0_px**2 + W0_py**2 + W0_pz**2), min=1e-16))
            W1_m = torch.sqrt(torch.clamp(W1_e**2 - (W1_px**2 + W1_py**2 + W1_pz**2), min=1e-16))

            W_phys = torch.stack([
                W0_px, W0_py, W0_pz,
                W1_px, W1_py, W1_pz,
                W0_m, W1_m
            ], dim=1)

            # Normalize W
            out = (W_phys - mean_ww) / std_ww

            # Jacobian: Reverse of forward
            logdet = torch.sum(torch.log(std_nu[:6]) - torch.log(std_ww[:6]))

        return out, logdet

