import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from losses import mmd_loss, higgs_loss, neu_mass_loss
from layers import WtoNeutrinoBlock, CondNet

class INN(nn.Module):
    def __init__(
        self, 
        x_dim, inputs_dim, y_dim, z_dim, c_dim,
        num_blocks, internal_dim,
        ww_scaler, lvlv_scaler,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.inputs_dim = inputs_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.internal_dim = internal_dim
        self.output_dim = y_dim + z_dim
        self.input_pad = internal_dim - x_dim

        assert self.output_dim == internal_dim, f"output_dim {self.output_dim} != internal_dim {internal_dim}"
        assert self.internal_dim == self.y_dim + self.z_dim, f"internal_dim {self.internal_dim} != y_dim {self.y_dim} + z_dim {self.z_dim}"
        assert self.internal_dim == self.x_dim + self.input_pad, f"internal_dim {self.internal_dim} != x_dim {self.x_dim} + input_pad {self.input_pad}"
        assert self.output_dim % 2 == 0, f"output_dim={self.output_dim} must be even"
        assert internal_dim >= x_dim, f"internal_dim {internal_dim} must be >= x_dim {x_dim}"
        assert self.input_pad >= 0, f"input_pad {self.input_pad} must be >= 0"

        # Physics block (unchanged)
        self.phys = WtoNeutrinoBlock(ww_scaler, lvlv_scaler)

        # Build GraphINN
        # Input node
        nodes = [Ff.InputNode(internal_dim, name='Input')]
        
        # condition node
        cond_node = Ff.ConditionNode(c_dim, name='Condition')
        
        # Add coupling blocks
        def subnet_constructor(in_channels, out_channels, c_dim=self.c_dim, d_model=256, nhead=16, dropout=0.1):
            return CondNet(in_channels=in_channels, out_channels=out_channels, c_dim=c_dim, d_model=d_model, nhead=nhead, dropout=dropout)
        
        for i in range(num_blocks):
            nodes.append(Ff.Node(
                nodes[-1],
                Fm.AllInOneBlock,
                {
                    'subnet_constructor': subnet_constructor,
                    'affine_clamping': 3.0,
                    "global_affine_init": 1.0,
                    'global_affine_type': 'SOFTPLUS',
                    'permute_soft': True
                },
                conditions=cond_node,
                name=f'CondAllInOne {i}'
            ))
            
        # Output node
        nodes.append(Ff.OutputNode(nodes[-1], name='Output'))

        # Create GraphINN
        self.flow = Ff.GraphINN(nodes + [cond_node])

    def forward(self, input, cond, target=None, reverse=False):
        batch_size = input.shape[0]
        device = input.device
        # logdet_total = torch.zeros(batch_size, device=device)

        if not reverse:
            assert target is not None
            assert cond is not None
            
            # ----- Forward -----
            phy_out, _ = self.phys(
                input,
                target[:, :8],
                reverse=False
            )
            # logdet_total += phy_logdet

            x = torch.cat([
                phy_out,
                torch.zeros(batch_size, self.input_pad, device=device)
            ], dim=1) # with size (batch_size, internal_dim)
            output, _ = self.flow(x, c=cond, rev=False)
            # logdet_total += logdet_flow

            y = output[:, :self.y_dim]
            z = output[:, self.y_dim:]
            return (y, z)

        else:
            assert cond is not None
        
            # ----- Reverse -----
            output, _ = self.flow(input, c=cond, rev=True)
            # logdet_total += logdet_flow

            recon, _ = self.phys(
                input[:, :8],
                output[:, :self.x_dim],
                reverse=True
            )
            # logdet_total += phy_logdet

            pad = output[:, self.x_dim:]

            x_out = torch.cat([recon, pad], dim=1)
            return x_out


class INNLightningModule(pl.LightningModule):
    def __init__(
        self, 
        x_dim=8, inputs_dim=10, 
        y_dim=10, z_dim=2, c_dim=20,
        num_blocks=6, internal_dim=12, 
        ww_scaler=None, lvlv_scaler=None,
        lr=1e-3, loss_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Make scalers available to loss functions
        self.ww_scaler = ww_scaler
        self.lvlv_scaler = lvlv_scaler

        self.inn = INN(
            x_dim, inputs_dim,
            y_dim, z_dim, c_dim,
            num_blocks, internal_dim,
            ww_scaler, lvlv_scaler,
        )

        default_loss_weights = {
            "L_x": 1.0,
            "L_y": 1.0,
            "L_z": 1.0,
            "L_pad": 1.0,
            "L_pad_noise": 1.0,
            "L_x_gen": 0.0,
            # physical losses
            "L_W": 0.0,
            "L_x_huber": 0.0,
            "L_higgs": 0.0,
            "L_neu_mass": 0.0,
        }
        self.loss_weights = loss_weights or default_loss_weights

    def forward(self, x, cond, target=None, reverse=False):
        return self.inn(x, cond, target=target, reverse=reverse)


    def _shared_step(self, batch):
        x, y_true = batch # without padding on x; sampling z on y
        cond = y_true[:, self.hparams.y_dim:]  # conditioning variables
        y_true = y_true[:, :self.hparams.y_dim]
        (y_pred, z_pred) = self(x, cond, target=y_true, reverse=False)  # Forward pass (full graph for L_y)
        z_sample = torch.randn_like(z_pred)  # random sample from N(0,1) ... generative mode
        yz = torch.cat([y_true, z_sample], dim=1)
        yz_pred = torch.cat([y_pred.detach(), z_pred], dim=1)
        x_recon = self(yz, cond, target=None, reverse=True)

        # Reverse consistency on generated samples (paper-style: block gradient to y branch)
        yz_gen = torch.cat([y_pred, z_sample], dim=1)
        x_gen_recon = self(yz_gen, cond, target=None, reverse=True)

        # noisy-padding (ensure 0, and noise can give the same outcomes)
        x_recon_noisy, _ = self.inn.flow(yz_gen, c=cond, rev=True) # x'
        x_recon_nopad = x_recon[:, :-(self.inn.input_pad+2)] # not include neutrino masses part
        pad_std = x_recon_nopad.detach().std(dim=0)
        noise_scale = pad_std.mean() + 1e-8 # protect against zero std
        x_recon_noisy[:, -self.inn.input_pad:] = torch.randn_like(x_recon_noisy[:, -self.inn.input_pad:]) * noise_scale
        yz_from_noisy_pad, _ = self.inn.flow(x_recon_noisy, c=cond, rev=False)
        
        # losses (paper Sec. 3.3)
        L_x = mmd_loss(
            x_recon[:, :-(self.inn.input_pad+2)], 
            x[:, :-2]
        ) # not include W masses
        L_y = F.huber_loss(
            y_pred, 
            y_true
        )
        # L_y_mmd = mmd_loss(y_pred, y_true) # complementary to L_y
        L_z = mmd_loss(
            yz_pred, 
            yz
        )
        # Penalize non-zero padding and noisy padding (sec 3.3)
        L_pad = (torch.square(x_recon[:, -self.inn.input_pad:])).mean()
        L_pad_noise = F.huber_loss(yz_from_noisy_pad, torch.cat([y_pred.detach(), z_pred.detach()], dim=1))
        
        # additional physical losses
        L_w = mmd_loss(
            x_recon[:, -2-self.inn.input_pad:-self.inn.input_pad], 
            x[:, -2:]
        ) # W mass part only
        L_higgs = higgs_loss(
            x_recon[:, :-self.inn.input_pad], 
            self.ww_scaler
        )
        L_neu_mass = neu_mass_loss(
            x_recon[:, :-self.inn.input_pad], 
            y_true, 
            self.ww_scaler,
            self.lvlv_scaler,
            y_dim=self.hparams.y_dim
        )
        L_x_huber = F.huber_loss(
            x_recon[:, :-(self.inn.input_pad+2)], 
            x[:, :-2]
        ) # not include neutrino masses
        L_x_gen = F.huber_loss(
            x_gen_recon[:, :-(self.inn.input_pad+2)],
            x[:, :-2]
        )
        
        return {
            "L_x": L_x,
            "L_y": L_y,
            "L_z": L_z,
            "L_pad": L_pad,
            "L_pad_noise": L_pad_noise,
            "L_x_gen": L_x_gen,
            "L_W": L_w,
            "L_higgs": L_higgs,
            "L_neu_mass": L_neu_mass,
            "L_x_huber": L_x_huber
        }


    def training_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
        loss = self.loss_weights["L_x"] * losses["L_x"] \
            + self.loss_weights["L_y"] * losses["L_y"] \
            + self.loss_weights["L_z"] * losses["L_z"] \
            + self.loss_weights["L_pad"] * losses["L_pad"] \
            + self.loss_weights["L_pad_noise"] * losses["L_pad_noise"] \
            + self.loss_weights["L_x_gen"] * losses["L_x_gen"] \
            + self.loss_weights["L_W"] * losses["L_W"] \
            + self.loss_weights["L_higgs"] * losses["L_higgs"] \
            + self.loss_weights["L_neu_mass"] * losses["L_neu_mass"] \
            + self.loss_weights["L_x_huber"] * losses["L_x_huber"]

        self.log_dict({
            "train_loss": loss.detach(), 
            "L_x": losses["L_x"].detach(), 
            "L_y": losses["L_y"].detach(), 
            "L_z": losses["L_z"].detach(), 
            "L_pad": losses["L_pad"].detach(),
            "L_pad_noise": losses["L_pad_noise"].detach(),
            "L_x_gen": losses["L_x_gen"].detach(),
            "L_W": losses["L_W"].detach(),
            "L_higgs": losses["L_higgs"].detach(),  
            "L_neu_mass": losses["L_neu_mass"].detach(),
            "L_x_huber": losses["L_x_huber"].detach(),
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss # return loss to do backprop and optimizer step

    def validation_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
            
        val_loss = self.loss_weights["L_x"] * losses["L_x"] \
            + self.loss_weights["L_y"] * losses["L_y"] \
            + self.loss_weights["L_z"] * losses["L_z"] \
            + self.loss_weights["L_pad"] * losses["L_pad"] \
            + self.loss_weights["L_pad_noise"] * losses["L_pad_noise"] \
            + self.loss_weights["L_x_gen"] * losses["L_x_gen"] \
            + self.loss_weights["L_W"] * losses["L_W"] \
            + self.loss_weights["L_higgs"] * losses["L_higgs"] \
            + self.loss_weights["L_neu_mass"] * losses["L_neu_mass"] \
            + self.loss_weights["L_x_huber"] * losses["L_x_huber"]

        self.log_dict({
            "val_loss": val_loss.detach(), 
            "val_L_x": losses["L_x"].detach(), 
            "val_L_y": losses["L_y"].detach(), 
            "val_L_z": losses["L_z"].detach(), 
            "val_L_pad": losses["L_pad"].detach(), 
            "val_L_pad_noise": losses["L_pad_noise"].detach(),
            "val_L_x_gen": losses["L_x_gen"].detach(),
            "val_L_W": losses["L_W"].detach(),
            "val_L_higgs": losses["L_higgs"].detach(),
            "val_L_neu_mass": losses["L_neu_mass"].detach(),
            "val_L_x_huber": losses["L_x_huber"].detach()
        }, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        return optimizer
