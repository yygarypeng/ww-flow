import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from utilities import mmd_loss, higgs_loss, neu_mass_loss
from layers import ResidualBlock, WtoNeutrinoBlock

class INN(nn.Module):
    def __init__(
        self, 
        x_dim, inputs_dim, y_dim, z_dim, c_dim,
        num_blocks, internal_dim,
        ww_scaler, lvlv_scaler
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

        # INN subnet constructor
        def subnet_fc(in_dim, out_dim):
            # print(f"subnet_fc called: in_dim={in_dim}, out_dim={out_dim}")
            return nn.Sequential(
                ResidualBlock(in_dim, 64, dropout=0.0),
                ResidualBlock(64, 128, dropout=0.0),
                ResidualBlock(128, 256, dropout=0.0),
                ResidualBlock(256, 128, dropout=0.0),
                ResidualBlock(128, 64, dropout=0.0),
                ResidualBlock(64, out_dim, dropout=0.0),
            )

        # Build GraphINN
        
        # Input node
        nodes = [Ff.InputNode(internal_dim, name='Input')]
        
        # condition node
        cond_node = Ff.ConditionNode(c_dim, name='Condition')

        # Add coupling blocks
        for i in range(num_blocks):
            nodes.append(Ff.Node(
                nodes[-1],
                Fm.PermuteRandom,
                {'seed': 114},
                name=f'Permute {i}'
            ))
            nodes.append(Ff.Node(
                nodes[-1],
                Fm.GLOWCouplingBlock,
                {'subnet_constructor': subnet_fc},
                conditions=cond_node,
                name=f'GLOW {i}'
            ))

        # Output node
        nodes.append(Ff.OutputNode(nodes[-1], name='Output'))

        # Create GraphINN
        self.flow = Ff.GraphINN(nodes + [cond_node])

    def forward(self, input, cond, target=None, reverse=False):
        batch_size = input.shape[0]
        device = input.device
        logdet_total = torch.zeros(batch_size, device=device)

        if not reverse:
            assert target is not None
            assert cond is not None
            
            # ----- Forward -----
            phy_out, phy_logdet = self.phys(
                input,
                target[:, :8],
                reverse=False
            )
            logdet_total += phy_logdet

            x = torch.cat([
                phy_out,
                torch.zeros(batch_size, self.input_pad, device=device)
            ], dim=1) # with size (batch_size, internal_dim)
            output, logdet_flow = self.flow(x, c=cond, rev=False)
            logdet_total += logdet_flow

            y = output[:, :self.y_dim]
            z = output[:, self.y_dim:]
            return (y, z), logdet_total

        else:
            assert cond is not None
        
            # ----- Reverse -----
            output, logdet_flow = self.flow(input, c=cond, rev=True)
            logdet_total += logdet_flow

            recon, phy_logdet = self.phys(
                input[:, :8],
                output[:, :self.x_dim],
                reverse=True
            )
            logdet_total += phy_logdet

            pad = output[:, self.x_dim:]

            x_out = torch.cat([recon, pad], dim=1)
            return x_out, logdet_total


class INNLightningModule(pl.LightningModule):
    def __init__(
        self, 
        x_dim=8, inputs_dim=10, 
        y_dim=10, z_dim=2, c_dim=20,
        num_blocks=6, internal_dim=12, 
        ww_scaler=None, lvlv_scaler=None,
        lr=1e-3, loss_weights=None
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
            ww_scaler, lvlv_scaler
        )

        self.loss_weights = loss_weights or {
            "L_y": 1.0,
            "L_z": 5.0,
            "L_x": 5.0,
            "L_pad": 1.0,
            "L_higgs": 0.0,
            "L_y_mmd": 0.0,
            "L_neu_mass": 0.0,
        }

    def forward(self, x, cond, target=None, reverse=False):
        return self.inn(x, cond, target=target, reverse=reverse)

    def training_step(self, batch, batch_idx):
        x, y_true = batch # without padding on x; sampling z on y
        cond = y_true[:, self.hparams.y_dim:]  # conditioning variables
        y_true = y_true[:, :self.hparams.y_dim]
        (y_pred, z_pred), logdet = self(x, cond, target=y_true, reverse=False)  # Forward pass (full graph for L_y)
        # z_sample = randmultin(z_pred, device=x.device) # random sample from N(0,1) ... generative mode
        z_sample = torch.randn_like(z_pred)  # random sample from N(0,1) ... generative mode
        yz = torch.cat([y_true, z_sample], dim=1)
        yz_pred = torch.cat([y_pred.detach(), z_pred], dim=1)
        x_recon, logdet_inv = self(yz, cond, target=None, reverse=True)
        
        # Losses (paper Sec. 3.3)
        L_y = F.l1_loss(y_pred, y_true)
        L_y_mmd = mmd_loss(y_pred, y_true) # complementary to L_y
        L_z = mmd_loss(yz_pred, yz)
        L_x = mmd_loss(x_recon[:, :-self.inn.input_pad], x) # x_recon include zeros paddings
        # Penalize non-zero padding
        L_pad = (torch.abs(x_recon[:, -self.inn.input_pad:])).mean()
        # specifically target Higgs and neutrino mass reconstruction
        L_higgs = higgs_loss(
            x_recon[:, :-self.inn.input_pad], 
            self.ww_scaler
        )
        L_neu_mass = neu_mass_loss(
            x_recon[:, :-self.inn.input_pad], 
            y_true, 
            self.ww_scaler,
            self.lvlv_scaler
        )
            
        loss = self.loss_weights["L_y"] * L_y \
            + self.loss_weights["L_y_mmd"] * L_y_mmd \
            + self.loss_weights["L_z"] * L_z \
            + self.loss_weights["L_x"] * L_x \
            + self.loss_weights["L_pad"] * L_pad \
            + self.loss_weights["L_higgs"] * L_higgs \
            + self.loss_weights["L_neu_mass"] * L_neu_mass

        self.log_dict({
            "train_loss": loss, "L_y": L_y, "L_z": L_z, "L_x": L_x,
            "L_pad": L_pad, "L_higgs": L_higgs, "L_y_mmd": L_y_mmd, "L_neu_mass": L_neu_mass,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss # return loss to do backprop and optimizer step

    def validation_step(self, batch, batch_idx):
        x, y_true = batch # without padding on x; sampling z on y
        cond = y_true[:, self.hparams.y_dim:]  # conditioning variables
        y_true = y_true[:, :self.hparams.y_dim]
        (y_pred, z_pred), _ = self(x, cond, target=y_true, reverse=False)  # Forward pass
        z_sample = torch.randn_like(z_pred)
        yz = torch.cat([y_true, z_sample], dim=1)
        yz_pred = torch.cat([y_pred, z_pred], dim=1)
        x_recon, _ = self(yz, cond, target=None, reverse=True) 

        L_y = F.l1_loss(y_pred, y_true)
        L_y_mmd = mmd_loss(y_pred, y_true)
        L_z = mmd_loss(yz_pred, yz)
        L_x = mmd_loss(x_recon[:, :-self.inn.input_pad], x)
        L_pad = (torch.abs(x_recon[:, -self.inn.input_pad:])).mean()
        L_higgs = higgs_loss(
            x_recon[:, :-self.inn.input_pad],
            self.ww_scaler
        )
        L_neu_mass = neu_mass_loss(
            x_recon[:, :-self.inn.input_pad], 
            y_true, 
            self.ww_scaler, 
            self.lvlv_scaler
        )
        val_loss = self.loss_weights["L_y"] * L_y \
            + self.loss_weights["L_y_mmd"] * L_y_mmd \
            + self.loss_weights["L_z"] * L_z \
            + self.loss_weights["L_x"] * L_x \
            + self.loss_weights["L_pad"] * L_pad \
            + self.loss_weights["L_higgs"] * L_higgs \
            + self.loss_weights["L_neu_mass"] * L_neu_mass
        
        self.log_dict({
            "val_loss": val_loss, "val_L_y": L_y, "val_L_z": L_z, "val_L_x": L_x,
            "val_L_pad": L_pad, "val_L_higgs": L_higgs, "val_L_y_mmd": L_y_mmd, "val_L_neu_mass": L_neu_mass,
        }, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer