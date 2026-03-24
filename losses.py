import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

tor = 1e-16

# MMD Loss
# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy

def mmd_loss(x, y):
    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        sel_dist = 1.0 if median_dist.item() < tor else median_dist
        # print("Median distance: ", sel_dist.item()) # debug
        bandwidth_range = [0.05*sel_dist, 0.5*sel_dist, sel_dist, 5*sel_dist, 50*sel_dist]

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = torch.clamp(rx.t() + rx - 2. * xx, min=tor, max=1e6)
    dyy = torch.clamp(ry.t() + ry - 2. * yy, min=tor, max=1e6)
    dxy = torch.clamp(rx.t() + ry - 2. * xy, min=tor, max=1e6)
    
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + tor))  # add tor to prevent division by zero
    def imq_kernel(a, d):
        return a**2 / (a**2 + d + tor)  # add tor for numerical stability
    
    for a in bandwidth_range:
        XX += imq_kernel(a, dxx)
        YY += imq_kernel(a, dyy)
        XY += imq_kernel(a, dxy)
        # print("bandwidth: ", a)
        # print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    mmd2 = torch.mean(XX + YY - 2. * XY)
    
    if torch.isfinite(mmd2):
        return mmd2  # add tor to prevent sqrt of zero
    else:
        print("\nWarning: MMD loss is not finite. Returning zero loss to avoid NaNs.")        
        # raise Exception("MMD loss is not finite.")
        print("Bandwidth range: ", [a.item() for a in bandwidth_range])
        print("Median distance: ", median_dist.item())
        print("x mean: ", x.mean().item())
        print("y mean: ", y.mean().item())
        print("XX mean: ", XX.mean().item())
        print("YY mean: ", YY.mean().item())
        print("XY mean: ", XY.mean().item(), "\n")
        raise Exception("DEBUG: stop here")

def higgs_loss(x_recon, scaler_X):

    scale = torch.tensor(scaler_X.scale_[:8], device=x_recon.device, dtype=x_recon.dtype)
    mean = torch.tensor(scaler_X.mean_[:8], device=x_recon.device, dtype=x_recon.dtype)
    x_original = x_recon * scale + mean
    px0_r, py0_r, pz0_r, px1_r, py1_r, pz1_r, m0_r, m1_r = x_original[:, :8].unbind(1)

    def higgs_mass(px0, py0, pz0, px1, py1, pz1, m0, m1):
        # Compute energies robustly. Use clamp instead of abs to avoid hiding
        # non-physical issues while still preventing negative values from
        # numerical round-off.
        # Also clamp the argument to avoid overflow -> inf and inf-inf -> NaN later.
        max_arg = torch.tensor(1e12, device=px0.device, dtype=px0.dtype)
        e0_arg = torch.clamp(m0**2 + px0**2 + py0**2 + pz0**2, min=tor, max=max_arg)
        e1_arg = torch.clamp(m1**2 + px1**2 + py1**2 + pz1**2, min=tor, max=max_arg)
        e0 = torch.sqrt(e0_arg)
        e1 = torch.sqrt(e1_arg)
        e_h = e0 + e1
        px_h, py_h, pz_h = px0 + px1, py0 + py1, pz0 + pz1
        p_h2 = px_h**2 + py_h**2 + pz_h**2
        # Avoid negative m^2 from round-off; avoid inf-inf from overflow.
        m2 = torch.clamp(e_h**2 - p_h2, min=0.0, max=max_arg)
        return torch.sqrt(m2 + tor)

    m_h_r = higgs_mass(px0_r, py0_r, pz0_r, px1_r, py1_r, pz1_r, m0_r, m1_r)
    m_h_t = torch.full_like(m_h_r, 125.0)  # target Higgs mass 125 GeV
    # print("Reconstructed Higgs mass:", m_h_r.mean().item())
    loss = F.huber_loss(m_h_r, m_h_t)
    if not torch.isfinite(loss):
        print("Warning: Higgs mass loss is not finite. Returning penalty loss to avoid NaNs.")
        print("m_h_r mean: ", m_h_r.mean().item())
        print("m_h_t mean: ", m_h_t.mean().item())
        # raise Exception("Higgs mass loss is not finite.")
        return torch.tensor(10.0, device=x_recon.device, dtype=x_recon.dtype)
    return loss


def neu_mass_loss(pred_ww, train_features, scaler_X, scaler_Y, y_dim=10):
    scale_ww = torch.tensor(scaler_X.scale_[:8], device=pred_ww.device, dtype=pred_ww.dtype)
    mean_ww = torch.tensor(scaler_X.mean_[:8], device=pred_ww.device, dtype=pred_ww.dtype)
    scale_lvlv = torch.tensor(scaler_Y.scale_[:y_dim], device=train_features.device, dtype=train_features.dtype)
    mean_lvlv = torch.tensor(scaler_Y.mean_[:y_dim], device=train_features.device, dtype=train_features.dtype)
    pred_ww = pred_ww * scale_ww + mean_ww
    train_features = train_features * scale_lvlv + mean_lvlv
    
    px_w0, py_w0, pz_w0 = pred_ww[:, 0], pred_ww[:, 1], pred_ww[:, 2]
    px_w1, py_w1, pz_w1 = pred_ww[:, 3], pred_ww[:, 4], pred_ww[:, 5]
    m_w0, m_w1 = pred_ww[:, 6], pred_ww[:, 7]
    e_w0, e_w1 = torch.sqrt(torch.clamp(m_w0**2 + px_w0**2 + py_w0**2 + pz_w0**2, min=tor)), torch.sqrt(torch.clamp(m_w1**2 + px_w1**2 + py_w1**2 + pz_w1**2, min=tor))

    lep0_px, lep0_py, lep0_pz, lep0_e = train_features[:, 0], train_features[:, 1], train_features[:, 2], train_features[:, 3] 
    lep1_px, lep1_py, lep1_pz, lep1_e = train_features[:, 4], train_features[:, 5], train_features[:, 6], train_features[:, 7]

    def nu_mass_calc(lep_px, lep_py, lep_pz, lep_e, px_w, py_w, pz_w, e_w):
        nu_px = px_w - lep_px
        nu_py = py_w - lep_py
        nu_pz = pz_w - lep_pz
        nu_e = e_w - lep_e
        nu_m2 = nu_e**2 - (nu_px**2 + nu_py**2 + nu_pz**2)
        return nu_m2
    
    nu0_m2 = nu_mass_calc(lep0_px, lep0_py, lep0_pz, lep0_e, px_w0, py_w0, pz_w0, e_w0)
    nu1_m2 = nu_mass_calc(lep1_px, lep1_py, lep1_pz, lep1_e, px_w1, py_w1, pz_w1, e_w1)
    loss = F.huber_loss(nu0_m2, torch.zeros_like(nu0_m2)) + F.huber_loss(nu1_m2, torch.zeros_like(nu1_m2))

    return loss

##### Archive of old code #####

# def randmultin(z_like_tensor, device):
#     batch_size, z_dim = z_like_tensor.shape
#     mean = torch.zeros(z_dim, device=device, dtype=z_like_tensor.dtype)
#     cov = torch.eye(z_dim, device=device, dtype=z_like_tensor.dtype)
#     dist = MultivariateNormal(mean, covariance_matrix=cov)
#     return dist.sample((batch_size,))  # [batch_size, z_dim]
# --> use torch.randn_like(z_pred) instead, see model.py