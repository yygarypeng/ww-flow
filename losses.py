import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# MMD Loss
# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy

def mmd_loss(x, y):
    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        sel_dist = 1.0 if median_dist.item() < 1e-16 else median_dist
        # print("Median distance: ", sel_dist.item()) # debug
        bandwidth_range = [0.05*sel_dist, 0.5*sel_dist, sel_dist, 5*sel_dist, 50*sel_dist]

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = torch.clamp(rx.t() + rx - 2. * xx, min=1e-16, max=1e6)
    dyy = torch.clamp(ry.t() + ry - 2. * yy, min=1e-16, max=1e6)
    dxy = torch.clamp(rx.t() + ry - 2. * xy, min=1e-16, max=1e6)
    
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + 1e-16))  # add 1e-16 to prevent division by zero
    def imq_kernel(a, d):
        return a**2 / (a**2 + d + 1e-16)  # add 1e-16 for numerical stability
    
    for a in bandwidth_range:
        XX += imq_kernel(a, dxx)
        YY += imq_kernel(a, dyy)
        XY += imq_kernel(a, dxy)
        # print("bandwidth: ", a)
        # print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    mmd2 = torch.mean(XX + YY - 2. * XY)
    
    if torch.isfinite(mmd2):
        return mmd2  # add 1e-16 to prevent sqrt of zero
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