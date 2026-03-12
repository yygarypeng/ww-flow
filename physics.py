import numpy as np

TOR = 1e-16

def pt(px, py):
    return np.sqrt(px**2 + py**2)

def eta(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    return 0.5 * np.log((p + pz) / (p - pz + TOR))

def phi(px, py):
    return np.arctan2(py, px)

def deta(eta1, eta2):
    return eta1 - eta2

def dphi(phi1, phi2):
    phi_diff = phi1 - phi2
    # Wrap to [-pi, pi]
    return (phi_diff + np.pi) % (2 * np.pi) - np.pi

def dr(deta, dphi):
    return np.sqrt(deta**2 + dphi**2)