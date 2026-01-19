import multiprocessing
import numpy as np
import ROOT
from ROOT import TLorentzVector, TVector3

class Booster:
    def __init__(self, particles):
        """
        :param particles: np.array of [W0, l0, W1, l1] 4-vectors (px, py, pz, energy) with shape (N, 4)
        """
        self.particles = particles

    def _boost_to_rest_frame(self, vecs, boost_vec):
        """Boosts all TLorentzVectors in vecs by -boost_vec."""
        for vec in vecs:
            vec.Boost(-boost_vec)

    def _construct_basis(self, WpBoson, Beam_p):
        """Constructs orthogonal basis (k, r, n) in Higgs rest frame."""
        k = TVector3(WpBoson.X(), WpBoson.Y(), WpBoson.Z()).Unit()
        p = TVector3(Beam_p.X(), Beam_p.Y(), Beam_p.Z()).Unit()
        y = p.Dot(k)
        r_length = np.sqrt(1 - y * y)
        r = (1 / r_length) * (p - y * k)
        n = (1 / r_length) * (p.Cross(k))
        return k, r, n

    def _map_to_basis(self, lepton, n, r, k):
        """Maps lepton momentum to (n, r, k) basis."""
        lepton_vec = lepton.Vect()
        return TLorentzVector(
            lepton_vec.Dot(n),
            lepton_vec.Dot(r),
            lepton_vec.Dot(k),
            lepton.E()
        )

    def w_rest_booster(self, part):
        WpBoson = TLorentzVector(*part[:4])
        WpLepton = TLorentzVector(*part[4:8])
        WnBoson = TLorentzVector(*part[8:12])
        WnLepton = TLorentzVector(*part[12:16])
        # Step 1: Construct Higgs 4-vector and boost all particles to Higgs rest frame
        Higgs = WpBoson + WnBoson
        Beam_p = TLorentzVector(0, 0, 1, 1) # spatial-axis
        Higgs_boost = Higgs.BoostVector()
        self._boost_to_rest_frame([WpBoson, WpLepton, WnBoson, WnLepton, Beam_p], Higgs_boost)

        # Step 2: Construct orthogonal basis (k, r, n)
        k, r, n = self._construct_basis(WpBoson, Beam_p)

        # Step 3: Boost to W+ and W- rest frames
        self._boost_to_rest_frame([WpBoson, WpLepton], WpBoson.BoostVector())
        self._boost_to_rest_frame([WnBoson, WnLepton], WnBoson.BoostVector())

        # Step 4: Map leptons to (n, r, k) basis
        WpLp_k = self._map_to_basis(WpLepton, n, r, k)
        WnLp_k = self._map_to_basis(WnLepton, n, r, k)

        w_rest_WpLepton = np.array([WpLp_k.X(), WpLp_k.Y(), WpLp_k.Z(), WpLp_k.T()]) # np.array([px, py, pz, energy])
        w_rest_WnLepton = np.array([WnLp_k.X(), WnLp_k.Y(), WnLp_k.Z(), WnLp_k.T()])

        return w_rest_WpLepton, w_rest_WnLepton

    def setup(self):
        # results = [self.w_rest_booster(p) for p in self.particles]
        with multiprocessing.Pool(8) as pool:
			# Retrieve the output from the pool
            results = list(pool.map(self.w_rest_booster, self.particles))
        w_rest_lp, w_rest_ln = zip(*results)
        self.w_rest_lp, self.w_rest_ln = np.vstack(w_rest_lp), np.vstack(w_rest_ln)
        
    def lep_4_in_w_rest(self):
        return self.w_rest_lp, self.w_rest_ln

    def lep_theta_phi_in_w_rest(self):
        
        def theta(p4):
            p3_mag = np.sqrt(np.sum(np.square(p4[:, 0:3]), axis=1))  # Calculate the magnitude of the spatial components
            pz = p4[:, 2]  # Extract the pz component
            return np.arccos(np.divide(pz, p3_mag)) / np.pi  # Normalize to [0, 1]

        def phi(p4):
            phi = np.arctan2(p4[:, 1], p4[:, 0])  # Calculate the azimuthal angle
            return phi / np.pi  # normalize to [-1, 1]

        lead_theta = theta(self.w_rest_lp)
        lead_phi = phi(self.w_rest_lp)
        sublead_theta = theta(self.w_rest_ln)
        sublead_phi = phi(self.w_rest_ln)

        return (lead_theta, lead_phi), (sublead_theta, sublead_phi)
    
    def lep_xi_in_w_rest(self):
        
        def xi(p4):
            xi_n = p4[:, 0] / np.linalg.norm(p4[:, :3], axis=1)
            xi_r = p4[:, 1] / np.linalg.norm(p4[:, :3], axis=1)
            xi_k = p4[:, 2] / np.linalg.norm(p4[:, :3], axis=1)
            return xi_n, xi_r, xi_k

        xi_pos = xi(self.w_rest_lp)
        xi_neg = xi(self.w_rest_ln)

        return xi_pos, xi_neg
    
    def _cglmp(self, z_xp, z_xn, z_yp, z_yn):
        """Calculate Bij (CGLMP values)."""
        tr_a = (np.divide(8, np.sqrt(3))) * (z_xp * z_xn + z_yp * z_yn)
        tr_b = 25 * (np.square(z_xp) - np.square(z_yp)) * (np.square(z_xn) - np.square(z_yn))
        tr_c = 100 * (z_xp * z_yp * z_xn * z_yn)
        return tr_a + tr_b + tr_c
    
    def cglmp_bij(self):
        xi_pos, xi_neg = self.lep_xi_in_w_rest()
        b_xy = self._cglmp(xi_pos[0], xi_neg[0], xi_pos[1], xi_neg[1])
        b_yz = self._cglmp(xi_pos[1], xi_neg[1], xi_pos[2], xi_neg[2])
        b_zx = self._cglmp(xi_pos[0], xi_neg[0], xi_pos[2], xi_neg[2])
        return b_xy, b_yz, b_zx

if __name__ == "__main__":
    import time
    t1 = time.time()
    import load_data
    data = load_data.load_particles_from_h5("/root/data/mc20_truth.h5")
    particles = np.concatenate(
		[
			data["lead_w"]["p4"],
			data["truth_lead_lep"]["p4"],
			data["sublead_w"]["p4"],
			data["truth_sublead_lep"]["p4"],
		],
		axis=-1,
	)
    print(particles.shape)
    booster = Booster(particles)
    booster.setup()
    print(booster.lep_theta_phi_in_w_rest())
    print(booster.lep_4_in_w_rest())
    print(booster.lep_xi_in_w_rest())
    print(booster.cglmp_bij())
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1:<.2f} seconds")