from ROOT import TLorentzVector, TVector3
import multiprocessing
import numpy as np

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
        k = WpBoson.Vect().Unit()
        p = Beam_p.Vect().Unit()
        y = p.Dot(k)
        r_length = np.sqrt(1 - y * y + 1e-16)
        r = (1 / r_length) * (p - y * k)
        n = (1 / r_length) * (p.Cross(k))
        # print("Norm of basis vectors:", np.sum(np.array(n)**2, axis=-1), np.sum(np.array(r)**2, axis=-1), np.sum(np.array(k)**2, axis=-1))
        return n, r, k

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
        part = np.asarray(part, dtype=float).reshape(-1)
        if part.size != 16:
            raise ValueError(f"Expected one event with 16 values, got shape {np.asarray(part).shape} and size {part.size}.")

        WpBoson = TLorentzVector(*part[:4])
        WpLepton = TLorentzVector(*part[4:8])
        WnBoson = TLorentzVector(*part[8:12])
        WnLepton = TLorentzVector(*part[12:16])
        # Step 1: Construct Higgs 4-vector and boost all particles to Higgs rest frame
        Higgs = WpBoson + WnBoson
        Beam_p = TLorentzVector(0, 0, 1, 1) # dummy time and assign beam direction along +z
        self._boost_to_rest_frame([WpBoson, WpLepton, WnBoson, WnLepton], Higgs.BoostVector())

        # Step 2: Construct orthogonal basis (k, r, n) 
        # k along W+ momentum, r in the plane of W- and beam, n orthogonal to both
        n, r, k = self._construct_basis(WnBoson, Beam_p)

        # Step 3: Boost to W+ and W- rest frames
        self._boost_to_rest_frame([WpLepton], WpBoson.BoostVector())
        self._boost_to_rest_frame([WnLepton], WnBoson.BoostVector())

        # Step 4: Map leptons to (n, r, k) basis
        WpLp_k = self._map_to_basis(WpLepton, n, r, k)
        WnLp_k = self._map_to_basis(WnLepton, n, r, k)

        # Keep a consistent 2D row shape for safe concatenation across all events.
        w_rest_WpLepton = np.array([WpLp_k.Px(), WpLp_k.Py(), WpLp_k.Pz(), WpLp_k.E()], dtype=float).reshape(1, -1)
        w_rest_WnLepton = np.array([WnLp_k.Px(), WnLp_k.Py(), WnLp_k.Pz(), WnLp_k.E()], dtype=float).reshape(1, -1)

        return w_rest_WpLepton, w_rest_WnLepton

    def setup(self):
        # results = [self.w_rest_booster(p) for p in self.particles]
        with multiprocessing.Pool(8) as pool:
			# Retrieve the output from the pool
            results = list(pool.map(self.w_rest_booster, self.particles))
        w_rest_lp, w_rest_ln = zip(*results)
        self.w_rest_lp, self.w_rest_ln = np.concatenate(w_rest_lp), np.concatenate(w_rest_ln)
        
    def lep_4_in_w_rest(self):
        return self.w_rest_lp, self.w_rest_ln

    def lep_theta_phi_in_w_rest(self):
        
        def theta(p4):
            p3_mag = np.sqrt(np.sum(np.square(p4[:, 0:3]), axis=1))
            pz = p4[:, 2]
            _clamped = np.clip(np.divide(pz, p3_mag), -1.0, 1.0)
            return np.arccos(_clamped) / np.pi

        def phi(p4):
            phi = np.arctan2(p4[:, 1], p4[:, 0])
            phi_norm = phi / np.pi
            return np.where(phi_norm < -0.5, phi_norm + 2.0, phi_norm)

        pos_theta = theta(self.w_rest_lp)
        pos_phi = phi(self.w_rest_lp)
        neg_theta = theta(self.w_rest_ln)
        neg_phi = phi(self.w_rest_ln)

        return (pos_theta, pos_phi), (neg_theta, neg_phi)
    
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
    from matplotlib import pyplot as plt
    t1 = time.time()
    import load_data
    data = load_data.load_particles_from_h5("/root/data/archived/mc20_truth.h5")
    presel = (data) 
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
    plt.hist(booster.lep_theta_phi_in_w_rest()[0][1], bins=50, alpha=0.5, label="pos_phi")
    plt.savefig("pos_phi_hist.png")
    # print(booster.lep_4_in_w_rest())
    # print(booster.lep_xi_in_w_rest())
    # print(booster.cglmp_bij())
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1:<.2f} seconds")