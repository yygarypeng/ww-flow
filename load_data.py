import numpy as np
import h5py

from physics import pt, eta, phi, deta, dphi, dr
from ohbboosting import Booster

def load_particles_from_h5(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        # For each category (ggF_train, ggF_test, VBF_train, etc.)
        for category_name in f.keys():
            category_data = {}
            
            # For each particle/object group within the category
            for group_name in f[category_name].keys():
                group_data = {}

                # Load datasets (numpy arrays)
                if isinstance(f[category_name][group_name], h5py.Group):
                    for dataset_name in f[category_name][group_name].keys():
                        group_data[dataset_name] = f[category_name][group_name][dataset_name][:]

                    # Load attributes (scalars)
                    for attr_name, attr_value in f[category_name][group_name].attrs.items():
                        group_data[attr_name] = attr_value
                else:
                    # Handle case where it's a dataset directly
                    group_data = f[category_name][group_name][:]

                category_data[group_name] = group_data

            result[category_name] = category_data

    return result

def load_data(data_path):
    
    data = load_particles_from_h5(data_path)

    def col(a):
        return a.reshape(a.shape[0], -1)

    # Collect all training and target objects from all categories
    all_train_objs = []
    all_target_objs = []
    
    # Iterate through all categories (ggF_train, ggF_test, VBF_train, etc.)
    for category in data.keys():
        category_data = data[category]
        
        # training features
        lep_pos_px = category_data["pos_lep"]["px"]
        lep_pos_py = category_data["pos_lep"]["py"]
        lep_pos_pz = category_data["pos_lep"]["pz"]
        lep_pos_energy = category_data["pos_lep"]["energy"]
        lep_neg_px = category_data["neg_lep"]["px"]
        lep_neg_py = category_data["neg_lep"]["py"]
        lep_neg_pz = category_data["neg_lep"]["pz"]
        lep_neg_energy = category_data["neg_lep"]["energy"]
        
        lep_pos_pt = category_data["pos_lep"]["pt"]
        lep_neg_pt = category_data["neg_lep"]["pt"]
        lep_pos_eta = category_data["pos_lep"]["eta"]
        lep_neg_eta = category_data["neg_lep"]["eta"]
        lep_pos_phi = category_data["pos_lep"]["phi"]
        lep_neg_phi = category_data["neg_lep"]["phi"]
        
        dilep_px = lep_pos_px + lep_neg_px
        dilep_py = lep_pos_py + lep_neg_py
        dilep_pz = lep_pos_pz + lep_neg_pz
        dilep_energy = lep_pos_energy + lep_neg_energy
        dilep_pt = pt(dilep_px, dilep_py)
        dilep_eta = eta(dilep_px, dilep_py, dilep_pz)
        dilep_phi = phi(dilep_px, dilep_py)
        m_ll = np.sqrt(dilep_energy**2 - dilep_px**2 - dilep_py**2 - dilep_pz**2)

        met_px = category_data["met"]["px"]
        met_py = category_data["met"]["py"]
        met_pt = category_data["met"]["pt"]
        met_phi = category_data["met"]["phi"]
        
        dphi_llmet = dphi(dilep_phi, met_phi)
        dphi_l1met = dphi(lep_pos_phi, met_phi)
        dphi_l2met = dphi(lep_neg_phi, met_phi)
        dphi_l1l2 = dphi(lep_pos_phi, lep_neg_phi)
        deta_l1l2 = deta(lep_pos_eta, lep_neg_eta)
        dr_l1l2 = dr(deta_l1l2, dphi_l1l2)
        
        
        # only select first 3 jets (leading/subleading/subsubleading)
        jet_px = category_data["jets"]["px"][:, 0:3]
        jet_py = category_data["jets"]["py"][:, 0:3]
        jet_pz = category_data["jets"]["pz"][:, 0:3]
        jet_energy = category_data["jets"]["energy"][:, 0:3]
        jet_btag = category_data["jets"]["btag"][:, 0:3]
        n_jets = category_data["jets"]["n_jets"]
        n_bjets = category_data["jets"]["n_bjets"]

        def safe_log_nonzero(x):
            out = np.zeros_like(x, dtype=np.float32)
            mask = x > 0
            out[mask] = np.log(x[mask])
            return out

        safe_log_jet_e0 = safe_log_nonzero(jet_energy[:, 0])
        safe_log_jet_e1 = safe_log_nonzero(jet_energy[:, 1])
        safe_log_jet_e2 = safe_log_nonzero(jet_energy[:, 2])

        # pack them
        # all training mass-like objects are in GeV unit
        
        train_obj = np.concatenate([
            # total 10 + 22 = 32 features for training (input to the model)
            # for y (observed variables)
            col(lep_pos_px), #0
            col(lep_pos_py), #1
            col(lep_pos_pz), #2
            col(np.log(lep_pos_energy)), #3
            col(lep_neg_px), #4
            col(lep_neg_py), #5
            col(lep_neg_pz), #6
            col(np.log(lep_neg_energy)), #7
            # for cond (conditional variables)
            col(met_px), #8
            col(met_py), #9
            col(jet_px[:, 0]), #0
            col(jet_py[:, 0]), #1
            col(jet_pz[:, 0]), #2
            col(safe_log_jet_e0), #3
            col(jet_px[:, 1]), #4
            col(jet_py[:, 1]), #5
            col(jet_pz[:, 1]), #6
            col(safe_log_jet_e1), #7
            col(jet_px[:, 2]), #8
            col(jet_py[:, 2]), #9
            col(jet_pz[:, 2]), #10
            col(safe_log_jet_e2), #11
            # col(dilep_px), #12
            # col(dilep_py), #13
            # col(dilep_pz), #14
            # col(dilep_energy), #15
            # col(deta_l1l2), #16
            # col(dphi_llmet), #17
            # col(dphi_l1met), #18 (l1 -> pos_lep; l2 -> neg_lep)
            # col(dphi_l2met), #19
            # col(dphi_l1l2), #20
            # col(dr_l1l2), #21
        ], axis=-1)
        
        # target objects
        true_w0_p4 = np.stack([
            category_data["truth_pos_w"]["px"],
            category_data["truth_pos_w"]["py"],
            category_data["truth_pos_w"]["pz"],
            category_data["truth_pos_w"]["energy"]
        ], axis=1)
        true_l0_p4 = np.stack([
            category_data["truth_pos_lep"]["px"],
            category_data["truth_pos_lep"]["py"],
            category_data["truth_pos_lep"]["pz"],
            category_data["truth_pos_lep"]["energy"]
        ], axis=1)
        true_w1_p4 = np.stack([
            category_data["truth_neg_w"]["px"],
            category_data["truth_neg_w"]["py"],
            category_data["truth_neg_w"]["pz"],
            category_data["truth_neg_w"]["energy"]
        ], axis=1)
        true_l1_p4 = np.stack([
            category_data["truth_neg_lep"]["px"],
            category_data["truth_neg_lep"]["py"],
            category_data["truth_neg_lep"]["pz"],
            category_data["truth_neg_lep"]["energy"]
        ], axis=1)
        particles = np.concatenate([
                true_w0_p4,
                true_l0_p4,
                true_w1_p4,
                true_l1_p4,
            ], axis=-1)
        booster = Booster(particles)
        booster.setup()
        true_l0_theta_phi, true_l1_theta_phi = booster.lep_theta_phi_in_w_rest()
        
        target_obj = np.concatenate([
            col(true_l0_theta_phi[0]),
            col(true_l0_theta_phi[1]),
            col(true_l1_theta_phi[0]),
            col(true_l1_theta_phi[1])
        ], axis=-1)
        
        all_train_objs.append(train_obj)
        all_target_objs.append(target_obj)
    
    # Concatenate all categories
    train_obj = np.concatenate(all_train_objs, axis=0)
    target_obj = np.concatenate(all_target_objs, axis=0)
    
    print("Training objects shape:", train_obj.shape)
    print("Target objects shape:", target_obj.shape)

    # After concatenating all categories, add this before the return statement:
    # Remove rows with NaN or infinite values
    valid_train = np.isfinite(train_obj).all(axis=1)
    valid_target = np.isfinite(target_obj).all(axis=1)
    valid_idx = valid_train & valid_target

    train_obj = train_obj[valid_idx]
    target_obj = target_obj[valid_idx]

    print("Removed", (~valid_idx).sum(), "rows with NaN or infinite values")

    return train_obj, target_obj

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5"
    train_obj, target_obj = load_data(data_path)
    print("train_obj shape:", train_obj.shape)
    print("target_obj shape:", target_obj.shape)
    print("train_obj example:", train_obj[0])
    print("target_obj example:", target_obj[0])
    print("Finished loading data.")
