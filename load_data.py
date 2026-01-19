import numpy as np
import h5py

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

        met_px = category_data["met"]["px"]
        met_py = category_data["met"]["py"]

        jet_px = category_data["jets"]["px"]
        jet_py = category_data["jets"]["py"]
        jet_pz = category_data["jets"]["pz"]
        jet_energy = category_data["jets"]["energy"]
        jet_btag = category_data["jets"]["btag"]
        n_jets = category_data["jets"]["n_jets"]
        n_bjets = category_data["jets"]["n_bjets"]

        # pack them
        # TODO: select 30 GeV threshold
        train_obj = np.concatenate([
            col(lep_pos_px),
            col(lep_pos_py),
            col(lep_pos_pz),
            col(lep_pos_energy),
            col(lep_neg_px),
            col(lep_neg_py),
            col(lep_neg_pz),
            col(lep_neg_energy),
            col(met_px),
            col(met_py),
            col(jet_px),
            col(jet_py),
            col(jet_pz),
            col(jet_energy),
            col(jet_btag), # selection variable
            # col(n_jets),
            # col(n_bjets),
        ], axis=-1)
        
        # target objects
        target_obj = np.concatenate([
            col(category_data["truth_pos_w"]["px"]),
            col(category_data["truth_pos_w"]["py"]),
            col(category_data["truth_pos_w"]["pz"]),
            col(category_data["truth_neg_w"]["px"]),
            col(category_data["truth_neg_w"]["py"]),
            col(category_data["truth_neg_w"]["pz"]),
            col(category_data["truth_pos_w"]["m"]),
            col(category_data["truth_neg_w"]["m"]),
            col(category_data["truth_nu"]["px"]),
			col(category_data["truth_nu"]["py"]),
            col(category_data["truth_nu"]["pz"]),
			col(category_data["truth_antinu"]["px"]),
			col(category_data["truth_antinu"]["py"]),
			col(category_data["truth_antinu"]["pz"]),
			col(np.zeros_like(category_data["truth_nu"]["energy"])), # massless neutino
            col(np.zeros_like(category_data["truth_antinu"]["energy"])),
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
    n_bjets = train_obj[:, -1]
    plt.hist(n_bjets, bins=10, range=(-0.5, 9.5), histtype='stepfilled', alpha=0.7)
    plt.xlabel("Number of b-jets")
    plt.ylabel("Entries")
    # w_pos_mass = target_obj[:, 8]
    # w_neg_mass = target_obj[:, 9]
    # plt.hist(w_pos_mass, bins=50, range=(0, 120), histtype='step', label='W+ mass')
    # plt.hist(w_neg_mass, bins=50, range=(0, 120), histtype='bar', label='W- mass')
    # plt.xlabel("W mass [GeV]")
    # plt.ylabel("Entries")
    # plt.legend()
    # plt.savefig("w_mass.png")
    # print("Train objects:", train_obj)
    # print("Target objects:", target_obj)