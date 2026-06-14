import numpy as np
from ohbboosting import Booster

def rmse(pred, truth):
	mask = np.isfinite(pred) & np.isfinite(truth)
	if np.sum(mask) != len(pred):
		print(f"Warning: {len(pred) - np.sum(mask)} invalid entries found")
	pred = pred[mask]
	truth = truth[mask]
	return np.sqrt(np.mean((pred - truth) ** 2))

def r2_score(pred, truth):
    mask = np.isfinite(pred) & np.isfinite(truth)
    if np.sum(mask) != len(pred):
        print(f"Warning: {len(pred) - np.sum(mask)} invalid entries found")
    pred = pred[mask]
    truth = truth[mask]
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    return 1 - ss_res / ss_tot

def eval(x_true, x_pred):
    rmse_value = rmse(x_pred, x_true)
    r2_value = r2_score(x_pred, x_true)
    print(f"RMSE: {rmse_value:.4f}")
    print(f"R^2 Score: {r2_value:.4f}")
    return rmse_value, r2_value

if __name__ == "__main__":
    # Prepare data
    import load_data
    data = load_data.load_particles_from_h5("/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5")
    
    #################
    # Example usage #
    #################
    
    # true labels
    particles_true = np.concatenate(
		[
			data["ggF_test"]["truth_pos_w"]["p4"],
			data["ggF_test"]["truth_pos_lep"]["p4"],
			data["ggF_test"]["truth_neg_w"]["p4"],
			data["ggF_test"]["truth_neg_lep"]["p4"],
		],
		axis=-1,
	)
    # print(particles_true.shape) # should be (N, 16)
    booster = Booster(particles_true)
    booster.setup()
    pos_ang_true, neg_ang_true = booster.lep_theta_phi_in_w_rest()
    pos_theta_true, pos_phi_true = pos_ang_true
    neg_theta_true, neg_phi_true = neg_ang_true
    
    # predictions (mock)
    particles_mock = particles_true + np.random.normal(size=particles_true.shape) * 0.1  # add some noise
    booster_mock = Booster(particles_mock)
    booster_mock.setup()
    pos_ang_mock, neg_ang_mock = booster_mock.lep_theta_phi_in_w_rest()
    pos_theta_mock, pos_phi_mock = pos_ang_mock
    neg_theta_mock, neg_phi_mock = neg_ang_mock
    
    # evaluate 
    eval(pos_theta_true, pos_theta_mock)