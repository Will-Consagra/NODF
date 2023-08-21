import torch 
import numpy as np
from ax.service.ax_client import AxClient
import copy 

from data_objects.data_objects import resample_data
from utility.utility import matern_spec_density

def _evaluate(device, parameterization, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test):
	## obtain optimization parameters based on values in `parameterization' dict
	if "r" in parameterization:
		field_model = field_model_(hidden_features=parameterization["r"]).to(device) 
	else:
		field_model = field_model_().to(device) 

	if "learning_rate" in parameterization:
		optim = optimizer_(params=field_model.parameters(), lr=parameterization["learning_rate"])
	else:
		optim = optimizer_(params=field_model.parameters())

	if "nu" in parameterization:
		eigs_root = hyper_params["eigs_root"]
		rho = hyper_params["rho"]
		nu = parameterization["nu"]
		R_tensor = torch.from_numpy(np.diag(1/matern_spec_density(eigs_root, rho, nu))).float().to(device)
		hyper_params["R_tensor"] = R_tensor
	else:
		R_tensor = hyper_params["R_tensor"]
		
	hyper_params["lambda_c"] = parameterization["lambda_c"]	
	## estimate parameters 
	algorithm(device,
				field_model,
				optim,
				hyper_params,
				dataloader_train, 
				num_epochs, 
				verbose=False)
	## get mode predictions at test locations 
	Phi_tensor = hyper_params["Phi_tensor"]
	coords_test = dataloader.dataset.X.to(device)
	Y_tensor_test = dataloader.dataset.Y.T.to(device)
	C_hat_test = field_model(coords_test)["model_out"]
	return float(((Y_tensor_test - Phi_tensor @ C_hat_test.T)**2).sum().cpu().detach().numpy())

def BO_optimization(device, Obs, field_model_, optimizer_, algorithm, hyper_params, parameter_map, Nexperiments, obj_func="MISE",
					num_epochs=1000, experiment_name="hyper_param_opt_experiment", train_prop=0.8, batch_frac=1):
	"""
	Implements BO-based hyper-parameter optimization: Algorithm 3 from arXiv:2307.08138
	"""
	map_ax_client = AxClient()
	map_ax_client.create_experiment(name=experiment_name,
									parameters=parameter_map,
									minimize=True, 
									)

	for _ in range(Nexperiments):
		parameters, trial_index = map_ax_client.get_next_trial()
		if "num_epochs" in parameters:
			num_epochs = parameters["num_epochs"]
		dataloader_train, dataloader_test = resample_data(Obs, train_prop=train_prop, batch_frac=batch_frac)
		map_ax_client.complete_trial(trial_index=trial_index, 
							raw_data=_evaluate(device, parameters, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test))
	best_parameters, metrics = map_ax_client.get_best_parameters()
	return best_parameters, map_ax_client

