import torch 
import numpy as np
from ax.service.ax_client import AxClient
import copy 

from data_objects.data_objects import ObservationPoints
from utility.utility import matern_spec_density

def _evaluate_fullyprob(device, parameterization, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test):

	field_model = field_model_() 

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
				init_params=None, 
				verbose=False)
   
	## compute mean 
	Phi_tensor = hyper_params["Phi_tensor"]
	K = R_tensor.shape[0]
	coordmap_train, data_train = dataloader_train.dataset.getfulldata()
	Y_tensor_train = data_train["yvals"].T 

	## update hyper params 
	hyper_params_update = copy.deepcopy(hyper_params)
	hyper_params_update["sigma2_w"] = parameterization["sigma2_w"]
	hyper_params_update["R_tensor"] = R_tensor
	hyper_params_update["Phi_tensor"] = Phi_tensor

	## get 1st + 2nd moments for GP from training locations 
	sigma2_e = hyper_params_update["sigma2_e"]
	sigma2_w = hyper_params_update["sigma2_w"]
	eta = field_model.get_basis()
	Xi_v = eta(coordmap_train["coords"]).T
	r = Xi_v.shape[0]

	Lambda = (1/sigma2_e) * ((sigma2_e/sigma2_w)*torch.kron(torch.eye(r), R_tensor) + torch.kron(Xi_v @ Xi_v.T, Phi_tensor.T @ Phi_tensor))
	Lambda_inv = torch.linalg.inv(Lambda)

	pyz_prod = (Phi_tensor.T @ Y_tensor_train @ Xi_v.T).T.reshape(-1,1) ## K * r x 1 column stacking 
	post_mean_w =  (1/sigma2_e)* Lambda_inv @ pyz_prod

	vec_W_post_mean = post_mean_w
	vec_W_post_cov = Lambda_inv

	## get mode predictions at test locations 
	coordmap_test, data_test = dataloader_test.dataset.getfulldata()
	Y_tensor_test = data_test["yvals"].T
	Xi_test = eta(coordmap_test["coords"]).T
	W_post_mean = torch.from_numpy(vec_W_post_mean.detach().numpy().reshape(K, r, order="F"))
	Ctensor_post_mean_test = (W_post_mean @ Xi_test).T
	return float(((Y_tensor_test - Phi_tensor @ Ctensor_post_mean_test.T)**2).sum().detach().numpy())

def _evaluate(device, parameterization, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test):

	field_model = field_model_() 

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
				init_params=None, 
				verbose=False)
   
	## compute mean 
	Phi_tensor = hyper_params["Phi_tensor"]
	K = R_tensor.shape[0]
	coordmap_train, data_train = dataloader_train.dataset.getfulldata()
	model_output_train = field_model(coordmap_train["coords"])
	C_mu_tensor_train = model_output_train["model_out"][:,0:1]
	Y_centered_tensor_train = data_train["yvals"].T - Phi_tensor[:,0:1] @ C_mu_tensor_train.T

	## update hyper params 
	hyper_params_update = copy.deepcopy(hyper_params)
	hyper_params_update["sigma2_w"] = parameterization["sigma2_w"]
	hyper_params_update["R_tensor"] = R_tensor[1:K,1:K]
	hyper_params_update["Phi_tensor"] = Phi_tensor[:,1:K]

	## get 1st + 2nd moments for GP from training locations 
	sigma2_e = hyper_params_update["sigma2_e"]
	sigma2_w = hyper_params_update["sigma2_w"]
	eta = field_model.get_basis()
	Xi_v = eta(coordmap_train["coords"]).T
	r = Xi_v.shape[0]

	Lambda = (1/sigma2_e) * ((sigma2_e/sigma2_w)*torch.kron(torch.eye(r), R_tensor[1:K,1:K]) + torch.kron(Xi_v @ Xi_v.T, Phi_tensor[:,1:K].T @ Phi_tensor[:,1:K]))
	Lambda_inv = torch.linalg.inv(Lambda)

	pyz_prod = (Phi_tensor[:,1:K].T @ Y_centered_tensor_train @ Xi_v.T).T.reshape(-1,1) ## K * r x 1 column stacking 
	post_mean_w =  (1/sigma2_e)* Lambda_inv @ pyz_prod

	vec_W_post_mean = post_mean_w
	vec_W_post_cov = Lambda_inv

	## get mode predictions at test locations 
	coordmap_test, data_test = dataloader_test.dataset.getfulldata()
	model_output_test = field_model(coordmap_test["coords"])
	C_mu_tensor_test = model_output_test["model_out"][:,0:1]
	Y_tensor_test = data_test["yvals"].T
	#Y_tensor_test_centered = data_test["yvals"].T - Phi_tensor[:,0:1] @ C_mu_tensor_test.T
	Xi_test = eta(coordmap_test["coords"]).T
	W_post_mean = torch.from_numpy(vec_W_post_mean.detach().numpy().reshape(K-1, r, order="F")) ##inefficient!! change eventually 
	Ctensor_post_mean_test = torch.column_stack((C_mu_tensor_test, 
											(W_post_mean @ Xi_test).T))

	return float(((Y_tensor_test - Phi_tensor @ Ctensor_post_mean_test.T)**2).sum().detach().numpy())

	## augment with a coverage penalty 
	#Nsamples = 200; alpha = 0.05; slack = 0.05 ## this can be extened to multiple quantiles to ensure coverage everywhere
	#post_vec_W_samples = np.random.multivariate_normal(vec_W_post_mean.detach().numpy().ravel(), 
	#													vec_W_post_cov.detach().numpy(), 
	#													   size=Nsamples)  	
	#post_samples = torch.zeros((Nsamples, Xi_test.shape[1], K))
	#for i in range(Nsamples):
	#	Wi = torch.from_numpy(post_vec_W_samples[i,:].reshape(K-1, r, order="F")).float()
	#	post_samples[i,:,:] = torch.column_stack((C_mu_tensor_test, 
	#						(Wi @ Xi_test).T))
	#Ytensor_post_predictive_test = post_samples @ Phi_tensor.T + torch.normal(0, np.sqrt(sigma2_e), size=(Nsamples, Xi_test.shape[1], Phi_tensor.shape[0]))##Nsamples x nverts x M
	
	#lp = torch.quantile(Ytensor_post_predictive_test, alpha/2, axis=0).T
	#up = torch.quantile(Ytensor_post_predictive_test, 1-(alpha/2), axis=0).T
	#covered = ((Y_tensor_test >= lp) & (Y_tensor_test <= up)).detach().numpy().astype(int)
	#ecp = np.mean(covered)
	#print(ecp)
	#if ecp >= (1-alpha - slack):
	#	return float(((Y_tensor_test - Phi_tensor @ Ctensor_post_mean_test.T)**2).sum().detach().numpy())
	#else:
	#	return float(((Y_tensor_test - Phi_tensor @ Ctensor_post_mean_test.T)**2).sum().detach().numpy())*(5/ecp)

def _evaluate_direct(device, parameterization, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test):
	field_model = field_model_() 

	if "learning_rate" in parameterization:
		optim = optimizer_(params=field_model.parameters(), lr=parameterization["learning_rate"])
	else:
		optim = optimizer_(params=field_model.parameters())

	hyper_params["lambda_c"] = parameterization["lambda_c"]
	Phi_tensor = hyper_params["Phi_tensor"]

	## estimate parameters 
	algorithm(device,
				field_model,
				optim,
				hyper_params,
				dataloader_train, 
				num_epochs, 
				init_params=None, 
				verbose=False)

	## get mode predictions at test locations 
	coordmap_test, data_test = dataloader_test.dataset.getfulldata()
	Y_tensor_test = data_test["yvals"].T
	C_hat_test = field_model(coordmap_test["coords"])
	return float(((Y_tensor_test - Phi_tensor @ C_hat_test.T)**2).sum().detach().numpy())

def _resample_data(Obs, strata=False, train_prop=0.8):
	if strata:
		raise NotImplementedError("Stratified sampling is not yet supported")
	else:
		dataset_size = Obs.N
		indices = list(range(dataset_size))
		split = int(np.floor((1-train_prop) * dataset_size))
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]

		N_train = len(train_indices)
		N_test = len(test_indices)
		Y_train = Obs.Y[train_indices,:]
		Y_test = Obs.Y[test_indices,:]
		V_train = Obs.X[train_indices, :]
		V_test = Obs.X[test_indices, :]
		mask_array_train = Obs.mask[train_indices, :]
		mask_array_test = Obs.mask[test_indices, :]

		O_train = ObservationPoints(V_train, Y_train, N_train, mask=mask_array_train)
		O_test = ObservationPoints(V_test, Y_test, N_test, mask=mask_array_test)

		dataloader_train = torch.utils.data.DataLoader(O_train, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
		dataloader_test = torch.utils.data.DataLoader(O_test, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

	return dataloader_train, dataloader_test

def BO_optimization(device, Obs, field_model_, optimizer_, algorithm, hyper_params, parameter_map, num_epochs,
					Nexperiments, experiment_name="hyper_param_opt_experiment", train_prop=0.8, mean_prob=False):
	"""
	Implements BO-based hyper-parameter optimization.
	Arguments:
		-parameter_map: list of dicts defining free parameters and ranges
	"""
	map_ax_client = AxClient()
	map_ax_client.create_experiment(name=experiment_name,
									parameters=parameter_map,
									minimize=True, 
									)

	for _ in range(Nexperiments):
		parameters, trial_index = map_ax_client.get_next_trial()
		dataloader_train, dataloader_test = _resample_data(Obs, strata=False, train_prop=train_prop)
		if mean_prob:
			map_ax_client.complete_trial(trial_index=trial_index, 
						raw_data=_evaluate_fullyprob(device, parameters, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test))
		else:
			map_ax_client.complete_trial(trial_index=trial_index, 
						raw_data=_evaluate(device, parameters, hyper_params, field_model_, optimizer_, algorithm, num_epochs, dataloader_train, dataloader_test))
	best_parameters, metrics = map_ax_client.get_best_parameters()
	return best_parameters, map_ax_client

