import numpy as np 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import copy 
import sys 
sys.path.append("../")
from optimization.opt_algos import MAP
from optimization.hyper_optim import _resample_data
from utility.utility import get_posterior_utility_centered, log_marginal_likelihood_w
from geom.utils import Wbarycenter as Wbarycenter_python

def build_ensemble_nodf(device, untrained_ensemble_lst, dataloader, hyper_params, num_epochs, K, r, detach_=False, calib_prop=0.1):
	T = len(untrained_ensemble_lst)
	Ensemble = []
	for t in range(T):
		torch.manual_seed(t)
		field_model_t, optim_t = untrained_ensemble_lst[t]
		dataloader_train, dataloader_calib = _resample_data(dataloader.dataset, strata=False, train_prop=1-calib_prop)
		MAP(device, field_model_t, optim_t, hyper_params, dataloader_train, num_epochs, verbose=False)
		## compute posterior object
		posterior_field_t = get_posterior_utility_centered(device, field_model_t, dataloader_train, hyper_params)
		## compute marginal likelihood 
		result_t = log_marginal_likelihood_w(device, field_model_t, dataloader_train, hyper_params)
		## get sigma2_mu 
		hyper_params["sigma2_mu"] = mean_calibration(device, field_model_t, dataloader_calib, hyper_params, svars=None)
		if detach_:
			trained_params_t = field_model_t.net.parameters()
			detached_params_t = [param.detach() for param in trained_params_t]
			field_model_t.net.load_state_dict({name: param for name, param in zip(field_model_t.net.state_dict().keys(), detached_params_t)})
	Ensemble.append((posterior_field_t, field_model_t, result_t["lml"], hyper_params))
	return Ensemble

def ensemble_nodf_predict(device, Ensemble, coords_predict, K, EUCAPPROX=False):
	T = len(Ensemble)
	## 1) get ensemble predictive posteriors 
	ensemble_predictive_posteriors = []
	for t in range(T):
		posterior_field_t, field_model_t, lml_t, hyper_params_t = Ensemble[t]
		Mus_nodf_t = field_model_t(coords_predict.to(device))["model_out"]
		Post_mean_c_t, Post_cov_c_t = posterior_field_t.compute_predictive_posterior(device, coords_predict.to(device))
		Post_mean_c_t_tensor = Post_mean_c_t.cpu().detach().numpy()
		Post_cov_c_t_tensor = Post_cov_c_t.cpu().detach().numpy()
		ensemble_predictive_posteriors.append((Post_mean_c_t_tensor, Post_cov_c_t_tensor, Mus_nodf_t, hyper_params_t))

	## 2) combine predictive posteriors
	N = coords_predict.shape[0]
	lmls = np.array([Ensemble[t][-2].cpu().detach().numpy() for t in range(T)])
	weights = lmls/np.sum(lmls)
	Post_mean_c_tensor_WC = np.zeros((N, K))
	Post_cov_c_tensor_WC = np.zeros((N, K, K))

	if T == 1:
		Mus_nodf = np.zeros((N, 1))
		Post_mean_c_tensor_, Post_cov_c_tensor_, Mus_nodf_, hyper_params_ = ensemble_predictive_posteriors[0]
		Post_mean_c_tensor_WC[:,0] = Mus_nodf_[:,0].cpu().detach().numpy()
		Post_mean_c_tensor_WC[:,1:K] = Post_mean_c_tensor_
		Post_cov_c_tensor_WC[:,1:K,1:K] = Post_cov_c_tensor_
		Post_cov_c_tensor_WC[:,0,0] = hyper_params_["sigma2_mu"] ## small variance to 0'th order harmonic to stabalize estimation

	else:
		if EUCAPPROX:
			Mus_nodf = np.zeros((N, T))
			for t in range(T):
				Post_mean_c_tensor_t, Post_cov_c_tensor_t, Mus_nodf_t, hyper_params_t = ensemble_predictive_posteriors[t]
				Post_mean_c_tensor_WC[:,0] = Post_mean_c_tensor_WC[:,0] + weights[t] * Mus_nodf_t[:,0].cpu().detach().numpy()
				Post_mean_c_tensor_WC[:,1:K] = Post_mean_c_tensor_WC[:,1:K] + weights[t] * Post_mean_c_tensor_t
				Post_cov_c_tensor_WC[:,1:K,1:K] = Post_cov_c_tensor_WC[:,1:K,1:K] + weights[t] * Post_cov_c_tensor_t
				Post_cov_c_tensor_WC[:,0,0] = Post_cov_c_tensor_WC[:,0,0] + weights[t] * hyper_params_t["sigma2_mu"]
				#Mus_nodf[:, t] = Mus_nodf_t[:,0].cpu().detach().numpy()
			#Post_cov_c_tensor_WC[:,0,0] = np.var(Mus_nodf, axis=1)

		else:
			for vi in range(N):
				gps_vi = []; Mu_0_vi = []
				Mus_nodf = torch.zeros((T, K)) 
				for t in range(T):
					Post_mean_c_tensor_t, Post_cov_c_tensor_t, Mus_nodf_t, hyper_params_t = ensemble_predictive_posteriors[t]
					gps_vi.append([Post_mean_c_tensor_t[vi,:], 
									Post_cov_c_tensor_t[vi,:,:]])
					Mu_0_vi.append(Mus_nodf_t[vi,0].cpu().detach().numpy())
					Mus_nodf[t, :] = Mus_nodf_t[vi,:]

				mu_wbc, cov_wbc, FLAG, c_err = Wbarycenter_python(gps_vi, weights)
				if FLAG:
					raise Exception("Warning, Barycenter did not converge for vi=%s, Fixed point-iteration error: %s"%(vi,c_err))
				Mu_wbc = np.zeros(K)
				Mu_wbc[0] = np.mean(Mu_0_vi)
				Mu_wbc[1:] = mu_wbc.ravel()
				Cov_wbc =  np.zeros((K,K))
				Cov_wbc[1:K, 1:K] = cov_wbc
				Cov_wbc[0,0] = np.var(Mu_0_vi)

				Post_mean_c_tensor_WC[vi,:] = Mu_wbc
				Post_cov_c_tensor_WC[vi,:,:] = Cov_wbc

	return Post_mean_c_tensor_WC, Post_cov_c_tensor_WC

def post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid=None):
	if var_grid is None:
		sigma2_mus = (1e-3, 5e-3, 1e-2, 5e-2) 
		sigma2_ws = (1e-1, 3e-1, 5e-1, 7e-1)
		var_grid = [(sm, sw) for sm in sigma2_mus for sw in sigma2_ws]

	llk = np.zeros(len(var_grid))
	coordmap_calib, data_calib = dataloader_calib.dataset.getfulldata()
	Phi_tensor = hyper_params["Phi_tensor"]
	K = Phi_tensor.shape[1]
	model_output_calib = field_model(coordmap_calib["coords"].to(device))
	C_mu_tensor_calib = model_output_calib["model_out"][:,0:1]

	for i, svar in enumerate(var_grid):
		sigma2_mu, sigma2_w = var_grid[i]
		hyper_params_i = copy.deepcopy(hyper_params)
		hyper_params_i["sigma2_w"] = sigma2_w
		posterior_field = get_posterior_utility_centered(device, field_model, dataloader_calib, hyper_params_i)
		Post_mean_c, Post_cov_c = posterior_field.compute_predictive_posterior(device, coordmap_calib["coords"].to(device))
		Post_mean_coefs = torch.column_stack((C_mu_tensor_calib,
												Post_mean_c))

		Post_cov_mats = torch.zeros((Post_mean_coefs.shape[0], K, K))
		Post_cov_mats[:,0,0] = sigma2_mu*torch.ones(Post_cov_mats.shape[0]) 
		Post_cov_mats[:,1:K, 1:K] = Post_cov_c
		Post_cov_mats = Post_cov_mats.to(device)
		I_N = torch.eye(Phi_tensor.shape[0]).to(device)
		I_N = I_N.reshape((1, Phi_tensor.shape[0], Phi_tensor.shape[0]))
		I_N = I_N.repeat(Post_cov_mats.shape[0], 1, 1)
		Y_pred_mean = Post_mean_coefs @ Phi_tensor.T
		Y_pred_cov = Phi_tensor @ Post_cov_mats @ Phi_tensor.T + hyper_params_i["sigma2_e"]*I_N
		mvn = MultivariateNormal(Y_pred_mean, Y_pred_cov)
		llk[i] = float(mvn.log_prob(data_calib["yvals"].to(device)).sum().cpu().detach().numpy())

	sigma2_mu_optim, sigma2_w_optim = var_grid[np.argmax(llk)]
	return sigma2_mu_optim, sigma2_w_optim

def infer_field(device, field_model, optim,  dataloader, hyper_params, num_epochs, var_grid=None, detach_=False, calib_prop=0.1, batch_frac=1):
	## split to train + calibration
	dataloader_train, dataloader_calib = _resample_data(dataloader.dataset, strata=False, train_prop=1-calib_prop, batch_frac=batch_frac)
	## learn deep-basis 
	MAP(device, field_model, optim, hyper_params, dataloader_train, num_epochs, verbose=False)
	## get variance parameters via calibration 
	sigma2_mu_optim, sigma2_w_optim = post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid=var_grid)
	hyper_params["sigma2_mu"] = sigma2_mu_optim
	hyper_params["sigma2_w"] = sigma2_w_optim
	## compute posterior object
	posterior_field = get_posterior_utility_centered(device, field_model, dataloader_train, hyper_params)
	if detach_:
		trained_params = field_model.net.parameters()
		detached_params = [param.detach() for param in trained_params]
		field_model.net.load_state_dict({name: param for name, param in zip(field_model.net.state_dict().keys(), detached_params)})
	return posterior_field, field_model, hyper_params
