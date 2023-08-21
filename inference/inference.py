import numpy as np 
import torch
from torch.distributions.multivariate_normal import Distribution, MultivariateNormal

import copy 

from data_objects.data_objects import resample_data

class FieldDistribution(Distribution):
	def __init__(self, basis_model, vec_W_post_mean, vec_W_post_cov, K, r):
		super().__init__()
		
		self.K = K
		self.r = r
		self.basis = basis_model
		self.vec_W_post_mean = vec_W_post_mean
		self.vec_W_post_cov = vec_W_post_cov

	def compute_predictive_posterior(self, coords):
		"""
		Compute the point-wise predictive posterior mean + covariance 
		"""
		Nv = coords.shape[0]
		device = torch.device(coords.device)
		Xi_evals = self.basis(coords).T.to(device)
		vec_W_post_mean = self.vec_W_post_mean.to(device)
		vec_W_post_cov =  self.vec_W_post_cov.to(device)
		Ik = torch.eye(self.K).to(device)

		post_mean_c_lst = []
		post_cov_c_lst = []
		for iv in range(Nv): 
			XiV_I = torch.kron(Xi_evals[:,iv:iv+1].T, Ik)
			post_mean_c = torch.squeeze(XiV_I @ vec_W_post_mean)
			post_cov_c = XiV_I @ vec_W_post_cov @ XiV_I.T
			post_mean_c_lst.append(post_mean_c.detach())
			post_cov_c_lst.append(post_cov_c.detach())

		Post_mean_c_tensor = torch.stack(post_mean_c_lst, 0)
		Post_cov_c_tensor = torch.stack(post_cov_c_lst, 0)
		return Post_mean_c_tensor, Post_cov_c_tensor
	
	def rsample(self, sample_shape=torch.Size(), coords=None): 
		"""
		Sample function-valued field at locations in coords  
		coords - torch.tensor: sample_shape x D-dimensional coordinate location to draw samples 
		"""
		shape = self._extended_shape(sample_shape)
		nsamples = shape.numel()
		Xi_v = self.basis(coords).T
		post_vec_W_samples = np.random.multivariate_normal(self.vec_W_post_mean.detach().numpy().ravel(), 
														   self.vec_W_post_cov.detach().numpy(), 
														   size=nsamples)  
		fsamples = torch.zeros((nsamples, Xi_v.shape[1], self.K))
		for i in range(nsamples):
			Wi = torch.from_numpy(post_vec_W_samples[i,:].reshape(self.K, self.r, order="F")).float()
			fsamples[i,:,:] = (Wi @ Xi_v).T
		return fsamples

	def log_prob(self):
		"""
		Multivatiate Gaussian likelihood over discretization
		"""
		raise NotImplementedError 

def get_gp_posterior_utility(device, field_model, dataloader, hyper_params):
	## train a NF posterior for the given partition 
	
	Phi_tensor = hyper_params["Phi_tensor"]
	sigma2_e = hyper_params["sigma2_e"]
	sigma2_w = hyper_params["sigma2_w"]
	R_tensor = hyper_params["R_tensor"] 

	coords = dataloader.dataset.X.to(device)
	Y_tensor = dataloader.dataset.Y.T.to(device)
	model_output = field_model(coords)
	C_mu_tensor = model_output["model_out"][:,0:1].to(device)
	
	Y_centered_tensor = Y_tensor - Phi_tensor[:,0:1] @ C_mu_tensor.T

	eta = field_model.get_basis()
	Xi_v = eta(coords).T
	
	K = Phi_tensor.shape[1]
	r = Xi_v.shape[0]

	Lambda = (1/sigma2_e) * ((sigma2_e/sigma2_w)*torch.kron(torch.eye(r).to(device), R_tensor[1:,1:]) + torch.kron(Xi_v @ Xi_v.T, Phi_tensor[:,1:].T @ Phi_tensor[:,1:K]))
	Lambda_inv = torch.linalg.inv(Lambda)

	pyz_prod = (Phi_tensor[:,1:K].T @ Y_centered_tensor @ Xi_v.T).T.reshape(-1,1) 
	post_mean_w =  (1/sigma2_e)* Lambda_inv @ pyz_prod

	vec_W_post_mean = post_mean_w
	vec_W_post_cov = Lambda_inv
	
	posterior_field = FieldDistribution(eta, vec_W_post_mean, vec_W_post_cov, K-1, r)
	return posterior_field

def post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid):
	llk = np.zeros(len(var_grid))

	coordmap_calib =  dataloader_calib.dataset.X.to(device)
	data_calib_yvals = dataloader_calib.dataset.Y.to(device)

	Phi_tensor = hyper_params["Phi_tensor"]
	K = Phi_tensor.shape[1]
	model_output_calib = field_model(coordmap_calib)
	C_mu_tensor_calib = model_output_calib["model_out"][:,0:1]

	for i, svar in enumerate(var_grid):
		sigma2_mu, sigma2_w = var_grid[i]
		hyper_params_i = copy.deepcopy(hyper_params)
		hyper_params_i["sigma2_w"] = sigma2_w
		posterior_field = get_gp_posterior_utility(device, field_model, dataloader_calib, hyper_params_i)
		Post_mean_c, Post_cov_c = posterior_field.compute_predictive_posterior(coordmap_calib)
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
		llk[i] = float(mvn.log_prob(data_calib_yvals).sum().cpu().detach().numpy())

	sigma2_mu_optim, sigma2_w_optim = var_grid[np.argmax(llk)]
	return sigma2_mu_optim, sigma2_w_optim