import numpy as np 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import copy 

from data_objects.data_objects import resample_data

class FVRF(torch.nn.Module):
	"""
	Function-valued random field.
	"""
	def __init__(self, field_model, coords, Y_tensor, hyper_params):
		super().__init__()

		device = torch.device(coords.device)

		Phi_tensor = hyper_params["Phi_tensor"]
		sigma2_e = hyper_params["sigma2_e"]
		sigma2_w = hyper_params["sigma2_w"]
		R_tensor = hyper_params["R_tensor"] 

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

		self.vec_W_post_mean = post_mean_w
		self.vec_W_post_cov = Lambda_inv
		
		self.K = K
		self.r = r
		self.field_model = field_model
		self.basis = eta
		self.hyper_params = hyper_params

	def compute_predictive_posterior(self, coords):
		"""
		Compute the point-wise predictive posterior mean + covariance 
		"""
		Nv = coords.shape[0]
		device = torch.device(coords.device)
		Xi_evals = self.basis(coords).T.to(device)
		vec_W_post_mean = self.vec_W_post_mean.to(device)
		vec_W_post_cov =  self.vec_W_post_cov.to(device)
		Ik = torch.eye(self.K-1).to(device)

		post_mean_c_lst = []
		post_cov_c_lst = []
		for iv in range(Nv): ##ToDo --> optimize this calculation
			XiV_I = torch.kron(Xi_evals[:,iv:iv+1].T, Ik)
			post_mean_c = torch.squeeze(XiV_I @ vec_W_post_mean)
			post_cov_c = XiV_I @ vec_W_post_cov @ XiV_I.T
			post_mean_c_lst.append(post_mean_c.detach())
			post_cov_c_lst.append(post_cov_c.detach())

		Post_mean_c = torch.stack(post_mean_c_lst, 0)
		Post_cov_c = torch.stack(post_cov_c_lst, 0)

		C_mu_tensor = self.field_model(coords)["model_out"][:,0:1].to(device)

		Post_mean_coefs = torch.column_stack((C_mu_tensor,
												Post_mean_c))
		Post_cov_mats = torch.zeros((Post_mean_coefs.shape[0], self.K, self.K)).to(device)
		Post_cov_mats[:,0,0] = self.hyper_params["sigma2_mu"]*torch.ones(Post_cov_mats.shape[0]) 
		Post_cov_mats[:,1:self.K, 1:self.K] = Post_cov_c

		return Post_mean_coefs, Post_cov_mats
	
	def sample(self, coords, nsamples): 
		"""
		Sample function-valued random field at locations in coords  
		coords - torch.tensor: sample_shape x D-dimensional coordinate location to draw samples 
		"""
		Post_mean_coefs, Post_cov_mats = self.compute_predictive_posterior(coords)
		posterior_field = MultivariateNormal(Post_mean_coefs, Post_cov_mats)
		return posterior_field.sample((nsamples,))

	def log_prob(self):
		"""
		Multivatiate Gaussian likelihood over discretization
		"""
		raise NotImplementedError 

def post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid):
	
	llk = np.zeros(len(var_grid))

	coordmap_calib =  dataloader_calib.dataset.X.to(device)
	data_calib_yvals = dataloader_calib.dataset.Y.to(device)

	Phi_tensor = hyper_params["Phi_tensor"]
	K = Phi_tensor.shape[1]
	
	for i, svar in enumerate(var_grid):
		sigma2_mu, sigma2_w = var_grid[i]
		hyper_params_i = copy.deepcopy(hyper_params)
		hyper_params_i["sigma2_w"] = sigma2_w
		hyper_params_i["sigma2_mu"] = sigma2_mu
		posterior_field = FVRF(field_model, coordmap_calib, data_calib_yvals.T, hyper_params_i)
		Post_mean_coefs, Post_cov_mats = posterior_field.compute_predictive_posterior(coordmap_calib)

		I_N = torch.eye(Phi_tensor.shape[0]).to(device)
		I_N = I_N.reshape((1, Phi_tensor.shape[0], Phi_tensor.shape[0]))
		I_N = I_N.repeat(Post_cov_mats.shape[0], 1, 1)
		Y_pred_mean = Post_mean_coefs @ Phi_tensor.T
		Y_pred_cov = Phi_tensor @ Post_cov_mats @ Phi_tensor.T + hyper_params_i["sigma2_e"]*I_N

		posterior_predictive = MultivariateNormal(Y_pred_mean, Y_pred_cov)
		llk[i] = float(posterior_predictive.log_prob(data_calib_yvals).sum().cpu().detach().numpy())

	sigma2_mu_optim, sigma2_w_optim = var_grid[np.argmax(llk)]
	return sigma2_mu_optim, sigma2_w_optim