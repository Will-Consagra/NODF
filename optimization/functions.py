import torch
from torch.autograd import grad

"""
Likelihoods + priors 
"""

def neg_log_likelihood(chat, data, Phi_tensor):
	"""
	chat: batch_size X N_c X Kprime of coefficient field evaluations 
	data: dict
			"yvals": batch_size X N_sample X M of (noisy + discretized) function samples
			"mask": batch_size X N_sample binary 
	Phi_tensor: torch.tensor (M x K) basis evaluation matrix 
	"""
	yvals = data["yvals"] 
	mask = data["mask"]
	yhat = chat @ Phi_tensor.T ## batch_size X N_c X M (flattened predicted tensor) 
	l2_loss = (mask * (yhat - yvals)**2).mean() 
	return l2_loss

def integrated_roughness(coords, chat, R_tensor, lambda_c):
	"""
	chat: batch_size X N_c X Kprime of coefficient field evaluations
	R_tensor: Kprime x Kprime (diagonal) prior precision matrix of functions
	lambda_c: (float) 
	"""
	energy = (lambda_c/chat.shape[1])*torch.sum((chat**2) * torch.diag(R_tensor)) 
	return energy

def separable_prior(coords, chat, R_tensor, lambda_c):
	"""
	chat: batch_size X N_c X Kprime of coefficient field evaluations
	R_tensor: Kprime x Kprime (diagonal) prior precision matrix of functions
	lambda_c: (float) 
	"""
	#coords = coords.clone().detach().requires_grad_(True) 
	grad_c = torch.ones_like(chat[..., 0]).to(chat.device)
	grad_evals = torch.zeros(tuple(list(chat.shape) + [coords.shape[-1]]), dtype=torch.float32).to(chat.device) ##... N_c \times K \times D 
	lap_evals = torch.zeros(chat.shape, dtype=torch.float32).to(chat.device) 
	for k in range(chat.shape[-1]):
		grad_evals[...,k,:] = grad(chat[...,k], coords, grad_c, create_graph=True)[0]
		for d in range(coords.shape[-1]):
			lap_evals[...,k] += grad(grad_evals[...,k,d], coords, grad_c, create_graph=True)[0][...,d]
	prior_roughness = lambda_c*torch.sum((lap_evals**2) * torch.diag(R_tensor))
	return prior_roughness

def neg_log_prior_W_theta(field_model, R_tensor, lambdas):
	psi = field_model.get_weights_and_biases()
	
	W = psi[0][-1]
	reg_w = lambdas["w"]*torch.trace(W.T @ R_tensor @ W)

	reg_theta_w = lambdas["theta_w"]*sum([(Wl**2).sum() for Wl in psi[0][:-1]])
	reg_theta_b = lambdas["theta_b"]*sum([(Wl**2).sum() for Wl in psi[1]])

	return reg_w, reg_theta_w, reg_theta_b

