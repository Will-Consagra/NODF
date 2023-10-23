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
	energy = lambda_c*torch.mean((chat**2) * torch.diag(R_tensor))
	return energy
