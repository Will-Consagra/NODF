import numpy as np 
import torch 

from optimization.functions import neg_log_likelihood, integrated_roughness
from utility.utility import verbosity

from tqdm.autonotebook import tqdm
import time 

## penalized maximum likelihood estimation for deep basis parameters
def PMLE(device, field_model, optim, hyper_params, dataloader, num_epochs, verbose=False):
	"""
	field_model - nn.Module: Defines the coordindate functions of the coefficient field 
	optim - torch.optim: Optimizer 
	hyper_params - dict:
				sigma2_e
				B_tensor
				mu_tensor 
				Phi_tensor
				R_tensor 

	init_params - dict:
				theta
				W 
	dataloader torch.utils.data.DataLoader: Wrapper of data object D = (V, Y) (Tensor or Irregular)
	num_epochs - int: number of iterations 
	verbose - bool: controls verbosity
	"""

	sigma2_e = hyper_params["sigma2_e"]
	Phi_tensor = hyper_params["Phi_tensor"] 
	R_tensor = hyper_params["R_tensor"]
	lambda_c = hyper_params["lambda_c"]

	def neg_pmle(coords, chat, obs_data):
		l2_loss = neg_log_likelihood(chat, obs_data, Phi_tensor)
		prior_energy = integrated_roughness(coords, chat, R_tensor, lambda_c)
		return {"l2_loss":l2_loss, "prior_energy":prior_energy}

	total_steps = 0
	train_losses = []
	with tqdm(total=len(dataloader) * num_epochs) as pbar:
		for epoch in range(num_epochs):
			for step, (model_input, data) in enumerate(dataloader):
				start_time = time.time()

				model_input = {key: value.to(device) for key, value in model_input.items()}
				data = {key: value.to(device) for key, value in data.items()}

				model_output = field_model(model_input["coords"])
				coords = model_output["model_in"]
				chat = model_output["model_out"] 
				losses = neg_pmle(coords, chat, data)
				train_loss = 0.
				for loss_name, loss in losses.items():
					single_loss = loss.mean()

					verbosity(verbose, loss_name + "_weight: %s, total steps: %s"%(single_loss, total_steps))
					train_loss += single_loss

				train_losses.append(train_loss.item())
				verbosity(verbose, "total_train_loss: %s, total steps: %s"%(train_loss, total_steps))

				optim.zero_grad()
				train_loss.backward()
				optim.step()

				pbar.update(1)

				verbosity(verbose, "Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				total_steps += 1

