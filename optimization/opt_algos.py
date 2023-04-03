import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

from functools import partial 

from optimization.functions import neg_log_likelihood, integrated_roughness, separable_prior

from tqdm.autonotebook import tqdm
import time 

def verbosity(verbose, message):
    if verbose:
        print(message)

##defining optimization routine 
def MAP(device, field_model, optim, hyper_params, dataloader, num_epochs, penalty="angular_roughness", init_params=None, verbose=False):
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
	## 0) define parameters 
	sigma2_e = hyper_params["sigma2_e"]
	Phi_tensor = hyper_params["Phi_tensor"] 
	R_tensor = hyper_params["R_tensor"]
	lambda_c = hyper_params["lambda_c"]

	## 1) build wrapper for  neg log MAP loss
	def neg_MAP_loss(coords, chat, obs_data):
		l2_loss = neg_log_likelihood(chat, obs_data, Phi_tensor)
		if penalty=="angular_roughness":
			prior_energy = integrated_roughness(coords, chat, R_tensor, lambda_c)
		elif penalty=="separable_prior":
			prior_energy = separable_prior(coords, chat, R_tensor, lambda_c)
		else:
			raise NotImplementedError("Penalty %s not available"%penalty)
		return {"l2_loss":l2_loss, "prior_energy":prior_energy}

	## 2) initialize field model  
	### todo: should allow custom inititialization parameters to re-create priors from 2.3.1 and eventually for ensembling/freezing parts of the network.
	###       for now, we use the default initialization scheme 
	if init_params is not None:
		raise NotImplementedError

	## 3) maximize MAP
	total_steps = 0
	train_losses = []
	with tqdm(total=len(dataloader) * num_epochs) as pbar:
		for epoch in range(num_epochs):
			for step, (model_input, data) in enumerate(dataloader):
				start_time = time.time()

				model_input = {key: value.to(device) for key, value in model_input.items()}
				data = {key: value.to(device) for key, value in data.items()}

				model_output = field_model(model_input["coords"])
				coords = model_output["model_in"] ##  batch_size X N_c X D
				chat = model_output["model_out"] 
				losses = neg_MAP_loss(coords, chat, data)
				train_loss = 0.
				for loss_name, loss in losses.items():
					single_loss = loss.mean()

					verbosity(verbose, loss_name + "_weight: %s, total steps: %s"%(single_loss, total_steps))
					#print(loss_name + "_weight", single_loss, total_steps)
					train_loss += single_loss

				train_losses.append(train_loss.item())
				verbosity(verbose, "total_train_loss: %s, total steps: %s"%(train_loss, total_steps))
				#print("total_train_loss", train_loss, total_steps)

				optim.zero_grad()
				train_loss.backward()
				optim.step()

				pbar.update(1)

				verbosity(verbose, "Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
				#print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				total_steps += 1

##defining optimization routine 
def MAPWs(device, field_model, optim, hyper_params, dataloader, num_epochs, init_params=None, verbose=False):
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
	## 0) define parameters 
	sigma2_e = hyper_params["sigma2_e"]
	sigma2_tau = hyper_params["sigma2_tau"]
	sigma2_w = hyper_params["sigma2_w"]
	Phi_tensor = hyper_params["Phi_tensor"] 
	R_tensor = hyper_params["R_tensor"]
	lambda_c = hyper_params["lambda_c"]

	## 1) build wrapper for  neg log MAP loss
	def neg_MAP_loss(chat, obs_data):
		l2_loss = neg_log_likelihood(chat, obs_data, Phi_tensor)
		energy = integrated_roughness(chat, R_tensor, lambda_c)
		return {"l2_loss":l2_loss, "energy":energy}

	## 2) initialize field model  
	### todo: should allow custom inititialization parameters to re-create priors from 2.3.1 and eventually for ensembling/freezing parts of the network.
	###       for now, we use the default initialization scheme 
	if init_params is not None:
		raise NotImplementedError

	## 3) maximize MAP
	total_steps = 0
	train_losses = []
	with tqdm(total=len(dataloader) * num_epochs) as pbar:
		for epoch in range(num_epochs):
			for step, (model_input, data) in enumerate(dataloader):
				start_time = time.time()

				coords = model_input["coords"]
				model_output = field_model(coords)
				losses = neg_MAP_loss(model_output, data)

				train_loss = 0.
				l2_loss = neg_log_likelihood(model_output, data, Phi_tensor) 
				
				
				for loss_name, loss in losses.items():
					single_loss = loss.mean()

					verbosity(verbose, loss_name + "_weight: %s, total steps: %s"%(single_loss, total_steps))
					#print(loss_name + "_weight", single_loss, total_steps)
					train_loss += single_loss

				train_losses.append(train_loss.item())
				verbosity(verbose, "total_train_loss: %s, total steps: %s"%(train_loss, total_steps))
				#print("total_train_loss", train_loss, total_steps)

				optim.zero_grad()
				train_loss.backward()
				optim.step()

				pbar.update(1)

				verbosity(verbose, "Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
				#print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				total_steps += 1

##defining optimization routine 
def MaxBayesEvidence(device, eta_model, optim, hyper_params, dataloader, num_epochs, init_params=None, verbose=False):
	"""
	eta_model - nn.Module: Defines the data-driven basis functons of the coordindate functions of the coefficient field 
	optim - torch.optim: Optimizer 
	hyper_params - dict:
				sigma2_e
				sigma2_w 
				Phi_tensor
				R_tensor 

	init_params - dict:
				theta
				W 
	dataloader torch.utils.data.DataLoader: Wrapper of data object D = (V, Y) (Tensor or Irregular)
	num_epochs - int: number of iterations 
	verbose - bool: controls verbosity
	"""
	## 0) define parameters 
	sigma2_e = hyper_params["sigma2_e"]
	sigma2_w = hyper_params["sigma2_w"]
	Phi_tensor = hyper_params["Phi_tensor"] 
	R_tensor = hyper_params["R_tensor"]
	lambda_c = hyper_params["lambda_c"]
	K = R_tensor.shape[0]

	## 1) build wrapper for  neg log marginal likelihood loss
	def neg_MAP_type2_loss(xihat, obs_data):

		r = xihat.shape[-1]
		Y = obs_data["yvals"]
		yvec = Y.T.reshape(-1,1) ## Y.reshape(-1,1,order="F")
		#yvec = torch.transpose(Y, 1, 2).reshape(-1,1)

		Vinv = (1/sigma2_e) * ((sigma2_w/sigma2_e) * torch.kron(torch.eye(r), R_tensor) + torch.kron(torch.transpose(xihat, 1, 2) @ xihat, Phi_tensor.T @ Phi_tensor))
		V = torch.linalg.inv(Vinv)
		XiPhi = torch.kron(xihat, Phi_tensor) 
		w_post = (1/sigma2_e) * V @ torch.transpose(XiPhi, 1, 2) @ yvec
		Wpost = w_post.reshape(r, K).T
		#log_marg_likelihood= - (r/2)*np.log(sigma2_w) - (n*M/2)*np.log(sigma2_e) - \
		#						(1/(2*sigma2_e))*((yvec - XiPhi@w_post)**2).sum() - (1/(2*sigma2_w))*(w_post**2).sum() -\
		#						0.5*torch.logdet(V) - (n*M/2)*np.log(np.pi)
		#t1 = (1/(2*sigma2_e))*((yvec - XiPhi@w_post)**2).sum() 
		#t2 = (1/(2*sigma2_w))*(w_post**2).sum()
		#t3 = 0.5*torch.logdet(V) 
		#log_marg_likelihood = -t1 -t2 -t3
		log_marg_likelihood = - (1/(2*sigma2_e))*((yvec - XiPhi@w_post)**2).sum() - (1/(2*sigma2_w))*(w_post**2).sum() -0.5*torch.logdet(V) 
		chat = xihat @ Wpost.T
		energy = integrated_roughness(coords, chat, R_tensor, lambda_c)		
		return {"neg_log_bayes_evidence":-log_marg_likelihood, "energy":energy}


	## 2) initialize field model  
	### todo: should allow custom inititialization parameters to re-create priors from 2.3.1 and eventually for ensembling/freezing parts of the network.
	###       for now, we use the default initialization scheme 
	if init_params is not None:
		raise NotImplementedError

	## 3) maximize MAP
	total_steps = 0
	train_losses = []
	with tqdm(total=len(dataloader) * num_epochs) as pbar:
		for epoch in range(num_epochs):
			for step, (model_input, data) in enumerate(dataloader):
				start_time = time.time()

				coords = model_input["coords"]
				model_output = eta_model(coords)
				losses = neg_MAP_type2_loss(model_output, data)
				train_loss = 0.
				for loss_name, loss in losses.items():
					single_loss = loss.mean()

					verbosity(verbose, loss_name + "_weight: %s, total steps: %s"%(single_loss, total_steps))
					#print(loss_name + "_weight", single_loss, total_steps)
					train_loss += single_loss

				train_losses.append(train_loss.item())
				verbosity(verbose, "total_train_loss: %s, total steps: %s"%(train_loss, total_steps))
				#print("total_train_loss", train_loss, total_steps)

				optim.zero_grad()
				train_loss.backward()
				optim.step()

				pbar.update(1)

				verbosity(verbose, "Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
				#print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				total_steps += 1


