import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

#### SIREN NETWORK (Sitzmann et. al. 2020, NeurIPS) ####
class SineLayer(nn.Module): 
	def __init__(self, in_features, out_features, bias=True,
				 is_first=False, omega_0=30):
		super().__init__()
		self.omega_0 = omega_0
		self.is_first = is_first
		
		self.in_features = in_features
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		
		self.init_weights()
	
	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_features, 
											 1 / self.in_features)      
			else:
				self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
											 np.sqrt(6 / self.in_features) / self.omega_0)
		
	def forward(self, input):
		return torch.sin(self.omega_0 * self.linear(input))
	
	def forward_with_intermediate(self, input): 
		# For visualization of activation distributions
		intermediate = self.omega_0 * self.linear(input)
		return torch.sin(intermediate), intermediate

class Siren(nn.Module):  
	def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
				 first_omega_0=30, hidden_omega_0=30.):
		super().__init__()
		
		self.net = []
		self.domain_dim = in_features
		self.range_dim = out_features
		self.net.append(SineLayer(in_features, hidden_features, 
								  is_first=True, omega_0=first_omega_0))

		for i in range(hidden_layers):
			self.net.append(SineLayer(hidden_features, hidden_features, 
									  is_first=False, omega_0=hidden_omega_0))

		if outermost_linear:
			final_linear = nn.Linear(hidden_features, out_features, bias=False) 
			
			with torch.no_grad():
				final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
											  np.sqrt(6 / hidden_features) / hidden_omega_0)
				
			self.net.append(final_linear)
		else:
			self.net.append(SineLayer(hidden_features, out_features, 
									  is_first=False, omega_0=hidden_omega_0))
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, coords):
		output = self.net(coords)
		return {"model_in":coords, "model_out":output}     

	def forward_with_activations(self, coords, retain_grad=False):
		'''Returns not only model output, but also intermediate activations.
		Only used for visualizing activations later!'''
		activations = OrderedDict()

		activation_count = 0
		x = coords.clone().detach().requires_grad_(True)
		activations['input'] = x
		for i, layer in enumerate(self.net):
			if isinstance(layer, SineLayer):
				x, intermed = layer.forward_with_intermediate(x)
				
				if retain_grad:
					x.retain_grad()
					intermed.retain_grad()
					
				activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
				activation_count += 1
			else: 
				x = layer(x)
				
				if retain_grad:
					x.retain_grad()
					
			activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
			activation_count += 1

		return {"model_in":coords, "model_out": activations.popitem(), "activations": activations}

	def get_basis(self):
		return nn.Sequential(*(list(self.net.children())[:-1]))

def init_weights_normal(m):
	if type(m) == nn.Linear:
		if hasattr(m, "weight"):
			nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def init_weights_selu(m):
	if type(m) == nn.Linear:
		if hasattr(m, "weight"):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	if type(m) == nn.Linear:
		if hasattr(m, "weight"):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	if type(m) == nn.Linear:
		if hasattr(m, "weight"):
			nn.init.xavier_normal_(m.weight)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, "weight"):
			num_input = m.weight.size(-1)
			# See supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
	with torch.no_grad():
		if hasattr(m, "weight"):
			num_input = m.weight.size(-1)
			m.weight.uniform_(-1 / num_input, 1 / num_input)
