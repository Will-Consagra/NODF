import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class FourierFeatures(nn.Module): ##taken from `nueral-function-distributions':https://github.com/EmilienDupont/neural-function-distributions
	"""Random Fourier features.

	Args:
		frequency_matrix (torch.Tensor): Matrix of frequencies to use
			for Fourier features. Shape (num_frequencies, num_coordinates).
			This is referred to as B in the paper.
		learnable_features (bool): If True, fourier features are learnable,
			otherwise they are fixed.
	"""
	def __init__(self, frequency_matrix, learnable_features=False):
		super(FourierFeatures, self).__init__()
		if learnable_features:
			self.frequency_matrix = nn.Parameter(frequency_matrix)
		else:
			# Register buffer adds a key to the state dict of the model. This will
			# track the attribute without registering it as a learnable parameter.
			# We require this so frequency matrix will also be moved to GPU when
			# we call .to(device) on the model
			self.register_buffer('frequency_matrix', frequency_matrix)
		self.learnable_features = learnable_features
		self.num_frequencies = frequency_matrix.shape[0]
		self.coordinate_dim = frequency_matrix.shape[1]
		# Factor of 2 since we consider both a sine and cosine encoding
		self.feature_dim = 2 * self.num_frequencies

	def forward(self, coordinates):
		"""Creates Fourier features from coordinates.

		Args:
			coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
		"""
		# The coordinates variable contains a batch of vectors of dimension
		# coordinate_dim. We want to perform a matrix multiply of each of these
		# vectors with the frequency matrix. I.e. given coordinates of
		# shape (num_points, coordinate_dim) we perform a matrix multiply by
		# the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
		# to obtain an output of shape (num_points, num_frequencies).
		prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
		# Calculate cosine and sine features
		cos_features = torch.cos(2 * math.pi * prefeatures)
		sin_features = torch.sin(2 * math.pi * prefeatures)
		# Concatenate sine and cosine features
		return torch.cat((cos_features, sin_features), dim=1)    

class SineLayer(nn.Module): ##taken from `siren':https://github.com/vsitzmann/siren
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

class RBFLayer(nn.Module): ##taken from `siren':https://github.com/vsitzmann/siren
	'''Transforms incoming data using a given radial basis function.
		- Input: (1, N, in_features) where N is an arbitrary batch size
		- Output: (1, N, out_features) where N is an arbitrary batch size'''

	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
		self.sigmas = nn.Parameter(torch.Tensor(out_features))
		self.reset_parameters()

		self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

	def reset_parameters(self):
		nn.init.uniform_(self.centres, -1, 1)
		nn.init.constant_(self.sigmas, 10)

	def forward(self, input):
		input = input[0, ...]
		size = (input.size(0), self.out_features, self.in_features)
		x = input.unsqueeze(1).expand(size)
		c = self.centres.unsqueeze(0).expand(size)
		distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
		return self.gaussian(distances).unsqueeze(0)

	def gaussian(self, alpha):
		phi = torch.exp(-1 * alpha.pow(2))
		return phi

class RealGaborLayer(nn.Module): ##taken from https://github.com/vishwa91/wire
	"""
		in_features: Input features
		out_features; Output features
		bias: if True, enable bias for the linear operation
		is_first: Legacy SIREN parameter
		omega_0: Legacy SIREN parameter
		omega: Frequency of Gabor sinusoid term
		scale: Scaling of Gabor Gaussian term
	"""
	
	def __init__(self, in_features, out_features, bias=True,
				 is_first=False, omega0=10.0, sigma0=10.0,
				 trainable=False):
		super().__init__()
		self.omega_0 = omega0
		self.scale_0 = sigma0
		self.is_first = is_first
		
		self.in_features = in_features
		
		self.freqs = nn.Linear(in_features, out_features, bias=bias)
		self.scale = nn.Linear(in_features, out_features, bias=bias)
		
	def forward(self, input):
		omega = self.omega_0 * self.freqs(input)
		scale = self.scale(input) * self.scale_0
		
		return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
	"""
			in_features: Input features
			out_features; Output features
			bias: if True, enable bias for the linear operation
			is_first: Legacy SIREN parameter
			omega_0: Legacy SIREN parameter
			omega0: Frequency of Gabor sinusoid term
			sigma0: Scaling of Gabor Gaussian term
			trainable: If True, omega and sigma are trainable parameters
	"""
	def __init__(self, in_features, out_features, bias=True,
				 is_first=False, omega0=10.0, sigma0=40.0,
				 trainable=False):
		super().__init__()
		self.omega_0 = omega0
		self.scale_0 = sigma0
		self.is_first = is_first
		
		self.in_features = in_features
		
		if self.is_first:
			dtype = torch.float
		else:
			dtype = torch.cfloat
			
		# Set trainable parameters if they are to be simultaneously optimized
		self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
		self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
		
		self.linear = nn.Linear(in_features,
								out_features,
								bias=bias,
								dtype=dtype)
		
	def forward(self, input):
		lin = self.linear(input)
		omega = self.omega_0 * lin
		scale = self.scale_0 * lin
		
		return torch.exp(1j*omega - scale.abs().square())

	
####ARCHITECTURES####
class Siren(nn.Module):  ##adapted from `siren':https://github.com/vsitzmann/siren
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
			final_linear = nn.Linear(hidden_features, out_features) 
			
			with torch.no_grad():
				final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
											  np.sqrt(6 / hidden_features) / hidden_omega_0)
				
			self.net.append(final_linear)
		else:
			self.net.append(SineLayer(hidden_features, out_features, 
									  is_first=False, omega_0=hidden_omega_0))
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, coords):
		coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
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


class Wire(nn.Module): ##adapted from https://github.com/vishwa91/wire
	def __init__(self, in_features, hidden_features, hidden_layers, 
				 out_features, wavelet="gabor", omega=10.0, sigma=10.0,
				 trainable=False):
		super().__init__()
		
		# All results in the paper were with the default complex 'gabor' nonlinearity
		if wavelet == "gabor":
			self.nonlin = ComplexGaborLayer
			
			# Since complex numbers are two real numbers, reduce the number of 
			# hidden parameters by 2
			hidden_features = int(hidden_features/np.sqrt(2))
			dtype = torch.cfloat
			self.complex = True
		elif wavelet == "realgabor":
			self.nonlin = RealGaborLayer
			dtype = torch.float
			self.complex = False
			
		# Legacy parameter
		self.pos_encode = False
			
		self.net = []
		self.wavelet = wavelet
		self.net.append(self.nonlin(in_features,
									hidden_features, 
									omega0=omega,
									sigma0=sigma,
									is_first=True,
									trainable=trainable))

		for i in range(hidden_layers):
			self.net.append(self.nonlin(hidden_features,
										hidden_features, 
										omega0=omega,
										sigma0=sigma))

		final_linear = nn.Linear(hidden_features,
								 out_features,
								 dtype=dtype)            
		self.net.append(final_linear)
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, coords):
		coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
		output = self.net(coords)
		if self.wavelet == "gabor":
			output = output.real
		return {"model_in":coords, "model_out":output}  

	def get_basis(self):
		return nn.Sequential(*(list(self.net.children())[:-1]))

####IN-DEVELOPMENT####
class FunctionValuedField(nn.Module):
	"""
	Notes:
		1) We are currently fixing the initial Fourier feature matrix to be fixed 
	"""
	def __init__(self, input_dimension, output_dimension, layer_sizes, basis_expansion, non_linearity="sin"):
		
		super().__init__()
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.layer_sizes = layer_sizes
		self.basis_expansion = basis_expansion
		self.non_linearity = non_linearity
		self.final_non_linearity = final_non_linearity
		self.net = []

		nls_and_inits = {"sine":(Sine(), sine_init),
						 "relu":(nn.ReLU(inplace=True), init_weights_normal),
						 "sigmoid":(nn.Sigmoid(), init_weights_xavier),
						 "tanh":(nn.Tanh(), init_weights_xavier),
						 "selu":(nn.SELU(inplace=True), init_weights_selu),
						 "softplus":(nn.Softplus(), init_weights_normal),
						 "elu":(nn.ELU(inplace=True), init_weights_elu)}

		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

		self.net = []
		#build network 
		## initial basis expansion 
		prev_num_units = self.basis_expansion.feature_dim
		## multi-layer perceptron
		forward_layers = []
		for num_units in self.layer_sizes:
			self.net.append(nn.Sequential(nn.Linear(prev_num_units, num_units), nl))
			prev_num_units = num_units
		## linear last layer
		self.net.append(nn.Sequential(nn.Linear(prev_num_units, output_dimension)))
		self.net = nn.Sequential(*self.net)
		## initalize weights 
		self.net.apply(self.weight_init)
	
	def forward(self, coords):
		coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
		basis_evals = self.basis_expansion(coords)
		output = self.net(basis_evals)
		return {"model_in":coords, "model_out":output}     

	def get_hidden_function(self):
		return nn.Sequential(*list(self.net.children())[:-1])

	def get_basis(self):
		basis_net = [self.basis_expansion] + list(self.net.children())[:-1]
		return nn.Sequential(*basis_net)

	#####write modulations to different portions of the network 
	def forward_with_resamplings(self, coords, params):
		"""
		Eventually this method will grow to handle all `modulations' of the NF parameters:
			((w_0,B), theta, W)
		For now, we just consider the linear last layer 
		"""
		if "linear_last_layer" in params:
			eta = self.get_basis()
			Eta_v = eta(coords["coords"]).T
			W = params["linear_last_layer"]
			return (Eta_v @ W).T

#######INITIALIZATIONS#######
## adapted from `siren':https://github.com/vsitzmann/siren

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
