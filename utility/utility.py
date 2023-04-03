import numpy as np 
import torch
from torch.distributions.multivariate_normal import Distribution, MultivariateNormal

from scipy.special import legendre, gamma
from dipy.reconst.shm import real_sym_sh_basis, sph_harm_ind_list, sf_to_sh
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.segment.mask import applymask

def measurement_error_var_estimator(data_b0, mask=None):
	if mask is not None:
		data_b0 = applymask(data_b0, mask)
	data_b0 = np.divide(data_b0, np.mean(data_b0, axis = 3)[:,:,:,np.newaxis])
	sigma2_v = np.nanstd(np.where(data_b0 != 0, data_b0, np.nan), axis = 3)**2
	sigma2_hat = np.nanmean(sigma2_v)
	return sigma2_hat

def ESR_design(n_pts, bv=2000, uc=1):
	theta_samps = np.pi * np.random.rand(n_pts)
	phi_samps = 2 * np.pi * np.random.rand(n_pts)
	hsph_initial = HemiSphere(theta=theta_samps, phi=phi_samps)
	hsph_updated, potential = disperse_charges(hsph_initial, 5000)
	vertices = hsph_updated.vertices
	values = np.ones(vertices.shape[0])
	bvecs = vertices
	bvals = (bv * values)/1000*uc
	bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
	bvals = np.insert(bvals, (0, bvals.shape[0]), 0)
	return bvecs, bvals 
	
def get_odf_transformation(n):
	T = np.zeros((len(n), len(n)))
	for i in range(T.shape[0]):
		P_n = legendre(n[i])
		T[i, i] = P_n(0)
	return T

def get_signal_transformation(n):
	Tinv = np.zeros((len(n), len(n)))
	for i in range(Tinv.shape[0]):
		P_n = legendre(n[i])
		Tinv[i, i] = 1./P_n(0)
	return Tinv

def cart2sphere(x):
	## Note: theta, phi convention here is flipped compared to dipy.core.geometry.cart2sphere
	r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
	theta = np.arctan2(x[:,1], x[:,0])
	phi = np.arccos(x[:,2]/r)
	return np.column_stack([theta, phi])

def sphere2cart(x):
	theta = x[:,0]
	phi = x[:,1]
	xx = np.sin(phi)*np.cos(theta)
	yy = np.sin(phi)*np.sin(theta)
	zz = np.cos(phi)
	return np.column_stack([xx, yy, zz]) 

def S2hemisphere(x):
	x_copy = np.copy(x)
	x_polar = cart2sphere(x_copy)
	ix = np.argwhere(x_polar[:,1] > np.pi/2).ravel()
	x_copy[ix, :] = -1*x_copy[ix, :] 
	return x_copy

def verbosity(verbose, message):
	if verbose:
		print(message)

def SHLS_residual_bootstrap(Y_noisy, X, Phi, N_boot, sh_order=8, gamma=0.006, rescale=False):
	"""
	Residual bootstrap for UQ of Laplacian-regularized spherical harmonic regression 
	"""
	D = len(Y_noisy.shape)-1
	M = Y_noisy.shape[-1]
	##1) obtain pilot estimator 
	C_hat = sf_to_sh(Y_noisy, Sphere(xyz=X), sh_order=8, basis_type="descoteaux07", smooth=gamma)
	##2) estimate residuals
	Epsilon_hat = Y_noisy - C_hat @ Phi.T
	##3) rescale MB residuals 
	if rescale:
		X_spherical = cart2sphere(X)
		theta = X_spherical[:,0]; phi = X_spherical[:,1]
		B, m, n = real_sym_sh_basis(sh_order, phi, theta)
		R = np.diag(np.power(n, 2)*np.power(n+1, 2))
		R[0,0] = 1e-3
		Hmat = B@np.linalg.inv((B.T @ B + gamma*R))@B.T
		resid_scales = 1/np.sqrt(np.diag(Hmat @ Hmat.T))
		Epsilon_hat = np.multiply(Epsilon_hat, resid_scales)
	##4/5) bootstrap datasets and estimate bootstrapped ODFs 
	Ctensor_boot = np.zeros(tuple([N_boot+1] + list(C_hat.shape)))
	for nb in range(N_boot):
		Epsilon_b = np.apply_along_axis(
									lambda epsilon_i: np.random.choice(epsilon_i, size=M, replace=True), -1, ##apply along spherical dimension (python index 0-base)
										Epsilon_hat
										)
		Y_b = C_hat @ Phi.T + Epsilon_b
		C_hat_b = sf_to_sh(Y_b, Sphere(xyz=X), sh_order=8, basis_type="descoteaux07", smooth=gamma)
		Ctensor_boot[nb,...] = C_hat_b
	Ctensor_boot[-1,...] = C_hat
	return Ctensor_boot

def posterior_credible_intervals(f, alpha):
	"""
	Computes the pointwise 1-alpha posterior credible interval from posterior function evaluation matrix f
	arguments:
		f: M x R numpy matrix, evaluation of R (approximate) posterior functions at each of M points in S2
		alpha: float, specify width of intervals  
	"""
	lp = np.quantile(f, alpha/2, axis=1)
	up = np.quantile(f, 1-(alpha/2), axis=1)
	return lp, up

def matern_spec_density(omega, rho, nu):
	"""
	Spectral density for a Matern covariance function. Form can be found in Dutordoir et. al 2020 supplement.
	arguments:
		omega: frequency 
		rho: lengthscale 
		nu: differentiability
	"""
	term1 = ((2**3) * (np.pi**(3/2)) * gamma(nu + (3/2)) * np.power(2*nu, nu))/(gamma(nu) * np.power(rho, 2*nu))
	term2 = np.power(((2*nu)/np.power(rho, 2)) + (4*(np.pi**2)*np.power(omega,2)), -(nu + (3/2)))
	return term1 * term2

def GCV_selector(X, Y, sh_order, lambdas, mask=None):
	"""
		X: spherical design 
		Y: length N list of signals (sampled from volume)
		lambdas: list of candidate smoothing values
	"""

	X_spherical = cart2sphere(X)
	theta_ = X_spherical[:,0]; phi_ = X_spherical[:,1]
	B, m, n = real_sym_sh_basis(sh_order, phi_, theta_)
	R = np.diag(np.power(n, 2)*np.power(n+1, 2))
	M = X_spherical.shape[0]
	gcvs = np.zeros(len(lambdas))

	for i, lam in enumerate(lambdas):
		H = B @ np.linalg.inv(B.T @ B + lam*R) @ B.T 
		residual_space_trace = np.sum(np.diag(np.identity(M) - H))

		SSE = np.linalg.norm(Y @ (np.identity(M) - H), axis=-1, ord=2)
		GCV_tensor = SSE/(residual_space_trace/M)

		if mask is not None:
			GCV_tensor = applymask(GCV_tensor, mask)

		gcvs[i] = np.nanmean(GCV_tensor)

	return lambdas[np.argmin(gcvs)]

def log_det(M):
    evals, evecs = np.linalg.eig(M)
    return np.sum(np.log(evals))

def get_posterior_utility(field_model, dataloader, hyper_params, K, r):
    ## train a NF posterior for the given partition 
    
    Phi_tensor = hyper_params["Phi_tensor"]
    coordmap, data = dataloader.dataset.getfulldata()
    model_output = field_model(coordmap["coords"])
    C_mu_tensor = model_output["model_out"][:,0:1]

    Y_centered_tensor = data["yvals"].T - Phi_tensor[:,0:1] @ C_mu_tensor.T

    eta = field_model.get_basis()
    Xi_v = eta(coordmap["coords"]).T

    sigma2_e = hyper_params["sigma2_e"]
    sigma2_w = hyper_params["sigma2_w"]
    R_tensor = hyper_params["R_tensor"] 

    Lambda = (1/sigma2_e) * ((sigma2_e/sigma2_w)*torch.kron(torch.eye(r), R_tensor[1:K,1:K]) + torch.kron(Xi_v @ Xi_v.T, Phi_tensor[:,1:K].T @ Phi_tensor[:,1:K]))
    Lambda_inv = torch.linalg.inv(Lambda)

    pyz_prod = (Phi_tensor[:,1:K].T @ Y_centered_tensor @ Xi_v.T).T.reshape(-1,1) ## K * r x 1 column stacking 
    post_mean_w =  (1/sigma2_e)* Lambda_inv @ pyz_prod

    vec_W_post_mean = post_mean_w
    vec_W_post_cov = Lambda_inv
    
    posterior_field = FieldDistribution(eta, vec_W_post_mean, vec_W_post_cov, K, r)
    return posterior_field

def get_posterior_utility_fullyprob(field_model, dataloader, hyper_params, K, r):
    ## train a NF posterior for the given partition 
    
    Phi_tensor = hyper_params["Phi_tensor"]
    coordmap, data = dataloader.dataset.getfulldata()
    Y_tensor = data["yvals"].T

    eta = field_model.get_basis()
    Xi_v = eta(coordmap["coords"]).T

    sigma2_e = hyper_params["sigma2_e"]
    sigma2_w = hyper_params["sigma2_w"]
    R_tensor = hyper_params["R_tensor"] 

    Lambda = (1/sigma2_e) * ((sigma2_e/sigma2_w)*torch.kron(torch.eye(r), R_tensor) + torch.kron(Xi_v @ Xi_v.T, Phi_tensor.T @ Phi_tensor))
    Lambda_inv = torch.linalg.inv(Lambda)

    pyz_prod = (Phi_tensor.T @ Y_tensor @ Xi_v.T).T.reshape(-1,1) ## K * r x 1 column stacking 
    post_mean_w =  (1/sigma2_e)* Lambda_inv @ pyz_prod

    vec_W_post_mean = post_mean_w
    vec_W_post_cov = Lambda_inv
    
    posterior_field = FieldDistribution(eta, vec_W_post_mean, vec_W_post_cov, K+1, r)
    return posterior_field


class FieldDistribution(Distribution):

    def __init__(self, basis_model, vec_W_post_mean, vec_W_post_cov, K, r):
        super().__init__()
        
        self.K = K
        self.r = r
        self.basis = basis_model
        self.vec_W_post_mean = vec_W_post_mean
        self.vec_W_post_cov = vec_W_post_cov
        self.W_post_mean = torch.from_numpy(vec_W_post_mean.detach().numpy().reshape(self.K-1, 
                                                                                     self.r, order="F"))
        
        super().__init__(torch.Size(), validate_args=False)
        
    def compute_predictive_posterior(self, coords):
        """
        Computer the point-wise predictive posterior mean + covariance 
        """
        ##inefficient + can be slow but works
        Nv = coords.shape[0]
        Xi_evals = self.basis(coords).T 
        post_mean_c_lst = []
        post_cov_c_lst = []
        for iv in range(Nv): ##todo: batchify this 
            XiV_I = torch.kron(Xi_evals[:,iv:iv+1].T, torch.eye(self.K-1))
            post_mean_c = torch.squeeze(XiV_I @ self.vec_W_post_mean)
            post_cov_c = XiV_I @ self.vec_W_post_cov @ XiV_I.T
            post_mean_c_lst.append(post_mean_c)
            post_cov_c_lst.append(post_cov_c)

        Post_mean_c_tensor = torch.stack(post_mean_c_lst, 0)
        Post_cov_c_tensor = torch.stack(post_cov_c_lst, 0)
        return Post_mean_c_tensor, Post_cov_c_tensor
    
    def rsample(self, sample_shape=torch.Size(), coords=None): 
        """
        Sample function-valued field at locations in coords  
        coords - torch.tensor: sample_shape x D-dimensional coordinate location to draw samples 
        """
        assert coords is not None, ValueError("'coords' must be a torch tensor of dimension num_locations X D")
        shape = self._extended_shape(sample_shape)
        nsamples = shape.numel()
        Xi_v = self.basis(coords).T
        post_vec_W_samples = np.random.multivariate_normal(self.vec_W_post_mean.detach().numpy().ravel(), 
                                                           self.vec_W_post_cov.detach().numpy(), 
                                                           size=nsamples)  
        fsamples = torch.zeros((nsamples, Xi_v.shape[1], self.K-1))
        for i in range(nsamples):
            Wi = torch.from_numpy(post_vec_W_samples[i,:].reshape(self.K-1, self.r, order="F")).float()
            fsamples[i,:,:] = (Wi @ Xi_v).T
        return fsamples

    def log_prob(self):
        raise NotImplementedError 



