import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import applymask
from dipy.core.sphere import Sphere
from dipy.reconst.csdeconv import odf_sh_to_sharp
from dipy.reconst.shm import real_sym_sh_basis, sph_harm_ind_list, sf_to_sh, real_sh_descoteaux_from_index
from dipy.reconst.odf import gfa
import nibabel as nib

from functools import partial 

from utility.utility import get_odf_transformation, cart2sphere, sphere2cart, S2hemisphere, measurement_error_var_estimator, matern_spec_density, get_signal_transformation
from data_objects.data_objects import ObservationPoints, resample_data
from optimization.opt_algos import PMLE
from optimization.hyper_optim import BO_optimization
from models.representation import Siren
from inference.posterior import FVRF, post_calibration

import argparse
import os 
import time 

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
							description="NODF Estimation.")

parser.add_argument("--out_folder", action='store', required=True,
			   type=str, help='Path to folder to store output.') 

parser.add_argument('--img_file', action='store', required=True,
			   type=str, help='Nifti file for diffusion image (nx X ny X nz x M)') 

parser.add_argument('--mask_file', action='store', required=True,
			   type=str, help='Nifti file for mask image (nx X ny X nz)') 

parser.add_argument('--bval_file', action='store', required=True,
			   type=str, help='Text file b-values.') 

parser.add_argument('--bvec_file', action='store', required=True,
			   type=str, help='Text file b-vectors.') 

parser.add_argument('--bval', action='store',  default=1000,
			   type=float, help='B-value to build field.') 

parser.add_argument('--sh_order', action='store', default=8,
			   type=int, help='Order of spherical harmonic basis')

parser.add_argument('--bmarg', action='store', default=20,
			   type=int, help='+= bmarg considered same b-value.')

parser.add_argument("--rho", action='store', default=0.5,
			   type=float, help='Length-scale parameter for Matern Prior.')

parser.add_argument("--nu", action='store', default=1.5,
			   type=float, help='Smoothness parameter for Matern Prior.')

parser.add_argument("--num_epochs", action='store', default=2000,
			   type=int, help='Number of trainging epochs.')

parser.add_argument("--learning_rate", action='store', default=1e-4,
			   type=float, help='Learning rate for optimizer.')

parser.add_argument("--r", action='store', default=256,
			   type=int, help='Rank of spatial basis.')

parser.add_argument("--depth", action='store', default=5,
			   type=int, help='Number of hidden layers.')

parser.add_argument("--train_prop", action='store', default=0.8,
			   type=float, help='Proportion of voxels to be used in training for each iteration of hyper-parameter optimization.')

parser.add_argument("--batch_frac", action='store', default=1,
			   	type=int, help='Fraction of total training voxels to be used in each batch.')

parser.add_argument("--Nexperiments", action='store', default=20,
			   type=int, help='Number of experiments for hyper-parameter optimization.')

parser.add_argument("--calib_prop", action='store', default=0.1,
			   type=float, help='Proportion of voxels to be used in posterior calibration.')

parser.add_argument("--lambda_c", help='Prior regularization strength.', type=float)

parser.add_argument("--sigma2_mu", help='Variance for isotropic harmonic.', type=float)

parser.add_argument("--sigma2_w", help='Variance parameter for GP prior.', type=float)

parser.add_argument("--verbose", action='store_true')

parser.add_argument("--deconvolve", action='store_true')

args = parser.parse_args()

out_folder = args.out_folder
img_file = args.img_file
bval_file = args.bval_file
bvec_file = args.bvec_file
mask_file = args.mask_file
bval = args.bval
sh_order = args.sh_order
bmarg = args.bmarg
rho = args.rho 
nu = args.nu 
num_epochs = args.num_epochs
learning_rate = args.learning_rate
r = args.r 
depth = args.depth
train_prop = args.train_prop
batch_frac = args.batch_frac
Nexperiments = args.Nexperiments
calib_prop = args.calib_prop
lambda_c = args.lambda_c
sigma2_mu = args.sigma2_mu
sigma2_w = args.sigma2_w
verbose = args.verbose
deconvolve = args.deconvolve

#### this parameterization should work O.K. 
"""
out_folder = "/path/2/where/data/is/written"
img_file = ".../flipped_dwi.nii.gz"
bval_file = "../flip_x.bval"
bvec_file = "../flip_x.bvec"
mask_file = "../part_mask.nii.gz"
bval = 1000
sh_order = 8
bmarg = 20
rho = 0.5
nu = 1.5
num_epochs = 2000
learning_rate = 1e-4
r = 256
depth = 5
train_prop = 0.8
batch_frac = 1
Nexperiments = 20
calib_prop = 0.1
#lambda_c = 3.76e-7
#sigma2_mu = 0.005
#sigma2_w = 0.5
lambda_c = None
sigma2_mu = None
sigma2_w = None
verbose = True
deconvolve = True
"""

## make reproducible 
torch.manual_seed(0)
np.random.seed(0)

## get on gpu if available 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load data 
img = nib.load(img_file)
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
mask = nib.load(mask_file).get_fdata().astype(bool)

Ydata = img.get_fdata()

b0_ix = np.where(bvals < bmarg)[0]
bix = np.where((bvals >= bval - bmarg) & (bvals <= bval + bmarg))[0]
bvecs_bix = bvecs[bix]

## b0-normalization 
Y0_mean = Ydata[:,:,:,b0_ix].mean(axis=3)
Ynorm = np.nan_to_num(Ydata / Y0_mean[...,None], posinf=0, neginf=0)
## apply maks 
Ynorm = applymask(Ynorm, mask)
## get b-shel data 
Ynorm = Ynorm[..., bix]

nx, ny, nz = Ydata.shape[:-1]
M = bvecs_bix.shape[0]

## estimate measurement error variance
sigma2 = measurement_error_var_estimator(Ydata[...,b0_ix], mask=mask)

## define harominc function space 
K = int((sh_order+1)*(sh_order+2)/2)
m, n = sph_harm_ind_list(sh_order)
T_n = get_odf_transformation(n)
T_n_inv = get_signal_transformation(n)
sphere = get_sphere("repulsion724")
x_grid = cart2sphere(sphere.vertices)
theta_grid = x_grid[:,0]; phi_grid = x_grid[:,1]
B, m, n = real_sym_sh_basis(sh_order, phi_grid, theta_grid)

## function mapping
x_obs = cart2sphere(bvecs_bix)
theta_obs = x_obs[:,0]; phi_obs = x_obs[:,1]
Phi, m, n = real_sym_sh_basis(sh_order, phi_obs, theta_obs)

hyper_params = {}
ODFSPACE = True ## esimate ODF field or Diffusion Signal Attenuation field 

## observation tensor 
## map index to location
XX,YY,ZZ = np.meshgrid(np.linspace(0,1,num=nx),
					   np.linspace(0,1,num=ny),
					   np.linspace(0,1,num=nz),
					indexing="ij")
V_xyz = torch.from_numpy(np.column_stack((XX[mask],YY[mask],ZZ[mask]))).float()

## create observation object 
Y_flat = torch.from_numpy(Ynorm[mask,:]).float()
N = V_xyz.shape[0]
mask_array = torch.from_numpy(np.repeat(np.ones(N).reshape(-1,1), M, axis=1)).float() ##use data at all gradient orientations
O = ObservationPoints(V_xyz, Y_flat, mask=mask_array)
dataloader = DataLoader(O, shuffle=True, batch_size=N//batch_frac, pin_memory=True, num_workers=0)

## fixed hyper parameters 
D = 3
## collect fixed hyper params 
hyper_params = {"eigs_root":  np.sqrt(n*(n+1))}
if ODFSPACE:
	Phi_tensor = torch.from_numpy(Phi @ T_n_inv).float().to(device)
else:
	Phi_tensor = torch.from_numpy(Phi).float().to(device)
hyper_params["Phi_tensor"] = Phi_tensor
hyper_params["sigma2_e"] = sigma2
hyper_params["rho"] = rho
hyper_params["nu"] = nu
hyper_params["R_tensor"] = torch.from_numpy(np.diag(1/matern_spec_density(np.sqrt(n*(n+1)), 
																		  hyper_params["rho"], 
																		  hyper_params["nu"]))).float().to(device)
hyper_params["R_tensor_inv"] = torch.from_numpy(np.diag(matern_spec_density(np.sqrt(n*(n+1)), 
																			  hyper_params["rho"], 
																			  hyper_params["nu"]))).float().to(device)

## hyperparameters optimization (note, can add add additional hyperparameters here if desired) 
if lambda_c is None:
	## run hyer-parameter optimization scheme to select prior roughness penalty strength
	parameter_map=[{"name": "lambda_c", "type": "range", "bounds": [1e-9, 1e-1], "log_scale": True},]
	## freeze model and optimizer
	field_model_ = partial(Siren, in_features=D, out_features=K, hidden_features=r, 
						  hidden_layers=depth, outermost_linear=True)
	optimizer_ = partial(torch.optim.Adam, lr=learning_rate)
	best_parameters, map_ax_client = BO_optimization(device, O, field_model_, optimizer_, PMLE, hyper_params, parameter_map, 
														Nexperiments, num_epochs = num_epochs, 
														experiment_name="Matern_Covar", train_prop=train_prop)
	## selected hyper parameters 
	hyper_params["lambda_c"] = best_parameters["lambda_c"]
else:
	hyper_params["lambda_c"] = lambda_c 

## define basis basis system
field_model = Siren(in_features=D, out_features=K, 
				hidden_features=r, hidden_layers=depth, 
				outermost_linear=True).to(device)
optim = torch.optim.Adam(params=field_model.parameters(), 
									lr=learning_rate)
dataloader_train, dataloader_calib = resample_data(dataloader.dataset, train_prop=1-calib_prop, batch_frac=batch_frac)

## roughness-penalized MLE
PMLE(device, field_model, optim, hyper_params, dataloader_train, num_epochs, verbose=verbose)

## posterior calibration
if sigma2_mu is None:
	##set up small grid of candidate values for posterior calibfation (might need to adjust the ranges here, but I found them to work well for a vew different experimental designs)
	sigma2_mus = (1e-4, 1e-3, 5e-3, 1e-2, 5e-2) 
	sigma2_ws = (1e-1, 2e-1, 3e-1, 4e-1, 5e-1)
	var_grid = [(sm, sw) for sm in sigma2_mus for sw in sigma2_ws]
	sigma2_mu_optim, sigma2_w_optim = post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid=var_grid)
	hyper_params["sigma2_mu"] = sigma2_mu_optim
	hyper_params["sigma2_w"] = sigma2_w_optim
else:
	hyper_params["sigma2_mu"] = sigma2_mu
	hyper_params["sigma2_w"] = sigma2_w

## compute point-wise posteriors (note: this can be performed on any arbitrary set of domain points, .e.g. 'super-resolution', for simplicity, predictions are formed on observed data grid)
coords = dataloader.dataset.X.to(device)
Y_tensor = dataloader.dataset.Y.T.to(device)
posterior_field = FVRF(field_model, coords, Y_tensor,  hyper_params)
Post_mean_coefs, Post_cov_mats = posterior_field.compute_predictive_posterior(coords) ##approx 0.0001 seconds per-point inference, ultimately want to speed this up ...

## save point-wise posterior moments
torch.save(Post_mean_coefs, os.path.join(out_folder, "pointwise_posterior_mean.pt"))
torch.save(Post_cov_mats,  os.path.join(out_folder, "pointwise_posterior_covarianece.pt"))

## collect some posterior samples for inference of non-linear summary  
npost_samps = 200
posterior_field = MultivariateNormal(Post_mean_coefs.cpu().detach(), Post_cov_mats.cpu().detach())
Post_samples = posterior_field.sample((npost_samps,)).to(device)
## e.g. build posterior credible intervals for GFA 
Post_sample_evals = Post_samples @ torch.from_numpy(B.T).float().to(device)
gfa_samples = gfa(Post_sample_evals.cpu().detach().numpy()) ##SLOW ... should compute statistic in torch 
gfa_quantiles = np.quantile(gfa_samples, [0.025, 0.975], axis=0)
GFA_lp_tensor = np.zeros((nx, ny, nz))
GFA_up_tensor = np.zeros((nx, ny, nz))
GFA_lp_tensor[mask] = gfa_quantiles[0,:]
GFA_up_tensor[mask] = gfa_quantiles[1,:]

gfa_lp_img = nib.Nifti1Image(GFA_lp_tensor, img.affine)
nib.save(gfa_lp_img,  os.path.join(out_folder,"gfa_lower.nii.gz"))
gfa_up_img = nib.Nifti1Image(GFA_up_tensor, img.affine)
nib.save(gfa_up_img,  os.path.join(out_folder,"gfa_upper.nii.gz"))

## save mean estimates 
ODF_post_mode = np.zeros((nx, ny, nz, K))

if ODFSPACE:
	Post_mean_odf_WC = Post_mean_coefs.cpu().detach().numpy()
else:
	Post_mean_odf_WC = Post_mean_coefs.cpu().detach().numpy() @ T_n

ODF_post_mode[mask, :] = Post_mean_odf_WC
odf_img = nib.Nifti1Image(ODF_post_mode, img.affine)
nib.save(odf_img,  os.path.join(out_folder,"odfs_nodf.nii.gz"))

## spherical deconvolution  
if deconvolve:
	FODFCoefs_NODF = odf_sh_to_sharp(Post_mean_odf_WC, sphere, 
									basis="descoteaux07", 
									ratio=0.3, ##should estimate this from the data or make it configurable
									sh_order=sh_order, 
									lambda_=1.0, tau=0.1, r2_term=False)
	FODFCoefs_NODF_tourn = sf_to_sh(FODFCoefs_NODF @ B.T, sphere, sh_order=8, basis_type="tournier07") ##change coordiante system for visualizing in mrtrix
	FODFCoefs_NODF_tourn = FODFCoefs_NODF_tourn/np.linalg.norm(FODFCoefs_NODF_tourn, axis=-1)[...,None]
	FODFtensor_NODF = np.zeros((nx, ny, nz, K))
	FODFtensor_NODF[mask, :] = FODFCoefs_NODF_tourn
	fodf_img = nib.Nifti1Image(FODFtensor_NODF, img.affine)
	nib.save(fodf_img,  os.path.join(out_folder, "fodfs_nodf_tournier07.nii.gz"))


