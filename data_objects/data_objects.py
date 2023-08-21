import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader


def resample_data(Obs, train_prop=0.8, batch_frac=1):

	dataset_size = Obs.N
	indices = list(range(dataset_size))
	split = int(np.floor((1-train_prop) * dataset_size))
	np.random.shuffle(indices)
	train_indices, test_indices = indices[split:], indices[:split]

	N_train = len(train_indices)
	N_test = len(test_indices)
	Y_train = Obs.Y[train_indices,:]
	Y_test = Obs.Y[test_indices,:]
	V_train = Obs.X[train_indices, :]
	V_test = Obs.X[test_indices, :]
	mask_array_train = Obs.mask[train_indices, :]
	mask_array_test = Obs.mask[test_indices, :]

	O_train = ObservationPoints(V_train, Y_train, mask=mask_array_train)
	O_test = ObservationPoints(V_test, Y_test, mask=mask_array_test)

	dataloader_train = DataLoader(O_train, shuffle=True, batch_size=N_train//batch_frac, pin_memory=True, num_workers=0)
	dataloader_test = DataLoader(O_test, shuffle=True, batch_size=N_test, pin_memory=True, num_workers=0)

	return dataloader_train, dataloader_test

class ObservationPoints(Dataset):
	"""
	X: torch.tensor (N x D) spatial coordinates 
	Y: torch.tensor (N x M) M angular samples at each spatial location 
	mask: torch.tensor (N x M) (optional) mask for angular samples
	"""
	def __init__(self, X, Y, mask=None):
		super().__init__()
		self.Y = Y
		self.X = X
		self.N = self.Y.shape[0]
		if mask is None:
			mask = torch.ones(self.Y.shape)
		self.mask = mask
	
	def __getitem__(self, idx):
		coords = self.X[idx,:] 
		yvals = self.Y[idx,:] 
		mask_rix = self.mask[idx,:]
		return {"coords":coords}, {"yvals":yvals, "mask":mask_rix}
	
	def __len__(self):
		return self.N