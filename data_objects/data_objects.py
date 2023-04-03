import numpy as np 
import torch 
from torch.utils.data import Dataset

##Tensor Data Object for Regular Spatial Grid Observations 
class ObservationTensor(Dataset):
	def __init__(self, xcoords, Y, batch_size, mask=None):
		super().__init__()
		
		self.N = np.prod([len(xcoords[d]) for d in range(len(xcoords))])
		self.D = len(xcoords)
		self.M = Y.shape[-1]
		assert batch_size <= self.N, "Batch size cannot be larger than number of observation points"
		self.indexer = np.array([ix for ix in np.ndindex(*[len(xcoords[d]) for d in range(self.D)])])

		self.xcoords = xcoords 
		self.Y = Y 
		if mask is None:
			self.mask = np.ones(Y.shape[:-1])
		self.mask = mask 
		
		self.batch_size = batch_size
		
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		rix = np.random.choice(self.N, size=self.batch_size, replace=False) 
		coords = np.array([[self.xcoords[d][self.indexer[i,d]] for d in range(self.D)] for i in rix]) ##batch_size X 3
		yvals = np.array([self.Y[tuple(self.indexer[i,:])] for i in rix]) ##batch_size X M
		mask_flat = np.array([self.mask[tuple(self.indexer[i,:])] for i in rix]) ##batch_size X M
		return {"coords":torch.from_numpy(coords).float()}, {"yvals":torch.from_numpy(yvals).float(), "mask":torch.from_numpy(mask_flat).float()}
	
	def getfulldata(self):
		rix = np.arange(0,self.N) 
		coords = np.array([[self.xcoords[d][self.indexer[i,d]] for d in range(self.D)] for i in rix]) ##batch_size X 3
		yvals = np.array([self.Y[tuple(self.indexer[i,:])] for i in rix]) ##batch_size X M
		mask_flat = np.array([self.mask[tuple(self.indexer[i,:])] for i in rix]) 
		return {"coords":torch.from_numpy(coords).float()}, {"yvals":torch.from_numpy(yvals).float(), "mask":torch.from_numpy(mask_flat).float()}

##Data Object for Irregular Spatial Observations 
class ObservationPoints(Dataset):
	def __init__(self, X, Y, batch_size, mask=None):
		super().__init__()
		
		self.N = X.shape[0]
		self.D = X.shape[1]
		self.M = Y.shape[1]
		assert self.N <= Y.shape[0], "X & Y must share the same number of rows"

		self.Y = Y 
		self.X = X 
		
		if mask is None:
			self.mask = np.ones(N,M)
		self.mask = mask 
		
		self.batch_size = batch_size
		
	def __len__(self):
		return 1
	
	def __getitem__(self, idx):
		rix = np.random.choice(self.N, size=self.batch_size, replace=False) 
		coords = self.X[rix,:] 
		yvals = self.Y[rix,:] 
		mask_rix = self.mask[rix,:]
		return {"coords":torch.from_numpy(coords).float()}, {"yvals":torch.from_numpy(yvals).float(), "mask":torch.from_numpy(mask_rix).float()}
	
	def getfulldata(self):
		return {"coords":torch.from_numpy(self.X).float()}, {"yvals":torch.from_numpy(self.Y).float(), "mask":torch.from_numpy(self.mask).float()}
