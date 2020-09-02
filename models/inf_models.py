import torch
import torch.nn as nn
import torch.nn.functional as F


class DILATE(torch.nn.Module):
	"""docstring for DILATE"""
	def __init__(self, model):
		super(DILATE, self).__init__()
		self.base_model = model

	def forward(self, x):
		return self.base_model(x)

class MSE(torch.nn.Module):
	"""docstring for MSE"""
	def __init__(self, model):
		super(MSE, self).__init__()
		self.base_model = model

	def forward(self, x):
		return self.base_model(x)
		

class DualTPP(torch.nn.Module):
	"""docstring for DualTPP"""
	def __init__(self, arg):
		super(DualTPP, self).__init__()
		self.arg = arg

	def forward(self,):
		raise NotImplementedError