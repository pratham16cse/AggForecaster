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
	def __init__(self, base_model_name, base_models_dict):
		'''
		base_model_name: str
			model name from args.base_model_names list
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(DualTPP, self).__init__()
		self.base_model_name = base_model_name
		self.base_models_dict = base_models_dict


	def forward(self, x):
		bottom_level_model = self.base_models_dict[0]
		return bottom_level_model(x[0])