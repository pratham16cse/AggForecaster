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

		if base_model_name in ['seq2seqmse']:
			self.point_estimates = True
		else:
			self.point_estimates = False


	def optimize(params_dict):
		raise NotImplementedError

	def forward(self, inputs_dict):
		bottom_level_model = self.base_models_dict[0]

		params_dict = dict()
		for level in range(len(base_models_dict)):
			model = base_models_dict[level]
			inputs = inputs_dict[level]
			params = model(inputs)
			if self.point_estimates:
				means = params
				sigmas = torch.ones_like(means)
				params = [means, sigmas]
			else:
				raise NotImplementedError

			params_dict[level] = params

		all_preds = []
		for i in range(params_dict[0][0].size()[0]):
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i:i+1], params_dict[lvl][1][i:i+1]]

			ex_preds_opt = optimize(ex_params_dict)
			all_preds.append(ex_preds_opt)

		return all_preds