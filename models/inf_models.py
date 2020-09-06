import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cvxpy as cp


class DILATE(torch.nn.Module):
	"""docstring for DILATE"""
	def __init__(self, model):
		super(DILATE, self).__init__()
		self.base_model = model

	def forward(self, x):
		means, _ = self.base_model(x) 
		return means

class MSE(torch.nn.Module):
	"""docstring for MSE"""
	def __init__(self, model):
		super(MSE, self).__init__()
		self.base_model = model

	def forward(self, x):
		means, _ = self.base_model(x)
		return means
		

class DualTPP(torch.nn.Module):
	"""docstring for DualTPP"""
	def __init__(self, K, base_model_name, base_models_dict):
		'''
		K: int
			number of steps to aggregate at each level
		base_model_name: str
			model name from args.base_model_names list
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(DualTPP, self).__init__()
		self.K = K
		self.base_model_name = base_model_name
		self.base_models_dict = base_models_dict

	def aggregate_seq_(self, seq):
		assert seq.shape[0]%self.K == 0
		agg_seq = np.array([[cp.sum(seq[i:i+self.K])] for i in range(0, seq.shape[0], self.K)])
		return agg_seq

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def optimize(self, params_dict):

		#params_dict_detached = dict()
		#for 

		ex_preds_dict = dict()

		for lvl, params in params_dict.items():
			if lvl==0:
				ex_preds = cp.Variable(params[0].size())
				ex_preds_dict[lvl] = ex_preds
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds = self.aggregate_seq_(ex_preds_dict[lvl-1])
			lvl_loss = self.log_prob(lvl_ex_preds, params[0].detach().numpy(), params[1].detach().numpy())
			#import ipdb
			#ipdb.set_trace()
			if lvl==0:
				opt_loss = lvl_loss
			else:
				opt_loss += lvl_loss


		objective = cp.Minimize(opt_loss)

		#constraints = [ex_preds>=0]

		prob = cp.Problem(objective)#, constraints)

		try:
			opt_loss = prob.solve()
		except cp.error.SolverError:
			opt_loss = prob.solve(solver='SCS')

		#if ex_preds.value is None:

		#import ipdb
		#ipdb.set_trace()

		return ex_preds.value


	def forward(self, inputs_dict):
		bottom_level_model = self.base_models_dict[0]

		params_dict = dict()
		for level in range(len(self.base_models_dict)):
			model = self.base_models_dict[level]
			inputs = inputs_dict[level]
			means, stds = model(inputs)

			if model.point_estimates:
				stds = torch.ones_like(means)
			params = [means, stds]
			params_dict[level] = params

		all_preds = []
		for i in range(params_dict[0][0].size()[0]):
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i], params_dict[lvl][1][i]]

			ex_preds_opt = self.optimize(ex_params_dict)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		return all_preds