import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cvxpy as cp


class DILATE(torch.nn.Module):
	"""docstring for DILATE"""
	def __init__(self, base_models_dict):
		super(DILATE, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict):
		return self.base_models_dict[0](inputs_dict[0])

class MSE(torch.nn.Module):
	"""docstring for MSE"""
	def __init__(self, base_models_dict):
		super(MSE, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict):
		return self.base_models_dict[0](inputs_dict[0])
		

class DualTPP(torch.nn.Module):
	"""docstring for DualTPP"""
	def __init__(self, K, base_models_dict):
		'''
		K: int
			number of steps to aggregate at each level
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(DualTPP, self).__init__()
		self.K = K
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
			#print(i)
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i], params_dict[lvl][1][i]]

			ex_preds_opt = self.optimize(ex_params_dict)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		return all_preds, None


class OPT_ls(torch.nn.Module):
	"""docstring for OPT_ls"""
	def __init__(self, K, base_models_dict):
		'''
		K: int
			number of steps to aggregate at each level
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(OPT_ls, self).__init__()
		self.K = K
		self.base_models_dict = base_models_dict

	def aggregate_seq_(self, seq):
		assert seq.shape[0]%self.K == 0
		agg_seq = np.array([[cp.sum(seq[i:i+self.K])] for i in range(0, seq.shape[0], self.K)])
		return agg_seq

	def fit_with_indices(self, seq):
		W, B = [], []
		for i in range(0, seq.shape[0], self.K):
			x = np.ones_like(seq[i:i+self.K])
			x = np.cumsum(x) - 1
			y = seq[i:i+self.K]
			m_x = cp.sum(x)/x.shape[0]
			m_y = cp.sum(y)/y.shape[0]
			s_xy = cp.sum((x-m_x)*(y-m_y))
			s_xx = cp.sum((x-m_x)**2)
			w = s_xy/s_xx
			b = m_y - w*m_x
			W.append(w)
			B.append(b)
		W = np.expand_dims(np.array(W), axis=1)
		B = np.expand_dims(np.array(B), axis=1)
		fit_params = np.concatenate([W, B], axis=1)
		return fit_params

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def optimize(self, params_dict):

		#params_dict_detached = dict()
		#for 

		ex_preds_dict = dict()

		preds = cp.Variable(params_dict[0][0].size())


		ls_params = self.fit_with_indices(preds)
		ls_loss = self.log_prob(
			ls_params,
			params_dict[1][0].detach().numpy(), params_dict[1][1].detach().numpy()
		)
		preds_loss = self.log_prob(
			preds,
			params_dict[0][0].detach().numpy(), params_dict[0][1].detach().numpy()
		)
		opt_loss = preds_loss + ls_loss

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

		return preds.value


	def forward(self, inputs_dict):

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
			#print(i)
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i], params_dict[lvl][1][i]]

			ex_preds_opt = self.optimize(ex_params_dict)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		return all_preds, None