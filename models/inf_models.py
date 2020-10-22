import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cvxpy as cp

from utils import normalize, unnormalize


class DILATE(torch.nn.Module):
	"""docstring for DILATE"""
	def __init__(self, base_models_dict):
		super(DILATE, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict, norm_dict):
		return self.base_models_dict[1](inputs_dict[1])

class MSE(torch.nn.Module):
	"""docstring for MSE"""
	def __init__(self, base_models_dict):
		super(MSE, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict, norm_dict):
		return self.base_models_dict[1](inputs_dict[1])

class NLLsum(torch.nn.Module):
	"""docstring for NLLsum"""
	def __init__(self, base_models_dict):
		super(NLLsum, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict, norm_dict):
		return self.base_models_dict[1](inputs_dict[1])

class NLLls(torch.nn.Module):
	"""docstring for NLLls"""
	def __init__(self, base_models_dict):
		super(NLLls, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, inputs_dict, norm_dict):
		return self.base_models_dict[1](inputs_dict[1])

class DualTPP(torch.nn.Module):
	"""docstring for DualTPP"""
	def __init__(self, K_list, base_models_dict):
		'''
		K: int
			number of steps to aggregate at each level
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(DualTPP, self).__init__()
		self.K_list = K_list
		self.base_models_dict = base_models_dict

	def aggregate_seq_(self, seq, K):
		assert seq.shape[0]%K == 0
		agg_seq = np.array([[cp.sum(seq[i:i+K])] for i in range(0, seq.shape[0], K)])
		return agg_seq

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def optimize(self, params_dict, norm_dict):

		for lvl, params in params_dict.items():
			#params[0] = unnormalize(params[0].detach().numpy(), norm_dict[lvl].detach().numpy())
			if lvl==1:
				ex_preds = cp.Variable(params[0].shape)
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds, _ = normalize(
					self.aggregate_seq_(unnormalize(ex_preds, norm_dict[1]), lvl),
					norm_dict[lvl]
				)
			lvl_loss = self.log_prob(lvl_ex_preds, params[0].detach().numpy(), params[1].detach().numpy())
			#import ipdb
			#ipdb.set_trace()
			if lvl==1:
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


	def forward(self, inputs_dict, norm_dict):
		bottom_level_model = self.base_models_dict[1]

		norm_dict_np = dict()
		for lvl in norm_dict.keys():
			norm_dict_np[lvl] = np.expand_dims(norm_dict[lvl].detach().numpy(), axis=0)

		params_dict = dict()
		for level in self.K_list:
			model = self.base_models_dict[level]
			inputs = inputs_dict[level]
			means, stds = model(inputs)

			if model.point_estimates:
				stds = torch.ones_like(means)
			params = [means, stds]
			params_dict[level] = params

		all_preds = []
		for i in range(params_dict[1][0].size()[0]):
			#print(i)
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i], params_dict[lvl][1][i]]

			ex_preds_opt = self.optimize(ex_params_dict, norm_dict_np)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		#all_preds, _ = normalize(all_preds, norm_dict[0])

		return all_preds, None


class OPT_ls(torch.nn.Module):
	"""docstring for OPT_ls"""
	def __init__(self, K_list, base_models_dict, intercept_type='intercept'):
		'''
		K: int
			number of steps to aggregate at each level
		base_models_dict: dict
			key: level in the hierarchy
			value: base model at the level 'key'
		'''
		super(OPT_ls, self).__init__()
		self.K_list = K_list
		self.base_models_dict = base_models_dict
		self.intercept_type = intercept_type


	def fit_with_indices(self, seq, K):
		W, B = [], []
		for i in range(0, seq.shape[0], K):
			x = np.cumsum(np.ones(seq[i:i+K].shape)) - 1.
			x = np.cumsum(x) - 1
			y = seq[i:i+K]
			m_x = cp.sum(x)/x.shape[0]
			m_y = cp.sum(y)/y.shape[0]
			s_xy = cp.sum((x-m_x)*(y-m_y))
			s_xx = cp.sum((x-m_x)**2)
			w = s_xy/s_xx
			if self.intercept_type in ['intercept']:
				b = m_y - w*m_x
			elif self.intercept_type in ['sum']:
				b = cp.sum(y)
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

	def optimize(self, params_dict, norm_dict):

		#params_dict_detached = dict()
		#for 

#		ex_preds_dict = dict()
#
#		preds = cp.Variable(params_dict[1][0].size())
#
#
#		ls_params, _ = normalize(
#			self.fit_with_indices(unnormalize(preds, norm_dict[0])),
#			norm_dict[1]
#		)
#		#ls_params = unnormalize(ls_params, norm_dict[1].detach().numpy())
#		#params_dict[0][0] = unnormalize(params_dict[0][0], norm_dict[0])
#		#params_dict[0][1] = unnormalize(params_dict[0][1], norm_dict[0])
#		#params_dict[1][0] = unnormalize(params_dict[1][0], norm_dict[1])
#		#params_dict[1][1] = unnormalize(params_dict[1][1], norm_dict[1])
#		ls_loss = self.log_prob(
#			ls_params,
#			params_dict[1][0].detach().numpy(), params_dict[1][1].detach().numpy()
#		)
#		preds_loss = self.log_prob(
#			preds,
#			params_dict[0][0].detach().numpy(), params_dict[0][1].detach().numpy()
#		)
#		opt_loss = preds_loss + ls_loss

		for lvl, params in params_dict.items():
			#params[0] = unnormalize(params[0].detach().numpy(), norm_dict[lvl].detach().numpy())
			if lvl==1:
				ex_preds = cp.Variable(params[0].shape)
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds, _ = normalize(
					self.fit_with_indices(unnormalize(ex_preds, norm_dict[1]), lvl),
					norm_dict[lvl]
				)
			lvl_loss = self.log_prob(lvl_ex_preds, params[0].detach().numpy(), params[1].detach().numpy())
			#import ipdb
			#ipdb.set_trace()
			if lvl==1:
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


	def forward(self, inputs_dict, norm_dict):

		norm_dict_np = dict()
		for lvl in norm_dict.keys():
			norm_dict_np[lvl] = np.expand_dims(norm_dict[lvl].detach().numpy(), axis=0)

		params_dict = dict()
		for level in self.K_list:
			model = self.base_models_dict[level]
			inputs = inputs_dict[level]
			means, stds = model(inputs)

			if model.point_estimates:
				stds = torch.ones_like(means)
			params = [means, stds]
			params_dict[level] = params

		all_preds = []
		for i in range(params_dict[1][0].size()[0]):
			#print(i)
			ex_params_dict = dict()
			for lvl, params in params_dict.items():
				ex_params_dict[lvl] = [params_dict[lvl][0][i], params_dict[lvl][1][i]]

			ex_preds_opt = self.optimize(ex_params_dict, norm_dict_np)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		#all_preds, _ = normalize(all_preds, norm_dict[0])

		return all_preds, None


class OPT_st(torch.nn.Module):
	"""docstring for OPT_st"""
	def __init__(self, K_list, base_models_dict, intercept_type='intercept'):
		'''
		K_list: list
			list of K-values used for aggregation
		base_models_dict: dict of dicts
			key: aggregation method
			value: dict
				key: level in the hierarchy
				value: base model at the level 'key'
		'''
		super(OPT_st, self).__init__()
		self.K_list = K_list
		self.base_models_dict = base_models_dict
		self.intercept_type = intercept_type


	def aggregate_seq_(self, seq, K):
		assert seq.shape[0]%K == 0
		agg_seq = np.array([[cp.sum(seq[i:i+K])] for i in range(0, seq.shape[0], K)])
		return agg_seq

	def fit_slope_with_indices(self, seq, K):
		W = []
		for i in range(0, seq.shape[0], K):
			x = np.cumsum(np.ones(seq[i:i+K].shape)) - 1.
			y = seq[i:i+K]
			m_x = cp.sum(x)/x.shape[0]
			m_y = cp.sum(y)/y.shape[0]
			s_xy = cp.sum((x-m_x)*(y-m_y))
			s_xx = cp.sum((x-m_x)**2)
			w = s_xy/s_xx
			if self.intercept_type in ['intercept']:
				b = m_y - w*m_x
			elif self.intercept_type in ['sum']:
				b = cp.sum(y)
			W.append(w)
		W = np.expand_dims(np.array(W), axis=1)
		return W

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def optimize(self, params_dict, norm_dict):

		ex_preds = cp.Variable(params_dict['sum'][1][0].shape)
		for lvl, params in params_dict['slope'].items():
			if lvl==1:
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds, _ = normalize(
					self.fit_slope_with_indices(
						unnormalize(ex_preds, norm_dict['slope'][1]),
						lvl
					),
					norm_dict['slope'][lvl]
				)
			lvl_loss = self.log_prob(
				lvl_ex_preds,
				params_dict['slope'][lvl][0].detach().numpy(),
				params_dict['slope'][lvl][1].detach().numpy()
			)
			if lvl==1:
				opt_loss = lvl_loss
			else:
				opt_loss += lvl_loss

		for lvl, params in params_dict['sum'].items():
			if lvl==1:
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds, _ = normalize(
					self.aggregate_seq_(
						unnormalize(ex_preds, norm_dict['sum'][1]),
						lvl
					),
					norm_dict['sum'][lvl]
				)
			lvl_loss = self.log_prob(
				lvl_ex_preds,
				params_dict['sum'][lvl][0].detach().numpy(),
				params_dict['sum'][lvl][1].detach().numpy()
			)
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


	def forward(self, inputs_dict, norm_dict):
		'''
		inputs_dict: [aggregation method][level]
		norm_dict: [aggregation method][level]
		'''

		norm_dict_np = dict()
		for agg_method in norm_dict.keys():
			norm_dict_np[agg_method] = dict()
			for lvl in norm_dict[agg_method].keys():
				norm_dict_np[agg_method][lvl] = np.expand_dims(norm_dict[agg_method][lvl].detach().numpy(), axis=0)

		params_dict = dict()
		for agg_method in self.base_models_dict.keys():
			params_dict[agg_method] = dict()
			for level in self.K_list:
				model = self.base_models_dict[agg_method][level]
				inputs = inputs_dict[agg_method][level]
				means, stds = model(inputs)

				if model.point_estimates:
					stds = torch.ones_like(means)
				params = [means, stds]
				params_dict[agg_method][level] = params

		all_preds = []
		for i in range(params_dict['sum'][1][0].size()[0]):
			#print(i)
			ex_params_dict = dict()
			for agg_method in params_dict.keys():
				ex_params_dict[agg_method] = dict()
				for lvl in params_dict[agg_method].keys():
					ex_params_dict[agg_method][lvl] = [params_dict[agg_method][lvl][0][i], params_dict[agg_method][lvl][1][i]]

			#import ipdb
			#ipdb.set_trace()
			ex_preds_opt = self.optimize(ex_params_dict, norm_dict_np)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		#all_preds, _ = normalize(all_preds, norm_dict[0])

		return all_preds, None

class OPT_KL_st(OPT_st):
	"""docstring for OPT_st"""
	def __init__(self, K_list, base_models_dict, intercept_type='intercept'):
		'''
		K_list: list
			list of K-values used for aggregation
		base_models_dict: dict of dicts
			key: aggregation method
			value: dict
				key: level in the hierarchy
				value: base model at the level 'key'
		'''
		super(OPT_KL_st, self).__init__(K_list, base_models_dict, intercept_type=intercept_type)


	def aggregate_seq_(self, mu, var, K):
		assert mu.shape[0]%K == 0
		agg_mu = np.array([[cp.sum(mu[i:i+K])] for i in range(0, mu.shape[0], K)])
		agg_var = np.array([[cp.sum(var[i:i+K])] for i in range(0, var.shape[0], K)])
		return agg_mu, agg_var

	def fit_slope_with_indices(self, mu, var, K):
		W_mu = []
		W_var = []
		for i in range(0, mu.shape[0], K):
			x = np.cumsum(np.ones(mu[i:i+K].shape)) - 1.
			y_mu = mu[i:i+K]
			y_var = var[i:i+K]
			m_x = cp.sum(x)/x.shape[0]
			m_y = cp.sum(y_mu)/y_mu.shape[0]
			s_xx = cp.sum((x-m_x)**2)
			a = (x-m_x) / s_xx
			w_mu = cp.sum(a*y_mu)
			w_var = cp.sum(a**2 * y_var)
			if self.intercept_type in ['intercept']:
				b = m_y - w_mu*m_x
			elif self.intercept_type in ['sum']:
				b = cp.sum(y_mu)
			W_mu.append(w_mu)
			W_var.append(w_var)
		W_mu = np.expand_dims(np.array(W_mu), axis=1)
		W_var = np.expand_dims(np.array(W_var), axis=1)
		return W_mu, W_var

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def KL(self, mu_1, var_1, mu_2, var_2, lvl):

		def single_eqn_kl(mu_1, var_1, mu_2, var_2):
			return cp.sum(cp.log(var_1)/2. - cp.log(var_2)/2. + (var_2 + (mu_2-mu_1)**2)/(2*var_1) - 0.5)

		kl_distance = 0.
		if lvl != 1:
			for i in range(mu_1.shape[0]):
				kl_distance += (single_eqn_kl(mu_1[i,0], var_1[i,0], mu_2[i,0], var_2[i,0]))
		else:
			kl_distance = single_eqn_kl(mu_1, var_1, mu_2, var_2)

		return kl_distance


	def optimize(self, params_dict, norm_dict):

		ex_mu = cp.Variable(params_dict['sum'][1][0].shape)
		ex_var = cp.Variable(params_dict['sum'][1][1].shape)
		for lvl, params in params_dict['slope'].items():
			if lvl==1:
				lvl_ex_mu = ex_mu
				lvl_ex_var = ex_var
			else:
				lvl_ex_mu, lvl_ex_var = self.fit_slope_with_indices(
					unnormalize(ex_mu, norm_dict['slope'][1]),
					unnormalize(ex_var, norm_dict['slope'][1]**2),
					lvl
				)
				lvl_ex_mu, _ = normalize(lvl_ex_mu, norm_dict['slope'][lvl])
				lvl_ex_var, _ = normalize(lvl_ex_var, norm_dict['slope'][lvl]**2)
			lvl_loss = self.KL(
				params_dict['slope'][lvl][0].detach().numpy(),
				params_dict['slope'][lvl][1].detach().numpy()**2,
				lvl_ex_mu, lvl_ex_var, lvl
			)
			if lvl==1:
				opt_loss = lvl_loss
			else:
				opt_loss += lvl_loss

		for lvl, params in params_dict['sum'].items():
			if lvl==1:
				lvl_ex_mu = ex_mu
				lvl_ex_var = ex_var
			else:
				lvl_ex_mu, lvl_ex_var = self.aggregate_seq_(
					unnormalize(ex_mu, norm_dict['sum'][1]),
					unnormalize(ex_var, norm_dict['sum'][1]**2),
					lvl
				)
				lvl_ex_mu, _ = normalize(lvl_ex_mu, norm_dict['sum'][lvl])
				lvl_ex_var, _ = normalize(lvl_ex_var, norm_dict['sum'][lvl]**2)
			lvl_loss = self.KL(
				params_dict['sum'][lvl][0].detach().numpy(),
				params_dict['sum'][lvl][1].detach().numpy()**2,
				lvl_ex_mu, lvl_ex_var, lvl
			)
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

		return ex_mu.value


	def forward(self, inputs_dict, norm_dict):
		'''
		inputs_dict: [aggregation method][level]
		norm_dict: [aggregation method][level]
		'''

		norm_dict_np = dict()
		for agg_method in norm_dict.keys():
			norm_dict_np[agg_method] = dict()
			for lvl in norm_dict[agg_method].keys():
				norm_dict_np[agg_method][lvl] = np.expand_dims(norm_dict[agg_method][lvl].detach().numpy(), axis=0)

		params_dict = dict()
		for agg_method in self.base_models_dict.keys():
			params_dict[agg_method] = dict()
			for level in self.K_list:
				model = self.base_models_dict[agg_method][level]
				inputs = inputs_dict[agg_method][level]
				means, stds = model(inputs)

				if model.point_estimates:
					stds = torch.ones_like(means)
				params = [means, stds]
				params_dict[agg_method][level] = params

		all_preds = []
		for i in range(params_dict['sum'][1][0].size()[0]):
			print(i)
			ex_params_dict = dict()
			for agg_method in params_dict.keys():
				ex_params_dict[agg_method] = dict()
				for lvl in params_dict[agg_method].keys():
					ex_params_dict[agg_method][lvl] = [params_dict[agg_method][lvl][0][i], params_dict[agg_method][lvl][1][i]]

			#import ipdb
			#ipdb.set_trace()
			ex_preds_opt = self.optimize(ex_params_dict, norm_dict_np)
			all_preds.append(ex_preds_opt)

		all_preds = torch.FloatTensor(all_preds)

		#all_preds, _ = normalize(all_preds, norm_dict[0])

		return all_preds, None