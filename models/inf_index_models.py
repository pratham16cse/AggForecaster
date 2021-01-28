import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cvxpy as cp
from bisect import bisect_left
#import pywt

from utils import normalize, unnormalize, sqz, expand



class MSE(torch.nn.Module):
	"""docstring for MSE"""
	def __init__(self, base_models_dict):
		super(MSE, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict,
		norm_dict, inputs_gaps_dict, N_input, N_output, targets_dict=None
	):
		return self.base_models_dict[1](feats_in_dict[1], inputs_dict[1], feats_tgt_dict[1])

class NLL(torch.nn.Module):
	"""docstring for NLL"""
	def __init__(self, base_models_dict):
		super(NLL, self).__init__()
		self.base_models_dict = base_models_dict

	def forward(
		self, feats_in_dict, inputs_dict, feats_tgt_dict,
		norm_dict, inputs_gaps_dict, N_input, N_output, targets_dict=None
	):
		return self.base_models_dict[1](feats_in_dict[1], inputs_dict[1], feats_tgt_dict[1])


class OPT_st(torch.nn.Module):
	"""docstring for OPT_st"""
	def __init__(self, K_list, base_models_dict, device, disable_sum=False, intercept_type='intercept'):
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
		self.disable_sum = disable_sum
		self.device = device


	def aggregate_seq_(self, seq, indices):
		#assert seq.shape[0]%K == 0
		#agg_seq = np.array([[1./K * cp.sum(seq[i:i+K])] for i in range(0, seq.shape[0], K)])
		agg_seq = []
		prev = 0
		for i in range(len(indices[0])):
			curr = int(indices[0][i].item())
			#s = 1./(len(seq[prev:curr])) * cp.sum(seq[prev:curr])
			s = 1./(curr-prev) * cp.sum(seq[prev:curr])
			agg_seq.append([s])
			prev = curr
		agg_seq = np.array(agg_seq)
		return agg_seq

	def fit_slope_with_indices(self, seq, indices):
		W = []
		#import ipdb
		#ipdb.set_trace()
		prev = 0
		#print(indices[0][:, 0])
		for i in range(len(indices[0])):
			curr = int(indices[0][i].item())
			#print(prev, curr, len(indices[0]))
			x = np.cumsum(np.ones(seq[prev:curr].shape)) - 1.
			y = seq[prev:curr]
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
			prev = curr
		W = np.expand_dims(np.array(W), axis=1)
		return W

	def log_prob(self, ex_preds, means, std):
		#import ipdb
		#ipdb.set_trace()
		return -cp.sum(np.sum(np.log(1/(((2*np.pi)**0.5)*std)) - (((ex_preds - means)**2) / (2*(std)**2))))

	def optimize(self, params_dict, params_idx_dict, norm_dict):

		ex_preds = cp.Variable(params_dict['sum'][1][0].shape)
		for lvl, params in params_dict['slope'].items():
			if lvl==1:
				lvl_ex_preds = ex_preds
			else:
				lvl_ex_preds, _ = normalize(
					self.fit_slope_with_indices(
						unnormalize(ex_preds, norm_dict['slope'][1]),
						params_idx_dict['slope'][lvl]
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

		if not self.disable_sum:
			for lvl, params in params_dict['sum'].items():
				if lvl==1:
					lvl_ex_preds = ex_preds
				else:
					lvl_ex_preds, _ = normalize(
						self.aggregate_seq_(
							unnormalize(ex_preds, norm_dict['sum'][1]),
							params_idx_dict['sum'][lvl]
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


	def forward(
		self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, 
		inputs_gaps_dict, N_input, N_output, targets_dict=None
	):
		'''
		inputs_dict: [aggregation method][level]
		norm_dict: [aggregation method][level]
		'''

		norm_dict_np = dict()
		for agg_method in norm_dict.keys():
			norm_dict_np[agg_method] = dict()
			for lvl in norm_dict[agg_method].keys():
				norm_dict_np[agg_method][lvl] = norm_dict[agg_method][lvl].detach().numpy()

		params_dict = dict()
		params_idx_dict = dict()
		for agg_method in self.base_models_dict.keys():
			params_dict[agg_method] = dict()
			params_idx_dict[agg_method] = dict()
			if agg_method in ['slope', 'sum']:
				for level in self.K_list:
					print(agg_method, level)
					model = self.base_models_dict[agg_method][level]
					inputs = inputs_dict[agg_method][level]
					feats_in, feats_tgt = feats_in_dict[agg_method][level], feats_tgt_dict[agg_method][level]
					inputs_gaps = inputs_gaps_dict[agg_method][level]
					if level == 1:
						means, stds = model(feats_in, inputs, feats_tgt)
					else:
						hidden = model.init_hidden(inputs_gaps.shape[0], self.device)
						end_idx = np.ones((inputs.shape[0], 1, 1)) * (N_input + N_output)
						means_gaps, stds_gaps, means, stds, _ = model.simulate(
							inputs_gaps, inputs, hidden, end_idx
						)
						#means_gaps = torch.ones_like(means_gaps)*2. #TODO: Remove this line
						means_idx = torch.round(torch.cumsum(means_gaps, dim=1))
						stds_idx = stds_gaps
						#import ipdb
						#ipdb.set_trace()

					if targets_dict is not None and level != 1:
						means = targets_dict[agg_method][level]

					if model.point_estimates:
						stds = torch.ones_like(means)
						if level != 1:
							stds_idx = torch.ones_like(means_idx)
					params = [means, stds]
					params_dict[agg_method][level] = params

					if level != 1:
						params_idx = [means_idx, stds_idx]
						params_idx_dict[agg_method][level] = params_idx

		all_preds_mu = []
		all_preds_std = []
		for i in range(params_dict['sum'][1][0].size()[0]):
			#print(i)
			ex_params_dict = dict()
			ex_params_idx_dict = dict()
			ex_norm_dict = dict()
			for agg_method in params_dict.keys():
				ex_params_dict[agg_method] = dict()
				ex_params_idx_dict[agg_method] = dict()
				ex_norm_dict[agg_method] = dict()
				for lvl in params_dict[agg_method].keys():
					if lvl != 1:
						# Discard all-but-first indices greater than end_idx
						stp = bisect_left(
							params_idx_dict[agg_method][lvl][0][i][:, 0].detach().numpy(),
							N_output
						) + 1

						#print(stp)
						ex_params_idx_dict[agg_method][lvl] = [
							params_idx_dict[agg_method][lvl][0][i][:stp],
							params_idx_dict[agg_method][lvl][1][i][:stp]
						]
					else:
						stp = len(params_dict[agg_method][lvl][0][i])

					ex_params_dict[agg_method][lvl] = [
						params_dict[agg_method][lvl][0][i][:stp],
						params_dict[agg_method][lvl][1][i][:stp]
					]
					ex_norm_dict[agg_method][lvl] = norm_dict_np[agg_method][lvl][i]

			#import ipdb
			#ipdb.set_trace()
			ex_preds_opt = self.optimize(ex_params_dict, ex_params_idx_dict, ex_norm_dict)
			all_preds_mu.append(ex_preds_opt)
			all_preds_std.append(params_dict['sum'][1][1][i])

		all_preds_mu = torch.FloatTensor(all_preds_mu)
		all_preds_std = torch.stack(all_preds_std)

		#all_preds, _ = normalize(all_preds, norm_dict[0])

		return all_preds_mu, all_preds_std