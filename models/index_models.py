import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.distributions.normal import Normal



class IndexModel(nn.Module):
	"""docstring for IndexModel"""
	def __init__(
		self,input_size, output_size, hidden_size, num_grulstm_layers, fc_units,
		point_estimates
	):
		super(IndexModel, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.num_grulstm_layers = num_grulstm_layers

		self.point_estimates = point_estimates

		self.gru_gaps = nn.GRU(
			input_size=1,
			hidden_size=hidden_size,
			num_layers=num_grulstm_layers,
			batch_first=True
		)
		self.gru_values = nn.GRU(
			input_size=1,
			hidden_size=hidden_size,
			num_layers=num_grulstm_layers,
			batch_first=True
		)
		self.fc_gaps = nn.Linear(hidden_size, fc_units)
		self.fc_values = nn.Linear(hidden_size, fc_units)
		self.out_mean_gaps = nn.Linear(fc_units, 1)
		self.out_std_gaps = nn.Linear(fc_units, 1)
		self.out_mean_values = nn.Linear(fc_units, 1)
		self.out_std_values = nn.Linear(fc_units, 1)

	def init_hidden(self, batch_size, device):
		#[num_layers*num_directions,batch,hidden_size]   
		return (
			torch.zeros(self.num_grulstm_layers, batch_size, self.hidden_size, device=device),
			torch.zeros(self.num_grulstm_layers, batch_size, self.hidden_size, device=device)
		)


	def forward(self, gaps, values, hidden, verbose=False):

		inputs = torch.cat((gaps, values), dim=-1)
		hidden_gaps, hidden_values = hidden

		output_gaps, hidden_gaps = self.gru_gaps(gaps, hidden_gaps)
		output_values, hidden_values = self.gru_values(values, hidden_values)
		output_gaps = self.fc_gaps(output_gaps)
		output_values = self.fc_values(output_values)

		means_gaps = F.softplus(self.out_mean_gaps(output_gaps)) + 1e-3 # gaps must be positive
		means_values = self.out_mean_values(output_values)
		stds_gaps = F.softplus(self.out_std_gaps(output_gaps)) + 1e-3
		stds_values = F.softplus(self.out_std_values(output_values)) + 1e-3
		#stds = F.softplus(stds) + 1e-3
		#means_gaps, means_values = means[:, :, 0:1], means[:, :, 1:]
		#means_gaps = 2. + F.softplus(means_gaps) + 1e-3 # gaps must be positive
		#stds_gaps, stds_values = stds[:, :, 0:1], stds[:, :, 1:]
		#import ipdb
		#ipdb.set_trace()
		hidden = (hidden_gaps, hidden_values)
		if self.point_estimates:
			stds_gaps, stds_values = None, None
		return means_gaps, stds_gaps, means_values, stds_values, hidden

	def simulate(self, gaps, values, hidden, end_idx):

		means_gaps, means_values, stds_gaps, stds_values = [], [], [], []
		if self.point_estimates:
			stds_gaps, stds_values = None, None
		means_gaps_t, stds_gaps_t, means_values_t, stds_values_t, hidden_t = self.forward(gaps, values, hidden)

		means_gaps_t = means_gaps_t[:, -1:]
		means_values_t = means_values_t[:, -1:]
		means_gaps.append(means_gaps_t)
		means_values.append(means_values_t)
		if not self.point_estimates:
			stds_gaps_t = stds_gaps_t[:, -1:]
			stds_values_t = stds_values_t[:, -1:]
			stds_gaps.append(stds_gaps_t)
			stds_values.append(stds_values_t)
		pred_idx = torch.sum(gaps, dim=1, keepdim=True).detach().numpy() + means_gaps_t.detach().numpy()
		while any(pred_idx < end_idx):
			#print(means_gaps_t)
			(
				means_gaps_t, stds_gaps_t, means_values_t, stds_values_t, hidden_t
			) = self.forward(means_gaps_t, means_values_t, hidden_t)

			means_gaps.append(means_gaps_t)
			means_values.append(means_values_t)
			if not self.point_estimates:
				stds_gaps.append(stds_gaps_t)
				stds_values.append(stds_values_t)
			pred_idx += means_gaps_t.detach().numpy()

		means_gaps = torch.cat(means_gaps, dim=1)
		means_values = torch.cat(means_values, dim=1)
		if not self.point_estimates:
			stds_gaps = torch.cat(stds_gaps, dim=1)
			stds_values = torch.cat(stds_values, dim=1)

		return means_gaps, stds_gaps, means_values, stds_values, hidden_t


def get_index_model(
	args, config, level,
	N_input, N_output, input_size, output_size,
	point_estimates
):
	idx_model = IndexModel(
		2, #input_size=2,
		2, #output_size=2,
		args.hidden_size,
		args.num_grulstm_layers, args.fc_units,
		point_estimates
	)

	return idx_model