from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from collections import OrderedDict
import pywt

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ECG5000, parse_Traffic, parse_Taxi, parse_Traffic911, parse_gc_datasets


def add_metrics_to_dict(
	metrics_dict, model_name, metric_mse, metric_dtw, metric_tdi, metric_crps
):
	if model_name not in metrics_dict:
		metrics_dict[model_name] = dict()

	metrics_dict[model_name]['mse'] = metric_mse
	metrics_dict[model_name]['dtw'] = metric_dtw
	metrics_dict[model_name]['tdi'] = metric_tdi
	metrics_dict[model_name]['crps'] = metric_crps

	return metrics_dict

def write_arr_to_file(output_dir, inf_model_name, inputs, targets, preds):

	# Files are saved in .npy format
	np.save(os.path.join(output_dir, inf_model_name + '_' + 'preds'), preds)

	for fname in os.listdir(output_dir):
	    if fname.endswith('targets.npy'):
	        break
	else:
		np.save(os.path.join(output_dir, 'inputs'), inputs)
		np.save(os.path.join(output_dir, 'targets'), targets)

def normalize(data, norm=None):
	if norm is None:
		norm = np.mean(data, axis=(0, 1))

	data_norm = data * 1.0/norm 
	return data_norm, norm

def unnormalize(data, norm):
	return data * norm

sqz = lambda x: np.squeeze(x, axis=-1)
expand = lambda x: np.expand_dims(x, axis=-1)

def fit_slope_with_indices(seq, K):
    x = np.reshape(np.ones_like(seq), (-1, K))
    x = np.cumsum(x, axis=1) - 1
    y = np.reshape(seq, (-1, K))
    m_x = np.mean(x, axis=1, keepdims=True)
    m_y = np.mean(y, axis=1, keepdims=True)
    s_xy = np.sum((x-m_x)*(y-m_y), axis=1, keepdims=True)
    s_xx = np.sum((x-m_x)**2, axis=1, keepdims=True)
    w = s_xy/s_xx
    return w

def fit_sum_and_trend_with_indices(seq, K):
    x = np.reshape(np.ones_like(seq), (-1, K))
    x = np.cumsum(x, axis=1) - 1
    y = np.reshape(seq, (-1, K))
    m_x = np.mean(x, axis=1, keepdims=True)
    m_y = np.mean(y, axis=1, keepdims=True)
    s_xy = np.sum((x-m_x)*(y-m_y), axis=1, keepdims=True)
    s_xx = np.sum((x-m_x)**2, axis=1, keepdims=True)
    w = s_xy/s_xx
    b = np.sum(y, axis=1, keepdims=True)

    agg_seq = np.concatenate((w, b), axis=1)
    return agg_seq

def fit_with_indices(seq, K):
    x = np.reshape(np.ones_like(seq), (-1, K))
    x = np.cumsum(x, axis=1) - 1
    y = np.reshape(seq, (-1, K))
    m_x = np.mean(x, axis=1, keepdims=True)
    m_y = np.mean(y, axis=1, keepdims=True)
    s_xy = np.sum((x-m_x)*(y-m_y), axis=1, keepdims=True)
    s_xx = np.sum((x-m_x)**2, axis=1, keepdims=True)
    w = s_xy/s_xx
    b = m_y - w*m_x

    agg_seq = np.concatenate((w, b), axis=1)
    return agg_seq

def aggregate_data_wavelet(
	wavelet_levels, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):

	agg_train_input = pywt.wavedec(sqz(train_input), 'haar', level=wavelet_levels, mode='periodic')
	agg_train_target = pywt.wavedec(sqz(train_target), 'haar', level=wavelet_levels, mode='periodic')
	agg_dev_input = pywt.wavedec(sqz(dev_input), 'haar', level=wavelet_levels, mode='periodic')
	agg_dev_target = pywt.wavedec(sqz(dev_target), 'haar', level=wavelet_levels, mode='periodic')
	agg_test_input = pywt.wavedec(sqz(test_input), 'haar', level=wavelet_levels, mode='periodic')
	agg_test_target = pywt.wavedec(sqz(test_target), 'haar', level=wavelet_levels, mode='periodic')

	agg_train_input = [expand(x) for x in agg_train_input]
	agg_train_target = [expand(x) for x in agg_train_target]
	agg_dev_input = [expand(x) for x in agg_dev_input]
	agg_dev_target = [expand(x) for x in agg_dev_target]
	agg_test_input = [expand(x) for x in agg_test_input]
	agg_test_target = [expand(x) for x in agg_test_target]

	#import ipdb
	#ipdb.set_trace()

	return (
		agg_train_input, agg_train_target, agg_dev_input, agg_dev_target,
		agg_test_input, agg_test_target
	)

def aggregate_data_slope(
	K, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):
	def aggregate_seqs_(seqs):
		agg_seqs = []
		for seq in seqs:
			assert len(seq)%K == 0
			agg_seq = fit_slope_with_indices(seq, K)
			agg_seqs.append(agg_seq)
		return np.array(agg_seqs)

	agg_train_input = aggregate_seqs_(train_input)
	agg_train_target = aggregate_seqs_(train_target)
	agg_dev_input = aggregate_seqs_(dev_input)
	agg_dev_target = aggregate_seqs_(dev_target)
	agg_test_input = aggregate_seqs_(test_input)
	agg_test_target = aggregate_seqs_(test_target)

	return (
		agg_train_input, agg_train_target, agg_dev_input, agg_dev_target,
		agg_test_input, agg_test_target
	)

def aggregate_data_sumwithtrend(
	K, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):
	def aggregate_seqs_(seqs):
		agg_seqs = []
		for seq in seqs:
			assert len(seq)%K == 0
			agg_seq = fit_sum_and_trend_with_indices(seq, K)
			agg_seqs.append(agg_seq)
		return np.array(agg_seqs)

	agg_train_input = aggregate_seqs_(train_input)
	agg_train_target = aggregate_seqs_(train_target)
	agg_dev_input = aggregate_seqs_(dev_input)
	agg_dev_target = aggregate_seqs_(dev_target)
	agg_test_input = aggregate_seqs_(test_input)
	agg_test_target = aggregate_seqs_(test_target)

	return (
		agg_train_input, agg_train_target, agg_dev_input, agg_dev_target,
		agg_test_input, agg_test_target
	)

def aggregate_data_leastsquare(
	K, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):
	def aggregate_seqs_(seqs):
		agg_seqs = []
		for seq in seqs:
			assert len(seq)%K == 0
			agg_seq = fit_with_indices(seq, K)
			agg_seqs.append(agg_seq)
		return np.array(agg_seqs)

	agg_train_input = aggregate_seqs_(train_input)
	agg_train_target = aggregate_seqs_(train_target)
	agg_dev_input = aggregate_seqs_(dev_input)
	agg_dev_target = aggregate_seqs_(dev_target)
	agg_test_input = aggregate_seqs_(test_input)
	agg_test_target = aggregate_seqs_(test_target)

	return (
		agg_train_input, agg_train_target, agg_dev_input, agg_dev_target,
		agg_test_input, agg_test_target
	)

def aggregate_data(
	K, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):

	def aggregate_seqs_(seqs):
		agg_seqs = []
		for seq in seqs:
			assert len(seq)%K == 0
			agg_seq = [np.sum(seq[i:i+K], axis=0) for i in range(0, len(seq), K)]
			agg_seqs.append(agg_seq)
		return np.array(agg_seqs)

	agg_train_input = aggregate_seqs_(train_input)
	agg_train_target = aggregate_seqs_(train_target)
	agg_dev_input = aggregate_seqs_(dev_input)
	agg_dev_target = aggregate_seqs_(dev_target)
	agg_test_input = aggregate_seqs_(test_input)
	agg_test_target = aggregate_seqs_(test_target)

	return (
		agg_train_input, agg_train_target, agg_dev_input, agg_dev_target,
		agg_test_input, agg_test_target
	)

def create_hierarchical_wavelet_data(
	args, train_input, train_target, dev_input, dev_target,
	test_input, test_target, train_bkp, dev_bkp, test_bkp,
	aggregation_type
):
	(
		train_input_coeffs, train_target_coeffs, dev_input_coeffs, dev_target_coeffs,
		test_input_coeffs, test_target_coeffs,
	)= aggregate_data_wavelet(
		args.wavelet_levels, train_input, train_target,
		dev_input, dev_target, test_input, test_target,
	)

	K2data = OrderedDict()
	# +1 : wavedec returns args.wavelet_levels+1 coefficients
	# +1 : Extra slot for base values
	# +1 : Because starting index is 1.
	for K in range(1, args.wavelet_levels+1+1+1):
		if K == 1:
			train_input_agg, train_target_agg = train_input, train_target
			dev_input_agg, dev_target_agg = dev_input, dev_target
			test_input_agg, test_target_agg = test_input, test_target
		else:
			train_input_agg, train_target_agg = train_input_coeffs[K-2], train_target_coeffs[K-2]
			dev_input_agg, dev_target_agg = dev_input_coeffs[K-2], dev_target_coeffs[K-2]
			test_input_agg, test_target_agg = test_input_coeffs[K-2], test_target_coeffs[K-2]

		if args.normalize:
			train_input_norm, norm = normalize(train_input_agg)
		else:
			train_input_norm = train_input_agg
			norm = np.ones_like(np.mean(train_input_agg, axis=(0, 1)))
		train_target_norm, _ = normalize(train_target_agg, norm)
		dev_input_norm, _ = normalize(dev_input_agg, norm)
		dev_target_norm = dev_target_agg
		test_input_norm, _ = normalize(test_input_agg, norm)
		test_target_norm = test_target_agg

		dataset_train = SyntheticDataset(train_input_norm, train_target_norm, train_bkp)
		dataset_dev = SyntheticDataset(dev_input_norm, dev_target_norm, dev_bkp)
		dataset_test  = SyntheticDataset(test_input_norm, test_target_norm, test_bkp)
		trainloader = DataLoader(
			dataset_train, batch_size=args.batch_size, shuffle=True,
			drop_last=True, num_workers=1
		)
		devloader = DataLoader(
			dataset_dev, batch_size=dev_input.shape[0], shuffle=False,
			drop_last=False, num_workers=1
		)
		testloader  = DataLoader(
			dataset_test, batch_size=test_input.shape[0], shuffle=False,
			drop_last=False, num_workers=1
		)
		norm = torch.FloatTensor(norm)
		K2data[K] = {
			'trainloader': trainloader,
			'devloader': devloader,
			'testloader': testloader,
			'N_output': test_target_norm.shape[1],
			'input_size': test_input_norm.shape[2],
			'output_size': test_target_norm.shape[2],
			'norm': norm
		}

	return K2data

def create_hierarchical_data(
	args, train_input, train_target, dev_input, dev_target,
	test_input, test_target, train_bkp, dev_bkp, test_bkp,
	aggregation_type
):

	aggregation2func = {}
	aggregation2func['sum'] = aggregate_data
	aggregation2func['leastsquare'] = aggregate_data_leastsquare
	aggregation2func['sumwithtrend'] = aggregate_data_sumwithtrend
	aggregation2func['slope'] = aggregate_data_slope

	aggregation_func = aggregation2func[aggregation_type]

	K2data = OrderedDict()
	for K in args.K_list:
		if K == 1:
			train_input_agg, train_target_agg = train_input, train_target
			dev_input_agg, dev_target_agg = dev_input, dev_target
			test_input_agg, test_target_agg = test_input, test_target
		else:
			(
				train_input_agg, train_target_agg, dev_input_agg, dev_target_agg,
				test_input_agg, test_target_agg,
			)= aggregation_func(
				K, train_input, train_target,
				dev_input, dev_target, test_input, test_target,
			)

		if args.normalize:
			train_input_norm, norm = normalize(train_input_agg)
		else:
			train_input_norm = train_input_agg
			norm = np.ones_like(np.mean(train_input_agg, axis=(0, 1)))
		train_target_norm, _ = normalize(train_target_agg, norm)
		dev_input_norm, _ = normalize(dev_input_agg, norm)
		dev_target_norm = dev_target_agg
		test_input_norm, _ = normalize(test_input_agg, norm)
		test_target_norm = test_target_agg

		dataset_train = SyntheticDataset(train_input_norm, train_target_norm, train_bkp)
		dataset_dev = SyntheticDataset(dev_input_norm, dev_target_norm, dev_bkp)
		dataset_test  = SyntheticDataset(test_input_norm, test_target_norm, test_bkp)
		trainloader = DataLoader(
			dataset_train, batch_size=args.batch_size, shuffle=True,
			drop_last=True, num_workers=1
		)
		devloader = DataLoader(
			dataset_dev, batch_size=dev_input.shape[0], shuffle=False,
			drop_last=False, num_workers=1
		)
		testloader  = DataLoader(
			dataset_test, batch_size=test_input.shape[0], shuffle=False,
			drop_last=False, num_workers=1
		)
		norm = torch.FloatTensor(norm)
		K2data[K] = {
			'trainloader': trainloader,
			'devloader': devloader,
			'testloader': testloader,
			'N_output': test_target_norm.shape[1],
			'input_size': test_input_norm.shape[2],
			'output_size': test_target_norm.shape[2],
			'norm': norm
		}

	return K2data


def get_processed_data(args):

	if args.dataset_name in ['synth']:
		# parameters
		N = 500
		sigma = 0.01

		# Load synthetic dataset
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = create_synthetic_dataset(N, args.N_input, args.N_output, sigma)

	elif args.dataset_name in ['sin']:
		N = 500
		sigma = 0.01

		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = create_sin_dataset(N, args.N_input, args.N_output, sigma)

	elif args.dataset_name in ['ECG5000']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = parse_ECG5000(args.N_input, args.N_output)

	elif args.dataset_name in ['Traffic']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = parse_Traffic(args.N_input, args.N_output)

	elif args.dataset_name in ['Taxi']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = parse_Taxi(args.N_input, args.N_output)

	elif args.dataset_name in ['Traffic911']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = parse_Traffic911(args.N_input, args.N_output)
	elif args.dataset_name in ['Exchange', 'Solar', 'Wiki']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
		) = parse_gc_datasets(args.dataset_name, args.N_input, args.N_output)


	K2data_sum = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		aggregation_type='sum'
	)
	print('sum done')
	K2data_ls = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		aggregation_type='leastsquare'
	)
	print('ls done')
	K2data_st = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		aggregation_type='sumwithtrend'
	)
	print('sumwithtrend done')
	K2data_slope = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		aggregation_type='slope'
	)
	print('slope done')
	K2data_wavelet = create_hierarchical_wavelet_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		aggregation_type='wavelet'
	)
	print('wavelet done')

	dataset = dict()
	dataset['sum'] = K2data_sum
	dataset['leastsquare'] = K2data_ls
	dataset['sumwithtrend'] = K2data_st
	dataset['slope'] = K2data_slope
	dataset['wavelet'] = K2data_wavelet

	return dataset
	#return {
		#'trainloader': trainloader,
		#'testloader': testloader,
		#'K2data_sum': K2data_sum,
		#'K2data_ls': K2data_ls,
		#'K2data': K2data
	#}