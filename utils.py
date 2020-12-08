from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from collections import OrderedDict
import pywt
import pandas as pd
import re
import time

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ECG5000, parse_Traffic, parse_Taxi, parse_Traffic911, parse_gc_datasets


def add_metrics_to_dict(
	metrics_dict, model_name, metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae
):
	if model_name not in metrics_dict:
		metrics_dict[model_name] = dict()

	metrics_dict[model_name]['mse'] = metric_mse
	metrics_dict[model_name]['dtw'] = metric_dtw
	metrics_dict[model_name]['tdi'] = metric_tdi
	metrics_dict[model_name]['crps'] = metric_crps
	metrics_dict[model_name]['mae'] = metric_mae

	return metrics_dict

def write_arr_to_file(output_dir, inf_model_name, inputs, targets, pred_mu, pred_std):

	# Files are saved in .npy format
	np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_mu'), pred_mu)
	np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_std'), pred_std)

	for fname in os.listdir(output_dir):
	    if fname.endswith('targets.npy'):
	        break
	else:
		np.save(os.path.join(output_dir, 'inputs'), inputs)
		np.save(os.path.join(output_dir, 'targets'), targets)

def write_aggregate_preds_to_file(
	output_dir, base_model_name, agg_method, level, inputs, targets, pred_mu, pred_std
):

	# Files are saved in .npy format
	sep = '__'
	model_str = base_model_name + sep + agg_method + sep  + str(level)
	agg_str = agg_method + sep  + str(level)

	np.save(os.path.join(output_dir, model_str + sep + 'pred_mu'), pred_mu)
	np.save(os.path.join(output_dir, model_str + sep + 'pred_std'), pred_std)

	suffix = agg_str + sep + 'targets.npy'
	for fname in os.listdir(output_dir):
	    if fname.endswith(suffix):
	        break
	else:
		np.save(os.path.join(output_dir, agg_str + sep + 'inputs'), inputs)
		np.save(os.path.join(output_dir, agg_str + sep + 'targets'), targets)

def normalize(data, norm=None):
	if norm is None:
		norm = np.mean(data, axis=(0, 1))
		#norm = np.quantile(data, axis=(0, 1), 0.90)

	data_norm = data * 1.0/norm 
	#data_norm = data * 10.0/norm
	return data_norm, norm

def unnormalize(data, norm):
	return data * norm

sqz = lambda x: np.squeeze(x, axis=-1)
expand = lambda x: np.expand_dims(x, axis=-1)

def shift_timestamp(ts, offset):
	result = ts + offset * ts.freq
	return pd.Timestamp(result, freq=ts.freq)

def get_date_range(start, seq_len):
	end = shift_timestamp(start, seq_len)
	full_date_range = pd.date_range(start, end, freq=start.freq)
	return full_date_range

def get_granularity(freq_str: str):
    """
    Splits a frequency string such as "7D" into the multiple 7 and the base
    granularity "D".

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    freq_regex = r'\s*((\d+)?)\s*([^\d]\w*)'
    m = re.match(freq_regex, freq_str)
    assert m is not None, "Cannot parse frequency string: %s" % freq_str
    groups = m.groups()
    multiple = int(groups[1]) if groups[1] is not None else 1
    granularity = groups[2]
    return multiple, granularity

class TimeFeature:
    """
    Base class for features that only depend on time.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'

class FourrierDateFeatures(TimeFeature):
    def __init__(self, freq: str) -> None:
        # reocurring freq
        freqs = [
            'month',
            'day',
            'hour',
            'minute',
            'weekofyear',
            'weekday',
            'dayofweek',
            'dayofyear',
            'daysinmonth',
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        #return np.vstack([np.cos(steps), np.sin(steps)])
        return np.stack([np.cos(steps), np.sin(steps)], axis=-1)

def time_features_from_frequency_str(freq_str):
    multiple, granularity = get_granularity(freq_str)

    features = {
        'M': ['weekofyear'],
        'W': ['daysinmonth', 'weekofyear'],
        'D': ['dayofweek'],
        'B': ['dayofweek', 'dayofyear'],
        'H': ['hour', 'dayofweek'],
        'min': ['minute', 'hour', 'dayofweek'],
        'T': ['minute', 'hour', 'dayofweek'],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes= [
        FourrierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes

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
		for i, seq in enumerate(seqs):
			print(i, len(seqs))
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
	data_train, data_dev, data_test,
	aggregation_type
):

	#aggregation2func = {}
	#aggregation2func['sum'] = aggregate_data
	#aggregation2func['leastsquare'] = aggregate_data_leastsquare
	#aggregation2func['sumwithtrend'] = aggregate_data_sumwithtrend
	#aggregation2func['slope'] = aggregate_data_slope
	#aggregation2func['wavelet'] = aggregate_data_wavelet

	#aggregation_func = aggregation2func[aggregation_type]

	if aggregation_type in ['wavelet']:
		wavelet_levels = args.wavelet_levels
		K_list = range(1, args.wavelet_levels+1+1+1)
		# +1 : wavedec returns args.wavelet_levels+1 coefficients
		# +1 : Extra slot for base values
		# +1 : Because starting index is 1.
	else:
		wavelet_levels = None
		K_list = args.K_list

	K2data = OrderedDict()
	for K in K_list:
		#if K == 1:
		#	train_input_agg, train_target_agg = train_input, train_target
		#	dev_input_agg, dev_target_agg = dev_input, dev_target
		#	test_input_agg, test_target_agg = test_input, test_target
		#else:
		#	(
		#		train_input_agg, train_target_agg, dev_input_agg, dev_target_agg,
		#		test_input_agg, test_target_agg,
		#	)= aggregation_func(
		#		K, train_input, train_target,
		#		dev_input, dev_target, test_input, test_target,
		#	)

		#if args.normalize:
		#	train_input_norm, norm = normalize(train_input_agg)
		#	_, norm = normalize(np.array(data_train))
		#else:
		#	train_input_norm = train_input_agg
		#	norm = np.ones_like(np.mean(train_input_agg, axis=(0, 1)))
		#train_target_norm, _ = normalize(train_target_agg, norm)
		#dev_input_norm, _ = normalize(dev_input_agg, norm)
		#dev_target_norm = dev_target_agg
		#test_input_norm, _ = normalize(test_input_agg, norm)
		#test_target_norm = test_target_agg

		#dataset_train = SyntheticDataset(train_input_norm, train_target_norm, train_bkp)
		#dataset_dev = SyntheticDataset(dev_input_norm, dev_target_norm, dev_bkp)
		#dataset_test  = SyntheticDataset(test_input_norm, test_target_norm, test_bkp)

		#trainloader = DataLoader(
		#	dataset_train, batch_size=args.batch_size, shuffle=True,
		#	drop_last=True, num_workers=1
		#)
		#devloader = DataLoader(
		#	dataset_dev, batch_size=dev_input.shape[0], shuffle=False,
		#	drop_last=False, num_workers=1
		#)
		#testloader  = DataLoader(
		#	dataset_test, batch_size=test_input.shape[0], shuffle=False,
		#	drop_last=False, num_workers=1
		#)

		_, norm = normalize(np.array([[s for seq in data_train for s in seq['target']]]))
		if not args.normalize:
			norm = np.ones_like(norm)

		lazy_dataset_train = TimeSeriesDataset(
			data_train, args.N_input, args.N_output, int(args.N_output/3),
			aggregation_type, K,
			input_norm=norm, target_norm=norm,
			use_time_features=args.use_time_features,
			wavelet_levels=wavelet_levels
		)
		lazy_dataset_dev = TimeSeriesDataset(
			data_dev, args.N_input, args.N_output, args.N_output,
			aggregation_type, K,
			input_norm=norm, target_norm=np.ones_like(norm),
			use_time_features=args.use_time_features,
			wavelet_levels=wavelet_levels
		)
		lazy_dataset_test = TimeSeriesDataset(
			data_test, args.N_input, args.N_output, args.N_output,
			aggregation_type, K,
			input_norm=norm, target_norm=np.ones_like(norm),
			use_time_features=args.use_time_features,
			wavelet_levels=wavelet_levels
		)
		trainloader = DataLoader(
			lazy_dataset_train, batch_size=args.batch_size, shuffle=True,
			drop_last=True, num_workers=2
		)
		devloader = DataLoader(
			lazy_dataset_dev, batch_size=len(lazy_dataset_dev), shuffle=False,
			drop_last=True, num_workers=1
		)
		testloader = DataLoader(
			lazy_dataset_test, batch_size=len(lazy_dataset_test), shuffle=False,
			drop_last=True, num_workers=1
		)
		#for i, b in enumerate(trainloader):
		#	print(b[0].shape, b[1].shape)
		#	if i>10: break
		#import ipdb
		#ipdb.set_trace()

		#for i, d in enumerate(lazy_dataset_train):
		#	print(d[0].shape, d[1].shape)
		#	if i>10:
		#		break
		#import ipdb
		#ipdb.set_trace()
		norm = torch.FloatTensor(norm)
		K2data[K] = {
			'trainloader': trainloader,
			'devloader': devloader,
			'testloader': testloader,
			'N_output': lazy_dataset_test.dec_len,
			'input_size': lazy_dataset_test.input_size,
			'output_size': lazy_dataset_test.output_size,
			'norm': norm
		}

	return K2data

class TimeSeriesDataset(torch.utils.data.Dataset):
	"""docstring for TimeSeriesDataset"""
	def __init__(
		self, data, enc_len, dec_len, stride, aggregation_type, K,
		input_norm, target_norm, use_time_features, wavelet_levels=None
	):
		super(TimeSeriesDataset, self).__init__()

		if aggregation_type not in ['wavelet']:
			assert enc_len%K == 0
			assert dec_len%K == 0

		self.data = data
		self._enc_len = enc_len
		self._dec_len = dec_len
		#self.num_values = len(data[0]['target'][0])
		self.stride = stride
		self.aggregation_type = aggregation_type
		self.K = K
		self.input_norm = input_norm
		self.target_norm = target_norm
		self.use_time_features = use_time_features
		self.wavelet_levels = wavelet_levels

		self.indices = []
		for i in range(0, len(data)):
			for j in range(0, len(data[i]['target']), stride):
				if j+self._enc_len+self._dec_len <= len(data[i]['target']):
					self.indices.append((i, j))

		if self.use_time_features:
			multiple, granularity = get_granularity(data[0]['freq_str'])
			freq_str = str(self.K * multiple) + granularity
			self.time_features_obj = time_features_from_frequency_str(granularity)

			self.date_range = []
			for i in range(0, len(data)):
				self.date_range.append(
					get_date_range(
						self.data[i]['start'],
						len(self.data[i]['target'])
					)
				)

	@property
	def enc_len(self):
		if self.wavelet_levels is not None:

			if self.K == self.wavelet_levels+2:
				return self._enc_len // 2**(self.K-2)

			return self._enc_len // 2**(self.K-1)

		return self._enc_len // self.K
	
	@property
	def dec_len(self):
		if self.wavelet_levels is not None:

			if self.K == self.wavelet_levels+2:
				return self._dec_len // 2**(self.K-2)

			return self._dec_len // 2**(self.K-1)

		return self._dec_len // self.K

	@property
	def input_size(self):
		input_size = len(self.data[0]['target'][0])
		if self.use_time_features:
			# Multiplied by 2 because of sin and cos
			input_size += (2 * len(self.time_features_obj))
		return input_size

	@property
	def output_size(self):
		output_size = len(self.data[0]['target'][0])
		return output_size
	

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		#print(self.indices)
		ts_id = self.indices[idx][0]
		pos_id = self.indices[idx][1]	

		ex_input = np.array(
			self.data[ts_id]['target'][ pos_id : pos_id+self._enc_len ]
		)
		ex_target = np.array(
			self.data[ts_id]['target'][ pos_id+self._enc_len : pos_id+self._enc_len+self._dec_len ]
		)

		if self.K != 1:

			#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
			if self.aggregation_type in ['sum']:
				#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
				ex_input_agg = map(
					self.aggregate_data,
					np.split(ex_input, np.arange(self.K, self._enc_len, self.K), axis=0),
				)
				ex_target_agg = map(
					self.aggregate_data,
					np.split(ex_target, np.arange(self.K, self._dec_len, self.K), axis=0)
				)
				#print('lengths of aggs:', len(list(ex_agg)[0]), len(list(ex_agg)[1]))
				ex_input_agg, ex_target_agg = list(ex_input_agg), list(ex_target_agg)

			elif self.aggregation_type in ['slope']:
				#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
				ex_input_agg = map(
					self.aggregate_data_slope,
					np.split(ex_input, np.arange(self.K, self._enc_len, self.K), axis=0),
				)
				ex_target_agg = map(
					self.aggregate_data_slope,
					np.split(ex_target, np.arange(self.K, self._dec_len, self.K), axis=0)
				)
				ex_input_agg, ex_target_agg = list(ex_input_agg), list(ex_target_agg)

			elif self.aggregation_type in ['wavelet']:
				ex_input_agg = self.aggregate_data_wavelet(ex_input, self.K)
				ex_target_agg = self.aggregate_data_wavelet(ex_target, self.K)


			#ex_input, ex_target = zip(*ex_agg)
			ex_input = torch.FloatTensor(ex_input_agg)
			ex_target = torch.FloatTensor(ex_target_agg)

		#print('after', ex_input.shape, ex_target.shape, ts_id, pos_id)
		ex_input, _ = normalize(ex_input, self.input_norm)
		ex_target, _ = normalize(ex_target, self.target_norm)

		if self.use_time_features:
			#st = time.time()
			#date_range = get_date_range(
			#	self.data[ts_id]['start'],
			#	len(self.data[ts_id]['target'])
			#)
			#et = time.time()
			#print('date range time', et-st)
			#ex_input_dates = date_range[ pos_id : pos_id+self._enc_len ]
			#ex_target_dates = date_range[ pos_id+self._enc_len : pos_id+self._enc_len+self._dec_len ]
			ex_input_dates = self.date_range[ts_id][ pos_id : pos_id+self._enc_len ]
			ex_target_dates = self.date_range[ts_id][ pos_id+self._enc_len : pos_id+self._enc_len+self._dec_len ]

			#st = time.time()
			ex_input_feats = np.concatenate(
				[feat(ex_input_dates) for feat in self.time_features_obj],
				axis=1
			)
			ex_target_feats = np.concatenate(
				[feat(ex_target_dates) for feat in self.time_features_obj],
				axis=1
			)
			#et = time.time()
			#print('feat creation time', et-st)
			if self.K != 1:
				#st = time.time()
				ex_input_feats = map(
					self.get_avg_feats,
					np.split(ex_input_feats, np.arange(self.K, self._enc_len, self.K), axis=0)
				)
				ex_target_feats = map(
					self.get_avg_feats,
					np.split(ex_target_feats, np.arange(self.K, self._dec_len, self.K), axis=0)
				)
				#et = time.time()
				#print('feat avg time', et-st)
				ex_input_feats, ex_target_feats = list(ex_input_feats), list(ex_target_feats)
				ex_input_feats = torch.FloatTensor(ex_input_feats)
				ex_target_feats = torch.FloatTensor(ex_target_feats)
		else:
			ex_input_feats, ex_target_feats = ex_input, ex_target


		return (
			ex_input, ex_target,
			ex_input_feats, ex_target_feats,
			torch.FloatTensor([ts_id, pos_id])
		)

	def aggregate_data(self, values):
		return np.sum(values, axis=0)

	def aggregate_data_slope(self, values):
		x = np.expand_dims(np.arange(self.K), axis=1)
		m_x = np.mean(x, axis=0)
		s_xx = np.sum((x-m_x)**2, axis=0)

		y = values
		m_y = np.mean(y, axis=0)
		s_xy = np.sum((x-m_x)*(y-m_y), axis=0)
		w = s_xy/s_xx

		return w

	def aggregate_data_wavelet(self, values, K):
		coeffs = pywt.wavedec(sqz(values), 'haar', level=self.wavelet_levels, mode='periodic')
		coeffs = [expand(x) for x in coeffs]
		coeffs = coeffs[-(K-1)]
		return coeffs

	def get_time_features(self, start, seqlen):
		end = shift_timestamp(start, seqlen)
		full_date_range = pd.date_range(start, end, freq=start.freq)
		chunk_range = full_date_range[ pos_id : pos_id+self._enc_len ]

	def get_avg_date(self, date_range):
		return date_range.mean(axis=0)

	def get_avg_feats(self, time_feats):
		return np.mean(time_feats, axis=0)



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
			data_train, data_dev, data_test,
		) = create_sin_dataset(N, args.N_input, args.N_output, sigma)

	elif args.dataset_name in ['ECG5000']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
			data_train, data_dev, data_test
		) = parse_ECG5000(args.N_input, args.N_output)

	elif args.dataset_name in ['Traffic']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
			data_train, data_dev, data_test
		) = parse_Traffic(args.N_input, args.N_output)

	elif args.dataset_name in ['Taxi']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
			data_train, data_dev, data_test
		) = parse_Taxi(args.N_input, args.N_output)

	elif args.dataset_name in ['Traffic911']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
			data_train, data_dev, data_test
		) = parse_Traffic911(args.N_input, args.N_output)
	elif args.dataset_name in ['Exchange', 'Solar', 'Wiki']:
		(
			X_train_input, X_train_target,
			X_dev_input, X_dev_target,
			X_test_input, X_test_target,
			train_bkp, dev_bkp, test_bkp,
			data_train, data_dev, data_test
		) = parse_gc_datasets(args.dataset_name, args.N_input, args.N_output)


	K2data_sum = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		data_train, data_dev, data_test,
		aggregation_type='sum'
	)
	print('sum done')
	#K2data_ls = create_hierarchical_data(
	#	args, X_train_input, X_train_target,
	#	X_dev_input, X_dev_target,
	#	X_test_input, X_test_target,
	#	train_bkp, dev_bkp, test_bkp,
	#	data_train, data_dev, data_test,
	#	aggregation_type='leastsquare'
	#)
	#print('ls done')
	#K2data_st = create_hierarchical_data(
	#	args, X_train_input, X_train_target,
	#	X_dev_input, X_dev_target,
	#	X_test_input, X_test_target,
	#	train_bkp, dev_bkp, test_bkp,
	#	data_train, data_dev, data_test,
	#	aggregation_type='sumwithtrend'
	#)
	#print('sumwithtrend done')
	K2data_slope = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		data_train, data_dev, data_test,
		aggregation_type='slope'
	)
	print('slope done')
	#K2data_wavelet = create_hierarchical_wavelet_data(
	#	args, X_train_input, X_train_target,
	#	X_dev_input, X_dev_target,
	#	X_test_input, X_test_target,
	#	train_bkp, dev_bkp, test_bkp,
	#	aggregation_type='wavelet'
	#)
	K2data_wavelet = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp,
		data_train, data_dev, data_test,
		aggregation_type='wavelet'
	)
	print('wavelet done')

	dataset = dict()
	dataset['sum'] = K2data_sum
	#dataset['leastsquare'] = K2data_ls
	#dataset['sumwithtrend'] = K2data_st
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