from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from collections import OrderedDict
import pywt
import pandas as pd
import re
import time
import shutil

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ECG5000, parse_Traffic, parse_Taxi, parse_Traffic911, parse_gc_datasets


to_tensor = lambda x: torch.FloatTensor(x.copy())

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def clean_trial_checkpoints(result):
	for trl in result.trials:
		trl_paths = result.get_trial_checkpoints_paths(trl,'metric')
		for path, _ in trl_paths:
			shutil.rmtree(path)

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

def normalize(data, norm=None, norm_type=None):
	if norm is None:
		assert norm_type is not None

		if norm_type in ['same']:
			norm = np.ones_like(np.mean(data, axis=(1), keepdims=True)) # No normalization
		if norm_type in ['avg']:
			norm = np.mean(data, axis=(0, 1))
			norm = np.ones_like(np.mean(data, axis=(1), keepdims=True)) * norm # mean of entire data
		elif norm_type in ['avg_per_series']:
			norm = np.std(data, axis=(1), keepdims=True) # per-series std
		elif norm_type in ['quantile90']:
			norm = np.quantile(data, 0.90, axis=(0, 1)) # 0.9 quantile of entire data
		elif norm_type in ['std']:
			norm = np.std(data, axis=(0,1)) # std of entire data

	data_norm = data * 1.0/norm 
	#data_norm = data * 10.0/norm
	#import ipdb
	#ipdb.set_trace()
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

def fit_slope_with_indices(seq, K, is_var):
    x = np.reshape(np.ones_like(seq), (-1, K))
    x = np.cumsum(x, axis=1) - 1
    y = np.reshape(seq, (-1, K))
    m_x = np.mean(x, axis=1, keepdims=True)
    m_y = np.mean(y, axis=1, keepdims=True)
    s_xy = np.sum((x-m_x)*(y-m_y), axis=1, keepdims=True)
    s_xx = np.sum((x-m_x)**2, axis=1, keepdims=True)
    #w = s_xy/s_xx
    a = (x - m_x) / s_xx
    #import ipdb
    #ipdb.set_trace()
    if is_var:
        w = np.sum(a**2 * y, axis=1, keepdims=True)
    else:
        w = np.sum(a * y, axis=1, keepdims=True)
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

def aggregate_seqs_sum(seqs, K, is_var):
    agg_seqs = []
    for i, seq in enumerate(seqs):
        #print(i, len(seqs))
        assert len(seq)%K == 0
        if is_var:
            agg_seq = [(1./(K*K)) * np.sum(seq[i:i+K], axis=0) for i in range(0, len(seq), K)]
        else:
            agg_seq = [np.mean(seq[i:i+K], axis=0) for i in range(0, len(seq), K)]
        agg_seqs.append(agg_seq)
    return np.array(agg_seqs)

def aggregate_seqs_slope(seqs, K, is_var=False):
    agg_seqs = []
    for seq in seqs:
        assert len(seq)%K == 0
        agg_seq = fit_slope_with_indices(seq, K, is_var)
        agg_seqs.append(agg_seq)
    return np.array(agg_seqs)

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

def create_hierarchical_data(
	args, data_train, data_dev, data_test,
	dev_tsid_map, test_tsid_map,
	aggregation_type, K
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


	lazy_dataset_train = TimeSeriesDatasetOfflineAggregate(
		data_train, args.N_input, args.N_output, -1,
		aggregation_type, K,
		norm_type=args.normalize,
		use_time_features=args.use_time_features,
		learnK=args.learnK
	)
	norm = lazy_dataset_train.input_norm
	dev_norm, test_norm = [], []
	for i in range(len(data_dev)):
		dev_norm.append(norm[dev_tsid_map[i]])
	for i in range(len(data_test)):
		test_norm.append(norm[test_tsid_map[i]])
	dev_norm, test_norm = np.stack(dev_norm), np.stack(test_norm)
	#import ipdb
	#ipdb.set_trace()
	lazy_dataset_dev = TimeSeriesDatasetOfflineAggregate(
		data_dev, args.N_input, args.N_output, args.N_output,
		aggregation_type, K,
		input_norm=dev_norm, target_norm=np.ones_like(dev_norm),
		use_time_features=args.use_time_features,
		tsid_map=dev_tsid_map,
		learnK=args.learnK
	)
	lazy_dataset_test = TimeSeriesDatasetOfflineAggregate(
		data_test, args.N_input, args.N_output, args.N_output,
		aggregation_type, K,
		input_norm=test_norm, target_norm=np.ones_like(test_norm),
		use_time_features=args.use_time_features,
		tsid_map=test_tsid_map,
		learnK=args.learnK,
	)
	trainloader = DataLoader(
		lazy_dataset_train, batch_size=args.batch_size, shuffle=True,
		drop_last=True, num_workers=0
	)
	devloader = DataLoader(
		lazy_dataset_dev, batch_size=args.batch_size, shuffle=False,
		drop_last=False, num_workers=0
	)
	testloader = DataLoader(
		lazy_dataset_test, batch_size=args.batch_size, shuffle=False,
		drop_last=False, num_workers=0
	)

	return {
		'trainloader': trainloader,
		'devloader': devloader,
		'testloader': testloader,
		'N_input': lazy_dataset_test.enc_len,
		'N_output': lazy_dataset_test.dec_len,
		'input_size': lazy_dataset_test.input_size,
		'output_size': lazy_dataset_test.output_size,
		'train_norm': norm,
		'dev_norm': dev_norm,
		'test_norm': test_norm
	}

class TimeSeriesDataset(torch.utils.data.Dataset):
	"""docstring for TimeSeriesDataset"""
	def __init__(
		self, data, enc_len, dec_len, stride, aggregation_type, K,
		use_time_features, tsid_map=None, input_norm=None, target_norm=None,
		norm_type=None,
		learnK=False,
		wavelet_levels=None
	):
		super(TimeSeriesDataset, self).__init__()

		if aggregation_type not in ['wavelet']:
			assert enc_len%K == 0
			assert dec_len%K == 0

		print('Creating dataset:', aggregation_type, K)
		self.data = data
		self._enc_len = enc_len
		self._dec_len = dec_len
		#self.num_values = len(data[0]['target'][0])
		self.stride = stride
		self.aggregation_type = aggregation_type
		self.K = K
		self.input_norm = input_norm
		self.target_norm = target_norm
		self.norm_type = norm_type
		self.use_time_features = use_time_features
		self.tsid_map = tsid_map
		self.learnK = learnK
		self.wavelet_levels = wavelet_levels

		#if self.K != 1:
		#	self._enc_len = self.K * enc_len
		#	for i in range(len(data)):
		#		if self.K  * enc_len > len(data[i]['target']):
		#			self._enc_len = enc_len
		#			break

		self.indices = []
		for i in range(0, len(data)):
			if stride == -1:
				j = 0
				while j < len(data[i]['target']):
					if j+self._enc_len+self._dec_len <= len(data[i]['target']):
						self.indices.append((i, j))
					s = (np.random.randint(0, self._dec_len))
					j += s
			else:
				j = len(data[i]['target']) - self._enc_len - self._dec_len
				self.indices.append((i, j))
				#for j in range(start_idx, len(data[i]['target']), stride):
				#	if j+self._enc_len+self._dec_len <= len(data[i]['target']):
				#		self.indices.append((i, j))

		if self.input_norm is None:
			assert norm_type is not None
			data_agg = []
			for i in range(0, len(data)):
				ex = self.data[i]['target']
				ex = ex[:len(ex)-len(ex)%self.K]
				if self.K != 1:
					if self.aggregation_type in ['sum']:
						ex_agg = map(
							self.aggregate_data,
							np.split(ex, np.arange(self.K, len(ex), self.K), axis=0),
						)
						ex_agg = list(ex_agg)

					elif self.aggregation_type in ['slope']:
						ex_agg = map(
							self.aggregate_data_slope,
							np.split(ex, np.arange(self.K, len(ex), self.K), axis=0),
						)
						ex_agg = list(ex_agg)
				else:
					ex_agg = ex

				data_agg.append(np.array(ex_agg))
			data_agg = np.array(data_agg)

			_, self.input_norm = normalize(data_agg, norm_type=self.norm_type)
			self.target_norm = self.input_norm

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

		return self._enc_len // max(self.K, 1)
	
	@property
	def dec_len(self):
		if self.wavelet_levels is not None:

			if self.K == self.wavelet_levels+2:
				return self._dec_len // 2**(self.K-2)

			return self._dec_len // 2**(self.K-1)

		return self._dec_len // max(self.K, 1)

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
		ex_input_bp = np.arange(1, self._enc_len, 1)
		ex_target_bp = np.arange(1, self._dec_len, 1)

		if self.K != 1:

			if self.K == -1:
				input_bp = self.get_piecewise_linear_breakpoints(ex_input)
				target_bp = self.get_piecewise_linear_breakpoints(ex_target)
				#print(input_bp, target_bp)
			else:
				input_bp = np.arange(self.K, self._enc_len, self.K)
				target_bp = np.arange(self.K, self._dec_len, self.K)

			#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
			#print(input_bp, target_bp)
			if self.aggregation_type in ['sum']:
				#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
				ex_input_agg = map(
					self.aggregate_data,
					np.split(ex_input, input_bp, axis=0),
				)
				ex_target_agg = map(
					self.aggregate_data,
					np.split(ex_target, target_bp, axis=0)
				)
				#print('lengths of aggs:', len(list(ex_agg)[0]), len(list(ex_agg)[1]))
				ex_input_agg, ex_target_agg = list(ex_input_agg), list(ex_target_agg)

			elif self.aggregation_type in ['slope']:
				#print('before', ex_input.shape, ex_target.shape, ts_id, pos_id)
				ex_input_agg = map(
					self.aggregate_data_slope,
					np.split(ex_input, input_bp, axis=0),
				)
				ex_target_agg = map(
					self.aggregate_data_slope,
					np.split(ex_target, target_bp, axis=0)
				)
				ex_input_agg, ex_target_agg = list(ex_input_agg), list(ex_target_agg)

			elif self.aggregation_type in ['wavelet']:
				ex_input_agg = self.aggregate_data_wavelet(ex_input, self.K)
				ex_target_agg = self.aggregate_data_wavelet(ex_target, self.K)


			#ex_input, ex_target = zip(*ex_agg)
			ex_input = np.array(ex_input_agg)
			ex_target = np.array(ex_target_agg)
		else:
			input_bp = np.arange(1, self._enc_len, 1)
			target_bp = np.arange(1, self._dec_len, 1)

		#print('after', ex_input.shape, ex_target.shape, ts_id, pos_id)
		if self.tsid_map is None:
			mapped_id = ts_id
		else:
			mapped_id = self.tsid_map[ts_id]
		ex_input, _ = normalize(ex_input, self.input_norm[mapped_id])
		ex_target, _ = normalize(ex_target, self.target_norm[mapped_id])
		ex_norm = self.input_norm[mapped_id]

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
				ex_input_feats = np.array(ex_input_feats)
				ex_target_feats = np.array(ex_target_feats)
		else:
			ex_input_feats, ex_target_feats = ex_input, ex_target

		#print(type(ex_input), type(ex_target), type(ex_input_feats), type(ex_target_feats))
		ex_input = to_tensor(ex_input)
		ex_target = to_tensor(ex_target)
		ex_input_feats = to_tensor(ex_input_feats)
		ex_target_feats = to_tensor(ex_target_feats)
		ex_norm = to_tensor(ex_norm)
		input_bp = np.concatenate((input_bp, np.array([self._enc_len])), axis=-1)
		target_bp = np.concatenate((target_bp, np.array([self._dec_len])), axis=-1)
		input_gaps = input_bp - np.concatenate((np.array([0]), input_bp[:-1]), axis=-1)
		target_gaps = target_bp - np.concatenate((np.array([0]), target_bp[:-1]), axis=-1)
		input_bp = to_tensor(np.expand_dims(input_bp, axis=-1))
		target_bp = to_tensor(np.expand_dims(target_bp, axis=-1))
		input_gaps = to_tensor(np.expand_dims(input_gaps, axis=-1))
		target_gaps = to_tensor(np.expand_dims(target_gaps, axis=-1))

		return (
			ex_input, ex_target,
			ex_input_feats, ex_target_feats,
			ex_norm,
			torch.FloatTensor([ts_id, pos_id]),
			input_bp, target_bp,
			input_gaps, target_gaps
		)

	def aggregate_data(self, values):
		return np.mean(values, axis=0)

	def aggregate_data_slope(self, values, compute_b=False):
		x = np.expand_dims(np.arange(values.shape[0]), axis=1)
		m_x = np.mean(x, axis=0)
		s_xx = np.sum((x-m_x)**2, axis=0)

		y = values
		m_y = np.mean(y, axis=0)
		s_xy = np.sum((x-m_x)*(y-m_y), axis=0)
		w = s_xy/s_xx

		if compute_b:
			b = m_y - w*m_x
			return w, b
		else:
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

	def calculate_error(self, segment):
		w, b = self.aggregate_data_slope(segment, compute_b=True)
		x = np.expand_dims(np.arange(len(segment)), axis=1)
		segment_pred = w*x+b

		return np.max(np.abs(segment - segment_pred)) # Using max error

	def get_piecewise_linear_breakpoints(self, values, max_err=2.0): # TODO: set appropriate max_err
		breakpoints = []
		approx_series = []
		anchor = 0
		while anchor < len(values)-1:
			i, j = 1, 1
			segment = values[anchor:anchor+i+1]
			err = self.calculate_error(segment)
			while err <= max_err and anchor+i<len(values):
				j = i
				i += 1
				segment = values[anchor:anchor+i+1]
				err = self.calculate_error(segment)
			bp = anchor+j+1
			breakpoints.append(bp)
			#segment = values[anchor:bp+1]
			#w = self.aggregate_data_slope(values[anchor:bp+1])
			#approx_series.append((bp, w))
			anchor = bp
			#print(len(values), bp, anchor, w, b)
			#print(len(segment))
			#if len(segment)<4:
			#	print(segment, np.arange(len(segment))*w+b)

		#print(breakpoints)
		return breakpoints[:-1]

class TimeSeriesDatasetOfflineAggregate(torch.utils.data.Dataset):
	"""docstring for TimeSeriesDatasetOfflineAggregate"""
	def __init__(
		self, data, enc_len, dec_len, stride, aggregation_type, K,
		use_time_features, tsid_map=None, input_norm=None, target_norm=None,
		norm_type=None,
		learnK=False
	):
		super(TimeSeriesDatasetOfflineAggregate, self).__init__()

		assert enc_len%K == 0
		assert dec_len%K == 0

		print('Creating dataset:', aggregation_type, K)
		self._enc_len = enc_len
		self._dec_len = dec_len
		#self.num_values = len(data[0]['target'][0])
		self.stride = stride
		self.aggregation_type = aggregation_type
		self.K = K
		self.input_norm = input_norm
		self.target_norm = target_norm
		self.norm_type = norm_type
		self.use_time_features = use_time_features
		self.tsid_map = tsid_map
		self.learnK = learnK

		# Perform aggregation if level != 1
		data_agg = []
		for i in range(0, len(data)):
			ex = data[i]['target']
			if self.K >= 1:
				ex = ex[ len(ex)%self.K: ]

			bp = np.arange(1,len(ex), 1)
			if self.K == -1:
				bp = self.get_piecewise_linear_breakpoints(ex)
				#print(input_bp, target_bp)
			else:
				bp = np.arange(self.K, len(ex), self.K)

			if self.K != 1:
				if self.aggregation_type in ['sum']:
					ex_agg = map(
						self.aggregate_data,
						np.split(ex, bp, axis=0),
					)
					ex_agg = list(ex_agg)

				elif self.aggregation_type in ['slope']:
					ex_agg = map(
						self.aggregate_data_slope,
						np.split(ex, bp, axis=0),
					)
					ex_agg = list(ex_agg)
			else:
				ex_agg = ex

			bp = np.concatenate((bp, np.array([len(ex)])), axis=-1)
			gaps = bp - np.concatenate((np.array([0]), bp[:-1]), axis=-1)

			data_agg.append(
				{
					'target':np.array(ex_agg),
					'bp': np.array(bp),
					'gaps': np.array(gaps)
				}
			)

			if self.use_time_features:
				raise NotImplementedError

		if self.input_norm is None:
			assert norm_type is not None
			data_for_norm = []
			for i in range(0, len(data)):
				ex = data[i]['target']
				data_for_norm.append(np.array(ex))
			data_for_norm = np.array(data_for_norm)

			_, self.input_norm = normalize(data_for_norm, norm_type=self.norm_type)
			self.target_norm = self.input_norm
			del data_for_norm

		if self.use_time_features:
			multiple, granularity = get_granularity(data[0]['freq_str'])
			freq_str = str(self.K * multiple) + granularity
			self.time_features_obj = time_features_from_frequency_str(granularity)

			self.date_range = []
			for i in range(0, len(data)):
				self.date_range.append(
					get_date_range(
						data[i]['start'],
						len(data[i]['target'])
					)
				)

		self.data = data_agg
		self.indices = []
		for i in range(0, len(self.data)):
			if stride == -1:
				j = 0
				while j < len(self.data[i]['target']):
					if j+self.enc_len+self.dec_len <= len(self.data[i]['target']):
						self.indices.append((i, j))
					s = (np.random.randint(0, self.dec_len))
					j += s
			else:
				j = len(self.data[i]['target']) - self.enc_len - self.dec_len
				self.indices.append((i, j))
				#for j in range(start_idx, len(self.data[i]['target']), stride):
				#	if j+self.enc_len+self.dec_len <= len(self.data[i]['target']):
				#		self.indices.append((i, j))

		#import ipdb
		#ipdb.set_trace()

	@property
	def enc_len(self):
		if self.K > 1:
			el = (self._enc_len // self.K) * 2
		else:
			el = self._enc_len
		return el
	
	@property
	def dec_len(self):
		if self.K > 1:
			dl = self._dec_len // self.K
		else:
			dl = self._dec_len
		return dl

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
			self.data[ts_id]['target'][ pos_id : pos_id+self.enc_len ]
		)
		ex_target = np.array(
			self.data[ts_id]['target'][ pos_id+self.enc_len : pos_id+self.enc_len+self.dec_len ]
		)

		input_bp = self.data[ts_id]['bp'][ pos_id : pos_id+self.enc_len ]
		target_bp = self.data[ts_id]['bp'][ pos_id+self.enc_len : pos_id+self.enc_len+self.dec_len ]
		input_gaps = self.data[ts_id]['gaps'][ pos_id : pos_id+self.enc_len ]
		target_gaps = self.data[ts_id]['gaps'][ pos_id+self.enc_len : pos_id+self.enc_len+self.dec_len ]

		#print('after', ex_input.shape, ex_target.shape, ts_id, pos_id)
		if self.tsid_map is None:
			mapped_id = ts_id
		else:
			mapped_id = self.tsid_map[ts_id]
		ex_input, _ = normalize(ex_input, self.input_norm[mapped_id])
		ex_target, _ = normalize(ex_target, self.target_norm[mapped_id])
		ex_norm = self.input_norm[mapped_id]

		if self.use_time_features:
			raise NotImplementedError
		else:
			ex_input_feats, ex_target_feats = ex_input, ex_target

		#print(type(ex_input), type(ex_target), type(ex_input_feats), type(ex_target_feats))
		ex_input = to_tensor(ex_input)
		ex_target = to_tensor(ex_target)
		ex_input_feats = to_tensor(ex_input_feats)
		ex_target_feats = to_tensor(ex_target_feats)
		ex_norm = to_tensor(ex_norm)
		input_bp = to_tensor(np.expand_dims(input_bp, axis=-1))
		target_bp = to_tensor(np.expand_dims(target_bp, axis=-1))
		input_gaps = to_tensor(np.expand_dims(input_gaps, axis=-1))
		target_gaps = to_tensor(np.expand_dims(target_gaps, axis=-1))
		#print(
		#	ex_input.shape, ex_target.shape,
		#	input_bp.shape, target_bp.shape,
		#	input_gaps.shape, target_gaps.shape
		#)

		return (
			ex_input, ex_target,
			ex_input_feats, ex_target_feats,
			ex_norm,
			torch.FloatTensor([ts_id, pos_id]),
			input_bp, target_bp,
			input_gaps, target_gaps
		)

	def aggregate_data(self, values):
		return np.mean(values, axis=0)

	def aggregate_data_slope(self, values, compute_b=False):
		x = np.expand_dims(np.arange(values.shape[0]), axis=1)
		m_x = np.mean(x, axis=0)
		s_xx = np.sum((x-m_x)**2, axis=0)

		y = values
		m_y = np.mean(y, axis=0)
		s_xy = np.sum((x-m_x)*(y-m_y), axis=0)
		w = s_xy/s_xx

		if compute_b:
			b = m_y - w*m_x
			return w, b
		else:
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

	def calculate_error(self, segment):
		w, b = self.aggregate_data_slope(segment, compute_b=True)
		x = np.expand_dims(np.arange(len(segment)), axis=1)
		segment_pred = w*x+b

		return np.max(np.abs(segment - segment_pred)) # Using max error

	def get_piecewise_linear_breakpoints(self, values, max_err=0.0): # TODO: set appropriate max_err
		breakpoints = []
		approx_series = []
		anchor = 0
		while anchor < len(values)-1:
			i, j = 1, 1
			segment = values[anchor:anchor+i+1]
			err = self.calculate_error(segment)
			while err <= max_err and anchor+i<len(values):
				j = i
				i += 1
				segment = values[anchor:anchor+i+1]
				err = self.calculate_error(segment)
			bp = anchor+j+1
			breakpoints.append(bp)
			#segment = values[anchor:bp+1]
			#w = self.aggregate_data_slope(values[anchor:bp+1])
			#approx_series.append((bp, w))
			anchor = bp
			#print(len(values), bp, anchor, w, b)
			#print(len(segment))
			#if len(segment)<4:
			#	print(segment, np.arange(len(segment))*w+b)

		#print(breakpoints)
		return breakpoints[:-1]



class DataProcessor(object):
	"""docstring for DataProcessor"""
	def __init__(self, args):
		super(DataProcessor, self).__init__()
		self.args = args

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
			N = 100
			sigma = 0.01
	
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
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
		elif args.dataset_name in ['Exchange', 'Solar', 'Wiki', 'taxi30min']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
			) = parse_gc_datasets(args.dataset_name, args.N_input, args.N_output)
	
		if args.use_time_features:
			assert 'start' in data_train[0].keys()

		self.data_train = data_train
		self.data_dev = data_dev
		self.data_test = data_test
		self.dev_tsid_map = dev_tsid_map
		self.test_tsid_map = test_tsid_map


	def get_processed_data(self, args, agg_method, K):
	
	
		dataset = create_hierarchical_data(
			args, self.data_train, self.data_dev, self.data_test,
			self.dev_tsid_map, self.test_tsid_map,
			aggregation_type=agg_method, K=K
		)
	
		return dataset