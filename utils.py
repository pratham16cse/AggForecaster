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
from tsmoothie.smoother import SpectralSmoother, ExponentialSmoother
import time

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ECG5000, parse_Traffic, parse_Taxi, parse_Traffic911, parse_gc_datasets, parse_weather, parse_bafu, parse_meteo, parse_azure, parse_ett


to_float_tensor = lambda x: torch.FloatTensor(x.copy())
to_long_tensor = lambda x: torch.FloatTensor(x.copy())

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

def normalize(data, norm=None, norm_type=None, is_var=False):
	if norm is None:
		assert norm_type is not None

		if norm_type in ['same']: # No normalization 
			scale = np.ones_like(np.mean(data, axis=(1), keepdims=True))
			shift = np.zeros_like(scale)
			norm = np.concatenate([shift, scale], axis=-1)
		if norm_type in ['avg']: # mean of entire data
			norm = np.mean(data, axis=(0, 1))
			scale = np.ones_like(np.mean(data, axis=(1), keepdims=True)) * norm
			shift = np.zeros_like(scale)
			norm = np.concatenate([shift, scale], axis=-1)
		elif norm_type in ['avg_per_series']: # per-series mean
			scale = np.mean(data, axis=(1), keepdims=True)
			shift = np.zeros_like(scale)
			norm = np.concatenate([shift, scale], axis=-1)
		elif norm_type in ['quantile90']: # 0.9 quantile of entire data
			scale = np.quantile(data, 0.90, axis=(0, 1))
			shift = np.zeros_like(scale)
			norm = np.concatenate([shift, scale], axis=-1)
		elif norm_type in ['std']: # std of entire data
			scale = np.std(data, axis=(0,1))
			shift = np.zeros_like(scale)
			norm = np.concatenate([shift, scale], axis=-1)
		elif norm_type in ['zscore_per_series']: # z-score at each series
			mean = np.mean(data, axis=(1), keepdims=True) # per-series mean
			std = np.std(data, axis=(1), keepdims=True) # per-series std
			norm = np.concatenate([mean, std], axis=-1)

	if is_var:
 		data_norm = data * 1.0 / norm[ ... , :, 1:2 ]
	else:
 		data_norm = (data - norm[...,:,0:1])* 1.0/norm[...,:,1:2]
	#data_norm = data * 10.0/norm
	#import ipdb
	#ipdb.set_trace()
	return data_norm, norm

def unnormalize(data, norm, is_var):
    if is_var:
        data_unnorm = data * norm[ ... , : , 1:2 ]
    else:
        data_unnorm = data * norm[ ... , : , 1:2 ] + norm[ ... , : , 0:1 ]

    return data_unnorm

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
        #'H': ['hour', 'dayofweek'],
        'H': ['hour'],
        #'min': ['minute', 'hour', 'dayofweek'],
        'min': ['minute', 'hour'],
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

def aggregate_seqs_sum(seqs, K, is_var):
    agg_seqs = []
    for i, seq in enumerate(seqs):
        #print(i, len(seqs))
        assert len(seq)%K == 0
        if is_var:
            agg_seq = [(1./(K*K)) * np.sum(seq[i:i+K], axis=0) for i in range(0, len(seq), K)]
        else:
            agg_seq = [np.sum(seq[i:i+K], axis=0) for i in range(0, len(seq), K)]
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


class TimeSeriesDatasetOfflineAggregate(torch.utils.data.Dataset):
	"""docstring for TimeSeriesDatasetOfflineAggregate"""
	def __init__(
		self, data, enc_len, dec_len, stride, aggregation_type, K,
		use_time_features, which_split, tsid_map=None, input_norm=None, target_norm=None,
		norm_type=None,
	):
		super(TimeSeriesDatasetOfflineAggregate, self).__init__()

		assert enc_len%K == 0
		assert dec_len%K == 0

		print('Creating dataset:', aggregation_type, K)
		self._enc_len = enc_len
		self._dec_len = dec_len
		self._base_enc_len = enc_len
		self._base_dec_len = dec_len
		#self.num_values = len(data[0]['target'][0])
		self.stride = stride
		self.aggregation_type = aggregation_type
		self.K = K
		self.input_norm = input_norm
		self.target_norm = target_norm
		self.norm_type = norm_type
		self.use_time_features = use_time_features
		self.tsid_map = tsid_map

		# Perform aggregation if level != 1
		data_agg = []
		for i in range(0, len(data)):
			ex = data[i]['target']
			ex = ex[ len(ex)%self.K: ]
			ex_f = data[i]['feats']
			ex_f = ex_f[ len(ex)%self.K: ]

			#bp = np.arange(1,len(ex), 1)
			bp = [(i, self.K) for i in np.arange(0, len(ex), self.K)]

			if self.K != 1:
				ex_agg, ex_f_agg = [], []
				if self.aggregation_type in ['sum']:
					for b in range(len(bp)):
						s, e = bp[b][0], bp[b][0]+bp[b][1]
						ex_agg.append(self.aggregate_data(ex[s:e]))

				elif self.aggregation_type in ['slope']:
					for b in range(len(bp)):
						s, e = bp[b][0], bp[b][0]+bp[b][1]
						ex_agg.append(self.aggregate_data_slope(ex[s:e]))
				ex_f_agg.append(self.aggregate_data(ex_f[s:e]))
			else:
				ex_agg = ex
				ex_f_agg = ex_f

			gaps = [bp_i[1] for bp_i in bp]

			data_agg.append(
				{
					'target':np.array(ex_agg),
					'feats':np.array(ex_f_agg),
					'bp': bp,
					'gaps': np.array(gaps),
				}
			)

			#if self.use_time_features:
			#	multiple, granularity = get_granularity(data[0]['freq_str'])
			#	freq_str = str(self.K * multiple) + granularity
			#	self.time_features_obj = time_features_from_frequency_str(
			#		granularity
			#	)
			#	self.date_range = []
			#	for i in range(0, len(data)):
			#		self.date_range.append(
			#			get_date_range(
			#				data[i]['start'],
			#				len(data[i]['target'])
			#			)
			#		)


		if self.input_norm is None:
			assert norm_type is not None
			data_for_norm = []
			for i in range(0, len(data)):
				ex = data_agg[i]['target']
				data_for_norm.append(np.array(ex))
			data_for_norm = np.array(data_for_norm)

			_, self.input_norm = normalize(data_for_norm, norm_type=self.norm_type)
			self.target_norm = self.input_norm
			del data_for_norm


		self.data = data_agg
		self.indices = []
		for i in range(0, len(self.data)):
			if stride == -1:
				j = 0
				while j < len(self.data[i]['target']):
					if j+self.enc_len+self.dec_len <= len(self.data[i]['target']):
						self.indices.append((i, j))
					#s = (np.random.randint(0, min(self.dec_len, 50)))
					s = 1
					j += s
			else:
				#if self.K > 1:
				#import ipdb
				#ipdb.set_trace()
				if which_split == 'dev':
					start_idx = len(self.data[i]['target']) - self.enc_len - self.dec_len
					for j in range(start_idx, len(self.data[i]['target']), 1):
						if j+self.enc_len+self.dec_len <= len(self.data[i]['target']):
							self.indices.append((i, j))
				if which_split == 'test':
					j = len(self.data[i]['target']) - self.enc_len - self.dec_len
					self.indices.append((i, j))

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
			input_size += len(self.data[0]['feats'][0])
		return input_size

	@property
	def output_size(self):
		output_size = len(self.data[0]['target'][0])
		return output_size

	@property
	def base_enc_len(self):
		return self._base_enc_len
	
	@property
	def base_dec_len(self):
		return self._base_dec_len

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

		ex_input_feats = self.data[ts_id]['feats'][ pos_id : pos_id+self.enc_len ]
		ex_target_feats = self.data[ts_id]['feats'][ pos_id+self.enc_len : pos_id+self.enc_len+self.dec_len ]

		#print(type(ex_input), type(ex_target), type(ex_input_feats), type(ex_target_feats))
		ex_input = to_float_tensor(ex_input)
		ex_target = to_float_tensor(ex_target)
		ex_input_feats = to_long_tensor(ex_input_feats)
		ex_target_feats = to_long_tensor(ex_target_feats)
		ex_norm = to_float_tensor(ex_norm)
		input_bp = to_float_tensor(np.expand_dims(input_bp, axis=-1))
		target_bp = to_float_tensor(np.expand_dims(target_bp, axis=-1))
		input_gaps = to_float_tensor(np.expand_dims(input_gaps, axis=-1))
		target_gaps = to_float_tensor(np.expand_dims(target_gaps, axis=-1))
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
		return np.sum(values, axis=0)

	def aggregate_data_slope(self, values, compute_b=False):
		x = np.expand_dims(np.arange(values.shape[0]), axis=1)
		m_x = np.mean(x, axis=0)
		s_xx = np.sum((x-m_x)**2, axis=0)

		y = values
		#m_y = np.mean(y, axis=0)
		#s_xy = np.sum((x-m_x)*(y-m_y), axis=0)
		#w = s_xy/s_xx

		a = (x - m_x) / s_xx
		w = np.sum(a*y, axis=0)

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

	def smooth(self, series):
		#smoother = SpectralSmoother(smooth_fraction=0.4, pad_len=10)
		smoother = ExponentialSmoother(window_len=10, alpha=0.15)
		series = np.concatenate((np.zeros((10, 1)), series), axis=0)
		series_smooth = np.expand_dims(smoother.smooth(series[:, 0]).smooth_data[0], axis=-1)
		return series_smooth


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
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
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

		elif args.dataset_name in ['weather']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
			) = parse_weather(args.dataset_name, args.N_input, args.N_output)
		elif args.dataset_name in ['bafu']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
			) = parse_bafu(args.dataset_name, args.N_input, args.N_output)
		elif args.dataset_name in ['meteo']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map
			) = parse_meteo(args.dataset_name, args.N_input, args.N_output)
		elif args.dataset_name in ['azure']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map,
                                feats_info
			) = parse_azure(args.dataset_name, args.N_input, args.N_output)
		elif args.dataset_name in ['ett']:
			(
				data_train, data_dev, data_test,
				dev_tsid_map, test_tsid_map,
                                feats_info
			) = parse_ett(args.dataset_name, args.N_input, args.N_output)

		if args.use_time_features:
			assert 'feats' in data_train[0].keys()

		self.data_train = data_train
		self.data_dev = data_dev
		self.data_test = data_test
		self.dev_tsid_map = dev_tsid_map
		self.test_tsid_map = test_tsid_map
		self.feats_info = feats_info


	def get_processed_data(self, args, agg_method, K):

 		if agg_method in ['wavelet']:
 			wavelet_levels = args.wavelet_levels
 			K_list = range(1, args.wavelet_levels+1+1+1)
 			# +1 : wavedec returns args.wavelet_levels+1 coefficients
 			# +1 : Extra slot for base values
 			# +1 : Because starting index is 1.
 		else:
 			wavelet_levels = None
 			K_list = args.K_list
 
 
 		lazy_dataset_train = TimeSeriesDatasetOfflineAggregate(
 			self.data_train, args.N_input, args.N_output, -1,
 			agg_method, K, which_split='train',
 			norm_type=args.normalize,
 			use_time_features=args.use_time_features,
 		)
 		print('Number of chunks in train data:', len(lazy_dataset_train))
 		norm = lazy_dataset_train.input_norm
 		dev_norm, test_norm = [], []
 		for i in range(len(self.data_dev)):
 			dev_norm.append(norm[self.dev_tsid_map[i]])
 		for i in range(len(self.data_test)):
 			test_norm.append(norm[self.test_tsid_map[i]])
 		dev_norm, test_norm = np.stack(dev_norm), np.stack(test_norm)
 		#import ipdb
 		#ipdb.set_trace()
 		lazy_dataset_dev = TimeSeriesDatasetOfflineAggregate(
 			self.data_dev, args.N_input, args.N_output, args.N_output,
 			agg_method, K,
 			input_norm=dev_norm, which_split='dev',
 			target_norm=np.concatenate(
 				[
 					np.zeros_like(dev_norm[..., :, 0:1]),
 					np.ones_like(dev_norm[..., :, 1:2])
 				],
 				axis=-1
 			),
 			use_time_features=args.use_time_features,
 			tsid_map=self.dev_tsid_map,
 		)
 		print('Number of chunks in dev data:', len(lazy_dataset_dev))
 		lazy_dataset_test = TimeSeriesDatasetOfflineAggregate(
 			self.data_test, args.N_input, args.N_output, args.N_output,
 			agg_method, K, which_split='test',
 			input_norm=test_norm,
 			target_norm=np.concatenate(
 				[
 					np.zeros_like(dev_norm[..., :, 0:1]),
 					np.ones_like(dev_norm[..., :, 1:2])
 				],
 				axis=-1
 			),
 			use_time_features=args.use_time_features,
 			tsid_map=self.test_tsid_map,
 		)
 		print('Number of chunks in test data:', len(lazy_dataset_test))
 		if K == 1:
 			batch_size = args.batch_size
 		else:
 			batch_size = 16
 		trainloader = DataLoader(
 			lazy_dataset_train, batch_size=batch_size, shuffle=True,
 			drop_last=True, num_workers=12, pin_memory=True
 		)
 		devloader = DataLoader(
 			lazy_dataset_dev, batch_size=batch_size, shuffle=False,
 			drop_last=False, num_workers=12, pin_memory=True
 		)
 		testloader = DataLoader(
 			lazy_dataset_test, batch_size=batch_size, shuffle=False,
 			drop_last=False, num_workers=12, pin_memory=True
 		)
 		#import ipdb
 		#ipdb.set_trace()
 
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
 			'test_norm': test_norm,
 			'feats_info': self.feats_info
 		}

