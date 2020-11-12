import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset, DataLoader

def generate_train_dev_test_data(data, N_input):
	train_per = 0.6
	dev_per = 0.2
	N = len(data)

	data_train = data[:int(train_per*N)]
	data_dev = data[int(train_per*N)-N_input:int((train_per+dev_per)*N)]
	data_test = data[int((train_per+dev_per)*N)-N_input:]

	return  (data_train, data_dev, data_test)

def create_forecast_io_seqs(data, enc_len, dec_len, stride):

	data_in, data_out = [], []
	for idx in range(0, len(data), stride):
		if idx+enc_len+dec_len <= len(data):
			data_in.append(data[idx:idx+enc_len])
			data_out.append(data[idx+enc_len:idx+enc_len+dec_len])

	data_in = np.array(data_in)
	data_out = np.array(data_out)
	return data_in, data_out

def parse_Traffic(N_input, N_output):
	with open('data/traffic/traffic.txt', 'r') as f:
		data = []
		# Taking only first series of length 17544
		# TODO: Add all series to the dataset
		for line in f:
			data.append(line.rstrip().split(',')[0])
		data = np.array(data).astype(np.float32)
		data = np.expand_dims(data, axis=-1)
	
	data_train, data_dev, data_test = generate_train_dev_test_data(data, N_input)

	data_train_in, data_train_out = create_forecast_io_seqs(data_train, N_input, N_output, int(N_output/3))
	data_dev_in, data_dev_out = create_forecast_io_seqs(data_dev, N_input, N_output, N_output)
	data_test_in, data_test_out = create_forecast_io_seqs(data_test, N_input, N_output, N_output)

	train_bkp = np.ones(data_train_in.shape[0]) * N_input
	dev_bkp = np.ones(data_dev_in.shape[0]) * N_input
	test_bkp = np.ones(data_test_in.shape[0]) * N_input

	return (
		data_train_in, data_train_out, data_dev_in, data_dev_out,
		data_test_in, data_test_out, train_bkp, dev_bkp, test_bkp,
	)

def parse_ECG5000(N_input, N_output):
	with open('data/ECG5000/ECG5000_TRAIN.tsv', 'r') as f:
		data = []
		for line in f:
			data.append(line.rstrip().split())
		data = np.array(data).astype(np.float32)
		data = np.expand_dims(data, axis=-1)
	with open('data/ECG5000/ECG5000_TEST.tsv', 'r') as f:
		data_test = []
		for line in f:
			data_test.append(line.rstrip().split())
		data_test = np.array(data_test).astype(np.float32)
		data_test = np.expand_dims(data_test, axis=-1)

	N = data.shape[0]
	dev_len = int(0.2*N)
	train_len = N - dev_len
	data_train, data_dev = data[:train_len], data[train_len:train_len+dev_len]

	data_train_in, data_train_out = data_train[:, :N_input], data_train[:, N_input:N_input+N_output]
	data_dev_in, data_dev_out = data_dev[:, :N_input], data_dev[:, N_input:N_input+N_output]
	data_test_in, data_test_out = data_test[:, :N_input], data_test[:, N_input:N_input+N_output]

	train_bkp = np.ones(data_train_in.shape[0]) * N_input
	dev_bkp = np.ones(data_dev_in.shape[0]) * N_input
	test_bkp = np.ones(data_test_in.shape[0]) * N_input

	return (
		data_train_in, data_train_out, data_dev_in, data_dev_out,
		data_test_in, data_test_out, train_bkp, dev_bkp, test_bkp,
	)

def create_bins(sequence, bin_size, num_bins):
	#num_bins = int(np.ceil((sequence[-1] - sequence[0]) * 1. / bin_size))
	counts = [0. for _ in range(num_bins)]
	curr_cnt = 0
	for ts in sequence:
		bin_id = int(ts // bin_size)
		counts[bin_id] += 1

	return counts

def parse_Taxi(N_input, N_output):
	# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
	# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-02.csv
	taxi_df_jan = pd.read_csv(
		'data/yellow_tripdata_2019-01.csv',
		usecols=["tpep_pickup_datetime", "PULocationID"])
	taxi_df_feb = pd.read_csv(
		'data/yellow_tripdata_2019-02.csv',
		usecols=["tpep_pickup_datetime", "PULocationID"])
	taxi_df = taxi_df_jan.append(taxi_df_feb)
	taxi_df['tpep_pickup_datetime'] = pd.to_datetime(
		taxi_df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce'
	)
	## Data cleaning
	# Dataset contains some spurious values, such as year 2038 and months other
	# than Jan and Feb. Following code purges such rows.
	taxi_df = taxi_df[(taxi_df['tpep_pickup_datetime'].dt.year == 2019)]
	taxi_df = taxi_df[(taxi_df['tpep_pickup_datetime'].dt.month < 3)]

	taxi_df = taxi_df.sort_values('tpep_pickup_datetime')
	taxi_df['timestamp'] = pd.DatetimeIndex(taxi_df['tpep_pickup_datetime']).astype(np.int64)/1000000000
	del taxi_df['tpep_pickup_datetime']
	taxi_df = taxi_df.sort_values(by=['timestamp'])
	#dataset_name = 'taxi'
	#if dataset_name in downsampling:
	#	taxi_timestamps = downsampling_dataset(taxi_timestamps, dataset_name)

	num_hrs = int(np.ceil((taxi_df['timestamp'].values[-1] - taxi_df['timestamp'].values[0])/3600.))
	loc2counts = dict()
	loc2numevents = dict()
	for loc_id, loc_df in taxi_df.groupby(['PULocationID']):
		timestamps = loc_df['timestamp'].values
		timestamps = timestamps - timestamps[0]
		loc2numevents[loc_id] = len(timestamps)
		# Select locations in which num_events per hour is >1
		if (len(timestamps) >= N_input+N_output and len(timestamps) / num_hrs > 1.):
			counts = create_bins(timestamps, bin_size=3600., num_bins=num_hrs)
			print(loc_id, len(timestamps), len(timestamps) / num_hrs, len(counts))
			loc2counts[loc_id] = counts

	data = np.array([val for val in loc2counts.values()])
	data = np.expand_dims(data, axis=2)
	data_train_in, data_train_out = [], []
	data_dev_in, data_dev_out = [], []
	data_test_in, data_test_out = [], []
	for seq in data:
		seq_train, seq_dev, seq_test = generate_train_dev_test_data(seq, N_input)
		batch_train_in, batch_train_out = create_forecast_io_seqs(seq_train, N_input, N_output, int(N_output/3))
		batch_dev_in, batch_dev_out = create_forecast_io_seqs(seq_dev, N_input, N_output, N_output)
		batch_test_in, batch_test_out = create_forecast_io_seqs(seq_test, N_input, N_output, N_output)
		data_train_in.append(batch_train_in)
		data_train_out.append(batch_train_out)
		data_dev_in.append(batch_dev_in)
		data_dev_out.append(batch_dev_out)
		data_test_in.append(batch_test_in)
		data_test_out.append(batch_test_out)

	data_train_in = np.concatenate(data_train_in, axis=0)
	data_train_out = np.concatenate(data_train_out, axis=0)
	data_dev_in = np.concatenate(data_dev_in, axis=0)
	data_dev_out = np.concatenate(data_dev_out, axis=0)
	data_test_in = np.concatenate(data_test_in, axis=0)
	data_test_out = np.concatenate(data_test_out, axis=0)

	train_bkp = np.ones(data_train_in.shape[0]) * N_input
	dev_bkp = np.ones(data_dev_in.shape[0]) * N_input
	test_bkp = np.ones(data_test_in.shape[0]) * N_input

	return (
		data_train_in, data_train_out, data_dev_in, data_dev_out,
		data_test_in, data_test_out, train_bkp, dev_bkp, test_bkp,
	)

def parse_Traffic911(N_input, N_output):
	call_df = pd.read_csv('data/911.csv')
	call_df = call_df[call_df['zip'].isnull()==False] # Ignore calls with NaN zip codes
	print('Types of Emergencies')
	print(call_df.title.apply(lambda x: x.split(':')[0]).value_counts())
	call_df['type'] = call_df.title.apply(lambda x: x.split(':')[0])
	print('Subtypes')
	for each in call_df.type.unique():
	    subtype_count = call_df[call_df.title.apply(lambda x: x.split(':')[0]==each)].title.value_counts()
	    print('For', each, 'type of Emergency, we have ', subtype_count.count(), 'subtypes')
	    print(subtype_count[subtype_count>100])
	print('Out of 3 types, considering only Traffic')
	call_data = call_df[call_df['type']=='Traffic']
	call_data['timeStamp'] = pd.to_datetime(call_data['timeStamp'], errors='coerce')
	print("We have timeline from", call_data['timeStamp'].min(), "to", call_data['timeStamp'].max())
	call_data = call_data.sort_values('timeStamp')
	call_data['timeStamp'] = pd.DatetimeIndex(call_data['timeStamp']).astype(np.int64)/1000000000

	num_hrs = int(
		np.ceil(
			(call_data['timeStamp'].values[-1] - call_data['timeStamp'].values[0])/(3600.)
		)
	)
	timestamps = call_data['timeStamp'].values
	timestamps = timestamps - timestamps[0]
	counts = create_bins(timestamps, bin_size=3600., num_bins=num_hrs)
	data = np.expand_dims(np.array(counts), axis=0)
	data = np.expand_dims(data, axis=2)
	data_train_in, data_train_out = [], []
	data_dev_in, data_dev_out = [], []
	data_test_in, data_test_out = [], []
	for seq in data:
		seq_train, seq_dev, seq_test = generate_train_dev_test_data(seq, N_input)
		batch_train_in, batch_train_out = create_forecast_io_seqs(seq_train, N_input, N_output, int(N_output/3))
		batch_dev_in, batch_dev_out = create_forecast_io_seqs(seq_dev, N_input, N_output, N_output)
		batch_test_in, batch_test_out = create_forecast_io_seqs(seq_test, N_input, N_output, N_output)
		data_train_in.append(batch_train_in)
		data_train_out.append(batch_train_out)
		data_dev_in.append(batch_dev_in)
		data_dev_out.append(batch_dev_out)
		data_test_in.append(batch_test_in)
		data_test_out.append(batch_test_out)

	data_train_in = np.concatenate(data_train_in, axis=0)
	data_train_out = np.concatenate(data_train_out, axis=0)
	data_dev_in = np.concatenate(data_dev_in, axis=0)
	data_dev_out = np.concatenate(data_dev_out, axis=0)
	data_test_in = np.concatenate(data_test_in, axis=0)
	data_test_out = np.concatenate(data_test_out, axis=0)

	train_bkp = np.ones(data_train_in.shape[0]) * N_input
	dev_bkp = np.ones(data_dev_in.shape[0]) * N_input
	test_bkp = np.ones(data_test_in.shape[0]) * N_input

	return (
		data_train_in, data_train_out, data_dev_in, data_dev_out,
		data_test_in, data_test_out, train_bkp, dev_bkp, test_bkp,
	)

def parse_exchange_rate(N_input, N_output):
	'''
	N_output = 30
	num_rolling_windows = 5
	'''
	num_rolling_windows = 5

	data = []
	with open('data/exchange_rate_nips/train/train.json') as f:
		for line in f:
			data.append(json.loads(line))

	data_test = []
	with open('data/exchange_rate_nips/test/test.json') as f:
		for line in f:
			data_test.append(json.loads(line))

	metadata = json.load(open('data/exchange_rate_nips/metadata/metadata.json'))

	data_train_in, data_train_out = [], []
	data_dev_in, data_dev_out = [], []
	data_test_in, data_test_out = [], []
	for entry in data:
		seq_train = entry['target'][ : -N_output*num_rolling_windows]
		seq_dev = entry['target'][ -N_output*num_rolling_windows - N_input : ]
		seq_train = np.expand_dims(seq_train, axis=-1)
		seq_dev = np.expand_dims(seq_dev, axis=-1)
		batch_train_in, batch_train_out = create_forecast_io_seqs(seq_train, N_input, N_output, int(N_output/3))
		batch_dev_in, batch_dev_out = create_forecast_io_seqs(seq_dev, N_input, N_output, N_output)
		data_train_in.append(batch_train_in)
		data_train_out.append(batch_train_out)
		data_dev_in.append(batch_dev_in)
		data_dev_out.append(batch_dev_out)
	for entry in data_test:
		seq_test = entry['target'][ -(N_input+N_output) : ]
		seq_test = np.expand_dims(seq_test, axis=-1)
		batch_test_in, batch_test_out = create_forecast_io_seqs(seq_test, N_input, N_output, N_output)
		data_test_in.append(batch_test_in)
		data_test_out.append(batch_test_out)

	data_train_in = np.concatenate(data_train_in, axis=0)
	data_train_out = np.concatenate(data_train_out, axis=0)
	data_dev_in = np.concatenate(data_dev_in, axis=0)
	data_dev_out = np.concatenate(data_dev_out, axis=0)
	data_test_in = np.concatenate(data_test_in, axis=0)
	data_test_out = np.concatenate(data_test_out, axis=0)

	train_bkp = np.ones(data_train_in.shape[0]) * N_input
	dev_bkp = np.ones(data_dev_in.shape[0]) * N_input
	test_bkp = np.ones(data_test_in.shape[0]) * N_input

	return (
		data_train_in, data_train_out, data_dev_in, data_dev_out,
		data_test_in, data_test_out, train_bkp, dev_bkp, test_bkp,
	)
