import numpy as np
import torch
import random
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
		if idx+enc_len+dec_len < len(data):
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
	
		data_train, data_dev, data_test = generate_train_dev_test_data(data, N_input)

		data_train_in, data_train_out = create_forecast_io_seqs(data_train, N_input, N_output, N_output)
		data_dev_in, data_dev_out = create_forecast_io_seqs(data_dev, N_input, N_output, N_output)
		data_test_in, data_test_out = create_forecast_io_seqs(data_test, N_input, N_output, N_output)

		train_bkp = np.ones(data_train_in.shape[0]) * N_input
		test_bkp = np.ones(data_test_in.shape[0]) * N_input

		return data_train_in, data_train_out, data_test_in, data_test_out, train_bkp, test_bkp

def parse_ECG5000(N_input, N_output):
	raise NotImplementedError


