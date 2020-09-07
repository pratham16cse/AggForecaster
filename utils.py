from torch.utils.data import DataLoader
import numpy as np

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ECG5000, parse_Traffic


def add_metrics_to_dict(
	metrics_dict,
	model_name,
	metric_mse,
):
	if model_name not in metrics_dict:
		metrics_dict[model_name] = dict()

	metrics_dict[model_name]['mse'] = metric_mse

	return metrics_dict

def aggregate_data(
	level, K, train_input, train_target, dev_input, dev_target,
	test_input, test_target
):

	def aggregate_seqs_(seqs):
		agg_seqs = []
		for seq in seqs:
			assert len(seq)%K == 0
			agg_seq = [np.sum(seq[i:i+K]) for i in range(0, len(seq), K)]
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
	args, train_input, train_target, dev_input, dev_target,
	test_input, test_target, train_bkp, dev_bkp, test_bkp
):
	level2data = dict()
	for level in range(args.L):
		if level > 0:
			(
				train_input, train_target, dev_input, dev_target,
				test_input, test_target,
			)= aggregate_data(
				level, args.K, train_input, train_target,
				dev_input, dev_target, test_input, test_target,
			)
		dataset_train = SyntheticDataset(train_input, train_target, train_bkp)
		dataset_dev = SyntheticDataset(dev_input, dev_target, dev_bkp)
		dataset_test  = SyntheticDataset(test_input, test_target, test_bkp)
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
		level2data[level] = {
			'trainloader': trainloader,
			'devloader': devloader,
			'testloader': testloader,
			'N_output': test_target.shape[1]
		}

	return level2data

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

	level2data = create_hierarchical_data(
		args, X_train_input, X_train_target,
		X_dev_input, X_dev_target,
		X_test_input, X_test_target,
		train_bkp, dev_bkp, test_bkp
	)

	return {
		#'trainloader': trainloader,
		#'testloader': testloader,
		'level2data': level2data,
	}