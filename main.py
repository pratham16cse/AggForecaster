import sys
import os
import argparse
import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.base_models import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from train import train_model
from eval import eval_base_model, eval_inf_model
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import json

from models import inf_models
import utils

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, help='dataset_name')
#parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--N_input', type=int, default=20,
                    help='number of input steps')
parser.add_argument('--N_output', type=int, default=20,
                    help='number of output steps')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs', default='Outputs')
parser.add_argument('--saved_models_dir', type=str,
                    help='Path to store all saved models', default='saved_models')

parser.add_argument('--epochs', type=int, default=500,
                    help='number of training epochs')
parser.add_argument('--print_every', type=int, default=50,
                    help='Print test output after every print_every epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, nargs='+',
                   help='Learning rate for the training algorithm')

parser.add_argument('-hls', '--hidden_size', type=int, default=128, nargs='+',
                   help='Number of units in RNN')
parser.add_argument('--num_grulstm_layers', type=int, default=1, nargs='+',
                   help='Number of layers in RNN')
parser.add_argument('--fc_units', type=int, default=16, nargs='+',
                   help='Number of fully connected units on top of RNN state')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Input batch size')
parser.add_argument('--gamma', type=float, default=0.01, nargs='+',
                   help='gamma parameter of DILATE loss')

# Hierarchical model arguments
parser.add_argument('--L', type=int, default=2,
                    help='number of levels in the hierarchy, leaves inclusive')
parser.add_argument('--K', type=int, default=2,
                    help='number of bins to aggregate')

#parser.add_argument('--patience', type=int, default=2,
#                    help='Number of epochs to wait for \
#                          before beginning cross-validation')
#parser.add_argument('--seed', type=int,
#                    help='Seed for parameter initialization',
#                    default=42)

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.base_model_names = [
    'seq2seqdilate',
    'seq2seqmse',
    'seq2seqnll'
]
args.inference_model_names = [
    'DILATE',
    'MSE',
    'seq2seqmse_dualtpp',
    'seq2seqnll_dualtpp',
    'seq2seqmse_optls',
    'seq2seqnll_optls',
]
args.aggregate_methods = ['sum', 'leastsquare']

base_models = {}
for name in args.base_model_names:
    base_models[name] = {}
inference_models = {}
for name in args.inference_model_names:
    inference_models[name] = {}


os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)

model2metrics = dict()
infmodel2preds = dict()


dataset = utils.get_processed_data(args)
level2data = dataset['level2data']
# ----- Start: base models training ----- #
for base_model_name in args.base_model_names:
    base_models[base_model_name] = {}

    levels = range(args.L)
    aggregate_methods = args.aggregate_methods
    if base_model_name in ['seq2seqdilate']:
        levels = [0]
        aggregate_methods = ['sum']

    for agg_method in aggregate_methods:
        base_models[base_model_name][agg_method] = {}
        level2data = dataset['level2data'][agg_method]

        for level in levels:
            trainloader = level2data[level]['trainloader']
            devloader = level2data[level]['devloader']
            testloader = level2data[level]['testloader']
            N_output = level2data[level]['N_output']
            input_size = level2data[level]['input_size']
            output_size = level2data[level]['output_size']

            if base_model_name in ['seq2seqmse', 'seq2seqdilate']:
                point_estimates = True
            elif base_model_name in ['seq2seqnll']:
                point_estimates = False

            saved_models_dir = os.path.join(
                args.saved_models_dir,
                args.dataset_name+'_'+base_model_name+'_'+agg_method+'_'+str(level)
            )
            os.makedirs(saved_models_dir, exist_ok=True)
            saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
            output_dir = os.path.join(args.output_dir, base_model_name)
            os.makedirs(output_dir, exist_ok=True)
            print('\n {} {} {}'.format(base_model_name, agg_method, str(level)))

            encoder = EncoderRNN(
                input_size=input_size, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
                batch_size=args.batch_size
            ).to(args.device)
            decoder = DecoderRNN(
                input_size=input_size, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
                fc_units=args.fc_units, output_size=output_size
            ).to(args.device)
            net_gru = Net_GRU(encoder,decoder, N_output, point_estimates, args.device).to(args.device)
            train_model(
                args, base_model_name, net_gru, trainloader, devloader, testloader,
                saved_models_path, output_dir, eval_every=50, verbose=1
            )

            base_models[base_model_name][agg_method][level] = net_gru
# ----- End: base models training ----- #

# ----- Start: Inference models ----- #
print('\n Starting Inference Models')

test_inputs_dict = dict()
test_targets_dict = dict()
for agg_method in args.aggregate_methods:
    test_inputs_dict[agg_method] = dict()
    test_targets_dict[agg_method] = dict()
    for level in range(args.L):
        gen_test = iter(dataset['level2data'][agg_method][level]['testloader'])
        test_inputs, test_targets, breaks = next(gen_test)

        test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(args.device)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).to(args.device)
        test_inputs_dict[agg_method][level] = test_inputs
        test_targets_dict[agg_method][level] = test_targets
#criterion = torch.nn.MSELoss()

for inf_model_name in args.inference_model_names:
    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']['sum']
        inf_net = inf_models.DILATE(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets = test_targets_dict['sum'][0]
    elif inf_model_name in ['MSE']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.MSE(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets = test_targets_dict['sum'][0]
    elif inf_model_name in ['seq2seqmse_dualtpp']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.DualTPP(args.K, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets = test_targets_dict['sum'][0]
    elif inf_model_name in ['seq2seqnll_dualtpp']:
        base_models_dict = base_models['seq2seqnll']['sum']
        inf_net = inf_models.DualTPP(args.K, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets = test_targets_dict['sum'][0]
    elif inf_model_name in ['seq2seqmse_optls']:
        base_models_dict = base_models['seq2seqmse']['leastsquare']
        inf_net = inf_models.OPT_ls(args.K, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['leastsquare']
        inf_test_targets = test_targets_dict['sum'][0]
    elif inf_model_name in ['seq2seqnll_optls']:
        base_models_dict = base_models['seq2seqnll']['leastsquare']
        inf_net = inf_models.OPT_ls(args.K, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['leastsquare']
        inf_test_targets = test_targets_dict['sum'][0]

    inf_net.eval()
    preds, metric_mse, metric_dtw, metric_tdi = eval_inf_model(
        args, inf_net, inf_test_inputs_dict, inf_test_targets, args.gamma, verbose=1
    )
    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('MSE metric for Inference model {}: {:f}'.format(inf_model_name, metric_mse))

    model2metrics = utils.add_metrics_to_dict(
        model2metrics, inf_model_name, metric_mse, metric_dtw, metric_tdi
    )
    infmodel2preds[inf_model_name] = preds
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    utils.write_arr_to_file(
        output_dir, inf_model_name,
        test_inputs_dict['sum'][0].detach().numpy(),
        test_targets_dict['sum'][0].detach().numpy(),
        preds.detach().numpy()
    )


# ----- End: Inference models ----- #

with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, DTW, TDI')
    for model_name, metrics_dict in model2metrics.items():
        fp.write(
            '\n{}, {:.3f}, {:.3f}, {:.3f}'.format(
                model_name,
                metrics_dict['mse'],
                metrics_dict['dtw'],
                metrics_dict['tdi'],
            )
        )

for model_name, metrics_dict in model2metrics.items():
    for metric, metric_val in metrics_dict.items():
        model2metrics[model_name][metric] = str(metric_val)
with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.json'), 'w') as fp:
    json.dump(model2metrics, fp)

# Visualize results


for ind in range(1,51):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)
    k = 1
    for inf_mdl_name, preds in infmodel2preds.items():

        input = test_inputs_dict['sum'][0].detach().cpu().numpy()[ind,:,:]
        target = test_targets_dict['sum'][0].detach().cpu().numpy()[ind,:,:]
        preds = preds.detach().cpu().numpy()[ind,:,:]

        plt.subplot(len(inference_models),1,k)
        plt.plot(range(0,args.N_input) ,input,label='input',linewidth=3)
        plt.plot(range(args.N_input-1,args.N_input+args.N_output), np.concatenate([ input[args.N_input-1:args.N_input], target ]) ,label='target',linewidth=3)
        plt.plot(range(args.N_input-1,args.N_input+args.N_output),  np.concatenate([ input[args.N_input-1:args.N_input], preds ])  ,label=inf_mdl_name,linewidth=3)
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.show()
