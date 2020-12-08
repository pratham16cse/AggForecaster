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
parser.add_argument('--ignore_ckpt', action='store_true', default=False,
                    help='Start the training without loading the checkpoint')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='Normalize the dataset using average')

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
parser.add_argument('--alpha', type=float, default=0.5,
                   help='alpha parameter of DILATE loss')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                   help='Probability of applying teacher forcing to a batch')
parser.add_argument('--deep_std', action='store_true', default=False,
                    help='Extra layers for prediction of standard deviation')
parser.add_argument('--train_twostage', action='store_true', default=False,
                    help='Train base model in two stages -- train only \
                          mean in first stage, train both in second stage')
parser.add_argument('--mse_loss_with_nll', action='store_true', default=False,
                    help='Add extra mse_loss when training with nll')
parser.add_argument('--second_moment', action='store_true', default=False,
                    help='compute std as std = second_moment - mean')
parser.add_argument('--variance_rnn', action='store_true', default=False,
                    help='Use second RNN to compute variance or variance related values')


parser.add_argument('--use_time_features', action='store_true', default=False,
                    help='Use time features derived from calendar-date')


# Hierarchical model arguments
parser.add_argument('--L', type=int, default=2,
                    help='number of levels in the hierarchy, leaves inclusive')
parser.add_argument('--K_list', type=int, nargs='*', default=[1],
                    help='List of bin sizes of each aggregation')
parser.add_argument('--wavelet_levels', type=int, default=2,
                    help='number of levels of wavelet coefficients')

parser.add_argument('--plot_anecdotes', action='store_true', default=False,
                    help='Plot the comparison of various methods')
parser.add_argument('--save_agg_preds', action='store_true', default=False,
                    help='Save inputs, targets, and predictions of aggregate base models')

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
    'NLLsum',
#    'NLLls',
    'seq2seqmse_dualtpp',
    'seq2seqnll_dualtpp',
#    'seq2seqmse_optls',
#    'seq2seqnll_optls',
    'seq2seqmse_optst',
    'seq2seqnll_optst',
    'seq2seqnll_optklst',
#    'seq2seqmse_wavelet',
#    'seq2seqnll_wavelet',
]
args.aggregate_methods = [
    'sum',
#    'leastsquare',
#    'sumwithtrend',
    'slope',
#    'wavelet'
]

if 1 not in args.K_list:
    args.K_list = [1] + args.K_list

if args.dataset_name in ['Traffic']:
    args.alpha = 0.8

if args.dataset_name in ['ECG5000']:
    args.teacher_forcing_ratio = 0.0

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
#level2data = dataset['level2data']
# ----- Start: base models training ----- #
for base_model_name in args.base_model_names:
    base_models[base_model_name] = {}

    levels = args.K_list
    aggregate_methods = args.aggregate_methods
    if base_model_name in ['seq2seqdilate']:
        levels = [1]
        aggregate_methods = ['sum']

    for agg_method in aggregate_methods:
        base_models[base_model_name][agg_method] = {}
        level2data = dataset[agg_method]

        if agg_method in ['wavelet']:
            levels = list(range(1, args.wavelet_levels+3))

        for level in levels:
            trainloader = level2data[level]['trainloader']
            devloader = level2data[level]['devloader']
            testloader = level2data[level]['testloader']
            N_output = level2data[level]['N_output']
            input_size = level2data[level]['input_size']
            output_size = level2data[level]['output_size']
            norm = level2data[level]['norm']

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
                fc_units=args.fc_units, output_size=output_size, deep_std=args.deep_std,
                second_moment=args.second_moment, variance_rnn=args.variance_rnn
            ).to(args.device)
            net_gru = Net_GRU(
                encoder,decoder, N_output, args.use_time_features,
                point_estimates, args.teacher_forcing_ratio, args.deep_std,
                args.device
            ).to(args.device)
            if agg_method in ['leastsquare', 'sumwithtrend', 'slope', 'wavelet'] and level == 1:
                base_models[base_model_name][agg_method][level] = base_models[base_model_name]['sum'][1]
            else:
                train_model(
                    args, base_model_name, net_gru,
                    trainloader, devloader, testloader, norm,
                    saved_models_path, output_dir, eval_every=50, verbose=1
                )

                base_models[base_model_name][agg_method][level] = net_gru

            if args.save_agg_preds:
                (
                    dev_inputs, dev_target, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae
                ) = eval_base_model(
                    args, base_model_name,
                    base_models[base_model_name][agg_method][level],
                    testloader, norm,
                    args.gamma, verbose=1
                )

                output_dir = os.path.join(args.output_dir, args.dataset_name + '_base')
                os.makedirs(output_dir, exist_ok=True)
                utils.write_aggregate_preds_to_file(
                    output_dir, base_model_name, agg_method, level,
                    utils.unnormalize(dev_inputs.detach().numpy(), norm.detach().numpy()),
                    dev_target.detach().numpy(),
                    pred_mu.detach().numpy(),
                    pred_std.detach().numpy()
                )

            #import ipdb
            #ipdb.set_trace()
# ----- End: base models training ----- #

# ----- Start: Inference models ----- #
print('\n Starting Inference Models')

test_inputs_dict = dict()
test_targets_dict = dict()
test_norm_dict = dict()
test_feats_in_dict = dict()
test_feats_tgt_dict = dict()
for agg_method in args.aggregate_methods:
    test_inputs_dict[agg_method] = dict()
    test_targets_dict[agg_method] = dict()
    test_norm_dict[agg_method] = dict()
    test_feats_in_dict[agg_method] = dict()
    test_feats_tgt_dict[agg_method] = dict()

    if agg_method in ['wavelet']:
        levels = list(range(1, args.wavelet_levels+3))
    else:
        levels = args.K_list

    for level in levels:
        gen_test = iter(dataset[agg_method][level]['testloader'])
        test_inputs, test_targets, test_feats_in, test_feats_tgt, breaks = next(gen_test)

        test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(args.device)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).to(args.device)
        test_feats_in  = torch.tensor(test_feats_in, dtype=torch.float32).to(args.device)
        test_feats_tgt = torch.tensor(test_feats_tgt, dtype=torch.float32).to(args.device)

        test_inputs_dict[agg_method][level] = test_inputs
        test_targets_dict[agg_method][level] = test_targets
        test_norm_dict[agg_method][level] = dataset[agg_method][level]['norm']
        test_feats_in_dict[agg_method][level] = test_feats_in
        test_feats_tgt_dict[agg_method][level] = test_feats_tgt
#criterion = torch.nn.MSELoss()

for inf_model_name in args.inference_model_names:
    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']['sum']
        inf_net = inf_models.DILATE(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
    elif inf_model_name in ['MSE']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.MSE(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
    elif inf_model_name in ['NLLsum']:
        base_models_dict = base_models['seq2seqnll']['sum']
        inf_net = inf_models.NLLsum(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
    elif inf_model_name in ['NLLls']:
        base_models_dict = base_models['seq2seqnll']['leastsquare']
        inf_net = inf_models.NLLls(base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['leastsquare']
        inf_test_norm_dict = test_norm_dict['leastsquare']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['leastsquare']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['leastsquare']
    elif inf_model_name in ['seq2seqmse_dualtpp']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
    elif inf_model_name in ['seq2seqnll_dualtpp']:
        base_models_dict = base_models['seq2seqnll']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
    elif inf_model_name in ['seq2seqmse_optls']:
        base_models_dict = base_models['seq2seqmse']['leastsquare']
        inf_net = inf_models.OPT_ls(args.K_list, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['leastsquare']
        inf_test_norm_dict = test_norm_dict['leastsquare']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['leastsquare']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['leastsquare']
    elif inf_model_name in ['seq2seqnll_optls']:
        base_models_dict = base_models['seq2seqnll']['leastsquare']
        inf_net = inf_models.OPT_ls(args.K_list, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict['leastsquare']
        inf_test_norm_dict = test_norm_dict['leastsquare']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['leastsquare']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['leastsquare']
    elif inf_model_name in ['seq2seqmse_optst']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, intercept_type='sum')
        inf_test_inputs_dict = test_inputs_dict
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict
    elif inf_model_name in ['seq2seqnll_optst']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, intercept_type='sum')
        inf_test_inputs_dict = test_inputs_dict
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict
    elif inf_model_name in ['seq2seqnll_optklst']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_KL_st(args.K_list, base_models_dict, intercept_type='sum')
        inf_test_inputs_dict = test_inputs_dict
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict
    elif inf_model_name in ['seq2seqmse_wavelet']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.WAVELET(args.wavelet_levels, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
    elif inf_model_name in ['seq2seqnll_wavelet']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.WAVELET(args.wavelet_levels, base_models_dict)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]

    inf_net.eval()
    pred_mu, pred_std, metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae = eval_inf_model(
        args, inf_net, inf_test_inputs_dict, inf_test_norm_dict,
        inf_test_targets, inf_norm,
        inf_test_feats_in_dict, inf_test_feats_tgt_dict,
        args.gamma, verbose=1
    )
    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('Metrics for Inference model {}: MAE:{:f}, CRPS:{:f}, MSE:{:f}, DTW:{:f}, TDI:{:f}'.format(
        inf_model_name, metric_mae, metric_crps, metric_mse, metric_dtw, metric_tdi)
    )

    model2metrics = utils.add_metrics_to_dict(
        model2metrics, inf_model_name,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae
    )
    infmodel2preds[inf_model_name] = pred_mu
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    utils.write_arr_to_file(
        output_dir, inf_model_name,
        utils.unnormalize(test_inputs_dict['sum'][1].detach().numpy(), inf_norm.detach().numpy()),
        test_targets_dict['sum'][1].detach().numpy(),
        pred_mu.detach().numpy(),
        pred_std.detach().numpy()
    )


# ----- End: Inference models ----- #

with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, DTW, TDI')
    for model_name, metrics_dict in model2metrics.items():
        fp.write(
            '\n{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                model_name,
                metrics_dict['mae'],
                metrics_dict['crps'],
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


if args.plot_anecdotes:
    for ind in range(1,51):
        plt.figure()
        plt.rcParams['figure.figsize'] = (16.0,8.0)
        k = 1
        for inf_mdl_name, pred_mu in infmodel2preds.items():

            input = test_inputs_dict['sum'][1].detach().cpu().numpy()[ind,:,:]
            target = test_targets_dict['sum'][1].detach().cpu().numpy()[ind,:,:]
            pred_mu = pred_mu.detach().cpu().numpy()[ind,:,:]

            plt.subplot(len(inference_models),1,k)
            plt.plot(range(0,args.N_input) ,input,label='input',linewidth=3)
            plt.plot(range(args.N_input-1,args.N_input+args.N_output), np.concatenate([ input[args.N_input-1:args.N_input], target ]) ,label='target',linewidth=3)
            plt.plot(range(args.N_input-1,args.N_input+args.N_output),  np.concatenate([ input[args.N_input-1:args.N_input], pred_mu ])  ,label=inf_mdl_name,linewidth=3)
            plt.xticks(range(0,40,2))
            plt.legend()
            k = k+1

        plt.show()
