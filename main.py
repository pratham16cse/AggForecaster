import sys
import os
import argparse
import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.base_models import EncoderRNN, DecoderRNN, Net_GRU, NetFullyConnected, get_base_model
from models.index_models import get_index_model
from loss.dilate_loss import dilate_loss
from train import train_model, get_optimizer
from eval import eval_base_model, eval_inf_model, eval_inf_index_model
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import json
from torch.utils.tensorboard import SummaryWriter
import shutil
import properscoring as ps
import scipy.stats

from functools import partial

from models import inf_models, inf_index_models
import utils

os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "1000000"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, help='dataset_name')
#parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--N_input', type=int, default=-1,
                    help='number of input steps')
parser.add_argument('--N_output', type=int, default=-1,
                    help='number of output steps')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs', default=None)
parser.add_argument('--saved_models_dir', type=str,
                    help='Path to store all saved models', default=None)

parser.add_argument('--ignore_ckpt', action='store_true', default=False,
                    help='Start the training without loading the checkpoint')

parser.add_argument('--normalize', type=str, default=None,
                    choices=['same', 'zscore_per_series', 'gaussian_copula', 'log'],
                    help='Normalization type (avg, avg_per_series, quantile90, std)')
parser.add_argument('--epochs', type=int, default=-1,
                    help='number of training epochs')

parser.add_argument('--print_every', type=int, default=50,
                    help='Print test output after every print_every epochs')

parser.add_argument('--learning_rate', type=float, default=-1.,# nargs='+',
                   help='Learning rate for the training algorithm')
parser.add_argument('--hidden_size', type=int, default=-1,# nargs='+',
                   help='Number of units in the encoder/decoder state of the model')
parser.add_argument('--num_grulstm_layers', type=int, default=-1,# nargs='+',
                   help='Number of layers of the model')

parser.add_argument('--fc_units', type=int, default=16, #nargs='+',
                   help='Number of fully connected units on top of the encoder/decoder state of the model')

parser.add_argument('--batch_size', type=int, default=-1,
                    help='Input batch size')

parser.add_argument('--gamma', type=float, default=0.01, nargs='+',
                   help='gamma parameter of DILATE loss')
parser.add_argument('--alpha', type=float, default=0.5,
                   help='alpha parameter of DILATE loss')
parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
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
parser.add_argument('--input_dropout', type=float, default=0.0,
                    help='Dropout on input layer')

parser.add_argument('--v_dim', type=int, default=-1,
                   help='Dimension of V vector in LowRankGaussian')
parser.add_argument('--b', type=int, default=-1,
                   help='Number of correlation terms to sample for loss computation during training')

#parser.add_argument('--use_feats', action='store_true', default=False,
#                    help='Use time features derived from calendar-date and other covariates')
parser.add_argument('--use_feats', type=int, default=-1,
                    help='Use time features derived from calendar-date and other covariates')

parser.add_argument('--t2v_type', type=str,
                    choices=['local', 'idx', 'mdh_lincomb', 'mdh_parti'],
                    help='time2vec type', default=None)

parser.add_argument('--use_coeffs', action='store_true', default=False,
                    help='Use coefficients obtained by decomposition, wavelet, etc..')


# Hierarchical model arguments
parser.add_argument('--L', type=int, default=2,
                    help='number of levels in the hierarchy, leaves inclusive')

parser.add_argument('--K_list', type=int, nargs='*', default=[],
                    help='List of bin sizes of each aggregation')

parser.add_argument('--wavelet_levels', type=int, default=2,
                    help='number of levels of wavelet coefficients')
parser.add_argument('--fully_connected_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a feed-forward network')
parser.add_argument('--transformer_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a Transformer')
parser.add_argument('--plot_anecdotes', action='store_true', default=False,
                    help='Plot the comparison of various methods')
parser.add_argument('--save_agg_preds', action='store_true', default=False,
                    help='Save inputs, targets, and predictions of aggregate base models')

parser.add_argument('--device', type=str,
                    help='Device to run on', default=None)

# parameters for ablation study
parser.add_argument('--leak_agg_targets', action='store_true', default=False,
                    help='If True, aggregate targets are leaked to inference models')
parser.add_argument('--patience', type=int, default=20,
                    help='Stop the training if no improvement shown for these many \
                          consecutive steps.')
#parser.add_argument('--seed', type=int,
#                    help='Seed for parameter initialization',
#                    default=42)

# Parameters for ARTransformerModel
parser.add_argument('--kernel_size', type=int, default=10,
                    help='Kernel Size of Conv (in ARTransformerModel)')
parser.add_argument('--nkernel', type=int, default=32,
                    help='Number of kernels of Conv (in ARTransformerModel)')
parser.add_argument('--dim_ff', type=int, default=512,
                    help='Dimension of Feedforward (in ARTransformerModel)')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads (in ARTransformerModel)')


args = parser.parse_args()

#args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.base_model_names = [
#    'seq2seqdilate',
#    'seq2seqnll',
#    'seq2seqmse',
#    'convmse',
#    'convmsenonar',
#    'convnll',
#    'rnn-aggnll-nar',
#    'rnn-q-nar',
#    'rnn-q-ar',
#    'trans-mse-nar',
#    'trans-q-nar',
#    'nbeats-mse-nar',
#    'nbeatsd-mse-nar'
#    'rnn-mse-ar',
#    'rnn-mse-nar',
#    'rnn-nll-ar',
#    'trans-mse-ar',
    'trans-nll-ar',
#    'trans-fnll-ar',
#    'rnn-nll-nar',
#    'rnn-fnll-nar',
#    'transm-nll-nar',
#    'transm-fnll-nar',
#    'transda-nll-nar',
#    'transda-fnll-nar',
]
args.aggregate_methods = [
    'sum',
#    'sumwithtrend',
    'slope',
#    'wavelet'
]

args.inference_model_names = []
if 'seq2seqdilate' in args.base_model_names:
    args.inference_model_names.append('DILATE')
if 'seq2seqmse' in args.base_model_names:
    args.inference_model_names.append('MSE')
    args.inference_model_names.append('seq2seqmse_dualtpp')
    args.inference_model_names.append('seq2seqmse_optst')
    args.inference_model_names.append('seq2seqmse_opttrend')
if 'seq2seqnll' in args.base_model_names:
    args.inference_model_names.append('NLL')
    args.inference_model_names.append('seq2seqnll_dualtpp')
    args.inference_model_names.append('seq2seqnll_optst')
    args.inference_model_names.append('seq2seqnll_opttrend')
    #args.inference_model_names.append('seq2seqnll_optklst')
    #args.inference_model_names.append('seq2seqnll_optkls')
    #args.inference_model_names.append('seq2seqnll_optklt')
if 'convmse' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-MSE')
    args.inference_model_names.append('convmse_dualtpp')
    #args.inference_model_names.append('convmse_dualtpp_cf')
    #args.inference_model_names.append('convmse_optst')
    #args.inference_model_names.append('convmse_opttrend')
if 'convnll' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-NLL')
    args.inference_model_names.append('convnll_dualtpp')
    #args.inference_model_names.append('convnll_optst')
    #args.inference_model_names.append('convnll_opttrend')
    #args.inference_model_names.append('convnll_optklst')
    #args.inference_model_names.append('convnll_optkls')
if 'convmsenonar' in args.base_model_names:
    args.inference_model_names.append('CNNRNN-NONAR-MSE')
    args.inference_model_names.append('convmse_nonar_dualtpp')
    #args.inference_model_names.append('convmse_nonar_dualtpp_cf')
    #args.inference_model_names.append('convmse_optst')
    #args.inference_model_names.append('convmse_opttrend')
   #args.inference_model_names.append('convnll_optklt')
if 'rnn-aggnll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-AGGNLL-NAR')
if 'rnn-q-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-NAR')
if 'rnn-mse-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-AR')
if 'rnn-q-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-AR')
if 'trans-mse-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-NAR')
    #args.inference_model_names.append('convmse_nonar_dualtpp')
if 'trans-q-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-Q-NAR')
    #args.inference_model_names.append('convmse_nonar_dualtpp')
if 'nbeats-mse-nar' in args.base_model_names:
    args.inference_model_names.append('NBEATS-MSE-NAR')
if 'nbeatsd-mse-nar' in args.base_model_names:
    args.inference_model_names.append('NBEATSD-MSE-NAR')
if 'rnn-mse-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-NAR')
    args.inference_model_names.append('rnn-mse-nar_opt-sum')
    args.inference_model_names.append('rnn-mse-nar_optcf-sum')
    #args.inference_model_names.append('rnn-mse-nar_opt-slope')
    #args.inference_model_names.append('rnn-mse-nar_kl-sum')
    #args.inference_model_names.append('rnn-mse-nar_kl-st')
if 'rnn-nll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-NAR')
    #args.inference_model_names.append('rnn-nll-nar_opt-sum')
    #args.inference_model_names.append('rnn-nll-nar_optcf-sum')
    #args.inference_model_names.append('rnn-nll-nar_opt-slope')
    #args.inference_model_names.append('rnn-nll-nar_opt-st')
    #args.inference_model_names.append('rnn-nll-nar_kl-sum')
    args.inference_model_names.append('rnn-nll-nar_kl-st')
if 'rnn-nll-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-AR')
    #args.inference_model_names.append('rnn-nll-ar_opt-sum')
    #args.inference_model_names.append('rnn-nll-ar_opt-slope')
    #args.inference_model_names.append('rnn-nll-ar_opt-st')
    args.inference_model_names.append('rnn-nll-ar_kl-sum')
    args.inference_model_names.append('rnn-nll-ar_kl-st')
if 'trans-mse-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-AR')
if 'trans-nll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-AR')
    #args.inference_model_names.append('trans-nll-ar_opt-sum')
    #args.inference_model_names.append('trans-nll-ar_optcf-sum')
    #args.inference_model_names.append('trans-nll-ar_opt-slope')
    #args.inference_model_names.append('trans-nll-ar_opt-st')
    args.inference_model_names.append('trans-nll-ar_kl-sum')
    args.inference_model_names.append('trans-nll-ar_kl-st')
if 'trans-fnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-FNLL-AR')
   #args.inference_model_names.append('trans-nll-ar_kl-st')
if 'rnn-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-FNLL-NAR')
if 'transm-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSM-NLL-NAR')
if 'transm-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSM-FNLL-NAR')
if 'transda-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSDA-NLL-NAR')
if 'transda-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSDA-FNLL-NAR')




if args.dataset_name in ['Traffic']:
    args.alpha = 0.8

if args.dataset_name in ['ECG5000']:
    args.teacher_forcing_ratio = 0.0

if args.dataset_name in ['Solar']:
    opt_normspace = False
else:
    opt_normspace = True

#import ipdb ; ipdb.set_trace()
if args.dataset_name == 'ett':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 192
    if args.N_output == -1: args.N_output = 192
    if args.K_list == []: args.K_list = []
    #args.K_list = [6]
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_ett_d192_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    if args.output_dir is None:
        args.output_dir = 'Outputs_ett_d192_klnorm_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size  == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = 'idx'
    if args.device is None: args.device = 'cuda:2'
    #python main.py ett --epochs 20 --N_input 192 --N_output 192 --K_list 6 --saved_models_dir saved_models_ett_d192 --output_dir Outputs_ett_d192_klnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --device cuda:0

elif args.dataset_name == 'taxi30min':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_taxi30min_d168_b48_pefix_e336_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2vglobal_mdh_parti'
    if args.output_dir is None:
        args.output_dir = 'Outputs_taxi30min_d168_klnorm_b48_pefix_e336_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2vglobal_mdh_parti'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    #args.t2v_type = 'mdh_parti'
    if args.device is None: args.device = 'cuda:2'

elif args.dataset_name == 'etthourly':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 168
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_etthourly_noextrafeats_d168_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_etthourly_noextrafeats_d168_klnorm_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    #args.print_every = 5 # TODO: Only for aggregate models
    if args.device is None: args.device = 'cuda:2'

elif args.dataset_name == 'azure':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 720
    if args.N_output == -1: args.N_output = 360
    #args.K_list = [60]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_azure_d360_e720_usefeats_bs128_normsame'
    if args.output_dir is None:   
        args.output_dir = 'Outputs_azure_d360_e720_usefeats_bs128_normsame'
    #args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'same'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 10
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = None
    if args.device is None: args.device = 'cuda:0'

elif args.dataset_name == 'Solar':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_Solar_d168_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_Solar_d168_normzscore_klnorm_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.device is None: args.device = 'cuda:1'

elif args.dataset_name == 'aggtest':
    if args.epochs == -1: args.epochs = 1
    if args.N_input == -1: args.N_input = 20
    if args.N_output == -1: args.N_output = 10
    #args.K_list = [12]
    if args.K_list == []: args.K_list = [1, 5]
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_aggtest_test'
    if args.output_dir is None:
        args.output_dir = 'Outputs_aggtest_test'
    if args.normalize is None: args.normalize = 'same'
    if args.learning_rate == -1.: args.learning_rate = 0.001
    if args.batch_size == -1: args.batch_size = 100
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = args.N_output
    if args.device is None: args.device = 'cuda:2'


elif args.dataset_name == 'Traffic911':
    args.epochs = 20
    args.N_input = 336
    args.N_output = 168
    args.K_list = [6]
    args.saved_models_dir = 'saved_models_Traffic911_d168'
    args.output_dir = 'Outputs_Traffic911_d168'
    args.normalize = 'zscore_per_series'
    args.learning_rate = 0.0001
    args.batch_size = 128
    args.hidden_size = 128
    args.num_grulstm_layers = 1
    args.v_dim = 1
    args.print_every = 5 # TODO: Only for aggregate models
    args.device = 'cuda:0'

if 1 not in args.K_list:
    args.K_list = [1] + args.K_list

print('Command Line Arguments:')
print(args)

#import ipdb ; ipdb.set_trace()

base_models = {}
base_models_preds = {}
for name in args.base_model_names:
    base_models[name] = {}
    base_models_preds[name] = {}
inference_models = {}
for name in args.inference_model_names:
    inference_models[name] = {}


DUMP_PATH = '/mnt/infonas/data/pratham/Forecasting/DILATE'
args.output_dir = os.path.join(DUMP_PATH, args.output_dir)
args.saved_models_dir = os.path.join(DUMP_PATH, args.saved_models_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)

model2metrics = dict()
infmodel2preds = dict()


#dataset = utils.get_processed_data(args)
data_processor = utils.DataProcessor(args)
#level2data = dataset['level2data']

# ----- Start: Load all datasets ----- #

dataset = {}
for agg_method in args.aggregate_methods:
    dataset[agg_method] = {}
    for level in args.K_list:
        if level==1 and agg_method is not 'sum':
            dataset[agg_method][level] = dataset['sum'][1]
        else:
            dataset[agg_method][level] = data_processor.get_processed_data(args, agg_method, level)

# ----- End : Load all datasets ----- #

# ----- Start: base models training ----- #
for base_model_name in args.base_model_names:
    base_models[base_model_name] = {}
    base_models_preds[base_model_name] = {}

    levels = args.K_list
    aggregate_methods = args.aggregate_methods
    if base_model_name in ['seq2seqdilate']:
        levels = [1]
        aggregate_methods = ['sum']

    for agg_method in aggregate_methods:
        base_models[base_model_name][agg_method] = {}
        base_models_preds[base_model_name][agg_method] = {}
        #level2data = dataset[agg_method]

        if agg_method in ['wavelet']:
            levels = list(range(1, args.wavelet_levels+3))

        for level in levels:
            level2data = dataset[agg_method][level]
            trainloader = level2data['trainloader']
            devloader = level2data['devloader']
            testloader = level2data['testloader']
            feats_info = level2data['feats_info']
            N_input = level2data['N_input']
            N_output = level2data['N_output']
            input_size = level2data['input_size']
            output_size = level2data['output_size']
            dev_norm = level2data['dev_norm']
            test_norm = level2data['test_norm']

            if base_model_name in [
                'seq2seqmse', 'seq2seqdilate', 'convmse', 'convmsenonar',
                'rnn-mse-nar', 'rnn-mse-ar', 'trans-mse-nar', 'nbeats-mse-nar',
                'nbeatsd-mse-nar', 'trans-mse-ar',
            ]:
                estimate_type = 'point'
            elif base_model_name in [
                'seq2seqnll', 'convnll', 'trans-q-nar', 'rnn-q-nar', 'rnn-q-ar',
                'rnn-nll-nar', 'rnn-nll-ar', 'rnn-aggnll-nar', 'trans-nll-ar',
                'transm-nll-nar', 'transda-nll-nar',
            ]:
                estimate_type = 'variance'
            elif base_model_name in [
                'rnn-fnll-nar', 'trans-fnll-ar', 'transm-nll-nar', 'transda-fnll-nar'
            ]:
                estimate_type = 'covariance'

            saved_models_dir = os.path.join(
                args.saved_models_dir,
                args.dataset_name+'_'+base_model_name+'_'+agg_method+'_'+str(level)
            )
            os.makedirs(saved_models_dir, exist_ok=True)
            writer = SummaryWriter(saved_models_dir)
            saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
            output_dir = os.path.join(args.output_dir, base_model_name)
            os.makedirs(output_dir, exist_ok=True)
            print('\n {} {} {}'.format(base_model_name, agg_method, str(level)))


            # Create the network
            net_gru = get_base_model(
                args, base_model_name, level,
                N_input, N_output, input_size, output_size,
                estimate_type, feats_info
            )
    
            # train the network
            if agg_method in ['sumwithtrend', 'slope', 'wavelet'] and level == 1:
                base_models[base_model_name][agg_method][level] = base_models[base_model_name]['sum'][1]
            else:
                train_model(
                    args, base_model_name, net_gru,
                    level2data, estimate_type,
                    saved_models_path, output_dir, writer, verbose=1
                )

                base_models[base_model_name][agg_method][level] = net_gru

            writer.flush()

            if args.save_agg_preds and level>=1:
                testloader = level2data['testloader']
                test_norm = level2data['test_norm']
                print(agg_method, level, level2data['N_output'])
                (
                    test_inputs, test_target, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae, metric_crps_part, metric_nll
                ) = eval_base_model(
                    args, base_model_name,
                    base_models[base_model_name][agg_method][level],
                    testloader, test_norm,
                    args.gamma, verbose=1
                )
                test_target = utils.unnormalize(test_target.detach().numpy(), test_norm, is_var=False)
                pred_mu = utils.unnormalize(pred_mu.detach().numpy(), test_norm, is_var=False)
                pred_std = utils.unnormalize(pred_std.detach().numpy(), test_norm, is_var=True)

                output_dir = os.path.join(args.output_dir, args.dataset_name + '_base')
                os.makedirs(output_dir, exist_ok=True)
                utils.write_aggregate_preds_to_file(
                    output_dir, base_model_name, agg_method, level,
                    utils.unnormalize(test_inputs.detach().numpy(), test_norm, is_var=False),
                    test_target,#.detach().numpy(),
                    pred_mu,#.detach().numpy(),
                    pred_std,#.detach().numpy()
                )

                # Aggregate level 1 predictions using current aggregation.
                base_models_preds[base_model_name][agg_method][level] = [pred_mu, pred_std]

                test_target = test_target#.detach().numpy()
                pred_mu = pred_mu#.detach().numpy()
                pred_std = pred_std#.detach().numpy()
                pred_mu_bottom = base_models_preds[base_model_name][agg_method][1][0]#.detach().numpy()
                pred_std_bottom = base_models_preds[base_model_name][agg_method][1][1]#.detach().numpy()
                if level != 1:
                    if agg_method in ['slope']:
                        pred_mu_agg = utils.aggregate_seqs_slope(pred_mu_bottom, level, is_var=False)
                        pred_std_agg = np.sqrt(utils.aggregate_seqs_slope(pred_std_bottom**2, level, is_var=True))
                    elif agg_method in ['sum']:
                        pred_mu_agg = utils.aggregate_seqs_sum(pred_mu_bottom, level, is_var=False)
                        pred_std_agg = np.sqrt(utils.aggregate_seqs_sum(pred_std_bottom**2, level, is_var=True))
                        #import ipdb
                        #ipdb.set_trace()
                else:
                    pred_mu_agg = pred_mu_bottom
                    pred_std_agg = pred_std_bottom

                mae_agg = np.mean(np.abs(test_target - pred_mu_agg))
                mae_base = np.mean(np.abs(test_target - pred_mu))
                mse_agg = np.mean((test_target - pred_mu_agg)**2)
                mse_base = np.mean((test_target - pred_mu)**2)

                crps_agg = ps.crps_gaussian(
                    test_target, mu=pred_mu_agg, sig=pred_std_agg
                ).mean()
                crps_base = ps.crps_gaussian(
                    test_target, mu=pred_mu, sig=pred_std
                ).mean()
                nll_agg = scipy.stats.norm(
                    pred_mu_agg, pred_std_agg
                ).pdf(test_target).mean()
                nll_base = scipy.stats.norm(
                    pred_mu, pred_std
                ).pdf(test_target).mean()

                if level!=1:
                    h_t = test_inputs.shape[1]
                    n_e = test_target.shape[1]
                    plt_dir = os.path.join(
                        output_dir, 'plots', agg_method,
                        'level_'+str(level),
                    )
                    os.makedirs(plt_dir, exist_ok=True)
                    for i in range(0, test_inputs.shape[0]):
                        plt.plot(
                            np.arange(1, h_t+n_e+1),
                            np.concatenate([test_inputs[i,:,0][-h_t:], test_target[i,:,0]]),
                            'ko-'
                        )
                        plt.plot(np.arange(h_t+1, h_t+n_e+1), pred_mu[i,:,0], 'bo-')
                        plt.plot(np.arange(h_t+1, h_t+n_e+1), pred_mu_agg[i,:,0], 'ro-')
                        plt.savefig(
                            os.path.join(plt_dir, str(i)+'.svg'),
                            format='svg', dpi=1200
                        )
                        plt.close()

                mae_base_parts = []
                mae_agg_parts = []
                mse_base_parts = []
                mse_agg_parts = []
                N = test_target.shape[1]
                p = max(int(N/4), 1)
                for i in range(0, N, p):
                    mae_base_parts.append(
                        np.mean(
                            np.abs(test_target[:, i:i+p] - pred_mu[:, i:i+p])
                        )
                    )
                    mae_agg_parts.append(
                        np.mean(
                            np.abs(test_target[:, i:i+p] - pred_mu_agg[:, i:i+p])
                        )
                    )
                    mse_base_parts.append(
                        np.mean(
                            (test_target[:, i:i+p] - pred_mu[:, i:i+p])**2
                        )
                    )
                    mse_agg_parts.append(
                        np.mean(
                            (test_target[:, i:i+p] - pred_mu_agg[:, i:i+p])**2
                        )
                    )


                print('-------------------------------------------------------')
                print('{0}, {1}, {2}, mae_base:{3}, mae_agg:{4}'.format(
                    base_model_name, agg_method, level, mae_base, mae_agg)
                )
                print('{0}, {1}, {2}, crps_base:{3}, crps_agg:{4}'.format(
                    base_model_name, agg_method, level, crps_base, crps_agg)
                )
                print('mae_base_parts:', mae_base_parts)
                print('mae_agg_parts:', mae_agg_parts)
                print('-------------------------------------------------------')
                print('{0}, {1}, {2}, mse_base:{3}, mse_agg:{4}'.format(
                    base_model_name, agg_method, level, mse_base, mse_agg)
                )
                print('{0}, {1}, {2}, nll_base:{3}, nll_agg:{4}'.format(
                    base_model_name, agg_method, level, nll_base, nll_agg)
                )
                print('mse_base_parts:', mse_base_parts)
                print('mse_agg_parts:', mse_agg_parts)
                print('-------------------------------------------------------')


writer.close()
            #import ipdb
            #ipdb.set_trace()
# ----- End: base models training ----- #

# ----- Start: Inference models for bottom level----- #
print('\n Starting Inference Models')

#test_inputs_dict = dict()
#test_targets_dict = dict()
#test_targets_dict_leak = dict()
#mapped_id_dict = dict()
#test_feats_in_dict = dict()
#test_feats_tgt_dict = dict()
#test_inputs_gaps_dict = dict()
#test_targets_gaps_dict = dict()
#test_coeffs_in_dict = dict()
#N_input, N_output = 0, 0
#for agg_method in args.aggregate_methods:
#    test_inputs_dict[agg_method] = dict()
#    test_targets_dict[agg_method] = dict()
#    test_targets_dict_leak[agg_method] = dict()
#    mapped_id_dict[agg_method] = dict()
#    test_feats_in_dict[agg_method] = dict()
#    test_feats_tgt_dict[agg_method] = dict()
#    test_inputs_gaps_dict[agg_method] = dict()
#    test_targets_gaps_dict[agg_method] = dict()
#    test_coeffs_in_dict[agg_method] = dict()
#
#    if agg_method in ['wavelet']:
#        levels = list(range(1, args.wavelet_levels+3))
#    else:
#        levels = args.K_list
#
#    for level in levels:
#        lvl_data = dataset[agg_method][level]
#        test_inputs, test_targets = [], []
#        test_feats_in, test_feats_tgt = [], []
#        test_coeffs_in = []
#        mapped_id = []
#        test_inputs_gaps, test_targets_gaps = [], []
#        for i, gen_test in enumerate(lvl_data['testloader']):
#            (
#                batch_test_inputs, batch_test_targets,
#                batch_test_feats_in, batch_test_feats_tgt,
#                batch_mapped_id,
#                _, batch_test_coeffs_in, _, batch_test_inputs_gaps, batch_test_targets_gaps
#            ) = gen_test
#
#            test_inputs.append(batch_test_inputs)
#            test_targets.append(batch_test_targets)
#            test_feats_in.append(batch_test_feats_in)
#            test_feats_tgt.append(batch_test_feats_tgt)
#            mapped_id.append(batch_mapped_id)
#            test_inputs_gaps.append(batch_test_inputs_gaps)
#            test_targets_gaps.append(batch_test_targets_gaps)
#            test_coeffs_in.append(batch_test_coeffs_in)
#
#        test_inputs  = torch.cat(test_inputs, dim=0)#, dtype=torch.float32).to(args.device)
#        test_targets = torch.cat(test_targets, dim=0)#, dtype=torch.float32).to(args.device)
#        test_feats_in  = torch.cat(test_feats_in, dim=0)#, dtype=torch.float32).to(args.device)
#        test_feats_tgt = torch.cat(test_feats_tgt, dim=0)#, dtype=torch.float32).to(args.device)
#        mapped_id = torch.cat(mapped_id, dim=0)#, dtype=torch.float32).to(args.device)
#        test_inputs_gaps  = torch.cat(test_inputs_gaps, dim=0)#, dtype=torch.float32).to(args.device)
#        test_targets_gaps = torch.cat(test_targets_gaps, dim=0)#, dtype=torch.float32).to(args.device)
#        test_coeffs_in  = torch.cat(test_coeffs_in, dim=0)#, dtype=torch.float32).to(args.device)
#
#        test_inputs_dict[agg_method][level] = test_inputs
#        test_targets_dict[agg_method][level] = test_targets
#        #test_targets_dict_leak[agg_method][level], _ = utils.normalize(
#        #    test_targets, test_norm
#        #)
#        #test_targets_dict_leak[agg_method][level] = lvl_data['test_norm'].normalize(
#        #    test_targets, torch.tensor(lvl_data['test_tsid_map'])
#        #)
#        test_targets_dict_leak = test_targets_dict # TODO: Normalization correction
#        mapped_id_dict[agg_method][level] = mapped_id
#        test_feats_in_dict[agg_method][level] = test_feats_in
#        test_feats_tgt_dict[agg_method][level] = test_feats_tgt
#        test_inputs_gaps_dict[agg_method][level] = test_inputs_gaps
#        test_targets_gaps_dict[agg_method][level] = test_targets_gaps
#        test_coeffs_in_dict[agg_method][level] = test_coeffs_in
#
#        if level == 1:
#            N_input = lvl_data['N_input']
#            N_output = lvl_data['N_output']
#
#assert N_input > 0
#assert N_output > 0
#criterion = torch.nn.MSELoss()

#import ipdb
#ipdb.set_trace()
for inf_model_name in args.inference_model_names:

    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']['sum']
        inf_net = inf_models.DILATE(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['MSE']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.MSE(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['NLL']:
        base_models_dict = base_models['seq2seqnll']['sum']
        inf_net = inf_models.NLL(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['CNNRNN-MSE']:
        base_models_dict = base_models['convmse']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['RNN-MSE-NAR']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        #inf_test_inputs_dict = test_inputs_dict['sum']
        #inf_test_targets_dict = test_targets_dict_leak['sum']
        #inf_test_norm_dict = test_norm_dict['sum']
        #inf_test_norm_dict = None
        #inf_test_targets = test_targets_dict['sum'][1]
        #inf_norm = test_norm_dict['sum'][1]
        #inf_norm = lvl_data['test_norm']
        #inf_test_feats_in_dict = test_feats_in_dict['sum']
        #inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
        #inf_test_coeffs_in_dict = test_coeffs_in_dict['sum']

    elif inf_model_name in ['RNN-NLL-NAR']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        #inf_test_inputs_dict = test_inputs_dict['sum']
        #inf_test_targets_dict = test_targets_dict_leak['sum']
        ##inf_test_norm_dict = test_norm_dict['sum']
        #inf_test_norm_dict = None
        #inf_test_targets = test_targets_dict['sum'][1]
        ##inf_norm = test_norm_dict['sum'][1]
        #inf_norm = lvl_data['test_norm']
        #inf_test_feats_in_dict = test_feats_in_dict['sum']
        #inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
        #inf_test_coeffs_in_dict = test_coeffs_in_dict['sum']

    elif inf_model_name in ['rnn-mse-nar_opt-sum']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum'], device=args.device)
        
    elif inf_model_name in ['rnn-nll-nar_opt-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum'], device=args.device)

    elif inf_model_name in ['rnn-nll-nar_optcf-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-nar_optcf-sum']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-nar_opt-slope']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-nar_opt-slope']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-nar_opt-st']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum', 'slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-nar_kl-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['rnn-nll-nar_kl-st']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum', 'slope'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['RNN-NLL-AR']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_opt-sum']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum'], device=args.device)
        
    elif inf_model_name in ['rnn-nll-ar_opt-sum']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum'], device=args.device)

    elif inf_model_name in ['rnn-mse-ar_optcf-sum']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_opt-slope']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-ar_opt-slope']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-ar_opt-st']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum', 'slope'], device=args.device)

    elif inf_model_name in ['rnn-nll-ar_kl-sum']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['rnn-nll-ar_kl-st']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum', 'slope'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['TRANS-MSE-AR']:
        base_models_dict = base_models['trans-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-NLL-AR']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['trans-nll-ar_opt-sum']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum'], device=args.device)

    elif inf_model_name in ['trans-nll-ar_optcf-sum']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['trans-nll-ar_opt-slope']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['slope'], device=args.device)

    elif inf_model_name in ['trans-nll-ar_opt-st']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, ['sum', 'slope'], device=args.device)

    elif inf_model_name in ['trans-nll-ar_kl-sum']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['trans-nll-ar_kl-st']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.KLInference(
            args.K_list, base_models_dict, ['sum', 'slope'], device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['TRANS-FNLL-AR']:
        base_models_dict = base_models['trans-fnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-FNLL-NAR']:
        base_models_dict = base_models['rnn-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSM-NLL-NAR']:
        base_models_dict = base_models['transm-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSM-FNLL-NAR']:
        base_models_dict = base_models['transm-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSDA-NLL-NAR']:
        base_models_dict = base_models['transda-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANSDA-FNLL-NAR']:
        base_models_dict = base_models['transda-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-AGGNLL-NAR']:
        base_models_dict = base_models['rnn-aggnll-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        #inf_test_norm_dict = test_norm_dict['sum']
        inf_test_norm_dict = None
        inf_test_targets = test_targets_dict['sum'][1]
        #inf_norm = test_norm_dict['sum'][1]
        inf_norm = lvl_data['test_norm']
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['RNN-Q-NAR']:
        base_models_dict = base_models['rnn-q-nar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['RNN-MSE-AR']:
        base_models_dict = base_models['rnn-mse-ar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['RNN-Q-AR']:
        base_models_dict = base_models['rnn-q-ar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']


    elif inf_model_name in ['CNNRNN-NONAR-MSE']:
        base_models_dict = base_models['convmsenonar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']


    elif inf_model_name in ['CNNRNN-NLL']:
        base_models_dict = base_models['convnll']['sum']
        inf_net = inf_models.NLL(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['TRANS-MSE-NAR']:
        base_models_dict = base_models['trans-mse-nar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['TRANS-Q-NAR']:
        base_models_dict = base_models['trans-q-nar']['sum']
        inf_net = inf_models.CNNRNN(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['NBEATS-MSE-NAR']:
        base_models_dict = base_models['nbeats-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        #inf_test_norm_dict = test_norm_dict['sum']
        inf_test_norm_dict = None
        inf_test_targets = test_targets_dict['sum'][1]
        #inf_norm = test_norm_dict['sum'][1]
        inf_norm = lvl_data['test_norm']
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
        inf_test_coeffs_in_dict = test_coeffs_in_dict['sum']

    elif inf_model_name in ['NBEATSD-MSE-NAR']:
        base_models_dict = base_models['nbeatsd-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        #inf_test_norm_dict = test_norm_dict['sum']
        inf_test_norm_dict = None
        inf_test_targets = test_targets_dict['sum'][1]
        #inf_norm = test_norm_dict['sum'][1]
        inf_norm = lvl_data['test_norm']
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']
        inf_test_coeffs_in_dict = test_coeffs_in_dict['sum']

    elif inf_model_name in ['seq2seqmse_dualtpp']:
        base_models_dict = base_models['seq2seqmse']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['seq2seqnll_dualtpp']:
        base_models_dict = base_models['seq2seqnll']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['convnll_dualtpp']:
        base_models_dict = base_models['convnll']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['convmse_dualtpp']:
        base_models_dict = base_models['convmse']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['rnn-mse-nar_dualtpp']:
        base_models_dict = base_models['rnn-mse-nar']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['convmse_dualtpp_cf']:
        base_models_dict = base_models['convmse']['sum']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['convmse_nonar_dualtpp']:
        base_models_dict = base_models['convmsenonar']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['convmse_nonar_dualtpp_cf']:
        base_models_dict = base_models['convmsenonar']['sum']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['rnn-mse-nar_dualtpp_cf']:
        base_models_dict = base_models['rnn-mse-nar']['sum']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict['sum']
        inf_test_targets_dict = test_targets_dict_leak['sum']
        inf_test_norm_dict = test_norm_dict['sum']
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict['sum']
        inf_test_feats_tgt_dict = test_feats_tgt_dict['sum']

    elif inf_model_name in ['seq2seqmse_optst']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqnll_optst']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convnll_optst']:
        base_models_dict = base_models['convnll']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convmse_optst']:
        base_models_dict = base_models['convmse']
        inf_net = inf_models.OPT_st(args.K_list, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqmse_opttrend']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.OPT_st(
            args.K_list, base_models_dict, disable_sum=True, device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqnll_opttrend']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_st(
            args.K_list, base_models_dict, disable_sum=True, device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convmse_opttrend']:
        base_models_dict = base_models['convmse']
        inf_net = inf_models.OPT_st(
            args.K_list, base_models_dict, disable_sum=True, device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convnll_opttrend']:
        base_models_dict = base_models['convnll']
        inf_net = inf_models.OPT_st(
            args.K_list, base_models_dict, disable_sum=True, device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqnll_optklst']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['sum', 'slope'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convnll_optklst']:
        base_models_dict = base_models['convnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['sum', 'slope'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqnll_optkls']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['sum'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convnll_optkls']:
        base_models_dict = base_models['convnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['sum'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqnll_optklt']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['slope'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['convnll_optklt']:
        base_models_dict = base_models['convnll']
        inf_net = inf_models.OPT_KL_st(
            args.K_list, base_models_dict,
            agg_methods=['slope'],
            device=args.device
        )
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]
        inf_test_feats_in_dict = test_feats_in_dict
        inf_test_feats_tgt_dict = test_feats_tgt_dict

    elif inf_model_name in ['seq2seqmse_wavelet']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.WAVELET(args.wavelet_levels, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]

    elif inf_model_name in ['seq2seqnll_wavelet']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.WAVELET(args.wavelet_levels, base_models_dict, device=args.device)
        inf_test_inputs_dict = test_inputs_dict
        inf_test_targets_dict = test_targets_dict_leak
        inf_test_norm_dict = test_norm_dict
        inf_test_targets = test_targets_dict['sum'][1]
        inf_norm = test_norm_dict['sum'][1]

    if not args.leak_agg_targets:
        inf_test_targets_dict = None

    inf_net.eval()
    #(
    #    pred_mu, pred_std, pred_d, pred_v,
    #    metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape
    #)= eval_inf_model(
    #    args, inf_net, inf_test_inputs_dict, inf_test_norm_dict,
    #    inf_test_targets, inf_norm, mapped_id_dict['sum'][1],
    #    inf_test_feats_in_dict, inf_test_feats_tgt_dict,
    #    inf_test_coeffs_in_dict,
    #    args.gamma, inf_test_targets_dict=inf_test_targets_dict, verbose=1
    #)

    (
        inputs, target, pred_mu, pred_std, pred_d, pred_v,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape,
        total_time
    )= eval_inf_model(args, inf_net, dataset, args.gamma, verbose=1)

    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('Metrics for Inference model {}: MAE:{:f}, CRPS:{:f}, MSE:{:f}, SMAPE:{:f}, Time:{:f}'.format(
        inf_model_name, metric_mae, metric_crps, metric_mse, metric_smape, total_time)
    )

    model2metrics = utils.add_metrics_to_dict(
        model2metrics, inf_model_name,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape
    )
    infmodel2preds[inf_model_name] = pred_mu
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    utils.write_arr_to_file(
        output_dir, inf_model_name,
        inputs.detach().numpy(),
        target.detach().numpy(),
        pred_mu.detach().numpy(),
        pred_std.detach().numpy(),
        pred_d.detach().numpy(),
        pred_v.detach().numpy()
    )


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

# ----- End: Inference models for bottom level----- #


# ----- Start: Base models for all aggreagations and levels --- #

model2metrics = {}
for base_model_name in args.base_model_names:
    for agg_method in args.aggregate_methods:
        for K in args.K_list:

            print('Base Model', base_model_name,'for', agg_method, K)
    
            loader = dataset[agg_method][K]['testloader']
            norm = dataset[agg_method][K]['test_norm']
            (
                test_inputs, test_target, pred_mu, pred_std,
                metric_dilate, metric_mse, metric_dtw, metric_tdi,
                metric_crps, metric_mae, metric_crps_part, metric_nll, metric_ql
            ) = eval_base_model(
                args, base_model_name, base_models[base_model_name][agg_method][K],
                loader, norm, args.gamma, verbose=1, unnorm=True
            )

            output_dir = os.path.join(args.output_dir, args.dataset_name + '_base')
            os.makedirs(output_dir, exist_ok=True)
            utils.write_aggregate_preds_to_file(
                output_dir, base_model_name, agg_method, K,
                test_inputs, test_target, pred_mu, pred_std
            )

            model2metrics = utils.add_base_metrics_to_dict(
                model2metrics, agg_method, K, base_model_name,
                metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae
            )


with open(os.path.join(args.output_dir, 'results_base_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, DTW, TDI')
    for agg_method in model2metrics.keys():
        for K in model2metrics[agg_method].keys():
            for model_name, metrics_dict in model2metrics[agg_method][K].items():
                fp.write(
                    '\n{}, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                        agg_method, K, model_name,
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
with open(os.path.join(args.output_dir, 'results_base_'+args.dataset_name+'.json'), 'w') as fp:
    json.dump(model2metrics, fp)

# ----- End: Base models for all aggreagations and levels --- #


# ----- Start: Aggreagation of Inference model outputs for all aggreagations and levels --- #

# ----- End: Aggreagation of Inference model outputs for all aggreagations and levels --- #


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
