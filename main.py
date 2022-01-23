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
from eval import eval_base_model, eval_inf_model, eval_inf_index_model, eval_aggregates
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
import itertools
from collections import OrderedDict

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
                    choices=[
                        'same', 'zscore_per_series', 'gaussian_copula', 'log', 'zeroshift_per_series'
                    ],
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
parser.add_argument('--kernel_size', type=int, default=-1,
                    help='Kernel Size of Conv (in ARTransformerModel)')
parser.add_argument('--nkernel', type=int, default=-1,
                    help='Number of kernels of Conv (in ARTransformerModel)')

parser.add_argument('--dim_ff', type=int, default=512,
                    help='Dimension of Feedforward (in ARTransformerModel)')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads (in ARTransformerModel)')


# Cross-validation parameters
parser.add_argument('--cv_inf', type=int, default=-1,
                    help='Cross-validate the Inference models based on score on dev data')

# Learning rate for Inference Model
parser.add_argument('--lr_inf', type=float, default=-1.,
                    help='Learning rate for SGD-based inference model')

# Regularization for SHARQ
parser.add_argument('--sharq_reg', type=float, default=1.,
                    help='Regularization Parameter for SHARQ')



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
#    'rnn-nll-ar',
#    'trans-mse-ar',
    'trans-nll-ar',
    'gpt-nll-ar',
#    'gpt-mse-ar',
#    'gpt-nll-nar',
#    'gpt-mse-nar',
    'informer-mse-nar',
#    'trans-bvnll-ar',
#    'trans-nll-atr',
#    'trans-fnll-ar',
#    'rnn-mse-nar',
#    'rnn-nll-nar',
#    'rnn-fnll-nar',
#    'transm-nll-nar',
#    'transm-fnll-nar',
#    'transda-nll-nar',
#    'transda-fnll-nar',
#    'oracle',
#    'oracleforecast'
#    'transsig-nll-nar',
    'sharq-nll-nar',
]
args.aggregate_methods = [
    'sum',
#    'sumwithtrend',
    'slope',
#    'haar',
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
    args.inference_model_names.append('rnn-mse-ar_opt-st')
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
    #args.inference_model_names.append('rnn-mse-nar_opt-sum')
    #args.inference_model_names.append('rnn-mse-nar_optcf-sum')
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
    #args.inference_model_names.append('rnn-nll-ar_kl-sum')
    args.inference_model_names.append('rnn-nll-ar_opt-st')
    args.inference_model_names.append('rnn-nll-ar_kl-st')
if 'trans-mse-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-AR')
if 'trans-nll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-AR')
    args.inference_model_names.append('trans-nll-ar_opt-sum')
    #args.inference_model_names.append('trans-nll-ar_optcf-sum')
    #args.inference_model_names.append('trans-nll-ar_optcf-slope')
    #args.inference_model_names.append('trans-nll-ar_optcf-haar')
    #args.inference_model_names.append('trans-nll-ar_optcf-st')
    #args.inference_model_names.append('trans-nll-ar_opt-slope')
    args.inference_model_names.append('trans-nll-ar_opt-st')
    args.inference_model_names.append('trans-nll-ar_kl-sum')
    args.inference_model_names.append('trans-nll-ar_kl-st')
    args.inference_model_names.append('trans-nll-ar_covkl-sum')
    args.inference_model_names.append('trans-nll-ar_covkl-st')
if 'gpt-nll-ar' in args.base_model_names:
    args.inference_model_names.append('GPT-NLL-AR')
    #args.inference_model_names.append('gpt-nll-ar_opt-st')
    #args.inference_model_names.append('gpt-nll-ar_kl-st')
if 'gpt-mse-ar' in args.base_model_names:
    args.inference_model_names.append('GPT-MSE-AR')
if 'gpt-nll-nar' in args.base_model_names:
    args.inference_model_names.append('GPT-NLL-NAR')
    args.inference_model_names.append('gpt-nll-nar_opt-st')
    args.inference_model_names.append('gpt-nll-nar_kl-st')
if 'gpt-mse-nar' in args.base_model_names:
    args.inference_model_names.append('GPT-MSE-NAR')
if 'informer-mse-nar' in args.base_model_names:
    args.inference_model_names.append('INFORMER-MSE-NAR')
if 'trans-bvnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-BVNLL-AR')
    #args.inference_model_names.append('trans-bvnll-ar_opt-sum')
    args.inference_model_names.append('trans-bvnll-ar_optcf-sum')
    args.inference_model_names.append('trans-bvnll-ar_optcf-slope')
    #args.inference_model_names.append('trans-bvnll-ar_optcf-haar')
    args.inference_model_names.append('trans-bvnll-ar_optcf-st')
    #args.inference_model_names.append('trans-bvnll-ar_opt-slope')
    #args.inference_model_names.append('trans-bvnll-ar_opt-st')
    #args.inference_model_names.append('trans-bvnll-ar_kl-sum')
    #args.inference_model_names.append('trans-bvnll-ar_kl-st')
if 'trans-nll-atr' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-ATR')
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
if 'oracle' in args.base_model_names:
    args.inference_model_names.append('oracle')
if 'oracleforecast' in args.base_model_names:
    args.inference_model_names.append('SimRetrieval')
if 'transsig-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSSIG-NLL-NAR')
if 'sharq-nll-nar' in args.base_model_names:
    args.inference_model_names.append('SHARQ-NLL-NAR')


args.bm2info = OrderedDict({
    'informer-mse-nar':{'aggregate_methods': ['sum'], 'K_list':[1]},
    'gpt-nll-ar':{'aggregate_methods':['sum'], 'K_list':[1]},
    'trans-nll-ar':{},
    'sharq-nll-nar':{'aggregate_methods':['sum']}
})


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
        args.saved_models_dir = 'saved_models_ett'
    if args.output_dir is None:
        args.output_dir = 'Outputs_ett'
    #if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size  == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = 'idx'
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = '15min'
    if 'trans-nll-ar' in args.base_model_names:
        args.bm2info['trans-nll-ar'] = {
            'aggregate_methods':['sum','slope'], 'K_list':[1,12]
        }
    if 'sharq-nll-nar' in args.base_model_names:
        args.bm2info['sharq-nll-nar']['K_list'] = [1,2,3,4,6,8,12,24]

elif args.dataset_name == 'taxi30min':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_taxi30min'
    if args.output_dir is None:
        args.output_dir = 'Outputs_taxi30min'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 128
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    #args.t2v_type = 'mdh_parti'
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32

elif args.dataset_name == 'etthourly':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 168
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_etthourly'
    if args.output_dir is None:
        args.output_dir = 'Outputs_etthourly'
    #if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.print_every = 5 # TODO: Only for aggregate models
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'
    if 'trans-nll-ar' in args.base_model_names:
        args.bm2info['trans-nll-ar'] = {
            'aggregate_methods':['sum','slope'], 'K_list':[1,6]
        }
    if 'sharq-nll-nar' in args.base_model_names:
        args.bm2info['sharq-nll-nar']['K_list'] = [1,2,3,4,6,8,12,24]

elif args.dataset_name == 'azure':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 720
    if args.N_output == -1: args.N_output = 360
    #args.K_list = [60]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_azure'
    if args.output_dir is None:   
        args.output_dir = 'Outputs_azure'
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
    if args.cv_inf == -1: args.cv_inf = 1
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32

elif args.dataset_name == 'Solar':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_Solar'
    if args.output_dir is None:
        args.output_dir = 'Outputs_Solar'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    #if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.0005
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'
    #args.patience = 5 # Only for sharq model
    if 'trans-nll-ar' in args.base_model_names:
        args.bm2info['trans-nll-ar'] = {
            'aggregate_methods':['sum','slope'], 'K_list':[1,6]
        }
    if 'sharq-nll-nar' in args.base_model_names:
        args.bm2info['sharq-nll-nar']['K_list'] = [1,2,3,4,6,8,12,24]

elif args.dataset_name == 'electricity':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_electricity'
    if args.output_dir is None:
        args.output_dir = 'Outputs_electricity'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    #if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'
    args.sharq_reg = 0.5 # For K=2,3
    args.sharq_reg = 0.1 # For K=4,6,8,12 24
    #args.patience = 50 # For sharq, K=2,3
    #args.patience = 5 # For sharq, K=4,6,8,12,24
    if 'trans-nll-ar' in args.base_model_names:
        args.bm2info['trans-nll-ar'] = {
            'aggregate_methods':['sum','slope'], 'K_list':[1,6,12]
        }
    if 'sharq-nll-nar' in args.base_model_names:
        args.bm2info['sharq-nll-nar']['K_list'] = [1,2,3,4,6,8,12,24]

elif args.dataset_name == 'aggtest':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 20
    if args.N_output == -1: args.N_output = 10
    #args.K_list = [12]
    if args.K_list == []: args.K_list = [1, 5]
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_aggtest_test'
    if args.output_dir is None:
        args.output_dir = 'Outputs_aggtest_test'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.005
    if args.batch_size == -1: args.batch_size = 10
    if args.hidden_size == -1: args.hidden_size = 32
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = args.N_output
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32


elif args.dataset_name == 'Traffic911':
    args.epochs = 20
    args.N_input = 336
    args.N_output = 168
    args.K_list = [6]
    args.saved_models_dir = 'saved_models_Traffic911'
    args.output_dir = 'Outputs_Traffic911'
    args.normalize = 'zscore_per_series'
    args.learning_rate = 0.0001
    args.batch_size = 128
    args.hidden_size = 128
    args.num_grulstm_layers = 1
    args.v_dim = 1
    args.print_every = 5 # TODO: Only for aggregate models
    args.device = 'cuda:0'

elif args.dataset_name == 'foodinflation':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_foodinflation'
    if args.output_dir is None:
        args.output_dir = 'Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zeroshift_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32

elif args.dataset_name == 'foodinflationmonthly':
    if args.epochs == -1: args.epochs = 100
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    #args.K_list = [12]
    if args.K_list == []: args.K_list = []
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_foodinflation'
    if args.output_dir is None:
        args.output_dir = 'Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zeroshift_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 32
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01    
    if args.kernel_size == -1: args.kernel_size = 2
    if args.nkernel == -1: args.nkernel = 32


if 1 not in args.K_list:
    args.K_list = [1] + args.K_list

def merge_info(info_arg):
    all_info = list(itertools.chain(*[bm[info_arg] for bm in args.bm2info.values()]))
    all_info = sorted(set(all_info),key=all_info.index)
    return all_info
all_agg_methods = merge_info('aggregate_methods')
all_K_list = merge_info('K_list')

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

if os.path.exists('bee'):
    DUMP_PATH = '/mnt/infonas/data/pratham/Forecasting/DILATE'
else:
    DUMP_PATH = '.'
args.output_dir = os.path.join(DUMP_PATH, args.output_dir)
args.saved_models_dir = os.path.join(DUMP_PATH, args.saved_models_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)



#dataset = utils.get_processed_data(args)
data_processor = utils.DataProcessor(args)
#level2data = dataset['level2data']

# ----- Start: Load all datasets ----- #

dataset = {}
for agg_method in all_agg_methods:
    dataset[agg_method] = {}
    for level in all_K_list:
        if level==1 and agg_method is not 'sum':
            dataset[agg_method][level] = dataset['sum'][1]
        else:
            dataset[agg_method][level] = data_processor.get_processed_data(args, agg_method, level)

# ----- End : Load all datasets ----- #

# ----- Start: base models training ----- #
for base_model_name in args.base_model_names:
    base_models[base_model_name] = {}
    base_models_preds[base_model_name] = {}

    levels = args.bm2info[base_model_name]['K_list']
    aggregate_methods = args.bm2info[base_model_name]['aggregate_methods']
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
                'rnn-mse-nar', 'rnn-mse-ar', 'trans-mse-nar',
                'gpt-mse-ar', 'gpt-mse-nar',
                'informer-mse-nar',
                'nbeats-mse-nar',
                'nbeatsd-mse-nar', 'trans-mse-ar', 'oracle', 'oracleforecast',
            ]:
                estimate_type = 'point'
            elif base_model_name in [
                'seq2seqnll', 'convnll', 'trans-q-nar', 'rnn-q-nar', 'rnn-q-ar',
                'rnn-nll-nar', 'rnn-nll-ar', 'rnn-aggnll-nar', 'trans-nll-ar',
                'gpt-nll-ar', 'gpt-nll-nar',
                'transm-nll-nar', 'transda-nll-nar', 'transsig-nll-nar', 'trans-nll-atr',
                'sharq-nll-nar',
            ]:
                estimate_type = 'variance'
            elif base_model_name in [
                'rnn-fnll-nar', 'trans-fnll-ar', 'transm-nll-nar', 'transda-fnll-nar'
            ]:
                estimate_type = 'covariance'
            elif base_model_name in ['trans-bvnll-ar']:
                estimate_type = 'bivariate'

            saved_models_dir = os.path.join(
                args.saved_models_dir,
                args.dataset_name+'_'+base_model_name+'_'+agg_method+'_'+str(level)
            )
            os.makedirs(saved_models_dir, exist_ok=True)
            writer = SummaryWriter(saved_models_dir)
            saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
            print('\n {} {} {}'.format(base_model_name, agg_method, str(level)))


            # Create the network
            net_gru = get_base_model(
                args, base_model_name, level,
                N_input, N_output, input_size, output_size,
                estimate_type, feats_info
            )
    
            # train the network
            if agg_method in ['sumwithtrend', 'slope', 'wavelet', 'haar'] and level == 1:
                base_models[base_model_name][agg_method][level] = base_models[base_model_name]['sum'][1]
            else:
                if base_model_name not in ['oracle', 'oracleforecast']:
                    if 'sharq' in base_model_name and level==1:
                        base_models[base_model_name][agg_method][level] = base_models['trans-nll-ar'][agg_method][level]
                    else:
                        if 'sharq' in base_model_name and level!=1:
                            train_model(
                                args, base_model_name, net_gru,
                                level2data, saved_models_path, writer, agg_method, level,
                                verbose=1,
                                bottom_net=base_models[base_model_name][agg_method][1],
                                bottom_data_dict=dataset[agg_method][1],
                                sharq_step=0
                            )
                            #train_model(
                            #    args, base_model_name, net_gru,
                            #    level2data, saved_models_path, writer, agg_method, level,
                            #    verbose=1,
                            #    bottom_net=base_models[base_model_name][agg_method][1],
                            #    bottom_data_dict=dataset[agg_method][1],
                            #    sharq_step=1
                            #)
                        else:
                            train_model(
                                args, base_model_name, net_gru,
                                level2data, saved_models_path, writer, agg_method, level,
                                verbose=1
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

#import ipdb ; ipdb.set_trace()


def run_inference_model(
    args, inf_model_name, base_models, which_split, opt_normspace, agg_method=None, K=None
):

    metric2val = dict()
    infmodel2preds = dict()

    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']['sum']
        inf_net = inf_models.DILATE(base_models_dict, device=args.device)
        raise NotImplementedError

    elif inf_model_name in ['RNN-MSE-NAR']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-NLL-NAR']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-nar_opt-sum']:
        base_models_dict = base_models['rnn-mse-nar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-nar_opt-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-nar_optcf-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-nar_optcf-sum']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-nar_opt-slope']:
        base_models_dict = base_models['rnn-mse-nar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-nar_opt-slope']:
        base_models_dict = base_models['rnn-nll-nar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-nar_opt-st']:
        base_models_dict = base_models['rnn-nll-nar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-nar_kl-sum']:
        base_models_dict = base_models['rnn-nll-nar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInference(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['rnn-nll-nar_kl-st']:
        base_models_dict = base_models['rnn-nll-nar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInference(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['RNN-NLL-AR']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_opt-sum']:
        base_models_dict = base_models['rnn-mse-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-ar_opt-sum']:
        base_models_dict = base_models['rnn-nll-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_optcf-sum']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_opt-slope']:
        base_models_dict = base_models['rnn-mse-ar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-ar_opt-slope']:
        base_models_dict = base_models['rnn-nll-ar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['rnn-nll-ar_opt-st']:
        base_models_dict = base_models['rnn-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )

    elif inf_model_name in ['rnn-nll-ar_kl-sum']:
        base_models_dict = base_models['rnn-nll-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInference(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['rnn-nll-ar_kl-st']:
        base_models_dict = base_models['rnn-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False,
        )

    elif inf_model_name in ['TRANS-MSE-AR']:
        base_models_dict = base_models['trans-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-NLL-AR']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['SHARQ-NLL-NAR']:
        base_models_dict = base_models['sharq-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['trans-nll-ar_opt-sum']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        #import ipdb ; ipdb.set_trace()
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )


    elif inf_model_name in ['trans-nll-ar_optcf-sum']:
        base_models_dict = base_models['trans-nll-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-nll-ar_optcf-slope']:
        base_models_dict = base_models['trans-nll-ar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-nll-ar_optcf-haar']:
        base_models_dict = base_models['trans-nll-ar']
        agg_method = ['haar'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-nll-ar_optcf-st']:
        base_models_dict = base_models['trans-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-nll-ar_opt-slope']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['trans-nll-ar_opt-st']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )

    elif inf_model_name in ['trans-nll-ar_kl-sum']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp'
        )

    elif inf_model_name in ['trans-nll-ar_kl-st']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models['trans-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp'
        )

    elif inf_model_name in ['trans-nll-ar_covkl-sum']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp',
            covariance=True
        )

    elif inf_model_name in ['trans-nll-ar_covkl-st']:
        bm = 'trans-nll-ar'
        base_models_dict = base_models[bm]
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.bm2info[bm]['K_list'] if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp',
            covariance=True
        )

    elif inf_model_name in ['GPT-NLL-AR']:
        base_models_dict = base_models['gpt-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['GPT-MSE-AR']:
        base_models_dict = base_models['gpt-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['gpt-nll-ar_opt-st']:
        base_models_dict = base_models['gpt-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )

    elif inf_model_name in ['gpt-nll-ar_kl-st']:
        base_models_dict = base_models['gpt-nll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp'
        )


    elif inf_model_name in ['GPT-NLL-NAR']:
        base_models_dict = base_models['gpt-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['gpt-nll-nar_opt-st']:
        base_models_dict = base_models['gpt-nll-nar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )

    elif inf_model_name in ['gpt-nll-nar_kl-st']:
        base_models_dict = base_models['gpt-nll-nar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=True, opt_normspace=False, kldirection='qp'
        )

    elif inf_model_name in ['GPT-MSE-NAR']:
        base_models_dict = base_models['gpt-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['INFORMER-MSE-NAR']:
        base_models_dict = base_models['informer-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-BVNLL-AR']:
        base_models_dict = base_models['trans-bvnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['trans-bvnll-ar_opt-sum']:
        agg_method = ['sum'] if agg_method is None else agg_method
        base_models_dict = base_models['trans-bvnll-ar']
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['trans-bvnll-ar_optcf-sum']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-bvnll-ar_optcf-slope']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-bvnll-ar_optcf-haar']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['haar'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-bvnll-ar_optcf-st']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP_CF(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=False
        )

    elif inf_model_name in ['trans-bvnll-ar_opt-slope']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['trans-bvnll-ar_opt-st']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)

    elif inf_model_name in ['trans-bvnll-ar_kl-sum']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['sum'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInference(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['trans-bvnll-ar_kl-st']:
        base_models_dict = base_models['trans-bvnll-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        inf_net = inf_models.KLInference(
            K_list, base_models_dict, agg_method, device=args.device, opt_normspace=opt_normspace
        )

    elif inf_model_name in ['TRANS-NLL-ATR']:
        base_models_dict = base_models['trans-nll-atr']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-FNLL-AR']:
        base_models_dict = base_models['trans-fnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-FNLL-NAR']:
        base_models_dict = base_models['rnn-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['oracle']:
        base_models_dict = base_models['oracle']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['SimRetrieval']:
        base_models_dict = base_models['oracleforecast']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['TRANSSIG-NLL-NAR']:
        base_models_dict = base_models['transsig-nll-nar']
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
        raise NotImplementedError

    elif inf_model_name in ['RNN-Q-NAR']:
        base_models_dict = base_models['rnn-q-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['RNN-MSE-AR']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['rnn-mse-ar_opt-st']:
        base_models_dict = base_models['rnn-mse-ar']
        agg_method = ['sum', 'slope'] if agg_method is None else agg_method
        K_list = args.K_list if K is None else K
        #inf_net = inf_models.DualTPP(K_list, base_models_dict, agg_method, device=args.device)
        inf_net = inf_models.KLInferenceSGD(
            K_list, base_models_dict, agg_method, args.lr_inf, device=args.device,
            solve_mean=True, solve_std=False, opt_normspace=False,
        )

    elif inf_model_name in ['RNN-Q-AR']:
        base_models_dict = base_models['rnn-q-ar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-MSE-NAR']:
        base_models_dict = base_models['trans-mse-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-Q-NAR']:
        base_models_dict = base_models['trans-q-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['NBEATS-MSE-NAR']:
        base_models_dict = base_models['nbeats-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        raise NotImplementedError

    elif inf_model_name in ['NBEATSD-MSE-NAR']:
        base_models_dict = base_models['nbeatsd-mse-nar']['sum']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)
        raise NotImplementedError

    elif inf_model_name in ['rnn-mse-nar_dualtpp']:
        base_models_dict = base_models['rnn-mse-nar']['sum']
        inf_net = inf_models.DualTPP(args.K_list, base_models_dict, device=args.device)
        raise NotImplementedError

    elif inf_model_name in ['rnn-mse-nar_dualtpp_cf']:
        base_models_dict = base_models['rnn-mse-nar']['sum']
        inf_net = inf_models.DualTPP_CF(args.K_list, base_models_dict, device=args.device)
        raise NotImplementedError

    if not args.leak_agg_targets:
        inf_test_targets_dict = None

    inf_net.eval()
    #import ipdb ; ipdb.set_trace()
    (
        inputs, target, pred_mu, pred_std, pred_d, pred_v,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape,
        total_time
    ) = eval_inf_model(args, inf_net, dataset, which_split, args.gamma, verbose=1)

    if inf_net.covariance == False:
        pred_v_foragg = None
    else:
        pred_v_foragg = pred_v
    #import ipdb ; ipdb.set_trace()
    agg2metrics = eval_aggregates(
        inputs, target, pred_mu, pred_std, pred_d, pred_v_foragg
    )

    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('Metrics for Inference model {}: MAE:{:f}, CRPS:{:f}, MSE:{:f}, SMAPE:{:f}, Time:{:f}'.format(
        inf_model_name, metric_mae, metric_crps, metric_mse, metric_smape, total_time)
    )

    metric2val = utils.add_metrics_to_dict(
        metric2val, inf_model_name,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape
    )
    infmodel2preds[inf_model_name] = pred_mu
    if which_split in ['test']:
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

    return metric2val, agg2metrics

model2metrics = dict()
model2aggmetrics = dict()
for inf_model_name in args.inference_model_names:

    if args.cv_inf:
        # Consider all possible combinations of aggregate methods
        aggregate_methods = []
        for l in range(1, len(args.aggregate_methods)+1):
            aggregate_methods += list(itertools.combinations(args.aggregate_methods, l))
        # Single value of K is used in a hyper-parameter config for inference model
        K_list = []
        if len(args.K_list) == 1:
            K_list = [args.K_list]
        else:
            for K in args.K_list:
                if K != 1:
                    K_list.append([1, K])

        hparam_configs = list(itertools.product(aggregate_methods, K_list))

        hparams2metrics = []
        for agg_method, K in hparam_configs:
            print('cv with agg_method, K:', agg_method, K)
            metric2val, agg2metrics = run_inference_model(
                args, inf_model_name, base_models, 'dev', opt_normspace,
                agg_method, K
            )
            hparams2metrics.append(metric2val)
        cv_metric =  'crps'
        best_cfg_idx, _ = min(enumerate(hparams2metrics), key=lambda x: x[1]['crps'])
        print(
            'best_cfg_idx:', best_cfg_idx,
            'best_agg_method and best K:', hparam_configs[best_cfg_idx],
        )
        metric2val, agg2metrics = run_inference_model(
            args, inf_model_name, base_models, 'test', opt_normspace,
            hparam_configs[best_cfg_idx][0],
            hparam_configs[best_cfg_idx][1]
        )
        model2metrics[inf_model_name] = metric2val
        model2aggmetrics[inf_model_name] = agg2metrics
    else:
        #raise NotImplementedError
        metric2val, agg2metrics = run_inference_model(
            args, inf_model_name, base_models, 'test', opt_normspace
        )
        model2metrics[inf_model_name] = metric2val
        model2aggmetrics[inf_model_name] = agg2metrics


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

with open(os.path.join(args.output_dir, 'results_agg_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, CRPS, MSE')
    for model_name, agg2metrics in model2aggmetrics.items():
        for agg, K2metrics in agg2metrics.items():
            for K, metrics_dict in K2metrics.items():
                fp.write(
                    '\n{}, {}, {}, {:.6f}, {:.6f}, {:.6f}'.format(
                        model_name,
                        agg, K,
                        round(metrics_dict['mae'], 3),
                        round(metrics_dict['crps'], 3),
                        round(metrics_dict['mse'], 3),
                        #metrics_dict['dtw'],
                        #metrics_dict['tdi'],
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

    for agg_method in args.bm2info[base_model_name]['aggregate_methods']:

        for K in args.bm2info[base_model_name]['K_list']:

            print('Base Model', base_model_name,'for', agg_method, K)
    
            loader = dataset[agg_method][K]['testloader']
            norm = dataset[agg_method][K]['test_norm']
            (
                test_inputs, test_target, pred_mu, pred_std,
                metric_dilate, metric_mse, metric_dtw, metric_tdi,
                metric_crps, metric_mae, metric_crps_part, metric_nll
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