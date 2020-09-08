import sys
import os
import argparse
import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.base_models import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train_model(
    args, model_name, net, trainloader, devloader, testloader,
    saved_models_path, output_dir,
    eval_every=50, verbose=1, Lambda=1, alpha=0.5
):

    optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    best_metric_mse = np.inf
    best_epoch = 0

    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            means, stds = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            if model_name in ['seq2seqmse']:
                loss_mse = criterion(target,means)
                loss = loss_mse
            if model_name in ['seq2seqdilate']:
                loss, loss_shape, loss_temporal = dilate_loss(target, means, alpha, args.gamma, device)
            if model_name in ['seq2seqnll']:
                dist = torch.distributions.normal.Normal(means, stds)
                loss = -torch.sum(dist.log_prob(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(verbose):
            if (epoch % args.print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                metric_mse, metric_dtw, metric_tdi = eval_base_model(net, devloader, args.gamma,verbose=1)

                if metric_mse < best_metric_mse:
                    best_metric_mse = metric_mse
                    best_epoch = epoch
                    torch.save(net.state_dict(), saved_models_path)
                    print('Model saved at epoch', epoch)

    print('Best model found at epoch', best_epoch)
    net.load_state_dict(torch.load(saved_models_path))
    net.eval()
    metric_mse, metric_dtw, metric_tdi = eval_base_model(net, devloader, args.gamma,verbose=1)


def eval_base_model(net,loader, gamma,verbose=1):
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        means, _ = net(inputs)

        # MSE
        loss_mse = criterion(target, means)
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = means[k,:,0:1].view(-1).detach().cpu().numpy()

            loss_dtw += dtw(target_k_cpu,output_k_cpu)
            path, sim = dtw_path(target_k_cpu, output_k_cpu)

            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)

        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    metric_mse = np.array(losses_mse).mean()
    metric_dtw = np.array(losses_dtw).mean()
    metric_tdi = np.array(losses_tdi).mean()

    print('Eval mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)

    return metric_mse, metric_dtw, metric_tdi

def eval_inf_model(net, lvl2testinputs, lvl2testtargets, gamma, verbose=1):
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    target = lvl2testtargets[0]
    batch_size, N_output = target.shape[0:2]
    preds, _ = net(lvl2testinputs)

    # MSE
    loss_mse = criterion(target, preds)
    loss_dtw, loss_tdi = 0,0
    # DTW and TDI
    for k in range(batch_size):
        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
        output_k_cpu = preds[k,:,0:1].view(-1).detach().cpu().numpy()

        loss_dtw += dtw(target_k_cpu,output_k_cpu)
        path, sim = dtw_path(target_k_cpu, output_k_cpu)

        Dist = 0
        for i,j in path:
                Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)

    loss_dtw = loss_dtw /batch_size
    loss_tdi = loss_tdi / batch_size

    # print statistics
    losses_mse.append( loss_mse.item() )
    losses_dtw.append( loss_dtw )
    losses_tdi.append( loss_tdi )

    metric_mse = np.array(losses_mse).mean()
    metric_dtw = np.array(losses_dtw).mean()
    metric_tdi = np.array(losses_tdi).mean()

    print('Eval mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)

    return preds, metric_mse, metric_dtw, metric_tdi




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
]

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
    levels = range(args.L)
    if base_model_name in ['seq2seqdilate']:
        levels = [0]
    for level in levels:
        trainloader = level2data[level]['trainloader']
        devloader = level2data[level]['devloader']
        testloader = level2data[level]['testloader']
        N_output = level2data[level]['N_output']

        if base_model_name in ['seq2seqmse', 'seq2seqdilate']:
            point_estimates = True
        elif base_model_name in ['seq2seqnll']:
            point_estimates = False

        saved_models_dir = os.path.join(
            args.saved_models_dir, args.dataset_name+'_'+base_model_name+'_'+str(level)
        )
        os.makedirs(saved_models_dir, exist_ok=True)
        saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
        output_dir = os.path.join(args.output_dir, base_model_name)
        os.makedirs(output_dir, exist_ok=True)

        encoder = EncoderRNN(
            input_size=1, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
            batch_size=args.batch_size
        ).to(device)
        decoder = DecoderRNN(
            input_size=1, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
            fc_units=args.fc_units, output_size=1
        ).to(device)
        net_gru = Net_GRU(encoder,decoder, N_output, point_estimates, device).to(device)
        train_model(
            args, base_model_name, net_gru, trainloader, devloader, testloader,
            saved_models_path, output_dir, eval_every=50, verbose=1
        )

        base_models[base_model_name][level] = net_gru
# ----- End: base models training ----- #

# ----- Start: Inference models ----- #

lvl2testinputs = dict()
lvl2testtargets = dict()
for level in range(args.L):
    gen_test = iter(level2data[level]['testloader'])
    test_inputs, test_targets, breaks = next(gen_test)

    test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
    lvl2testinputs[level] = test_inputs
    lvl2testtargets[level] = test_targets
#criterion = torch.nn.MSELoss()

for inf_model_name in args.inference_model_names:
    if inf_model_name in ['DILATE']:
        base_models_dict = base_models['seq2seqdilate']
        inf_net = inf_models.DILATE(base_models_dict)
        #pred = inf_net(lvl2testinputs[0]).to(device)
        #metric_mse = criterion(lvl2testtargets[0], pred)
    elif inf_model_name in ['MSE']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.MSE(base_models_dict)
        #pred = inf_net(lvl2testinputs[0]).to(device)
        #metric_mse = criterion(lvl2testtargets[0], pred)
    elif inf_model_name in ['seq2seqmse_dualtpp']:
        base_models_dict = base_models['seq2seqmse']
        inf_net = inf_models.DualTPP(args.K, base_models_dict)
        #pred = inf_net(lvl2testinputs).to(device)
        #metric_mse = criterion(lvl2testtargets[0], pred)
    elif inf_model_name in ['seq2seqnll_dualtpp']:
        base_models_dict = base_models['seq2seqnll']
        inf_net = inf_models.DualTPP(args.K, base_models_dict)
        #pred = inf_net(lvl2testinputs).to(device)
        #metric_mse = criterion(lvl2testtargets[0], pred)

    inf_net.eval()
    preds, metric_mse, metric_dtw, metric_tdi = eval_inf_model(
        inf_net, lvl2testinputs, lvl2testtargets, args.gamma, verbose=1
    )
    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('MSE metric for Inference model {}: {:f}'.format(inf_model_name, metric_mse))

    model2metrics = utils.add_metrics_to_dict(
        model2metrics, inf_model_name, metric_mse, metric_dtw, metric_tdi
    )
    infmodel2preds[inf_model_name] = preds


# ----- End: Inference models ----- #

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

        input = lvl2testinputs[0].detach().cpu().numpy()[ind,:,:]
        target = lvl2testtargets[0].detach().cpu().numpy()[ind,:,:]
        preds = preds.detach().cpu().numpy()[ind,:,:]

        plt.subplot(len(inference_models),1,k)
        plt.plot(range(0,args.N_input) ,input,label='input',linewidth=3)
        plt.plot(range(args.N_input-1,args.N_input+args.N_output), np.concatenate([ input[args.N_input-1:args.N_input], target ]) ,label='target',linewidth=3)
        plt.plot(range(args.N_input-1,args.N_input+args.N_output),  np.concatenate([ input[args.N_input-1:args.N_input], preds ])  ,label=inf_mdl_name,linewidth=3)
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.show()
