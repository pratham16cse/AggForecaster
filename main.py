import sys
import os
import argparse
import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

from models import inf_models

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# parameters
N = 500
N_input = 20
N_output = 20  
sigma = 0.01


def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    
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
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, args.gamma, device)
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
        
        if(verbose):
            if (epoch % args.print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                metric_mse, metric_dtw, metric_tdi = eval_model(net,testloader, args.gamma,verbose=1)

                if metric_mse < best_metric_mse:
                    best_metric_mse = metric_mse
                    best_epoch = epoch
                    torch.save(net.state_dict(), os.path.join(args.saved_models_dir, 'state_dict_model.pt'))
                    print('Model saved at epoch', epoch)

    print('Best model found at epoch', best_epoch)
    net.load_state_dict(torch.load(os.path.join(args.saved_models_dir, 'state_dict_model.pt')))
    net.eval()
    metric_mse, metric_dtw, metric_tdi = eval_model(net,testloader, args.gamma,verbose=1)
  

def eval_model(net,loader, gamma,verbose=1):   
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
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

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




parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, help='dataset_name')
#parser.add_argument('model_name', type=str, help='model_name')

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

#parser.add_argument('--patience', type=int, default=2,
#                    help='Number of epochs to wait for \
#                          before beginning cross-validation')
#parser.add_argument('--seed', type=int,
#                    help='Seed for parameter initialization',
#                    default=42)

args = parser.parse_args()

args.base_model_names = ['Seq2Seq_dilate', 'Seq2Seq_mse']
args.inference_model_names = ['DILATE', 'MSE', 'DualTPP']

base_models = {}
for name in args.base_model_names:
    base_models[name] = None

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)


# Load synthetic dataset
X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)
dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
dataset_test  = SyntheticDataset(X_test_input,X_test_target, test_bkp)
trainloader = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False, num_workers=1)


# ----- Start: base models training ----- #
for base_model_name in args.base_model_names:
    if base_model_name in ['Seq2Seq_mse']:
        loss_type = 'mse'
    elif base_model_name in ['Seq2Seq_dilate']:
        loss_type = 'dilate'

    encoder = EncoderRNN(
        input_size=1, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
        batch_size=args.batch_size
    ).to(device)
    decoder = DecoderRNN(
        input_size=1, hidden_size=args.hidden_size, num_grulstm_layers=args.num_grulstm_layers,
        fc_units=args.fc_units, output_size=1
    ).to(device)
    net_gru = Net_GRU(encoder,decoder, N_output, device).to(device)
    train_model(
        net_gru, loss_type=loss_type, learning_rate=args.learning_rate,
        epochs=args.epochs, gamma=args.gamma, print_every=args.print_every, eval_every=50, verbose=1
    )

    base_models[base_model_name] = net_gru
# ----- End: base models training ----- #

# ----- Start: Inference models ----- #

gen_test = iter(testloader)
test_inputs, test_targets, breaks = next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

for inf_model_name in args.inference_model_names:
    if inf_model_name in ['DILATE']:
        net = base_models['Seq2Seq_dilate']
        inf_net = inf_models.DILATE(net)
        pred = inf_net(test_inputs).to(device)
        metric_mse = criterion(test_targets, pred)
    elif inf_model_name in ['MSE']:
        net = base_models['Seq2Seq_mse']
        inf_net = inf_models.MSE(net)
        pred = inf_net(test_inputs).to(device)
        metric_mse = criterion(test_targets, pred)
    elif inf_model_name in ['DualTPP']:
        raise NotImplementedError

    print('MSE metric for Inference model {}: {:f}'.format(inf_model_name, metric_mse.item()))


# ----- End: Inference models ----- #

# Visualize results

#nets = [net_gru_mse,net_gru_dilate]

#for ind in range(1,51):
#    plt.figure()
#    plt.rcParams['figure.figsize'] = (17.0,5.0)  
#    k = 1
#    for net in nets:
#        pred = net(test_inputs).to(device)
#
#        input = test_inputs.detach().cpu().numpy()[ind,:,:]
#        target = test_targets.detach().cpu().numpy()[ind,:,:]
#        preds = pred.detach().cpu().numpy()[ind,:,:]
#
#        plt.subplot(1,3,k)
#        plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
#        plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
#        plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ])  ,label='prediction',linewidth=3)       
#        plt.xticks(range(0,40,2))
#        plt.legend()
#        k = k+1
#
#    plt.show()
#