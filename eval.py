import numpy as np
import torch
from tslearn.metrics import dtw, dtw_path
from utils import unnormalize
from loss.dilate_loss import dilate_loss
import properscoring as ps


def eval_base_model(args, model_name, net, loader, norm, gamma, verbose=1):
    criterion = torch.nn.MSELoss()
    losses_dilate = []
    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
        target = torch.tensor(target, dtype=torch.float32).to(args.device)
        batch_size, N_output = target.shape[0:2]
        means, _ = net(inputs)

        # Unnormalize the data
        means = unnormalize(means, norm)

        # MSE
        loss_mse = criterion(target, means)
        if model_name in ['seq2seqdilate']:
            loss_dilate, loss_shape, loss_temporal = dilate_loss(target, means, args.alpha, args.gamma, args.device)
        else:
            loss_dilate = torch.zeros([])
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
        losses_dilate.append( loss_dilate.item() )
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    metric_dilate = np.array(losses_dilate).mean()
    metric_mse = np.array(losses_mse).mean()
    metric_dtw = np.array(losses_dtw).mean()
    metric_tdi = np.array(losses_tdi).mean()

    print('Eval dilateloss= ', metric_dilate, \
        'mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)

    return metric_dilate, metric_mse, metric_dtw, metric_tdi

def eval_inf_model(args, net, inf_test_inputs_dict, inf_test_norm_dict, target, norm, gamma, verbose=1):
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []
    losses_crps = []

    batch_size, N_output = target.shape[0:2]
    pred_mu, pred_std = net(inf_test_inputs_dict, inf_test_norm_dict)

    # Unnormalize
    pred_mu = unnormalize(pred_mu, norm)
    if pred_std is not None:
        pred_std = unnormalize(pred_std, norm)
    else:
        pred_std = torch.ones_like(pred_mu) * 1e-9

    # MSE
    loss_mse = criterion(target, pred_mu)
    loss_dtw, loss_tdi = 0,0
    # DTW and TDI
    for k in range(batch_size):
        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
        output_k_cpu = pred_mu[k,:,0:1].view(-1).detach().cpu().numpy()

        loss_dtw += dtw(target_k_cpu,output_k_cpu)
        path, sim = dtw_path(target_k_cpu, output_k_cpu)

        Dist = 0
        for i,j in path:
                Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)

    loss_dtw = loss_dtw /batch_size
    loss_tdi = loss_tdi / batch_size

    # CRPS
    loss_crps = ps.crps_gaussian(
        target, mu=pred_mu.detach().numpy(), sig=pred_std.detach().numpy()
    )

    # print statistics
    losses_crps.append( loss_crps )
    losses_mse.append( loss_mse.item() )
    losses_dtw.append( loss_dtw )
    losses_tdi.append( loss_tdi )

    metric_mse = np.array(losses_mse).mean()
    metric_dtw = np.array(losses_dtw).mean()
    metric_tdi = np.array(losses_tdi).mean()
    metric_crps = np.array(losses_crps).mean()

    #print('Eval mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)

    return pred_mu, pred_std, metric_mse, metric_dtw, metric_tdi, metric_crps