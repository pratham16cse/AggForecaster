import numpy as np
import torch
from tslearn.metrics import dtw, dtw_path
from utils import unnormalize
from loss.dilate_loss import dilate_loss
import properscoring as ps


def eval_base_model(args, model_name, net, loader, norm, gamma, verbose=1):

    inputs, target, pred_mu, pred_std = [], [], [], []

    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    losses_dilate = []
    losses_mse = []
    losses_mae = []
    losses_dtw = []
    losses_tdi = []
    losses_crps = []

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi, loss_mae = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        # get the inputs
        batch_inputs, batch_target, feats_in, feats_tgt, norm, breakpoints = data
        #inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
        #batch_target = torch.tensor(batch_target, dtype=torch.float32).to(args.device)
        #feats_in = torch.tensor(feats_in, dtype=torch.float32).to(args.device)
        #feats_tgt = torch.tensor(feats_tgt, dtype=torch.float32).to(args.device)
        #norm = torch.tensor(norm, dtype=torch.float32).to(args.device)
        batch_size, N_output = batch_target.shape[0:2]
        # DO NOT PASS TARGET during forward pass
        batch_pred_mu, batch_pred_std = net(feats_in, batch_inputs, feats_tgt)

        # Unnormalize the data
        batch_pred_mu = unnormalize(batch_pred_mu, norm)
        if batch_pred_std is not None:
            batch_pred_std = unnormalize(batch_pred_std, norm)
        else:
            batch_pred_std = torch.ones_like(batch_pred_mu) * 1e-9

        inputs.append(batch_inputs)
        target.append(batch_target)
        pred_mu.append(batch_pred_mu)
        pred_std.append(batch_pred_std)

    inputs = torch.cat(inputs, dim=0)
    target = torch.cat(target, dim=0)
    pred_mu = torch.cat(pred_mu, dim=0)
    pred_std = torch.cat(pred_std, dim=0)

    # MSE
    loss_mse = criterion(target, pred_mu).item()
    loss_mae = criterion_mae(target, pred_mu).item()

    # DILATE loss
    if model_name in ['seq2seqdilate']:
        loss_dilate, loss_shape, loss_temporal = dilate_loss(target, pred_mu, args.alpha, args.gamma, args.device)
    else:
        loss_dilate = torch.zeros([])
    loss_dilate = loss_dilate.item()

    # DTW and TDI
    loss_dtw, loss_tdi = 0,0
    M = target.shape[0]
    for k in range(M):
        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
        output_k_cpu = pred_mu[k,:,0:1].view(-1).detach().cpu().numpy()

        loss_dtw += dtw(target_k_cpu,output_k_cpu)
        path, sim = dtw_path(target_k_cpu, output_k_cpu)

        Dist = 0
        for i,j in path:
                Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)

    loss_dtw = loss_dtw / M
    loss_tdi = loss_tdi / M

    # CRPS
    loss_crps = ps.crps_gaussian(
        target, mu=pred_mu.detach().numpy(), sig=pred_std.detach().numpy()
    ).mean()

    # CRPS in parts of horizon
    loss_crps_part = []
    N = target.shape[1]
    p = max(int(N/4), 1)
    for i in range(0, N, p):
        if i+p<=N:
            loss_crps_part.append(
                ps.crps_gaussian(
                    target[:, i:i+p],
                    mu=pred_mu[:, i:i+p].detach().numpy(),
                    sig=pred_std[:, i:i+p].detach().numpy()
                ).mean()
            )
    loss_crps_part = np.array(loss_crps_part)


    metric_dilate = loss_dilate
    metric_mse = loss_mse
    metric_mae = loss_mae
    metric_dtw = loss_dtw
    metric_tdi = loss_tdi
    metric_crps = loss_crps
    metric_crps_part = loss_crps_part

    print('Eval dilateloss= ', metric_dilate, \
        'mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi,
        'crps=', metric_crps, 'crps_parts=', metric_crps_part)

    return (
        inputs, target, pred_mu, pred_std,
        metric_dilate, metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae, metric_crps_part
    )

def eval_inf_model(
    args, net, inf_test_inputs_dict, inf_test_norm_dict, target, norm,
    inf_test_feats_in_dict, inf_test_feats_tgt_dict,
    gamma, inf_test_targets_dict=None, verbose=1):
    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    losses_mse = []
    losses_mae = []
    losses_dtw = []
    losses_tdi = []
    losses_crps = []

    batch_size, N_output = target.shape[0:2]
    pred_mu, pred_std = net(
        inf_test_feats_in_dict, inf_test_inputs_dict,
        inf_test_feats_tgt_dict, inf_test_norm_dict,
        targets_dict=inf_test_targets_dict,
    )

    # Unnormalize
    pred_mu = unnormalize(pred_mu, norm)
    if pred_std is not None:
        pred_std = unnormalize(pred_std, norm)
    else:
        pred_std = torch.ones_like(pred_mu) * 1e-9

    # MSE
    loss_mse = criterion(target, pred_mu)
    loss_mae = criterion_mae(target, pred_mu)
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
    losses_mae.append( loss_mae.item() )
    losses_dtw.append( loss_dtw )
    losses_tdi.append( loss_tdi )

    metric_mse = np.array(losses_mse).mean()
    metric_mae = np.array(losses_mae).mean()
    metric_dtw = np.array(losses_dtw).mean()
    metric_tdi = np.array(losses_tdi).mean()
    metric_crps = np.array(losses_crps).mean()

    #print('Eval mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)

    return (
        pred_mu, pred_std,
        metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae
    )