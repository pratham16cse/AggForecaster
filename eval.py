import numpy as np
import torch
from tslearn.metrics import dtw, dtw_path
from utils import unnormalize, normalize
from loss.dilate_loss import dilate_loss
import properscoring as ps
import train


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
    losses_nll = []
    losses_ql = []

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi, loss_mae, losses_nll, losses_ql = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        # get the inputs
        batch_inputs, batch_target, feats_in, feats_tgt, norm, _, _, _, _, _ = data
        #inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
        #batch_target = torch.tensor(batch_target, dtype=torch.float32).to(args.device)
        #feats_in = torch.tensor(feats_in, dtype=torch.float32).to(args.device)
        #feats_tgt = torch.tensor(feats_tgt, dtype=torch.float32).to(args.device)
        #norm = torch.tensor(norm, dtype=torch.float32).to(args.device)
        batch_size, N_output = batch_target.shape[0:2]
        #if N_output == 24:
        #    import ipdb
        #    ipdb.set_trace()
        # DO NOT PASS TARGET during forward pass
        batch_pred_mu, batch_pred_std = net(feats_in.to(args.device), batch_inputs.to(args.device), feats_tgt.to(args.device))
        batch_pred_mu = batch_pred_mu.cpu()
        if batch_pred_std is not None:
            batch_pred_std = batch_pred_std.cpu()

        batch_target, _ = normalize(batch_target, norm, is_var=False)

        # Unnormalize the data
        #batch_pred_mu = unnormalize(batch_pred_mu, norm, is_var=False)
        if batch_pred_std is not None:
            #batch_pred_std = unnormalize(batch_pred_std, norm, is_var=True)
            pass
        else:
            batch_pred_std = torch.ones_like(batch_pred_mu) #* 1e-9

        inputs.append(batch_inputs)
        target.append(batch_target)
        pred_mu.append(batch_pred_mu)
        pred_std.append(batch_pred_std)

    inputs = torch.cat(inputs, dim=0)
    target = torch.cat(target, dim=0)
    pred_mu = torch.cat(pred_mu, dim=0)
    pred_std = torch.cat(pred_std, dim=0)

    # MSE
    print(target.shape, pred_mu.shape)
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

    dist = torch.distributions.normal.Normal(pred_mu, pred_std)
    loss_nll = -torch.mean(dist.log_prob(target)).item()

    quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float)
    #quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float)
    #quantiles = torch.tensor([0.45, 0.5, 0.55], dtype=torch.float)
    quantile_weights = torch.ones_like(quantiles, dtype=torch.float)
    #quantile_weights = torch.tensor([1., 1., 1.], dtype=torch.float)
    loss_ql = train.QuantileLoss(
        quantiles, quantile_weights
    )(target, pred_mu, pred_std).item()


    metric_dilate = loss_dilate
    metric_mse = loss_mse
    metric_mae = loss_mae
    metric_dtw = loss_dtw
    metric_tdi = loss_tdi
    metric_crps = loss_crps
    metric_crps_part = loss_crps_part
    metric_nll = loss_nll
    metric_ql = loss_ql

    print('Eval dilateloss= ', metric_dilate, \
        'mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi,
        'crps=', metric_crps, 'crps_parts=', metric_crps_part,
        'nll=', metric_nll, 'ql=', metric_ql)

    return (
        inputs, target, pred_mu, pred_std,
        metric_dilate, metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae, metric_crps_part, metric_nll, metric_ql
    )

def eval_index_model(args, model_name, net, loader, norm, gamma, N_input, N_output, verbose=1):

    inputs_idx, inputs, target_gaps, target = [], [], [], []
    pred_mu_gaps, pred_mu, pred_std_gaps, pred_std = [], [], [], []

    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    losses_mse_idx, losses_mse = [], []
    losses_mae_idx, losses_mae = [], []
    losses_crps_idx, losses_crps = [], []

    for i, data in enumerate(loader, 0):
        # get the inputs
        (
            batch_inputs, batch_target,
            feats_in, feats_tgt, norm, _,
            batch_indices_in, batch_indices_tgt,
            batch_gaps_in, batch_gaps_tgt
        ) = data
        batch_size, _ = batch_target.shape[0:2]

        # TODO: temporarily using indices_in as the sequence for
        # one-step-ahead prediction task
        batch_indices = torch.cat((batch_indices_in, batch_indices_tgt), dim=1)
        batch_gaps = torch.cat((batch_gaps_in, batch_gaps_tgt), dim=1)
        batch_seq = torch.cat((batch_inputs, batch_target), dim=1)
        batch_indices_in = batch_indices[:, :-1]
        batch_indices_tgt = batch_indices[:, 1:]
        batch_gaps_in = batch_gaps[:, :-1]
        batch_gaps_tgt = batch_gaps[:, 1:]
        batch_inputs = batch_seq[:, :-1]
        batch_target = batch_seq[:, 1:]

        end_idx = np.ones((batch_size, 1, 1)) * (N_input+N_output)
        hidden = net.init_hidden(batch_inputs.shape[0], args.device)
        (
            batch_pred_mu_gaps, batch_pred_std_gaps,
            batch_pred_mu, batch_pred_std, _
        ) = net(batch_gaps_in, batch_inputs, hidden)
        #) = net.simulate(batch_gaps_in, batch_inputs, hidden, end_idx)

        # Unnormalize the data
        #batch_pred_mu_gaps = unnormalize(batch_pred_mu_gaps, norm)
        #batch_pred_mu = unnormalize(batch_pred_mu, norm)
        if batch_pred_std is not None:
            #batch_pred_std_gaps = unnormalize(batch_pred_std_gaps, norm)
            #batch_pred_std = unnormalize(batch_pred_std, norm)
            pass
        else:
            batch_pred_std_gaps = torch.ones_like(batch_pred_mu_gaps) * 1e-9
            batch_pred_std = torch.ones_like(batch_pred_mu) * 1e-9

        inputs_idx.append(batch_indices_in)
        inputs.append(batch_inputs)
        target_gaps.append(batch_gaps_tgt)
        target.append(batch_target)
        pred_mu_gaps.append(batch_pred_mu_gaps)
        pred_mu.append(batch_pred_mu)
        pred_std_gaps.append(batch_pred_std_gaps)
        pred_std.append(batch_pred_std)

    #import ipdb
    #ipdb.set_trace()
    inputs_idx = torch.cat(inputs_idx, dim=0)
    inputs = torch.cat(inputs, dim=0)
    target_gaps = torch.cat(target_gaps, dim=0)
    target = torch.cat(target, dim=0)
    pred_mu_gaps = torch.cat(pred_mu_gaps, dim=0)
    pred_mu = torch.cat(pred_mu, dim=0)
    pred_std_gaps = torch.cat(pred_std_gaps, dim=0)
    pred_std = torch.cat(pred_std, dim=0)

    # MSE
    print(target.shape, pred_mu.shape)
    metric_mse_idx = criterion(target_gaps, pred_mu_gaps).item()
    metric_mse = criterion(target, pred_mu).item()
    metric_mae_idx = criterion_mae(target_gaps, pred_mu_gaps).item()
    metric_mae = criterion_mae(target, pred_mu).item()

#    # DILATE loss
#    if model_name in ['seq2seqdilate']:
#        loss_dilate, loss_shape, loss_temporal = dilate_loss(target, pred_mu, args.alpha, args.gamma, args.device)
#    else:
#        loss_dilate = torch.zeros([])
#    loss_dilate = loss_dilate.item()

#    # DTW and TDI
#    loss_dtw, loss_tdi = 0,0
#    M = target.shape[0]
#    for k in range(M):
#        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
#        output_k_cpu = pred_mu[k,:,0:1].view(-1).detach().cpu().numpy()
#
#        loss_dtw += dtw(target_k_cpu,output_k_cpu)
#        path, sim = dtw_path(target_k_cpu, output_k_cpu)
#
#        Dist = 0
#        for i,j in path:
#                Dist += (i-j)*(i-j)
#        loss_tdi += Dist / (N_output*N_output)
#
#    loss_dtw = loss_dtw / M
#    loss_tdi = loss_tdi / M

    # CRPS
    metric_crps_idx = ps.crps_gaussian(
        target_gaps, mu=pred_mu_gaps.detach().numpy(), sig=pred_std_gaps.detach().numpy()
    ).mean()
    metric_crps = ps.crps_gaussian(
        target, mu=pred_mu.detach().numpy(), sig=pred_std.detach().numpy()
    ).mean()

#    # CRPS in parts of horizon
#    loss_crps_part = []
#    N = target.shape[1]
#    p = max(int(N/4), 1)
#    for i in range(0, N, p):
#        if i+p<=N:
#            loss_crps_part.append(
#                ps.crps_gaussian(
#                    target[:, i:i+p],
#                    mu=pred_mu[:, i:i+p].detach().numpy(),
#                    sig=pred_std[:, i:i+p].detach().numpy()
#                ).mean()
#            )
#    loss_crps_part = np.array(loss_crps_part)


    print('mse_idx= ', metric_mse_idx, 'mse= ', metric_mse,
          'mae_idx= ', metric_mae_idx, 'mae= ', metric_mae,
          'crps_idx=', metric_crps_idx, 'crps=', metric_crps)

    return (
        inputs_idx, inputs, target_gaps, target,
        pred_mu_gaps, pred_std_gaps, pred_mu, pred_std,
        metric_mse_idx, metric_mse,
        metric_mae_idx, metric_mae,
        metric_crps_idx, metric_crps
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
    pred_mu = pred_mu.cpu()
    if pred_std is not None:
        pred_std = pred_std.cpu()

    # Unnormalize
    pred_mu = unnormalize(pred_mu, norm, is_var=False)
    if pred_std is not None:
        pred_std = unnormalize(pred_std, norm, is_var=True)
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

def eval_inf_index_model(
    args, net, inf_test_inputs_dict, inf_test_norm_dict, target, norm,
    inf_test_feats_in_dict, inf_test_feats_tgt_dict, inf_test_inputs_gaps_dict,
    gamma, N_input, N_output, inf_test_targets_dict=None, verbose=1
):
    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    losses_mse = []
    losses_mae = []
    losses_dtw = []
    losses_tdi = []
    losses_crps = []

    batch_size, N_output = target.shape[0:2]
    end_idx = np.ones((batch_size, 1, 1)) * (N_input+N_output)
    pred_mu, pred_std = net(
        inf_test_feats_in_dict, inf_test_inputs_dict,
        inf_test_feats_tgt_dict, inf_test_norm_dict,
        inf_test_inputs_gaps_dict, N_input, N_output,
        targets_dict=inf_test_targets_dict,
    )

    # Unnormalize
    pred_mu = unnormalize(pred_mu, norm, is_var=False)
    if pred_std is not None:
        pred_std = unnormalize(pred_std, norm, is_var=True)
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
    pred_mu = pred_mu.cpu()
    if pred_std is not None:
        pred_std = pred_std.cpu()

    # Unnormalize
    pred_mu = unnormalize(pred_mu, norm, is_var=False)
    if pred_std is not None:
        pred_std = unnormalize(pred_std, norm, is_var=True)
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

