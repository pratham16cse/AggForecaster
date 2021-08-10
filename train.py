import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.dilate_loss import dilate_loss
from eval import eval_base_model, eval_index_model
import time
from models.base_models import get_base_model
from utils import DataProcessor
import random
from torch.distributions.normal import Normal

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles, quantile_weights):
        super().__init__()
        self.quantiles = quantiles
        self.quantile_weights = quantile_weights
        
    def forward(self, target, pred_mu, pred_sigma):
        assert not target.requires_grad
        assert pred_mu.size(0) == target.size(0)
        losses = []
        for i, (q, w) in enumerate(zip(self.quantiles, self.quantile_weights)):
            errors = target - Normal(pred_mu, pred_sigma).icdf(q)
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ) * w)
        #loss = torch.mean(
        #    torch.sum(torch.cat(losses, dim=1), dim=1))
        loss = torch.mean(torch.cat(losses, dim=1))
        return loss


def get_optimizer(args, lr, net):
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)
    return optimizer, scheduler


def train_model(
    args, model_name, net, data_dict, point_estimates, saved_models_path,
    output_dir, writer, verbose=1,
):

    lr = args.learning_rate
    epochs = args.epochs

    trainloader = data_dict['trainloader']
    devloader = data_dict['devloader']
    testloader = data_dict['testloader']
    norm = data_dict['dev_norm']
    N_input = data_dict['N_input']
    N_output = data_dict['N_output']
    input_size = data_dict['input_size']
    output_size = data_dict['output_size']
    Lambda=1

    optimizer, scheduler = get_optimizer(args, lr, net)

    criterion = torch.nn.MSELoss()

    if (not args.ignore_ckpt) and os.path.isfile(saved_models_path):
        print('Loading from saved model')
        checkpoint = torch.load(saved_models_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_metric = checkpoint['metric']
        epochs = 0
    else:
        if args.ignore_ckpt:
            print('Ignoring saved checkpoint')
        else:
            print('No saved model found')
        best_epoch = -1
        best_metric = np.inf
    net.train()

    if net.estimate_type in ['point']:
        mse_loss = torch.nn.MSELoss()

    curr_patience = args.patience
    curr_step = 0
    for curr_epoch in range(best_epoch+1, best_epoch+1+epochs):
        epoch_loss, epoch_time = 0., 0.
        for i, data in enumerate(trainloader, 0):
            st = time.time()
            inputs, target, feats_in, feats_tgt, _, _ = data
            target = target.to(args.device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            teacher_forcing_ratio = args.teacher_forcing_ratio
            teacher_force = True if random.random() <= teacher_forcing_ratio else False
            if 'nar' in model_name:
                out = net(
                    feats_in.to(args.device), inputs.to(args.device),
                    feats_tgt.to(args.device), target.to(args.device)
                )
                if net.estimate_type in ['point']:
                    means = out
                elif net.estimate_type in ['variance']:
                    means, stds = out
                elif net.estimate_type in ['covariance']:
                    means, stds, vs = out
            else:
                out = net(
                    feats_in.to(args.device), inputs.to(args.device),
                    feats_tgt.to(args.device), target.to(args.device),
                    teacher_force=teacher_force
                )
                if net.estimate_type in ['point']:
                    means = out
                elif net.estimate_type in ['variance']:
                    means, stds = out
                elif net.estimate_type in ['covariance']:
                    means, stds, vs = out

            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            if model_name in [
                'seq2seqmse', 'convmse', 'convmsenonar',
                'rnn-mse-nar', 'trans-mse-nar', 'rnn-mse-ar',
                'nbeats-mse-nar', 'nbeatsd-mse-nar'
            ]:
                loss_mse = criterion(target.to(args.device), means.to(args.device))
                loss = loss_mse
            if model_name in ['seq2seqdilate']:
                loss, loss_shape, loss_temporal = dilate_loss(target, means, args.alpha, args.gamma, args.device)
            if model_name in [
                    'seq2seqnll', 'convnll', 'rnn-nll-nar', 'rnn-nll-ar',
                    'trans-mse-ar', 'trans-nll-ar',
                    'trans-fnll-ar', 'rnn-fnll-nar',
                    'transm-nll-nar', 'transm-fnll-nar',
                    'transda-nll-nar', 'transda-fnll-nar',
                ]:
                if args.train_twostage:
                    if curr_epoch < epochs/2:
                        stds = torch.ones_like(stds)
                    if curr_epoch-1 <= epochs/2 and curr_epoch > epochs/2:
                        best_metric = np.inf

                if net.estimate_type == 'covariance':
                    order = torch.randperm(target.shape[1])
                    #order = torch.arange(target.shape[1])
                    #means_shuffled = means[..., order, :].view(-1, args.b).squeeze(dim=-1)
                    #stds_shuffled = stds[..., order, :].view(-1, args.b).squeeze(dim=-1)
                    #vs_shuffled = vs[..., order, :].view(-1, args.b, vs.shape[-1])
                    #target_shuffled = target[..., order, :].view(-1, args.b).squeeze(dim=-1)
                    means_shuffled = torch.cat(
                        torch.split(means[..., order, :], args.b, dim=1), dim=0
                    ).squeeze(dim=-1)
                    stds_shuffled = torch.cat(
                        torch.split(stds[..., order, :], args.b, dim=1), dim=0
                    ).squeeze(dim=-1)
                    vs_shuffled = torch.cat(
                        torch.split(vs[..., order, :], args.b, dim=1), dim=0
                    )
                    target_shuffled = torch.cat(
                        torch.split(target[..., order, :], args.b, dim=1), dim=0
                    ).squeeze(dim=-1)
                    dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                        means_shuffled, vs_shuffled, stds_shuffled
                    )
                    loss = -torch.mean(dist.log_prob(target_shuffled))
                    #import ipdb ; ipdb.set_trace()
                elif net.estimate_type == 'variance':
                    dist = torch.distributions.normal.Normal(means, stds)
                    loss = torch.mean(-dist.log_prob(target))
                elif net.estimate_type in ['point']:
                    loss = mse_loss(target, means)

                if args.mse_loss_with_nll:
                    loss += criterion(target, means)
            if model_name in ['rnn-aggnll-nar']:
                order = torch.randperm(target.shape[1])
                means_shuffled = torch.squeeze(means[..., order, :].view(-1, args.b), dim=-1)
                stds_shuffled = torch.squeeze(stds[..., order, :].view(-1, args.b), dim=-1)
                vs_shuffled = vs[..., order, :].view(-1, args.b, vs.shape[-1])
                target_shuffled = torch.squeeze(target[..., order, :].view(-1, args.b), dim=-1)
                dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                    means_shuffled, vs_shuffled, stds_shuffled
                )
                loss = -torch.mean(dist.log_prob(target_shuffled))

                K = 4
                bs, horizon, groups = target.shape[0], target.shape[1], int(target.shape[1]/K)
                newBS = bs * groups
                target_grouped = target[..., 0].reshape(newBS, K)
                #print(mean[..., 0].is_contiguous(), newBS, time_in_1)
                means_grouped = means[..., 0].reshape(newBS, K)
                stds_grouped = stds[..., 0].reshape(newBS, K)
                vs_grouped = vs[..., :].reshape(newBS, K, -1)


                dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                    means_grouped, vs_grouped, stds_grouped
                )
                dist_sum = torch.distributions.normal.Normal(
                    torch.mean(dist.mean), torch.sqrt(1./K * torch.sum(dist.covariance_matrix, dim=(-2,-1)))
                )
                target_sum = torch.mean(target_grouped, dim=-1)
                sum_nll_loss = -torch.mean(dist_sum.log_prob(target_sum))
                loss += sum_nll_loss

                if args.mse_loss_with_nll:
                    loss += criterion(target, means)

            if model_name in ['trans-q-nar', 'rnn-q-nar', 'rnn-q-ar']:
                quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float)
                #quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float)
                #quantiles = torch.tensor([0.45, 0.5, 0.55], dtype=torch.float)
                quantile_weights = torch.ones_like(quantiles, dtype=torch.float)
                #quantile_weights = torch.tensor([1., 1., 1.], dtype=torch.float)
                loss = QuantileLoss(
                    quantiles, quantile_weights
                )(target, means, stds)

                #loss += torch.mean(stds)
                #import ipdb
                #ipdb.set_trace()

            #if i==0:
            #    import ipdb
            #    ipdb.set_trace()

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            et = time.time()
            epoch_time += (et-st)
            print('Time required for batch ', i, ':', et-st, 'loss:', loss.item(), teacher_forcing_ratio, teacher_force, curr_patience)
            #if i>=100:
            #    break
            if (curr_step % args.print_every == 0):
                (
                    _, _, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae, metric_crps_part, metric_nll, metric_ql
                )= eval_base_model(
                    args, model_name, net, devloader, norm, args.gamma, verbose=1
                )

                if model_name in ['seq2seqdilate']:
                    metric = metric_dilate
                elif 'mse' in model_name:
                    #metric = metric_crps
                    metric = metric_mse
                elif 'nll' in model_name:
                    metric = metric_nll
                    #metric = metric_crps
                elif '-q-' in model_name:
                    metric = metric_ql

                #if True:
                if metric < best_metric:
                    curr_patience = args.patience
                    best_metric = metric
                    best_epoch = curr_epoch
                    state_dict = {
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': best_epoch,
                                'metric': best_metric,
                                }
                    torch.save(state_dict, saved_models_path)
                    print('Model saved at epoch', curr_epoch, 'step', curr_step)
                else:
                    curr_patience -= 1

                scheduler.step(metric)

                # ...log the metrics
                if model_name in ['seq2seqdilate']:
                    writer.add_scalar('dev_metrics/dilate', metric_dilate, curr_step)
                writer.add_scalar('dev_metrics/crps', metric_crps, curr_step)
                writer.add_scalar('dev_metrics/mae', metric_mae, curr_step)
                writer.add_scalar('dev_metrics/mse', metric_mse, curr_step)
                writer.add_scalar('dev_metrics/nll', metric_nll, curr_step)
                writer.add_scalar('dev_metrics/ql', metric_ql, curr_step)

            curr_step += 1 # Increment the step
            if curr_patience == 0:
                break

        # ...log the epoch_loss
        if model_name in ['seq2seqdilate']:
            writer.add_scalar('training_loss/DILATE', epoch_loss, curr_epoch)
        if model_name in [
            'seq2seqmse', 'convmse', 'convmsenonar', 'rnn-mse-nar', 'trans-mse-nar', 'rnn-mse-ar',
            'nbeats-mse-nar', 'nbeatsd-mse-nar'
        ]:
            writer.add_scalar('training_loss/MSE', epoch_loss, curr_epoch)
        if model_name in ['seq2seqnll', 'convnll', 'trans-q-nar', 'rnn-q-nar', 'rnn-q-ar']:
            writer.add_scalar('training_loss/NLL', epoch_loss, curr_epoch)
        writer.add_scalar('training_time/epoch_time', epoch_time, curr_epoch)


        if(verbose):
            if (curr_step % args.print_every == 0):
                print('curr_epoch ', curr_epoch, \
                      ' epoch_loss ', epoch_loss, \
                      ' loss shape ',loss_shape.item(), \
                      ' loss temporal ',loss_temporal.item(), \
                      'learning_rate:', optimizer.param_groups[0]['lr'])

        if curr_patience == 0:
            break

    print('Best model found at epoch', best_epoch)
    #net.load_state_dict(torch.load(saved_models_path))
    checkpoint = torch.load(saved_models_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()
    (
        _, _, pred_mu, pred_std,
        metric_dilate, metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae, metric_crps_part, metric_nll, metric_ql
    ) = eval_base_model(
        args, model_name, net, devloader, norm, args.gamma, verbose=1
    )

    if model_name in ['seq2seqdilate']:
        metric = metric_dilate
    else:
        metric = metric_crps

    return metric
