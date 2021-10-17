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


def get_optimizer(args, lr, net):
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)
    return optimizer, scheduler


def train_model(
    args, model_name, net, data_dict, saved_models_path, writer, verbose=1,
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
    cos_sim = torch.nn.CosineSimilarity(dim=2)

    if (not args.ignore_ckpt) and os.path.isfile(saved_models_path):
        print('Loading from saved model')
        checkpoint = torch.load(saved_models_path, map_location=args.device)
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
            #teacher_force = True if random.random() <= teacher_forcing_ratio else False
            if model_name in [
                'trans-nll-atr', 'rnn-mse-ar', 'rnn-nll-ar',
                'gpt-nll-ar', 'gpt-mse-ar'
            ]:
                teacher_force = True
            else:
                teacher_force = False
            out = net(
                feats_in.to(args.device), inputs.to(args.device),
                feats_tgt.to(args.device), target.to(args.device),
                teacher_force=teacher_force
            )
            if net.is_signature:
                if net.estimate_type in ['point']:
                    means, dec_state, sig_state = out
                elif net.estimate_type in ['variance']:
                    means, stds, dec_state, sig_state = out
                elif net.estimate_type in ['covariance']:
                    means, stds, vs, dec_state, sig_state = out
                elif net.estimate_type in ['bivariate']:
                    means, stds, rho, dec_state, sig_state = out
            else:
                if net.estimate_type in ['point']:
                    means = out
                elif net.estimate_type in ['variance']:
                    means, stds = out
                elif net.estimate_type in ['covariance']:
                    means, stds, vs = out
                elif net.estimate_type in ['bivariate']:
                    means, stds, rho = out

            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)


            if model_name in ['seq2seqdilate']:
                raise NotImplementedError
                loss, loss_shape, loss_temporal = dilate_loss(
                    target, means, args.alpha, args.gamma, args.device
                )
            if net.estimate_type == 'covariance':
                order = torch.randperm(target.shape[1])
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
            elif net.estimate_type in ['bivariate']:
                means_avg = 0.5 * (means[..., :-1, :] + means[..., 1:, :])
                var_a, var_b = stds[..., :-1, :]**2, stds[..., 1:, :]**2
                var_avg = var_a/4. + var_b/4. + rho * var_a * var_b / 2.
                stds_avg = var_avg**0.5
                target_avg = 0.5 * (target[..., :-1, :] + target[..., 1:, :])

                dist = torch.distributions.normal.Normal(means, stds)
                dist_avg = torch.distributions.normal.Normal(means_avg, stds_avg)

                loss = torch.mean(-dist.log_prob(target))
                loss += torch.mean(-dist_avg.log_prob(target_avg))
                #import ipdb ; ipdb.set_trace()
            if net.is_signature:
                sig_loss = torch.mean(1. - cos_sim(dec_state, sig_state))
                loss += sig_loss


            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            et = time.time()
            epoch_time += (et-st)
            print('Time required for batch ', i, ':', \
                  et-st, 'loss:', loss.item(), \
                  teacher_forcing_ratio, teacher_force, curr_patience)
            #if i>=100:
            #    break
            if (curr_step % args.print_every == 0):
                (
                    _, _, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae, metric_crps_part, metric_nll
                )= eval_base_model(
                    args, model_name, net, devloader, norm, args.gamma, verbose=1
                )

                if model_name in ['seq2seqdilate']:
                    raise NotImplementedError
                    metric = metric_dilate

                if net.estimate_type in ['point']:
                    metric = metric_mse
                elif net.estimate_type in ['variance', 'covariance', 'bivariate']:
                    metric = metric_nll
                    #metric = metric_crps

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
                    raise NotImplementedError
                    writer.add_scalar('dev_metrics/dilate', metric_dilate, curr_step)
                writer.add_scalar('dev_metrics/crps', metric_crps, curr_step)
                writer.add_scalar('dev_metrics/mae', metric_mae, curr_step)
                writer.add_scalar('dev_metrics/mse', metric_mse, curr_step)
                writer.add_scalar('dev_metrics/nll', metric_nll, curr_step)

            curr_step += 1 # Increment the step
            if curr_patience == 0:
                break

        # ...log the epoch_loss
        if model_name in ['seq2seqdilate']:
            raise NotImplementedError
            writer.add_scalar('training_loss/DILATE', epoch_loss, curr_epoch)
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
    checkpoint = torch.load(saved_models_path, map_location=args.device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()
    (
        _, _, pred_mu, pred_std,
        metric_dilate, metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae, metric_crps_part, metric_nll
    ) = eval_base_model(
        args, model_name, net, devloader, norm, args.gamma, verbose=1
    )
