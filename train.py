import os
import numpy as np
import torch
from loss.dilate_loss import dilate_loss
from eval import eval_base_model


def train_model(
    args, model_name, net, trainloader, devloader, testloader, norm,
    saved_models_path, output_dir,
    eval_every=50, verbose=1, Lambda=1
):

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
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
        epochs = args.epochs
    net.train()

    for curr_epoch in range(best_epoch+1, best_epoch+1+epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
            target = torch.tensor(target, dtype=torch.float32).to(args.device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            means, stds = net(inputs, target)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            if model_name in ['seq2seqmse']:
                loss_mse = criterion(target,means)
                loss = loss_mse
            if model_name in ['seq2seqdilate']:
                loss, loss_shape, loss_temporal = dilate_loss(target, means, args.alpha, args.gamma, args.device)
            if model_name in ['seq2seqnll']:
                dist = torch.distributions.normal.Normal(means, stds)
                loss = -torch.sum(dist.log_prob(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(verbose):
            if (curr_epoch % args.print_every == 0):
                print('curr_epoch ', curr_epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                metric_dilate, metric_mse, metric_dtw, metric_tdi = eval_base_model(
                    args, model_name, net, devloader, norm, args.gamma, verbose=1
                )

                if model_name in ['seq2seqdilate']:
                    metric = metric_dilate
                else:
                    metric = metric_mse

                if metric < best_metric:
                    best_metric = metric
                    best_epoch = curr_epoch
                    state_dict = {
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': best_epoch,
                                'metric': best_metric,
                                }
                    torch.save(state_dict, saved_models_path)
                    print('Model saved at epoch', curr_epoch)

    print('Best model found at epoch', best_epoch)
    #net.load_state_dict(torch.load(saved_models_path))
    checkpoint = torch.load(saved_models_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()
    metric_dilate, metric_mse, metric_dtw, metric_tdi = eval_base_model(
        args, model_name, net, devloader, norm, args.gamma,verbose=1
    )
