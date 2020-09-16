import numpy as np
import torch
from loss.dilate_loss import dilate_loss
from eval import eval_base_model


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
            inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
            target = torch.tensor(target, dtype=torch.float32).to(args.device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            means, stds = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            if model_name in ['seq2seqmse']:
                loss_mse = criterion(target,means)
                loss = loss_mse
            if model_name in ['seq2seqdilate']:
                loss, loss_shape, loss_temporal = dilate_loss(target, means, alpha, args.gamma, args.device)
            if model_name in ['seq2seqnll']:
                dist = torch.distributions.normal.Normal(means, stds)
                loss = -torch.sum(dist.log_prob(target))

            print(i, loss)
            #if torch.isnan(loss):
            #    import ipdb
            #    ipdb.set_trace()
            optimizer.zero_grad()
            #if torch.sum(torch.isnan(net.decoder.fc.weight)):
            #    import ipdb
            #    ipdb.set_trace()
            #print(optimizer.param_groups)
            loss.backward()
            if i >= 30:
                import ipdb
                ipdb.set_trace()
            optimizer.step()
            #if torch.sum(torch.isnan(net.decoder.fc.weight)):
            #    import ipdb
            #    ipdb.set_trace()

        if(verbose):
            if (epoch % args.print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                metric_mse, metric_dtw, metric_tdi = eval_base_model(args, net, devloader, args.gamma,verbose=1)

                if metric_mse < best_metric_mse:
                    best_metric_mse = metric_mse
                    best_epoch = epoch
                    torch.save(net.state_dict(), saved_models_path)
                    print('Model saved at epoch', epoch)

    print('Best model found at epoch', best_epoch)
    net.load_state_dict(torch.load(saved_models_path))
    net.eval()
    metric_mse, metric_dtw, metric_tdi = eval_base_model(args, net, devloader, args.gamma,verbose=1)
