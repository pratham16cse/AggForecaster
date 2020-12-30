import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.distributions.normal import Normal


class NetFullyConnected(torch.nn.Module):
    """docstring for NetFullyConnected"""
    def __init__(
        self, input_length, input_size, target_length, output_size,
        hidden_size, num_hidden_layers, fc_units,
        point_estimates, deep_std, variance_net, second_moment,
        device
    ):
        super(NetFullyConnected, self).__init__()

        self.point_estimates = point_estimates
        self.deep_std = deep_std
        self.variance_net = variance_net
        self.second_moment = second_moment
        self.device = device

        self.input_size = input_size
        self.input_length = input_length
        self.output_size = output_size
        self.target_length = target_length

        self.input_dim = self.input_size*self.input_length
        self.output_dim = self.output_size*self.target_length

        self.hidden_size = hidden_size
        self.fc_units = fc_units
        self.num_hidden_layers = num_hidden_layers

        self.layers = self.get_fully_connected_net()
        self.fc = nn.Linear(self.hidden_size, self.fc_units)
        self.out_mean = nn.Linear(self.fc_units, self.output_dim)
        if self.variance_net:
            self.layers_var = self.get_fully_connected_net()
            self.fc_var = nn.Linear(self.hidden_size, self.fc_units)
        if not self.deep_std:
            self.out_std = nn.Linear(self.fc_units, self.output_dim)
        else:
            self.out_std_1 = nn.Linear(self.fc_units, self.fc_units)
            self.out_std_2 = nn.Linear(self.fc_units, self.fc_units)
            self.out_std_3 = nn.Linear(self.fc_units, self.output_dim)

    def get_fully_connected_net(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.input_dim, self.hidden_size))
        for i in range(self.num_hidden_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        return layers

    def get_std(self, output, means):
        if not self.deep_std:
            stds = self.out_std(output)
            stds = F.softplus(stds) + 1e-3
        else:
            stds_1 = F.relu(self.out_std_1(output))
            stds_2 = F.relu(self.out_std_2(stds_1))
            stds = F.softplus(self.out_std_3(stds_2)) + 1e-3

        if self.second_moment:
            stds = stds - torch.pow(means, 2)
            stds = torch.sqrt(F.softplus(stds)) + 1e-3

        return stds

    def forward(self, feats_in, x_in, feats_tgt, x_tgt=None):
        '''
        Parameters of forward(..) are kept same as that of Net_GRU for
        code sharing. Other parameters tha x_in are ignored.
        '''
        x_in_ = x_in.view(-1, self.input_dim)

        output = x_in_
        for i, layer in enumerate(self.layers):
            output = layer(output)
            output = F.relu(output)
        output = F.relu( self.fc(output) )
        means = self.out_mean(output)

        if self.variance_net:
            output_var = x_in_
            for i, layer in enumerate(self.layers_var):
                output_var = layer(output_var)
                output_var = F.relu(output_var)
            output_var = F.relu( self.fc_var(output_var) )

        if self.variance_net:
            stds = self.get_std(output_var, means)
        else:
            stds = self.get_std(output, means)

        means = means.view(-1, self.target_length, self.output_size)
        stds = stds.view(-1, self.target_length, self.output_size)

        return means, stds


class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)      
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        #[num_layers*num_directions,batch,hidden_size]   
        return torch.zeros(self.num_grulstm_layers, batch_size, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_grulstm_layers,
        fc_units, output_size, deep_std, second_moment,
        variance_rnn
    ):
        super(DecoderRNN, self).__init__()      

        self.output_size = output_size
        self.deep_std = deep_std
        self.second_moment = second_moment
        self.variance_rnn = variance_rnn

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out_mean = nn.Linear(fc_units, output_size)
        if self.variance_rnn:
            self.gru_var = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
            self.fc_var = nn.Linear(hidden_size, fc_units)
        if not self.deep_std:
            self.out_std = nn.Linear(fc_units, output_size)
        else:
            self.out_std_1 = nn.Linear(fc_units, fc_units)
            self.out_std_2 = nn.Linear(fc_units, fc_units)
            self.out_std_3 = nn.Linear(fc_units, output_size)

    def get_std(self, output, means):
        if not self.deep_std:
            stds = self.out_std(output)
            stds = F.softplus(stds) + 1e-3
        else:
            stds_1 = F.relu(self.out_std_1(output))
            stds_2 = F.relu(self.out_std_2(stds_1))
            stds = F.softplus(self.out_std_3(stds_2)) + 1e-3

        if self.second_moment:
            stds = stds - torch.pow(means, 2)
            stds = torch.sqrt(F.softplus(stds)) + 1e-3

        return stds

    def forward(self, input, hidden, hidden_var):
        output, hidden = self.gru(input, hidden)
        output = F.relu( self.fc(output) )
        means = self.out_mean(output)

        if self.variance_rnn:
            output_var, hidden_var = self.gru_var(input, hidden_var)
            output_var = F.relu( self.fc_var(output_var) )
        else:
            hidden_var = None

        if self.variance_rnn:
            stds = self.get_std(output_var, means)
        else:
            stds = self.get_std(output, means)

        return (means, stds), (hidden, hidden_var)
    
class Net_GRU(nn.Module):
    def __init__(
        self, encoder, decoder, target_length, use_time_features,
        point_estimates, teacher_forcing_ratio, deep_std, device
    ):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.use_time_features = use_time_features
        self.point_estimates = point_estimates
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.deep_std = deep_std
        
    def forward(self, feats_in, x_in, feats_tgt, x_tgt=None, sample_variance=False):

        # Encoder
        input_length  = x_in.shape[1]
        if self.use_time_features:
            enc_in = torch.cat((x_in, feats_in), dim=-1)
        else:
            enc_in = x_in
        encoder_hidden = self.encoder.init_hidden(enc_in.shape[0], self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(enc_in[:,ei:ei+1,:]  , encoder_hidden)
            

        # Decoder
        if sample_variance and x_tgt is None:
            M = 40
        else:
            M = 1
        means = torch.zeros([M, x_in.shape[0], self.target_length, self.decoder.output_size]).to(self.device)
        stds = torch.zeros([M, x_in.shape[0], self.target_length, self.decoder.output_size]).to(self.device)

        for m in range(M):
            dec_in = enc_in[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
            decoder_hidden = encoder_hidden
            decoder_hidden_var = encoder_hidden
            for di in range(self.target_length):

                (step_means, step_stds), (decoder_hidden, decoder_hidden_var) \
                    = self.decoder(dec_in, decoder_hidden, decoder_hidden_var)

                # Training Mode
                if x_tgt is not None:
                    if random.random() < self.teacher_forcing_ratio:
                        # Teacher Forcing
                        dec_in = x_tgt[:, di:di+1]
                    else:
                        dec_in = step_means
                else:
                    if sample_variance:
                        dist = Normal(step_means, step_stds)
                        sample = dist.sample()
                        dec_in  = sample
                    else:
                        dec_in = step_means


                # Add features
                if self.use_time_features:
                    dec_in = torch.cat((dec_in, feats_tgt[:, di:di+1]), dim=-1)


                if x_tgt is None and sample_variance:
                    means[m, :, di:di+1, :] = sample
                else:
                    means[m, :, di:di+1, :] = step_means
                stds[m, :, di:di+1, :] = step_stds
                #print('di', di)
        if sample_variance and x_tgt is None:
            var, means = torch.var_mean(means, 0)
            stds = torch.sqrt(var)
        else:
            means, stds = torch.squeeze(means, 0), torch.squeeze(stds, 0)
        if self.point_estimates:
            stds = None
        return means, stds


def get_base_model(
    args, config, level, N_input, N_output,
    input_size, output_size, point_estimates
):

    #hidden_size = max(int(config['hidden_size']*1.0/int(np.sqrt(level))), args.fc_units)

    if args.fully_connected_agg_model:
        net_gru = NetFullyConnected(
            input_length=N_input, input_size=input_size,
            target_length=N_output, output_size=output_size,
            hidden_size=hidden_size, num_hidden_layers=args.num_grulstm_layers,
            fc_units=args.fc_units,
            point_estimates=point_estimates, deep_std=args.deep_std,
            variance_net=args.variance_rnn, second_moment=args.second_moment,
            device=args.device
        )
    else:
        encoder = EncoderRNN(
            input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=args.num_grulstm_layers,
            batch_size=args.batch_size
        ).to(args.device)
        decoder = DecoderRNN(
            input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=args.num_grulstm_layers,
            fc_units=args.fc_units, output_size=output_size, deep_std=args.deep_std,
            second_moment=args.second_moment, variance_rnn=args.variance_rnn
        ).to(args.device)
        net_gru = Net_GRU(
            encoder,decoder, N_output, args.use_time_features,
            point_estimates, args.teacher_forcing_ratio, args.deep_std,
            args.device
        ).to(args.device)

    return net_gru
