import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
        self, encoder, decoder, target_length, point_estimates,
        teacher_forcing_ratio, deep_std, device
    ):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.point_estimates = point_estimates
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.deep_std = deep_std
        
    def forward(self, x, target=None):
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(x.shape[0], self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
            
        decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden
        decoder_hidden_var = encoder_hidden
        
        means = torch.zeros([x.shape[0], self.target_length, self.decoder.output_size]).to(self.device)
        stds = torch.zeros([x.shape[0], self.target_length, self.decoder.output_size]).to(self.device)
        for di in range(self.target_length):
            (step_means, step_stds), (decoder_hidden, decoder_hidden_var) \
                = self.decoder(decoder_input, decoder_hidden, decoder_hidden_var)
            if target is not None:
                if random.random() < self.teacher_forcing_ratio:
                    # Teacher Forcing
                    decoder_input = target[:, di:di+1]
                else:
                    decoder_input = step_means
            else:
                decoder_input = step_means
            means[:, di:di+1, :] = step_means
            stds[:, di:di+1, :] = step_stds
        if self.point_estimates:
            stds = None
        return means, stds