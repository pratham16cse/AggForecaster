import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.distributions.normal import Normal
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)

    
class NARTransformerModel(nn.Module):
    def __init__(self, dec_len, feats_info, deep_std, device):
        super(NARTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.deep_std = deep_std
        self.device = device
        self.use_covariate_var_model = False

        self.kernel_size = 10
        nkernel = 32

        self.warm_start = self.kernel_size * 5

        self.embed_feat_layers = []
        for idx, (card, emb_size) in self.feats_info.items():
            if card is not 0:
                self.embed_feat_layers.append(nn.Embedding(card, emb_size))
            else:
                self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
        self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

        in_channels = sum([s for (_, s) in self.feats_info.values()])
        self.conv_feats = nn.Conv1d(
            kernel_size=self.kernel_size,stride=1, in_channels=in_channels, out_channels=nkernel
        )
        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=1,
            in_channels=1, out_channels=nkernel
        )

        self.linearMap = nn.Sequential(nn.ReLU(),nn.Linear(2*nkernel,nkernel))
        self.positional = PositionalEncoding(d_model=nkernel)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=nkernel, nhead=1, dropout=0.5, dim_feedforward=256
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=4)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=1, dropout=0.5, dim_feedforward=256
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=4)

        self.finalLinearMap = nn.Sequential(nn.ReLU(),nn.Linear(nkernel, 2))

        if self.deep_std:
            self.deepstd_layer = nn.Sequential(
                nn.Linear(nkernel, nkernel), nn.ReLU(),
                nn.Linear(nkernel, nkernel), nn.ReLU(),
                nn.Linear(nkernel, 1)
            )

        self.dec_pos_embed_layer = nn.Sequential(
            nn.Embedding(self.dec_len, nkernel)
            #nn.Linear(nkernel, nkernel), nn.ReLU(),
            #nn.Linear
        )

        if self.use_covariate_var_model:

            self.covvar_embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not 0:
                    self.covvar_embed_feat_layers.append(nn.Embedding(card, emb_size))
                else:
                    self.covvar_embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.covvar_embed_feat_layers = nn.ModuleList(self.covvar_embed_feat_layers)

            in_channels = sum([s for (_, s) in self.feats_info.values()])
            self.covvar_conv_feats = nn.Conv1d(
                kernel_size=self.kernel_size,stride=1, in_channels=in_channels, out_channels=nkernel
            )
            self.covvvar_dec_pos_embed_layer = nn.Sequential(
                nn.Embedding(self.dec_len, nkernel)
                #nn.Linear(nkernel, nkernel), nn.ReLU(),
                #nn.Linear
            )
            self.covvar_finalLinearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
       
    def forward(self, feats_in, X_in, feats_out, X_out=None):


        mean = X_in.mean(dim=1,keepdim=True)
        std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)/std

        feats_in_merged, feats_out_merged = [], []
        for i in range(feats_in.shape[-1]):
            card = self.feats_info[i][0]
            if card is not 0:
                feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
            else:
                feats_in_ = feats_in[..., i:i+1]
            feats_in_merged.append(
                self.embed_feat_layers[i](feats_in_)
            )
        feats_in_merged = torch.cat(feats_in_merged, dim=2)
        for i in range(feats_out.shape[-1]):
            card = self.feats_info[i][0]
            if card is not 0:
                feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
            else:
                feats_out_ = feats_out[..., i:i+1]
            feats_out_merged.append(
                self.embed_feat_layers[i](feats_out_)
            )
        feats_out_merged = torch.cat(feats_out_merged, dim=2)

        feats_in_embed = self.conv_feats(
            feats_in_merged.transpose(1,2)
        ).transpose(1,2).clamp(min=0)[:,::self.kernel_size,:]      
        X_in_embed = self.conv_data(X_in.transpose(1,2)).transpose(1,2).clamp(min=0)[:, ::self.kernel_size,:]
        feats_in_embed = self.linearMap(torch.cat([feats_in_embed,X_in_embed],dim=-1)).transpose(0,1)
        encoder_output = self.encoder(self.positional(feats_in_embed))
       
        feats_out_merged = torch.cat([feats_in_merged[:,-self.warm_start+1:, :],feats_out_merged],dim=1)
        feats_out_embed = self.conv_feats(
            feats_out_merged.transpose(1,2)
        ).transpose(1,2).clamp(min=0)
        X_out_embed = self.conv_data(
            torch.cat(
                [
                    X_in[..., -self.warm_start+1:, :],
                    torch.zeros([X_in.shape[0], self.dec_len, X_in.shape[-1]], dtype=torch.float, device=self.device)
                ],
                dim=1
            ).transpose(1, 2)
        ).transpose(1, 2).clamp(min=0)
        feats_out_embed = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1))

        X_out = self.decoder(
            self.positional(feats_out_embed.transpose(0,1)), encoder_output
        ).clamp(min=0)
        X_out = X_out.transpose(0,1)

        mean_out = self.finalLinearMap(X_out)[:,:,0:1]
        mean_out = mean_out*std+mean

        dec_pos = torch.ones((X_in.shape[0], 1)) * torch.unsqueeze(torch.range(0, self.dec_len-1), dim=0)
        dec_pos = torch.cat([torch.zeros(X_in.shape[0], X_out.shape[1]-self.dec_len), dec_pos], dim=1)
        dec_pos_embed = self.dec_pos_embed_layer(dec_pos.type(torch.LongTensor).to(self.device))
        if self.deep_std:
            std_out = self.finalLinearMap(X_out+dec_pos_embed)[:,:,1:2]
        elif self.use_covariate_var_model:
            feats_out_merged = []
            for i in range(feats_out.shape[-1]):
                card = self.feats_info[i][0]
                if card is not 0:
                    feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                else:
                    feats_out_ = feats_out[..., i:i+1]
                feats_out_merged.append(
                    self.embed_feat_layers[i](feats_out_)
                )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_out_embed = self.conv_feats(
                feats_out_merged.transpose(1,2)
            ).transpose(1,2).clamp(min=0)

            covvar_dec_pos_embed = self.dec_pos_embed_layer(dec_pos.type(torch.LongTensor).to(self.device))

            std_out = self.covvar_finalLinearMap(feats_out_embed+covvar_dec_pos_embed)

        else:
            std_out = self.deepstd_layer(X_out)
        std_out = F.softplus((std_out*std)/2)


        return mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :]


class RNNNARModel(nn.Module):
    def __init__(
            self, dec_len, num_rnn_layers, feat_set_size, embed_size, hidden_size, batch_size,
            point_estimates, device
        ):
        super(RNNNARModel, self).__init__()

        self.dec_len = dec_len
        self.num_rnn_layers = num_rnn_layers
        self.feat_set_size = feat_set_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.point_estimates = point_estimates
        self.device = device

        #self.embed_y_layer = nn.Linear(1, self.embed_size)
        #self.embed_feats_layer = nn.Linear(1, embed_size, bias=False)
        self.embed_feats_layer = nn.Embedding(self.feat_set_size, self.embed_size)

        self.encoder = nn.LSTM(1+self.embed_size, self.hidden_size, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size+self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )

    def forward(self, feats_in, X_in, feats_out, X_out=None):

        feats_in = torch.squeeze(feats_in, dim=-1)
        feats_out = torch.squeeze(feats_out, dim=-1)

        feats_in_embed = self.embed_feats_layer(feats_in)
        #X_in_embed = self.embed_y_layer(X_in)
        feats_out_embed = self.embed_feats_layer(feats_out)

        enc_input = torch.cat([feats_in_embed, X_in], dim=-1)

        enc_hidden = self.init_hidden(X_in.shape[0])
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        enc_output_tile = enc_output[..., -1:, :].repeat(1, self.dec_len, 1)
        dec_input = torch.cat([feats_out_embed, enc_output_tile], dim=-1)
        means = self.decoder(dec_input)

        if self.point_estimates:
            stds = None

        return means, stds

class ConvModelNonAR(torch.nn.Module):
    """docstring for ConvModel non autoregressive"""
    def __init__(
        self, input_length, input_size, target_length, output_size,
        window_size, feat_set_size, embed_size,
        hidden_size, num_rnn_layers, fc_units, use_time_features,
        point_estimates, teacher_forcing_ratio, deep_std, second_moment,
        variance_rnn, device, input_dropout,
        dropout=0.5
    ):
        super(ConvModelNonAR, self).__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.input_length = input_length
        self.output_size = output_size
        self.target_length = target_length
        self.conv_out_channels = hidden_size
        self.feat_set_size = feat_set_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.fc_units = fc_units
        self.num_rnn_layers = num_rnn_layers

        self.use_time_features = use_time_features
        self.point_estimates = point_estimates
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.deep_std = deep_std
        self.input_dropout = input_dropout

        self.embed_feats_layer = nn.Embedding(self.feat_set_size, self.embed_size)

        #conv_window_size = min(15, self.target_length)
        conv_window_size = self.window_size
        self.x_conv_layer = nn.Conv1d(
            1, self.conv_out_channels, conv_window_size, stride=conv_window_size
        )
        self.feats_conv_layer = nn.Conv1d(
            self.embed_size, self.conv_out_channels, conv_window_size, stride=conv_window_size
        )
        #self.deconv_layer = nn.ConvTranspose1d(
        #    hidden_size, 1, self.target_length - self.target_length//self.K + 1, stride=1
        #)
        self.deconv_layer = nn.ConvTranspose1d(
            hidden_size, hidden_size, conv_window_size, stride=conv_window_size
        )
        self.deagg_linear = nn.Linear(hidden_size, 1)
        if not self.point_estimates:
            self.deconv_layer_std = nn.ConvTranspose1d(
            hidden_size, 1, self.target_length - self.target_length//self.window_size + 1, stride=1
        )

        self.encoder = nn.LSTM(
            2*self.conv_out_channels, self.hidden_size, self.num_rnn_layers, batch_first=True
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.conv_out_channels+self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.deconv_mean = nn.ConvTranspose1d(
            hidden_size, hidden_size, conv_window_size, stride=conv_window_size
        )
        self.out_mean = nn.Linear(hidden_size, 1)

        self.decoder_std = nn.Sequential(
            nn.Linear(self.conv_out_channels+self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.deconv_std = nn.ConvTranspose1d(
            hidden_size, hidden_size, conv_window_size, stride=conv_window_size
        )
        self.out_std = nn.Linear(hidden_size, 1)


    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )


    def forward(self, feats_in, x_in, feats_out, x_tgt=None, teacher_force=False):

        feats_in = torch.squeeze(feats_in, dim=-1)
        feats_out = torch.squeeze(feats_out, dim=-1)

        feats_in_embed = self.embed_feats_layer(feats_in)
        feats_out_embed = self.embed_feats_layer(feats_out)

        feats_in_conv = self.feats_conv_layer(feats_in_embed.transpose(1, 2)).transpose(1, 2)
        feats_out_conv = self.feats_conv_layer(feats_out_embed.transpose(1, 2)).transpose(1, 2)

        x_conv_output = self.x_conv_layer(x_in.transpose(1, 2)).transpose(1, 2)
        conv_output = torch.cat([feats_in_conv, x_conv_output], dim=-1)
        #if self.use_time_features:
        #    aggregates_feats_in = self.conv_feats_layer(feats_in.transpose(1, 2)).transpose(1, 2)
        #    aggregates_feats_tgt = self.conv_feats_layer(feats_tgt.transpose(1, 2)).transpose(1, 2)
        #if x_tgt is not None:
        #    aggregates_tgt = self.conv_layer(x_tgt.transpose(1, 2)).transpose(1, 2)

        encoder_hidden = self.init_hidden(x_in.shape[0])
        encoder_output, encoder_hidden = self.encoder(conv_output, encoder_hidden)

        means, stds = [], []
        decoder_hidden = encoder_output[..., -1:, :].repeat(
            1, self.target_length//self.window_size, 1,
        )
        #import ipdb
        #ipdb.set_trace()
        decoder_hidden = torch.cat([feats_out_conv, decoder_hidden], dim=-1)

        means = self.decoder_mean(decoder_hidden)
        stds = self.decoder_std(decoder_hidden)

        #import ipdb
        #ipdb.set_trace()

        means = self.deconv_mean(means.transpose(1, 2)).transpose(1, 2)
        stds = self.deconv_std(stds.transpose(1, 2)).transpose(1, 2)
        stds = F.softplus(stds) + 1e-3

        #import ipdb
        #ipdb.set_trace()
        #means = means.reshape(means.shape[0], means.shape[1]*means.shape[2], 1)
        means = self.out_mean(means)
        #stds = stds.reshape(stds.shape[0], stds.shape[1]*stds.shape[2], 1)
        stds = self.out_std(stds)

        #import ipdb
        #ipdb.set_trace()

        if self.point_estimates:
            stds = None

        return means, stds


class ConvModel(torch.nn.Module):
    """docstring for ConvModel"""
    def __init__(
        self, input_length, input_size, target_length, output_size,
        K, hidden_size, num_enc_layers, fc_units, use_time_features,
        point_estimates, teacher_forcing_ratio, deep_std, second_moment,
        variance_rnn, device, input_dropout,
        dropout=0.5
    ):
        super(ConvModel, self).__init__()
        self.K = K
        self.input_size = input_size
        self.input_length = input_length
        self.output_size = output_size
        self.target_length = target_length
        self.conv_out_channels = hidden_size

        self.fc_units = fc_units
        self.use_time_features = use_time_features
        self.point_estimates = point_estimates
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.deep_std = deep_std
        self.input_dropout = input_dropout

        #conv_window_size = min(15, self.target_length)
        conv_window_size = self.K
        self.aggregate_layer = nn.Conv1d(
            1, self.conv_out_channels, conv_window_size, stride=conv_window_size
        )
        #self.deaggregate_layer = nn.ConvTranspose1d(
        #    hidden_size, 1, self.target_length - self.target_length//self.K + 1, stride=1
        #)
        self.deaggregate_layer = nn.ConvTranspose1d(
            hidden_size, hidden_size, conv_window_size, stride=conv_window_size
        )
        self.deagg_linear = nn.Linear(hidden_size, 1)
        if not self.point_estimates:
            self.deaggregate_layer_std = nn.ConvTranspose1d(
            hidden_size, 1, self.target_length - self.target_length//self.K + 1, stride=1
        )

        encoder_layers = TransformerEncoderLayer(1, 1, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_enc_layers)

        self.encoder = EncoderRNN(
            self.conv_out_channels, hidden_size, num_enc_layers, batch_size=None,
            input_dropout=self.input_dropout
        )
        self.decoder = DecoderRNN(
            self.conv_out_channels, hidden_size, num_enc_layers,
            fc_units, self.conv_out_channels, deep_std, second_moment,
            variance_rnn,
            input_dropout=self.input_dropout
        )

    def forward(self, feats_in, x_in, feats_tgt, x_tgt=None, teacher_force=False):

        enc_in = x_in
        if self.use_time_features:
            enc_in = torch.cat((enc_in, feats_in), dim=-1)
        aggregates = self.aggregate_layer(enc_in.transpose(1, 2)).transpose(1, 2)
        if x_tgt is not None:
            aggregates_tgt = self.aggregate_layer(x_tgt.transpose(1, 2)).transpose(1, 2)

        encoder_hidden = self.encoder.init_hidden(x_in.shape[0], self.device)
        encoder_output, encoder_hidden = self.encoder(aggregates, encoder_hidden)


        means, stds = [], []
        decoder_hidden = encoder_hidden
        decoder_hidden_var = encoder_hidden
        for di in range(self.target_length//self.K):
            dec_in = aggregates[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
            dec_in_var = enc_in[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence

            (step_means, step_stds), (decoder_hidden, decoder_hidden_var) \
                = self.decoder(dec_in, dec_in_var, decoder_hidden, decoder_hidden_var)

            # Training Mode
            if x_tgt is not None:
                if teacher_force:
                    # Teacher Forcing
                    dec_in = aggregates_tgt[:, di:di+1]
                else:
                    dec_in = step_means
            else:
                dec_in = step_means
            dec_in_var = step_stds

                # Add features
            if self.use_time_features:
                dec_in = torch.cat((dec_in, feats_tgt[:, di:di+1]), dim=-1)
                dec_in_var = torch.cat((dec_in_var, feats_tgt[:, di:di+1]), dim=-1)


            #means.append(step_means)
            #stds.append(step_stds)
            means.append(decoder_hidden.transpose(0, 1))
            stds.append(decoder_hidden.transpose(0, 1))

        means = torch.cat(means, dim=1)
        stds = torch.cat(stds, dim=1)

        #import ipdb
        #ipdb.set_trace()

        means = self.deaggregate_layer(means.transpose(1, 2)).transpose(1, 2)
        if self.point_estimates:
            stds = self.deaggregate_layer(stds.transpose(1, 2)).transpose(1, 2)
        else:
            stds = self.deaggregate_layer_std(stds.transpose(1, 2)).transpose(1, 2)
        stds = F.softplus(stds) + 1e-3

        #import ipdb
        #ipdb.set_trace()
        #means = means.reshape(means.shape[0], means.shape[1]*means.shape[2], 1)
        means = self.deagg_linear(means)
        stds = stds.reshape(stds.shape[0], stds.shape[1]*stds.shape[2], 1)

        #import ipdb
        #ipdb.set_trace()

        if self.point_estimates:
            stds = None

        return means, stds


# class PositionalEncoding(torch.nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class TransformerDecoder(torch.nn.Module):

    def __init__(self, input_length, ninp, target_length, output_size, hidden_size):

        super(TransformerDecoder, self).__init__()

        self.input_length = input_length
        self.ninp = ninp
        self.target_length = target_length
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.hidden_layer = nn.Linear(self.input_length*self.ninp, hidden_size)
        self.output_layer = nn.Linear(
            self.hidden_size,
            self.target_length*self.output_size
        )

    def forward(self, x): # x: (batch, input_length, ninp)
        #print(x.shape, self.input_length, self.ninp)
        out = self.hidden_layer(x.reshape(-1, self.input_length*self.ninp))
        out = F.relu(out)
        out = self.output_layer(out)
        out = out.view(-1, self.target_length, self.output_size)
        return out

class TransformerModel(torch.nn.Module):

    def __init__(
        self, input_length, input_size, target_length, output_size,
        hidden_size, num_enc_layers, fc_units,
        use_time_features, point_estimates, teacher_forcing_ratio,
        deep_std, device,
        ninp, nhead, dropout=0.5
    ):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.input_length = input_length
        self.output_size = output_size
        self.target_length = target_length

        self.target_length = target_length
        self.use_time_features = use_time_features
        self.point_estimates = point_estimates
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.deep_std = deep_std

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_enc_layers)
        #self.encoder = nn.Embedding(input_size, ninp)
        self.encoder = nn.Linear(input_size, ninp)
        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, output_size)
        self.decoder_mean = TransformerDecoder(input_length, ninp, target_length, output_size, hidden_size)
        self.decoder_std = TransformerDecoder(input_length, ninp, target_length, output_size, hidden_size)

        self.kernel_size = max(input_length//5, 2)

        self.conv_layer = torch.nn.Conv1d(input_size, input_size, self.kernel_size)

        self.init_weights()

        self.src_mask = self.generate_square_subsequent_mask(self.input_length)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    #def forward(self, src, src_mask):
    def forward(self, feats_in, x_in, feats_tgt, x_tgt=None, sample_variance=False):

        # Apply Convolution
        #c_in = x_in.transpose(2, 1)
        #c_in = F.pad(c_in, (self.kernel_size-1, 0), "constant", 0)
        #c_in = self.conv_layer(c_in)
        #x_in = c_in.transpose(2, 1)

        src = self.encoder(x_in) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        input_length  = x_in.shape[1]
        if self.use_time_features:
            enc_in = torch.cat((src, feats_in), dim=-1)
        else:
            enc_in = src

        # Apply Transformer Encoder
        enc_in = enc_in.transpose(1, 0)
        output = self.transformer_encoder(enc_in)#, self.src_mask)
        output = output.transpose(1, 0)

        # Apply Transformer Decoder
        means = self.decoder_mean(output)
        stds = self.decoder_std(output)
        stds = F.softplus(stds) + 1e-3

        return means, stds

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
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size, input_dropout):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.input_dropout = input_dropout
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.dropout_layer = torch.nn.Dropout(p=self.input_dropout)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(self.dropout_layer(input), hidden)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        #[num_layers*num_directions,batch,hidden_size]   
        return torch.zeros(self.num_grulstm_layers, batch_size, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_grulstm_layers,
        fc_units, output_size, deep_std, second_moment,
        variance_rnn, input_dropout
    ):
        super(DecoderRNN, self).__init__()      

        self.output_size = output_size
        self.deep_std = deep_std
        self.second_moment = second_moment
        self.variance_rnn = variance_rnn
        self.input_dropout = input_dropout
        fc_units = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.dropout_layer = torch.nn.Dropout(p=self.input_dropout)
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

    def forward(self, input, input_var, hidden, hidden_var):
        output, hidden = self.gru(self.dropout_layer(input), hidden)
        #output = F.relu( self.fc(output) )
        means = self.out_mean(output)

        if self.variance_rnn:
            output_var, hidden_var = self.gru_var(input_var, hidden_var)
            #output_var = F.relu( self.fc_var(output_var) )
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
        
    def forward(self, feats_in, x_in, feats_tgt, x_tgt=None, sample_variance=False, teacher_force=False):

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
            dec_in_var = enc_in[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
            decoder_hidden = encoder_hidden
            decoder_hidden_var = encoder_hidden
            for di in range(self.target_length):

                (step_means, step_stds), (decoder_hidden, decoder_hidden_var) \
                    = self.decoder(dec_in, dec_in_var, decoder_hidden, decoder_hidden_var)

                # Training Mode
                if x_tgt is not None:
                    if teacher_force:
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
                dec_in_var = step_stds


                # Add features
                if self.use_time_features:
                    dec_in = torch.cat((dec_in, feats_tgt[:, di:di+1]), dim=-1)
                    dec_in_var = torch.cat((dec_in_var, feats_tgt[:, di:di+1]), dim=-1)


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
    args, base_model_name, level, N_input, N_output,
    input_size, output_size, point_estimates, feats_info
):

    #hidden_size = max(int(config['hidden_size']*1.0/int(np.sqrt(level))), args.fc_units)
    hidden_size = args.hidden_size

    if level != 1 and args.fully_connected_agg_model:
        net_gru = NetFullyConnected(
            input_length=N_input, input_size=input_size,
            target_length=N_output, output_size=output_size,
            hidden_size=hidden_size, num_hidden_layers=args.num_grulstm_layers,
            fc_units=args.fc_units,
            point_estimates=point_estimates, deep_std=args.deep_std,
            variance_net=args.variance_rnn, second_moment=args.second_moment,
            device=args.device
        )
    elif level != 1 and args.transformer_agg_model:
        net_gru = TransformerModel(
            input_length=N_input, input_size=input_size,
            target_length=N_output, output_size=output_size,
            hidden_size=hidden_size, num_enc_layers=args.num_grulstm_layers,
            fc_units=args.fc_units,
            use_time_features=args.use_time_features,
            point_estimates=point_estimates,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            deep_std=args.deep_std, device=args.device,
            ninp=args.fc_units, nhead=4, dropout=0.5
        )
    else:
        if 'seq2seq' in base_model_name:
            encoder = EncoderRNN(
                input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=args.num_grulstm_layers,
                batch_size=args.batch_size,
                input_dropout=args.input_dropout
            ).to(args.device)
            decoder = DecoderRNN(
                input_size=input_size, hidden_size=hidden_size, num_grulstm_layers=args.num_grulstm_layers,
                fc_units=args.fc_units, output_size=output_size, deep_std=args.deep_std,
                second_moment=args.second_moment, variance_rnn=args.variance_rnn,
                input_dropout=args.input_dropout
            ).to(args.device)
            net_gru = Net_GRU(
                encoder,decoder, N_output, args.use_time_features,
                point_estimates, args.teacher_forcing_ratio, args.deep_std,
                args.device
            ).to(args.device)
        elif 'conv' in base_model_name:
            if 'nar' in base_model_name:
                net_gru = ConvModelNonAR(
                    input_length=N_input, input_size=input_size,
                    target_length=N_output, output_size=output_size,
                    window_size=60, feat_set_size=4, embed_size=64,
                    hidden_size=hidden_size,
                    num_rnn_layers=args.num_grulstm_layers, fc_units=args.fc_units,
                    use_time_features=args.use_time_features,
                    point_estimates=point_estimates,
                    teacher_forcing_ratio=args.teacher_forcing_ratio,
                    deep_std=args.deep_std,
                    second_moment=args.second_moment,
                    variance_rnn=args.variance_rnn, device=args.device,
                    input_dropout=args.input_dropout
                ).to(args.device)
            else:
                net_gru = ConvModel(
                    input_length=N_input, input_size=input_size,
                    target_length=N_output, output_size=output_size,
                    K=60, hidden_size=hidden_size,
                    num_enc_layers=args.num_grulstm_layers, fc_units=args.fc_units,
                    use_time_features=args.use_time_features,
                    point_estimates=point_estimates,
                    teacher_forcing_ratio=args.teacher_forcing_ratio,
                    deep_std=args.deep_std,
                    second_moment=args.second_moment,
                    variance_rnn=args.variance_rnn, device=args.device,
                    input_dropout=args.input_dropout
                ).to(args.device)
        elif 'rnn' in base_model_name:
            net_gru = RNNNARModel(
                dec_len=N_output,
                num_rnn_layers=args.num_grulstm_layers,
                feat_set_size=60, embed_size=64, hidden_size=hidden_size,
                batch_size=args.batch_size,
                point_estimates=point_estimates,
                device=args.device
            ).to(args.device)
        elif 'trans' in base_model_name:
            net_gru = NARTransformerModel(
                N_output, feats_info, deep_std=args.deep_std, device=args.device
            ).to(args.device)

    return net_gru
