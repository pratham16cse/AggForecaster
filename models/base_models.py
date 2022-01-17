import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.distributions.normal import Normal
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from models import transformer_manual_attn, transformer_dual_attn
from models import informer

class NBEATS_D(nn.Module):
    def __init__(
            self, input_width, dec_len, num_blocks, block_width, block_numlayers,
            point_estimates, feats_info, coeffs_info, use_coeffs
        ):
        super(NBEATS_D, self).__init__()

        self.num_blocks = num_blocks
        self.block_numlayers = block_numlayers
        self.point_estimates = point_estimates
        self.feats_info = feats_info
        self.coeffs_info = coeffs_info
        assert use_coeffs == True
        self.use_coeffs = use_coeffs

        input_size = input_width
        self.num_coeffs = len(self.coeffs_info)
        self.blocks = []
        for i in range(self.num_coeffs):
            coeff_block = []
            for j in range(num_blocks):
                block = nn.ModuleList(
                    [nn.Linear(input_size, block_width)]
                    + [nn.Linear(block_width, block_width) for _ in range(block_numlayers - 1)]
                )
                coeff_block.append(block)
            coeff_block = nn.ModuleList(coeff_block)
            self.blocks.append(coeff_block)
        self.blocks = nn.ModuleList(self.blocks)

        self.backcast_layer = []
        self.forecast_layer = []
        for i in range(self.num_coeffs):
            self.backcast_layer.append(nn.Linear(block_width, input_size))
            self.forecast_layer.append(nn.Linear(block_width, dec_len))
        self.backcast_layer = nn.ModuleList(self.backcast_layer)
        self.forecast_layer = nn.ModuleList(self.forecast_layer)
        
    def forward(self, feats_in, X_in, coeffs_in, feats_out, X_out=None, coeffs_out=None):
        block_input = X_in.squeeze(-1)
        forecasts = []
        for i in range(self.num_coeffs):
            coeff_input = coeffs_in[..., i]
            coeff_forecasts = []
            for j in range(self.num_blocks):
                curr_block = self.blocks[i][j]
                fc_input = coeff_input
                for k in range(self.block_numlayers):
                    fc_output = curr_block[k](fc_input)
                    fc_input = fc_output
                block_output = fc_output
                backcast_output = self.backcast_layer[i](block_output)
                forecast_output = self.forecast_layer[i](block_output)
                residual = coeff_input - backcast_output
                coeff_input = residual
                coeff_forecasts.append(forecast_output)
            forecasts.append(coeff_forecasts)

        for i in range(self.num_coeffs):
            forecasts[i] = torch.stack(forecasts[i], dim=1).sum(1)

        forecasts = torch.stack(forecasts, dim=1).sum(1).unsqueeze(-1)

        if self.point_estimates:
            stds = None
            v = None

        return forecasts, stds, v


class NBEATS(nn.Module):
    def __init__(
            self, input_width, dec_len, num_blocks, block_width, block_numlayers,
            point_estimates, feats_info, coeffs_info, use_coeffs
        ):
        super(NBEATS, self).__init__()
        
        self.num_blocks = num_blocks
        self.block_numlayers = block_numlayers
        self.point_estimates = point_estimates
        self.feats_info = feats_info
        self.coeffs_info = coeffs_info
        self.use_coeffs = use_coeffs

        input_size = input_width
        self.num_coeffs = len(self.coeffs_info)
        if self.use_coeffs:
            input_size += (self.num_coeffs * input_width)
        self.blocks = []
        for i in range(num_blocks):
            block = nn.ModuleList(
                [nn.Linear(input_size, block_width)]
                + [nn.Linear(block_width, block_width) for _ in range(block_numlayers - 1)]
            )
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
        self.backcast_layer = nn.Linear(block_width, input_size)
        self.forecast_layer = nn.Linear(block_width, dec_len)

    def forward(self, feats_in, X_in, coeffs_in, feats_out, X_out=None, coeffs_out=None):
        block_input = X_in.squeeze(-1)
        if self.use_coeffs:
            for i in range(self.num_coeffs):
                block_input = torch.cat([block_input, coeffs_in[..., i]], dim=1)
        forecasts = []
        for i in range(self.num_blocks):
            curr_block = self.blocks[i]
            fc_input = block_input
            for j in range(self.block_numlayers):
                fc_output = curr_block[j](fc_input)
                fc_input = fc_output
            block_output = fc_output
            backcast_output = self.backcast_layer(block_output)
            forecast_output = self.forecast_layer(block_output)
            residual = block_input - backcast_output
            block_input = residual
            forecasts.append(forecast_output)
            
        forecasts = torch.stack(forecasts, dim=1).sum(1).unsqueeze(-1)

        if self.point_estimates:
            stds = None
            v = None

        return forecasts, stds, v

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

    def forward(self, x, start_idx=0):
        x = x + self.pe[start_idx:start_idx+x.size(0), :].unsqueeze(1)
        return self.dropout(x)

    
class ARTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, device, is_signature=False
        ):
        super(ARTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.is_signature = is_signature
        self.use_covariate_var_model = False

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.warm_start = self.kernel_size * 5

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            in_channels = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            self.conv_feats = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=in_channels, out_channels=nkernel,
                bias=False,
                #padding=self.kernel_size//2
            )

        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            #bias=False,
            #padding=self.kernel_size//2
        )
        self.data_dropout = nn.Dropout(p=0.2)

        if self.use_feats:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
        else:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
        self.positional = PositionalEncoding(d_model=nkernel)

        enc_input_size = nkernel

        if self.t2v_type:
            if self.t2v_type not in ['local']:  
                self.t_size = sum([1 for (_, s) in self.feats_info.values() if s==-1])
            else:
                self.t_size = 1
            if self.t2v_type in ['mdh_lincomb']:
                self.t2v_layer_list = []
                for i in range(self.t_size):
                    self.t2v_layer_list.append(nn.Linear(1, nkernel))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                if self.t_size > 1:
                    self.t2v_linear =  nn.Linear(self.t_size*nkernel, nkernel)
                else:
                    self.t2v_linear = None
            elif self.t2v_type in ['local', 'mdh_parti', 'idx']:
                self.part_sizes = [nkernel//self.t_size]*self.t_size
                for i in range(nkernel%self.t_size):
                    self.part_sizes[i] += 1
                self.t2v_layer_list = []
                for i in range(len(self.part_sizes)):
                    self.t2v_layer_list.append(nn.Linear(1, self.part_sizes[i]))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                self.t2v_dropout = nn.Dropout(p=0.2)
                self.t2v_linear = None
            #import ipdb ; ipdb.set_trace()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['bivariate']:
            self.decoder_bv = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))
        if self.estimate_type in ['bivariate']:
            self.rho_layer = nn.Linear(nkernel, 2)

    def apply_signature(self, mean, X_in, feats_out, X_out):
        X_out = X_out - mean
        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)

            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_out.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_out.shape[0], self.kernel_size-1, X_out.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_out
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_out_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+X_out.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_out.shape[1], X_out.shape[0]).unsqueeze(-1)
                t_in = t_in / X_out.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            #if self.t2v_type is 'mdh_lincomb':
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            #import ipdb ; ipdb.set_trace()
            #t2v = torch.cat([t2v[0:1], torch.sin(t2v[1:])], dim=0)
            #enc_input = self.data_dropout(enc_input) + self.t2v_dropout(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)
        encoder_output = encoder_output.transpose(0, 1)

        return encoder_output

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=None
    ):

        #X_in = X_in[..., -X_in.shape[1]//5:, :]
        #feats_in = feats_in[..., -feats_in.shape[1]//5:, :]

        mean = X_in.mean(dim=1, keepdim=True)
        #std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)

        #import ipdb ; ipdb.set_trace()
        if self.use_feats:
            feats_in_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)

            feats_in_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_in_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_in_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_in_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_in.shape[0], self.kernel_size-1, X_in.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_in
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_in_embed,X_in_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_in_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_in.shape[1], X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_in[..., :, -self.t_size:].transpose(0,1)

            t2v = []
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)

        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_out_merged = torch.cat(
                [feats_in_merged[:,-self.warm_start+1:, :],feats_out_merged],
                dim=1
            )
            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)

        #import ipdb ; ipdb.set_trace()
        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    ),
                    X_in[..., -self.warm_start+1:, :],
                    torch.zeros(
                        [X_in.shape[0], self.dec_len, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    )
                ],
                dim=1
            ).transpose(1, 2)
        ).transpose(1, 2)

        if self.use_feats:
            dec_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            dec_input = X_out_embed.transpose(0,1)
        #import ipdb ; ipdb.set_trace()
        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+self.dec_len, dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(self.dec_len, X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            dec_input = dec_input + self.t2v_dropout(t2v)
        else:
            dec_input = self.positional(dec_input, start_idx=X_in.shape[1])
        #import ipdb ; ipdb.set_trace()

        decoder_output = self.decoder_mean(dec_input, encoder_output).clamp(min=0)
        decoder_output = decoder_output.transpose(0,1)
        mean_out = self.linear_mean(decoder_output)

        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            X_pred = self.decoder_std(dec_input, encoder_output).clamp(min=0)
            X_pred = X_pred.transpose(0,1)
            std_out = F.softplus(self.linear_std(X_pred))
            if self.estimate_type in ['covariance']:
                v_out = self.linear_v(X_pred)
            if self.estimate_type in ['bivariate']:
                X_pred = self.decoder_bv(dec_input, encoder_output)
                X_pred = X_pred.transpose(0,1)
                rho_out = self.rho_layer(X_pred)
                rho_out = rho_out[..., -self.dec_len:, :]
                rho_1, rho_2 = rho_out[..., 1:, :], rho_out[..., :-1, :]
                #rho_out = torch.einsum("ijk,ijk->ij", (rho_1, rho_2)).unsqueeze(-1)
                rho_out = (rho_1 * rho_2).sum(dim=-1, keepdims=True)
                #import ipdb ; ipdb.set_trace()
                rho_out = torch.tanh(rho_out)
            #import ipdb ; ipdb.set_trace()

        mean_out = mean_out + mean

        if self.is_signature:
            signature_state = self.apply_signature(mean, X_in, feats_out, X_out)
            decoder_output = decoder_output[..., -self.dec_len:, :]

        #import ipdb ; ipdb.set_trace()

        if self.is_signature:
            if self.estimate_type in ['point']:
                return mean_out[..., -self.dec_len:, :], decoder_output, signature_state
            elif self.estimate_type in ['variance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], decoder_output, signature_state)
            elif self.estimate_type in ['covariance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :], decoder_output, signature_state)
            elif self.estimate_type in ['bivariate']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out, decoder_output, signature_state)
        else:
            if self.estimate_type in ['point']:
                return mean_out[..., -self.dec_len:, :]
            elif self.estimate_type in ['variance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :])
            elif self.estimate_type in ['covariance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :])
            elif self.estimate_type in ['bivariate']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out)

class GPTTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, is_nar, device, is_signature=False
        ):
        super(GPTTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.is_signature = is_signature
        self.d_transform_typ = 'conv'
        self.f_transform_typ = 'conv'
        self.is_nar = is_nar

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.positional = PositionalEncoding(d_model=nkernel)
        if self.is_nar:
            self.warm_start = self.dec_len - self.kernel_size
        else:
            self.warm_start = 0

        if self.use_feats:
            self.use_local_weights = False
            self.embed_feat_layers = {}
            for idx, (card, emb_size) in self.feats_info.items():
                if card != -1 and card != 0 and emb_size > 0:
                    self.embed_feat_layers[str(idx)] = nn.Embedding(card, emb_size)
                elif emb_size == -2:
                    self.use_local_weights = True
                    self.tsid_idx = idx
                    self.num_local_weights = card
            self.embed_feat_layers = nn.ModuleDict(self.embed_feat_layers)
            feats_dim = sum([s for (_, s) in self.feats_info.values() if s>-1])

            if self.f_transform_typ in ['linear']:
                self.f_transform_lyr = nn.Linear(feats_dim, feats_dim)
            elif self.f_transform_typ in ['conv']:
                self.f_transform_lyr = nn.Conv1d(
                    kernel_size=self.kernel_size, stride=1,
                    in_channels=feats_dim, out_channels=feats_dim,
                )

            self.linear_map = nn.Linear(feats_dim+self.nkernel, self.nkernel)

        if self.d_transform_typ in ['linear']:
            self.d_transform_lyr = nn.Linear(1, nkernel)
        elif self.d_transform_typ in ['conv']:
            self.d_transform_lyr = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            )

        enc_input_size = nkernel
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        dec_input_size = nkernel
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['bivariate']:
            self.decoder_bv = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        if self.use_local_weights:
            #self.linear_mean = []
            #for i in range(self.num_local_weights):
            #    self.linear_mean.append(nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1)))
            #self.linear_mean = nn.ModuleList(self.linear_mean)
            self.linear_mean = nn.Embedding(self.num_local_weights, nkernel)
        else:
            self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            if self.use_local_weights:
                #self.linear_std = []
                #for i in range(self.num_local_weights):
                #    self.linear_std.append(nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1)))
                #self.linear_std = nn.ModuleList(self.linear_std)
                self.linear_std = nn.Embedding(self.num_local_weights, nkernel)
            else:
                self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))
        if self.estimate_type in ['bivariate']:
            self.rho_layer = nn.Linear(nkernel, 2)

    def merge_feats(self, feats):
        feats_merged = []
        for idx, efl in self.embed_feat_layers.items():
            feats_i = feats[..., int(idx)].type(torch.LongTensor).to(self.device)
            feats_merged.append(efl(feats_i))
        for idx, (card, emb_size) in self.feats_info.items():
            if card == 0:
                feats_merged.append(feats[..., idx:idx+1])
        feats_merged = torch.cat(feats_merged, dim=2)
        return feats_merged

    def pad_for_conv(self, x, trf_type):
        if trf_type in ['conv']:
            x_padded = torch.cat(
                [
                    torch.zeros(
                        (x.shape[0], self.kernel_size-1, x.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    x
                ],
                dim=1
            )
        elif trf_type in ['linear']:
            x_padded = x
        return x_padded

    def d_transform(self, x):
        if self.d_transform_typ in ['linear']:
            x_transform = self.d_transform_lyr(x)
        elif self.d_transform_typ in ['conv']:
            x_transform = self.d_transform_lyr(x.transpose(1,2)).transpose(1,2)

        return x_transform

    def f_transform(self, x):
        if self.f_transform_typ in ['linear']:
            x_transform = self.f_transform_lyr(x)
        elif self.f_transform_typ in ['conv']:
            x_transform = self.f_transform_lyr(x.transpose(1,2)).transpose(1,2)

        return x_transform

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=None
    ):

        mean = X_in.mean(dim=1, keepdim=True)
        #mean, _ = X_in.min(dim=1, keepdim=True)
        X_in = (X_in - mean)

        X_in_transformed = self.d_transform(self.pad_for_conv(X_in, self.d_transform_typ))
        if self.use_feats:
            feats_in_merged = self.merge_feats(feats_in)
            feats_in_transformed = self.f_transform(
                self.pad_for_conv(feats_in_merged, self.f_transform_typ)
            )
            enc_input = self.linear_map(torch.cat([feats_in_transformed, X_in_transformed], dim=-1))
        else:
            enc_input = X_in_transformed
        encoder_output = self.encoder(self.positional(enc_input.transpose(0,1)))

        #import ipdb ; ipdb.set_trace()
        if self.d_transform_typ in ['linear']: ps = 1 + self.warm_start
        elif self.d_transform_typ in ['conv']: ps = self.kernel_size + self.warm_start
        lps = X_in.shape[1] - ps + int(self.is_nar)
        if self.use_feats:
            if self.f_transform_typ in ['linear']: f_ps = 1 + self.warm_start
            elif self.f_transform_typ in ['conv']: f_ps = self.kernel_size + self.warm_start
            lf_ps = X_in.shape[1] - f_ps + int(self.is_nar)

        if self.is_nar: out_len = self.dec_len
        else: out_len = self.dec_len - 1

        #if X_out is not None:
        if self.is_nar==True or X_out is not None:
            if self.is_nar==True:
                X_out = torch.zeros(
                    (X_in.shape[0], self.dec_len, X_in.shape[2]),
                    dtype=torch.float, device=self.device
                )
            X_out_padded = torch.cat([X_in[..., lps:, :], X_out[..., :out_len, :]], dim=1)
            X_out_transformed = self.d_transform(X_out_padded)
            if self.use_feats:
                feats_out_padded = torch.cat(
                    [feats_in[..., lf_ps:, :], feats_out[..., :out_len, :]],
                    dim=1
                )
                feats_out_merged = self.merge_feats(feats_out_padded)
                feats_out_transformed = self.f_transform(feats_out_merged)
                #import ipdb ; ipdb.set_trace()
                dec_input = self.linear_map(
                    torch.cat([feats_out_transformed, X_out_transformed], dim=-1)
                )
            else:
                dec_input = X_out_transformed
            #dec_input = torch.cat([enc_input[..., ps:, :], dec_input[..., :out_len, :]], dim=1)
            #dec_input = dec_input[..., :, :]
            dec_input = self.positional(dec_input.transpose(0,1))
            #import ipdb ; ipdb.set_trace()

            decoder_output = self.decoder_mean(dec_input, encoder_output)#.clamp(min=0)
            decoder_output = decoder_output.transpose(0,1)
            if self.use_local_weights:
                local_indices = feats_out[..., self.tsid_idx].type(torch.LongTensor).to(self.device)
                local_weights = self.linear_mean(local_indices)
                decoder_output = decoder_output[..., -self.dec_len:, :]
                #import ipdb ; ipdb.set_trace()
                mean_out = (decoder_output * local_weights).sum(-1, keepdims=True)
                #mean_out = torch.einsum('ijk,ilk->ij', (local_weights, decoder_output)).unsqueeze(-1)
            else:
                mean_out = self.linear_mean(decoder_output)

            if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                X_pred = self.decoder_std(dec_input, encoder_output)#.clamp(min=0)
                X_pred = X_pred.transpose(0,1)
                if self.use_local_weights:
                    local_indices = feats_out[..., self.tsid_idx].type(torch.LongTensor).to(self.device)
                    local_weights = self.linear_std(local_indices)
                    X_pred = X_pred[..., -self.dec_len:, :]
                    std_out = (X_pred * local_weights).sum(-1, keepdims=True)
                    #std_out = torch.einsum('ijk,ilk->ij', (local_weights, X_pred)).unsqueeze(-1)
                    std_out = F.softplus(std_out)
                else:
                    std_out = F.softplus(self.linear_std(X_pred))

        else:
            std_out = []
            #import ipdb ; ipdb.set_trace()
            mean_out = list(torch.split(X_in[..., -ps:, :], 1, dim=1))
            if self.use_feats:
                f_padded = torch.cat([feats_in[..., -f_ps:, :], feats_out[..., :out_len, :]], dim=1)
            for i in range(0, self.dec_len):
                x_i = torch.cat(mean_out[-ps:], dim=1)
                x_i_transformed = self.d_transform(x_i)
                if self.use_feats:
                    f_i = f_padded[..., i:i+f_ps, :]
                    f_i_merged = self.merge_feats(f_i)
                    f_i_transformed = self.f_transform(f_i_merged)
                    dec_input = self.linear_map(torch.cat([f_i_transformed, x_i_transformed], dim=-1))
                else:
                    dec_input = x_i_transformed
                dec_input = self.positional(dec_input.transpose(0,1), start_idx=i)
                #import ipdb ; ipdb.set_trace()

                decoder_output = self.decoder_mean(dec_input, encoder_output)#.clamp(min=0)
                decoder_output = decoder_output.transpose(0,1)
                if self.use_local_weights:
                    local_indices = f_padded[..., f_ps+i:f_ps+i+1, self.tsid_idx].type(torch.LongTensor).to(self.device)
                    local_weights = self.linear_mean(local_indices)
                    decoder_output = decoder_output[..., -self.dec_len:, :]
                    mean_out_ = (decoder_output * local_weights).sum(-1, keepdims=True)
                else:
                    mean_out_ = self.linear_mean(decoder_output)

                if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                    X_pred = self.decoder_std(dec_input, encoder_output)#.clamp(min=0)
                    X_pred = X_pred.transpose(0,1)
                    if self.use_local_weights:
                        local_indices = f_padded[..., f_ps+i:f_ps+i+1, self.tsid_idx].type(torch.LongTensor).to(self.device)
                        local_weights = self.linear_std(local_indices)
                        X_pred = X_pred[..., -self.dec_len:, :]
                        std_out_ = (X_pred * local_weights).sum(-1, keepdims=True)
                        std_out_ = F.softplus(std_out_)
                    else:
                        std_out_ = F.softplus(self.linear_std(X_pred))

                mean_out.append(mean_out_)
                if self.estimate_type in ['variance']:
                    std_out.append(std_out_)

            #import ipdb ; ipdb.set_trace()
            mean_out = torch.cat(mean_out, 1)
            if self.estimate_type in ['variance']:
                std_out = torch.cat(std_out, 1)

        mean_out = mean_out + mean

        mean_out = mean_out[..., -self.dec_len:, :]
        if self.estimate_type in ['variance']:
            std_out = std_out[..., -self.dec_len:, :]

        if self.estimate_type in ['point']:
            return mean_out
        elif self.estimate_type in ['variance']:
            return (mean_out, std_out)

class ATRTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, device, is_signature=False
        ):
        super(ATRTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.is_signature = is_signature
        self.use_covariate_var_model = False

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.warm_start = self.kernel_size * 5

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            in_channels = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            self.conv_feats = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=in_channels, out_channels=nkernel,
                bias=False,
                #padding=self.kernel_size//2
            )

        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            #bias=False,
            #padding=self.kernel_size//2
        )
        self.data_dropout = nn.Dropout(p=0.2)

        if self.use_feats:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
        else:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
        self.positional = PositionalEncoding(d_model=nkernel)

        enc_input_size = nkernel

        if self.t2v_type:
            if self.t2v_type not in ['local']:  
                self.t_size = sum([1 for (_, s) in self.feats_info.values() if s==-1])
            else:
                self.t_size = 1
            if self.t2v_type in ['mdh_lincomb']:
                self.t2v_layer_list = []
                for i in range(self.t_size):
                    self.t2v_layer_list.append(nn.Linear(1, nkernel))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                if self.t_size > 1:
                    self.t2v_linear =  nn.Linear(self.t_size*nkernel, nkernel)
                else:
                    self.t2v_linear = None
            elif self.t2v_type in ['local', 'mdh_parti', 'idx']:
                self.part_sizes = [nkernel//self.t_size]*self.t_size
                for i in range(nkernel%self.t_size):
                    self.part_sizes[i] += 1
                self.t2v_layer_list = []
                for i in range(len(self.part_sizes)):
                    self.t2v_layer_list.append(nn.Linear(1, self.part_sizes[i]))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                self.t2v_dropout = nn.Dropout(p=0.2)
                self.t2v_linear = None
            #import ipdb ; ipdb.set_trace()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['bivariate']:
            self.decoder_bv = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))
        if self.estimate_type in ['bivariate']:
            self.rho_layer = nn.Linear(nkernel, 2)

    def apply_signature(self, mean, X_in, feats_out, X_out):
        X_out = X_out - mean
        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)

            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_out.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_out.shape[0], self.kernel_size-1, X_out.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_out
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_out_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+X_out.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_out.shape[1], X_out.shape[0]).unsqueeze(-1)
                t_in = t_in / X_out.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            #if self.t2v_type is 'mdh_lincomb':
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            #import ipdb ; ipdb.set_trace()
            #t2v = torch.cat([t2v[0:1], torch.sin(t2v[1:])], dim=0)
            #enc_input = self.data_dropout(enc_input) + self.t2v_dropout(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)
        encoder_output = encoder_output.transpose(0, 1)

        return encoder_output

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=False
    ):

        if teacher_force == True:
            assert X_out is not None
        elif teacher_force == False:
            X_out = torch.zeros_like(X_out)

        #X_in = X_in[..., -X_in.shape[1]//5:, :]
        #feats_in = feats_in[..., -feats_in.shape[1]//5:, :]

        mean = X_in.mean(dim=1, keepdim=True)
        #std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)

        #import ipdb ; ipdb.set_trace()
        if self.use_feats:
            feats_in_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)

            feats_in_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_in_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_in_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_in_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_in.shape[0], self.kernel_size-1, X_in.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_in
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_in_embed,X_in_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_in_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_in.shape[1], X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_in[..., :, -self.t_size:].transpose(0,1)

            t2v = []
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)

        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_out_merged = torch.cat(
                [feats_in_merged[:,-self.warm_start+1:, :],feats_out_merged],
                dim=1
            )
            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)

        #import ipdb ; ipdb.set_trace()
        if teacher_force == True:

            #if self.use_feats:
            #    feats_out_embed = feats_out_embed_all

            X_out_embed = self.conv_data(
                torch.cat(
                    [
                        torch.zeros(
                            [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
                            dtype=torch.float, device=self.device
                        ),
                        X_in[..., -self.warm_start+1:, :],
                        X_out
                    ],
                    dim=1
                ).transpose(1, 2)
            ).transpose(1, 2)
            #import ipdb ; ipdb.set_trace()

            if self.use_feats:
                dec_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
            else:
                dec_input = X_out_embed.transpose(0,1)
            #import ipdb ; ipdb.set_trace()
            if self.t2v_type:
                if self.t2v_type in ['local']:
                    t_in = torch.arange(
                        X_in.shape[1], X_in.shape[1]+self.dec_len, dtype=torch.float, device=self.device
                    ).unsqueeze(1).expand(self.dec_len, X_in.shape[0]).unsqueeze(-1)
                    t_in = t_in / X_in.shape[1] * 10.
                else:
                    t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
                t2v = []
                if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                    for i in range(self.t_size):
                        t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                        t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                        t2v.append(t2v_part)
                    t2v = torch.cat(t2v, dim=-1)
                    if self.t2v_linear is not None:
                        t2v = self.t2v_linear(t2v)
                dec_input = dec_input + self.t2v_dropout(t2v)
            else:
                dec_input = self.positional(dec_input, start_idx=X_in.shape[1])
            #import ipdb ; ipdb.set_trace()

            decoder_output = self.decoder_mean(dec_input, encoder_output).clamp(min=0)
            decoder_output = decoder_output.transpose(0,1)
            mean_out = self.linear_mean(decoder_output)

            if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                X_pred = self.decoder_std(dec_input, encoder_output).clamp(min=0)
                X_pred = X_pred.transpose(0,1)
                std_out = F.softplus(self.linear_std(X_pred))
                if self.estimate_type in ['covariance']:
                    v_out = self.linear_v(X_pred)
                if self.estimate_type in ['bivariate']:
                    X_pred = self.decoder_bv(dec_input, encoder_output)
                    X_pred = X_pred.transpose(0,1)
                    rho_out = self.rho_layer(X_pred)
                    rho_out = rho_out[..., -self.dec_len:, :]
                    rho_1, rho_2 = rho_out[..., 1:, :], rho_out[..., :-1, :]
                    #rho_out = torch.einsum("ijk,ijk->ij", (rho_1, rho_2)).unsqueeze(-1)
                    rho_out = (rho_1 * rho_2).sum(dim=-1, keepdims=True)
                    #import ipdb ; ipdb.set_trace()
                    rho_out = torch.tanh(rho_out)
                #import ipdb ; ipdb.set_trace()

            mean_out = mean_out + mean

            if self.is_signature:
                signature_state = self.apply_signature(mean, X_in, feats_out, X_out)
                decoder_output = decoder_output[..., -self.dec_len:, :]

        #import ipdb ; ipdb.set_trace()

            if self.is_signature:
                if self.estimate_type in ['point']:
                    return mean_out[..., -self.dec_len:, :], decoder_output, signature_state
                elif self.estimate_type in ['variance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], decoder_output, signature_state)
                elif self.estimate_type in ['covariance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :], decoder_output, signature_state)
                elif self.estimate_type in ['bivariate']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out, decoder_output, signature_state)
            else:
                if self.estimate_type in ['point']:
                    return mean_out[..., -self.dec_len:, :]
                elif self.estimate_type in ['variance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :])
                elif self.estimate_type in ['covariance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :])
                elif self.estimate_type in ['bivariate']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out)
        else:
            #X_out_embed = self.conv_data(
            #    torch.cat(
            #        [
            #            torch.zeros(
            #                [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
            #                dtype=torch.float, device=self.device
            #            ),
            #            X_in[..., -self.warm_start+1:, :],
            #            torch.zeros(
            #                [X_in.shape[0], self.dec_len, X_in.shape[-1]],
            #                dtype=torch.float, device=self.device
            #            )
            #        ],
            #        dim=1
            #    ).transpose(1, 2)
            #).transpose(1, 2)

            if self.use_feats:
                feats_out_embed_all = feats_out_embed

            X_out_st = torch.cat(
                [
                    torch.zeros(
                        [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    ),
                    X_in[..., -self.warm_start+1:, :],
                    torch.zeros(
                        [X_in.shape[0], 1, X_in.shape[-1]], dtype=torch.float, device=self.device
                    )
                ],
                dim=1
            )

            mean_out, std_out = [], []
            for i in range(self.dec_len):
                if i>0:
                    X_out_i = torch.cat([X_out_st, mean_out[-1]], dim=1)
                else:
                    X_out_i = X_out_st

                X_out_embed = self.conv_data(X_out_i.transpose(1, 2)).transpose(1, 2)
                if self.use_feats:
                    feats_out_embed = feats_out_embed_all[..., :X_out_embed.shape[1], :]

                #import ipdb ; ipdb.set_trace()

                if self.use_feats:
                    dec_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
                else:
                    dec_input = X_out_embed.transpose(0,1)

                #import ipdb ; ipdb.set_trace()
                if self.t2v_type:
                    if self.t2v_type in ['local']:
                        t_in = torch.arange(
                            X_in.shape[1], X_in.shape[1]+self.dec_len, dtype=torch.float, device=self.device
                        ).unsqueeze(1).expand(self.dec_len, X_in.shape[0]).unsqueeze(-1)
                        t_in = t_in / X_in.shape[1] * 10.
                    else:
                        t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
                    t2v = []
                    if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                        for i in range(self.t_size):
                            t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                            t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                            t2v.append(t2v_part)
                        t2v = torch.cat(t2v, dim=-1)
                        if self.t2v_linear is not None:
                            t2v = self.t2v_linear(t2v)
                    dec_input = dec_input + self.t2v_dropout(t2v)
                else:
                    dec_input = self.positional(dec_input, start_idx=X_in.shape[1])
                #import ipdb ; ipdb.set_trace()

                decoder_output = self.decoder_mean(dec_input, encoder_output).clamp(min=0)
                decoder_output = decoder_output.transpose(0,1)
                mean_out_i = self.linear_mean(decoder_output)
                #import ipdb ; ipdb.set_trace()

                if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                    X_pred = self.decoder_std(dec_input, encoder_output).clamp(min=0)
                    X_pred = X_pred.transpose(0,1)
                    std_out_i = F.softplus(self.linear_std(X_pred))
                    std_out.append(std_out_i[..., -1:, :])
                    if self.estimate_type in ['covariance']:
                        v_out = self.linear_v(X_pred)
                    if self.estimate_type in ['bivariate']:
                        X_pred = self.decoder_bv(dec_input, encoder_output)
                        X_pred = X_pred.transpose(0,1)
                        rho_out = self.rho_layer(X_pred)
                        rho_out = rho_out[..., -self.dec_len:, :]
                        rho_1, rho_2 = rho_out[..., 1:, :], rho_out[..., :-1, :]
                        #rho_out = torch.einsum("ijk,ijk->ij", (rho_1, rho_2)).unsqueeze(-1)
                        rho_out = (rho_1 * rho_2).sum(dim=-1, keepdims=True)
                        #import ipdb ; ipdb.set_trace()
                        rho_out = torch.tanh(rho_out)
                    #import ipdb ; ipdb.set_trace()

                mean_out_i = mean_out_i + mean
                mean_out.append(mean_out_i[..., -1:, :])

                if self.is_signature:
                    signature_state = self.apply_signature(mean, X_in, feats_out, X_out)
                    decoder_output = decoder_output[..., -self.dec_len:, :]

            #import ipdb ; ipdb.set_trace()

            mean_out = torch.cat(mean_out, dim=1)
            if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                std_out = torch.cat(std_out, dim=1)

            if self.is_signature:
                if self.estimate_type in ['point']:
                    return mean_out[..., -self.dec_len:, :], decoder_output, signature_state
                elif self.estimate_type in ['variance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], decoder_output, signature_state)
                elif self.estimate_type in ['covariance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :], decoder_output, signature_state)
                elif self.estimate_type in ['bivariate']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out, decoder_output, signature_state)
            else:
                if self.estimate_type in ['point']:
                    return mean_out[..., -self.dec_len:, :]
                elif self.estimate_type in ['variance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :])
                elif self.estimate_type in ['covariance']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :])
                elif self.estimate_type in ['bivariate']:
                    return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out)

            
class OracleModel(nn.Module):
    def __init__(
            self, dec_len, estimate_type
            ):
        super(OracleModel, self).__init__()

        self.dec_len = dec_len
        self.estimate_type = estimate_type
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=None):
        assert X_out is not None
        dists = []
        for i in range(X_in.shape[1]-self.dec_len):
            dist = torch.pow(X_in[:, i:i+self.dec_len] - X_out, 2).mean(dim=1)
            #dist = torch.abs(X_in[:, i:i+self.dec_len] - X_out).mean(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=1)
        #import ipdb; ipdb.set_trace()
        min_indices_ = torch.argmin(dists, dim=1).unsqueeze(-1).unsqueeze(-1)
        min_indices = []
        for i in range(self.dec_len):
            min_indices.append(min_indices_+i)
        min_indices = torch.cat(min_indices, dim=1)
        mean_out = X_in.gather(1, min_indices)

        #import ipdb; ipdb.set_trace()

        return mean_out

class OracleForecastModel(nn.Module):
    def __init__(
            self, dec_len, estimate_type
            ):
        super(OracleForecastModel, self).__init__()

        self.dec_len = dec_len
        self.estimate_type = estimate_type
        self.dummy_layer = nn.Linear(1, 1)
        self.warm_start = self.dec_len * 2

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=None):
        dists = []
        key = X_in[:, -self.warm_start:]
        for i in range(X_in.shape[1]-2*self.warm_start):
            dist = torch.pow(X_in[:, i:i+self.warm_start] - key, 2).mean(dim=1)
            #dist = torch.abs(X_in[:, i:i+self.warm_start] - key).mean(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=1)
        #import ipdb; ipdb.set_trace()
        min_indices_ = torch.argmin(dists, dim=1).unsqueeze(-1).unsqueeze(-1)
        min_indices = []
        for i in range(self.dec_len):
            min_indices.append(min_indices_+self.warm_start+i)
        min_indices = torch.cat(min_indices, dim=1)
        mean_out = X_in.gather(1, min_indices)

        #import ipdb; ipdb.set_trace()

        return mean_out


class ARCNNTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, coeffs_info, estimate_type, use_coeffs, v_dim,
            kernel_size, nkernel, dim_ff, nhead, device
        ):
        super(ARCNNTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.coeffs_info = coeffs_info
        self.estimate_type = estimate_type
        self.use_coeffs = use_coeffs
        self.v_dim = v_dim
        self.device = device
        self.use_covariate_var_model = False

        self.kernel_size = kernel_size
        self.nkernel = nkernel
        self.dim_ff = dim_ff
        self.nhead = nhead

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
            kernel_size=self.kernel_size, stride=self.kernel_size, in_channels=in_channels, out_channels=nkernel,
            #padding=self.kernel_size//2
        )
        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=self.kernel_size, in_channels=1, out_channels=nkernel,
            #padding=self.kernel_size//2
        )

        self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel))
        self.positional = PositionalEncoding(d_model=nkernel)

        if self.use_coeffs:
            enc_input_size = nkernel# + 2
        else:
            enc_input_size = nkernel
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=self.nhead, dropout=0, dim_feedforward=self.dim_ff
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=self.nhead, dropout=0, dim_feedforward=self.dim_ff
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.deconv_mean = nn.ConvTranspose1d(
            self.nkernel, self.nkernel,
            self.dec_len-(self.warm_start//self.kernel_size+self.dec_len//self.kernel_size)+1
        )
        self.deconv_std = nn.ConvTranspose1d(
            self.nkernel, self.nkernel,
            self.dec_len-(self.warm_start//self.kernel_size+self.dec_len//self.kernel_size)+1
        )

        self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))

    def forward(self, feats_in, X_in, coeffs_in, feats_out, X_out=None, coeffs_out=None, teacher_force=None):

        #X_in = X_in[..., -X_in.shape[1]//5:, :]
        #feats_in = feats_in[..., -feats_in.shape[1]//5:, :]
        #coeffs_in = coeffs_in[..., -coeffs_in.shape[1]//5:, :]

        #if self.use_coeffs:
        #    X_in = coeffs_in[..., 0:1]

        mean = X_in.mean(dim=1, keepdim=True)
        #std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)

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
        ).transpose(1,2).clamp(min=0)[..., :X_in.shape[1],:]      
        X_in_embed = self.conv_data(
            X_in.transpose(1,2)
        ).transpose(1,2).clamp(min=0)[..., :X_in.shape[1], :]
        feats_in_embed = self.linearMap(torch.cat([feats_in_embed, X_in_embed],dim=-1)).transpose(0,1)
        enc_input = self.positional(feats_in_embed)
        #if self.use_coeffs:
        #    enc_input = torch.cat([enc_input, coeffs_in[..., 1:].transpose(0, 1)], dim=-1)
        encoder_output = self.encoder(enc_input)

        feats_out_merged = torch.cat([feats_in_merged[:,-self.warm_start:, :],feats_out_merged],dim=1)
        feats_out_embed = self.conv_feats(
            feats_out_merged.transpose(1,2)
        ).transpose(1,2).clamp(min=0)
        X_out_embed = self.conv_data(
            torch.cat(
                [
                    X_in[..., -self.warm_start:, :],
                    torch.zeros([X_in.shape[0], self.dec_len, X_in.shape[-1]], dtype=torch.float, device=self.device)
                ],
                dim=1
            ).transpose(1, 2)
        ).transpose(1, 2).clamp(min=0)
        feats_out_embed = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1))

        X_out = self.decoder_mean(
            self.positional(feats_out_embed.transpose(0,1)), encoder_output
        ).clamp(min=0)
        X_out = X_out.transpose(0,1)
        #import ipdb ; ipdb.set_trace()
        X_out = self.deconv_mean(X_out.transpose(1, 2)).transpose(1, 2)
        mean_out = self.linear_mean(X_out)
        #mean_out = mean_out*std+mean

        X_out = self.decoder_std(
            self.positional(feats_out_embed.transpose(0,1)), encoder_output
        ).clamp(min=0)
        X_out = X_out.transpose(0,1)
        X_out = self.deconv_std(X_out.transpose(1, 2)).transpose(1, 2)
        std_out = F.softplus(self.linear_std(X_out))

        std_out = self.linear_std(X_out)
        #std_out = F.softplus((std_out*std)/2)
        std_out = F.softplus(std_out)
        v_out = self.linear_v(X_out)

        #import ipdb ; ipdb.set_trace()

        mean_out = mean_out + mean

        return (
            mean_out[..., -self.dec_len:, :],
            std_out[..., -self.dec_len:, :],
            v_out[..., -self.dec_len:, :]
        )


class RNNNARModel(nn.Module):
    def __init__(
            self, dec_len, num_rnn_layers, feats_info, hidden_size, batch_size,
            estimate_type, use_feats, v_dim, device
        ):
        super(RNNNARModel, self).__init__()

        self.dec_len = dec_len
        self.num_rnn_layers = num_rnn_layers
        self.feats_info = feats_info
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.estimate_type = estimate_type
        self.device = device
        self.use_feats = use_feats
        self.v_dim = v_dim

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            feats_embed_dim = sum([s for (_, s) in self.feats_info.values() if s is not -1])
        else:
            feats_embed_dim = 0
        enc_input_size = 1 + feats_embed_dim
        self.encoder = nn.LSTM(enc_input_size, self.hidden_size, batch_first=True)

        if self.use_feats:
            dec_input_size = self.hidden_size + feats_embed_dim
        else:
            dec_input_size = self.hidden_size

        self.decoder_mean = nn.Sequential(
            nn.Linear(dec_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        if self.estimate_type in ['variance', 'covariance']:
            self.decoder_std = nn.Sequential(
                nn.Linear(dec_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Softplus()
            )

        if self.estimate_type in ['covariance']:
            self.decoder_v = nn.Sequential(
                nn.Linear(dec_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.v_dim),
                #nn.Softplus()
            )

    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )

    def forward(self, feats_in, X_in, feats_out, X_out=None):

        if self.use_feats:
            feats_in_merged, feats_out_merged = [], []
            for i in range(feats_in.shape[-1]):
                card = self.feats_info[i][0]
                if card != -1:
                    if card != 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)
            for i in range(feats_out.shape[-1]):
                card = self.feats_info[i][0]
                if card != -1:
                    if card != 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_in_embed = feats_in_merged
            feats_out_embed = feats_out_merged
            enc_input = torch.cat([feats_in_embed, X_in], dim=-1)
        else:
            enc_input = X_in

        enc_hidden = self.init_hidden(X_in.shape[0])
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        enc_output_tile = enc_output[..., -1:, :].repeat(1, self.dec_len, 1)
        if self.use_feats:
            dec_input = torch.cat([feats_out_embed, enc_output_tile], dim=-1)
        else:
            dec_input = enc_output_tile
        means = self.decoder_mean(dec_input)
        if self.estimate_type in ['variance', 'covariance']:
            stds = self.decoder_std(dec_input)
        if self.estimate_type in ['covariance']:
            v = self.decoder_v(dec_input)

        if self.estimate_type in ['point']:
            return means
        elif self.estimate_type in ['variance']:
            return (means, stds)
        elif self.estimate_type in ['covariance']:
            return (means, stds, v)
        

class RNNARModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, num_rnn_layers, hidden_size, batch_size, device, is_signature=False
        ):
        super(RNNARModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.is_signature = is_signature

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            feats_embed_dim = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            enc_input_size = 1 + feats_embed_dim
        else:
            enc_input_size = 1

        self.encoder = nn.LSTM(enc_input_size, self.hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(enc_input_size, self.hidden_size,  batch_first=True)
        self.decoder_mean = nn.Linear(hidden_size, 1)
        self.decoder_std = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())
        self.decoder_v = nn.Linear(hidden_size, self.v_dim)

    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=True):

        if self.use_feats:
            feats_in_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)
        feats_in_embed = feats_in_merged

        enc_input = torch.cat([feats_in_embed, X_in], dim=-1)
        enc_hidden = self.init_hidden(X_in.shape[0])
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
        feats_out_embed = feats_out_merged

        dec_state = enc_state
        if X_out is not None:
            X_prev = torch.cat([X_in[..., -1:, :], X_out[..., :-1, :]], dim=1)
            feats_prev = torch.cat([feats_in_embed[..., -1:, :], feats_out_embed[..., :-1, :]], dim=1)
            dec_input = torch.cat([feats_prev, X_prev], dim=-1)
            dec_output, dec_state = self.decoder_lstm(dec_input, dec_state)
            means = self.decoder_mean(dec_output)
            if self.estimate_type in ['covariance', 'variance']:
                stds = self.decoder_std(dec_output)
                if self.estimate_type in ['covariance']:
                    v = self.decoder_v(dec_output)
        else:
            X_prev = X_in[..., -1:, :]
            feats_prev = feats_in_embed[..., -1:, :]
            means, stds, v = [], [], []
            for i in range(self.dec_len):
                dec_input = torch.cat([feats_prev, X_prev], dim=-1)
                dec_output, dec_state = self.decoder_lstm(dec_input, dec_state)
                step_pred_mu = self.decoder_mean(dec_output)
                means.append(step_pred_mu)
                if self.estimate_type in ['covariance', 'variance']:
                    step_pred_std = self.decoder_std(dec_output)
                    stds.append(step_pred_std)
                    if self.estimate_type in ['covariance']:
                        step_pred_v = self.decoder_v(dec_output)
                        v.append(step_pred_v)
                X_prev = step_pred_mu
                feats_prev = feats_out_embed[..., i:i+1, :]

            means = torch.cat(means, dim=1)
            if self.estimate_type in ['covariance', 'variance']:
                stds = torch.cat(stds, dim=1)
                if self.estimate_type in ['covariance']:
                    v = torch.cat(v, dim=1)

        if self.estimate_type in ['point']:
            return means
        elif self.estimate_type in ['variance']:
            return means, stds
        elif self.estimate_type in ['covariance']:
            return means, stds, v

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
    input_size, output_size, estimate_type, feats_info
):

    #hidden_size = max(int(config['hidden_size']*1.0/int(np.sqrt(level))), args.fc_units)
    hidden_size = args.hidden_size

    if base_model_name in ['rnn-mse-nar', 'rnn-nll-nar', 'rnn-fnll-nar']:
        net_gru = RNNNARModel(
            dec_len=N_output,
            num_rnn_layers=args.num_grulstm_layers,
            feats_info=feats_info,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            estimate_type=estimate_type,
            use_feats=args.use_feats,
            v_dim=args.v_dim,
            device=args.device
        ).to(args.device)
    elif base_model_name in ['rnn-mse-ar', 'rnn-nll-ar', 'rnn-fnll-ar']:
        net_gru = RNNARModel(
            dec_len=N_output,
            feats_info=feats_info,
            estimate_type=estimate_type,
            use_feats=args.use_feats,
            t2v_type=args.t2v_type,
            v_dim=args.v_dim,
            num_rnn_layers=args.num_grulstm_layers,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            device=args.device
        ).to(args.device)
    elif base_model_name in ['trans-mse-ar', 'trans-nll-ar', 'trans-fnll-ar', 'trans-bvnll-ar', 'sharq-nll-nar']:
            net_gru = ARTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=10, nkernel=32, device=args.device
            ).to(args.device)
    elif base_model_name in ['gpt-nll-ar', 'gpt-mse-ar']:
            net_gru = GPTTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=args.kernel_size, nkernel=args.nkernel, is_nar=False, device=args.device
            ).to(args.device)
    elif base_model_name in ['gpt-nll-nar', 'gpt-mse-nar']:
            net_gru = GPTTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=args.kernel_size, nkernel=args.nkernel, is_nar=True, device=args.device
            ).to(args.device)
    elif base_model_name in ['informer-mse-nar']:
            net_gru = informer.Informer(
                enc_in=1,
                dec_in=1,
                c_out=1,
                seq_len=N_input,
                label_len=N_output,
                out_len=N_output,
                factor=5,
                d_model=512,
                n_heads=8,
                e_layers=2,
                d_layers=1,
                d_ff=2048,
                dropout=0.05,
                attn='prob',
                embed='fixed',
                freq=args.freq,
                activation='gelu',
                output_attention=False,
                distil=True,
                mix=True,
                feats_info=feats_info,
                device=args.device
            ).to(args.device)
    elif base_model_name in ['trans-nll-atr']:
            net_gru = ATRTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=10, nkernel=32, device=args.device
            ).to(args.device)
    elif base_model_name in ['transsig-nll-nar']:
        net_gru = ARTransformerModel(
            N_output, feats_info, estimate_type, args.use_feats,
            args.t2v_type, args.v_dim,
            kernel_size=10, nkernel=32, device=args.device,
            is_signature=True
        ).to(args.device)
    elif base_model_name in ['transm-nll-nar', 'transm-fnll-nar']:
        net_gru = transformer_manual_attn.TransformerManualAttn(
            N_output, feats_info, coeffs_info, estimate_type, args.use_feats,
            args.use_coeffs, args.v_dim, kernel_size=10, nkernel=32, device=args.device
        ).to(args.device)
    elif base_model_name in ['transda-nll-nar', 'transda-fnll-nar']:
        net_gru = transformer_dual_attn.TransformerDualAttn(
            N_output, feats_info, coeffs_info, estimate_type, args.use_feats,
            args.use_coeffs, args.v_dim, kernel_size=10, nkernel=32, device=args.device
        ).to(args.device)

    elif base_model_name in ['nbeatsd-mse-nar']:
        net_gru = NBEATS_D(
            N_input, N_output, num_blocks=8, block_width=128, block_numlayers=4,
            point_estimates=point_estimates, feats_info=feats_info, coeffs_info=coeffs_info,
            use_coeffs=args.use_coeffs
        ).to(args.device)
    elif base_model_name in ['nbeats-mse-nar']:
        net_gru = NBEATS(
            N_input, N_output, num_blocks=8, block_width=128, block_numlayers=4,
            point_estimates=point_estimates, feats_info=feats_info, coeffs_info=coeffs_info,
            use_coeffs=args.use_coeffs
        ).to(args.device)
    elif base_model_name in ['oracle']:
        net_gru = OracleModel(N_output, estimate_type).to(args.device)
    elif base_model_name in ['oracleforecast']:
        net_gru = OracleForecastModel(N_output, estimate_type).to(args.device)

    return net_gru

def get_base_model_bak(
    args, base_model_name, level, N_input, N_output,
    input_size, output_size, estimate_type, feats_info
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
            if 'nar' in base_model_name:
                net_gru = RNNNARModel(
                    dec_len=N_output,
                    num_rnn_layers=args.num_grulstm_layers,
                    feats_info=feats_info,
                    hidden_size=hidden_size,
                    batch_size=args.batch_size,
                    estimate_type=estimate_type,
                    use_feats=args.use_feats,
                    v_dim=args.v_dim,
                    device=args.device
                ).to(args.device)
            elif 'ar' in base_model_name:
                net_gru = RNNARModel(
                    dec_len=N_output,
                    num_rnn_layers=args.num_grulstm_layers,
                    feats_info=feats_info, coeffs_info=coeffs_info,
                    hidden_size=hidden_size,
                    batch_size=args.batch_size,
                    estimate_type=estimate_type,
                    use_coeffs=args.use_coeffs,
                    use_time_features=args.use_time_features,
                    v_dim=args.v_dim,
                    device=args.device
                ).to(args.device)
        elif 'trans' in base_model_name:
            if 'transm' in base_model_name:
                net_gru = transformer_manual_attn.TransformerManualAttn(
                    N_output, feats_info, coeffs_info, estimate_type, args.use_feats,
                    args.use_coeffs, args.v_dim, kernel_size=10, nkernel=32, device=args.device
                ).to(args.device)
            elif 'transda' in base_model_name:
                net_gru = transformer_dual_attn.TransformerDualAttn(
                    N_output, feats_info, coeffs_info, estimate_type, args.use_feats,
                    args.use_coeffs, args.v_dim, kernel_size=10, nkernel=32, device=args.device
                ).to(args.device)
            else:
                net_gru = ARTransformerModel(
                    N_output, feats_info, estimate_type, args.use_feats,
                    args.t2v_type, args.v_dim,
                    kernel_size=10, nkernel=32, device=args.device
                ).to(args.device)
        elif 'nbeatsd' in base_model_name:
            net_gru = NBEATS_D(
                N_input, N_output, num_blocks=8, block_width=128, block_numlayers=4,
                point_estimates=point_estimates, feats_info=feats_info, coeffs_info=coeffs_info,
                use_coeffs=args.use_coeffs
            ).to(args.device)
        elif 'nbeats' in base_model_name:
            net_gru = NBEATS(
                N_input, N_output, num_blocks=8, block_width=128, block_numlayers=4,
                point_estimates=point_estimates, feats_info=feats_info, coeffs_info=coeffs_info,
                use_coeffs=args.use_coeffs
            ).to(args.device)
        elif base_model_name in ['oracle']:
            net_gru = OracleModel(N_output, estimate_type).to(args.device)
        elif base_model_name in ['oracleforecast']:
            net_gru = OracleForecastModel(N_output, estimate_type).to(args.device)

    return net_gru
