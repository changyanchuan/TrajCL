import sys
sys.path.append('..')
import math
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Transformer example: https://github.com/pytorch/examples/blob/master/word_language_model/model.py

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])



class DualSTB(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput):
        super(DualSTB, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(ninput, pos_droput)
        
        structural_attn_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer)
        
        self.spatial_attn = SpatialMSM(4, 32, 1, 3, attn_dropout, pos_droput)
        self.gamma_param = nn.Parameter(data = torch.tensor(0.5), requires_grad = True)

    def forward(self, src, attn_mask, src_padding_mask, src_len, srcspatial):
        # src: [seq_len, batch_size, emb_size]
        # attn_mask: [seq_len, seq_len]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]
        # srcspatial : [seq_len, batch_size, 4]

        if srcspatial is not None:
            _, attn_spatial = self.spatial_attn(srcspatial, attn_mask, src_padding_mask, src_len)
            attn_spatial = attn_spatial.repeat(self.nhead, 1, 1)
            gamma = torch.sigmoid(self.gamma_param) * 10
            attn_spatial = gamma * attn_spatial
        else:
            attn_spatial = None

        src = self.pos_encoder(src)
        rtn = self.structural_attn(src, attn_spatial, src_padding_mask)

        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        return rtn # return traj embeddings


class SpatialMSM(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput):
        super(SpatialMSM, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.pos_encoder = PositionalEncoding(ninput, pos_droput)
        trans_encoder_layers = SpatialMSMLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = SpatialMSMEncoder(trans_encoder_layers, nlayer)
        
    
    def forward(self, src, attn_mask, src_padding_mask, src_len):
        # src: [seq_len, batch_size, emb_size]
        # attn_mask: [seq_len, seq_len]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]

        src = self.pos_encoder(src)
        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)

        # average pooling
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        # rtn = [batch_size, traj_emb]
        # attn = [batch_size, seq_len, seq_len]
        return rtn, attn


# ref: torch.nn.modules.transformer
class SpatialMSMEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SpatialMSMEncoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


# ref: torch.nn.modules.transformer
class SpatialMSMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(SpatialMSMLayer, self).__init__()
        self.self_attn = nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SpatialMSMLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # attn = [batch, seq, seq] # masked
        return src, attn



