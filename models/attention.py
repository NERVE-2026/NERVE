#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from math import sqrt
from einops import rearrange, repeat


#%%
class ScaledDotProductAttention(nn.Module):
    # self-attention for temporal axis
    def __init__(self, d_model, num_heads, attn_dropout=0.1, output_attention=False):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = nn.Parameter(torch.tensor((d_model//num_heads)** -0.5), requires_grad=False) 
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, queries, keys, values, w_ep=None, attn_mask=None):
        # q: (bs, num_heads, l, d_q)
        # k: (bs, num_heads, l, d_k)
        # v: (bs, num_heads, l, d_v)
        # w_ep: (bs, num_heads, d_q, d_q)
        if w_ep is None:
            attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale # (bs, num_heads, l, l)
        else:
            attn_scores = torch.einsum('bhld,bhde,bhke->bhlk', queries, w_ep, keys) * self.scale # (bs, num_heads, l, l)
        # Attention mask (optional)
        if attn_mask is not None:    # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1)) # (bs, num_heads, l, l)
        output = torch.matmul(attn_weights, values) # (bs, num_heads, l, d_v)

        if self.output_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model, num_heads, res_attention=False, attn_dropout=0.1, proj_dropout=0, 
                 qkv_bias=False, qk_norm=None, qk_scale=None):
        super(MultiHeadAttention, self).__init__()

        d_head = d_model // num_heads
        self.n_heads = num_heads
        self.scale = qk_scale or d_head ** -0.5

        self.attention = ScaledDotProductAttention(d_model, num_heads, attn_dropout, res_attention)
        self.W_Q = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.W_out = nn.Sequential(nn.Linear(d_head * num_heads, d_model), nn.Dropout(proj_dropout))

        if qk_norm is not None:
            self.q_norm = qk_norm(d_head)
            self.k_norm = qk_norm(d_head)
        else:
            self.q_norm = None
            self.k_norm = None        

        self.res_attention = res_attention

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.W_Q(queries).view(B, L, H, -1).transpose(1, 2)
        keys = self.W_K(keys).view(B, S, H, -1).transpose(1, 2)
        values = self.W_V(values).view(B, S, H, -1).transpose(1, 2)
        if self.q_norm is not None:
            queries = self.q_norm(queries).type_as(values)
        if self.k_norm is not None:
            keys = self.k_norm(keys).type_as(values)

        if self.res_attention:
            out, attn_weights, attn_scores = self.attention(queries, keys, values, attn_mask=attn_mask)
        else:
            out, attn_weights = self.attention(queries, keys, values, attn_mask=attn_mask)

        out = out.transpose(-2, -1).contiguous().view(B, L, -1)
        out = self.W_out(out)

        # if self.res_attention:
        #     return out, attn_weights, attn_scores
        # else:
        #     return out, attn_weights
        return out

class ElectrodePositionAttention(nn.Module):
    def __init__(self, 
                 d_model, num_heads, res_attention=False, attn_dropout=0.1, proj_dropout=0, 
                 qkv_bias=False, qk_norm=None, qk_scale=None):
        super(ElectrodePositionAttention, self).__init__()

        d_head = d_model // num_heads
        self.n_heads = num_heads
        self.scale = qk_scale or d_head ** -0.5

        self.attention = ScaledDotProductAttention(d_model, num_heads, attn_dropout, res_attention)
        self.W_Q = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        
        self.Wl_Q = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        self.Wl_K = nn.Linear(d_model, d_head * num_heads, bias=qkv_bias)
        
        self.W_out = nn.Sequential(nn.Linear(d_head * num_heads, d_model), nn.Dropout(proj_dropout))

        if qk_norm is not None:
            self.q_norm = qk_norm(d_head)
            self.k_norm = qk_norm(d_head)
        else:
            self.q_norm = None
            self.k_norm = None        

        self.res_attention = res_attention

    def forward(self, queries, keys, values, routers, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        _, G, _ = routers.shape
        H = self.n_heads

        queries = self.W_Q(queries).view(B, L, H, -1).transpose(1, 2)
        keys = self.W_K(keys).view(B, S, H, -1).transpose(1, 2)
        values = self.W_V(values).view(B, S, H, -1).transpose(1, 2)
        router_queries = self.Wl_Q(routers).view(B, G, H, -1).transpose(1, 2)
        router_keys = self.Wl_K(routers).view(B, G, H, -1).transpose(1, 2)
        W_ep = torch.einsum('bhgd,bhge->bhde', router_queries, router_keys) # (B, H, d_model, d_model)

        
        if self.q_norm is not None:
            queries = self.q_norm(queries).type_as(values)
        if self.k_norm is not None:
            keys = self.k_norm(keys).type_as(values)

        if self.res_attention:
            out, attn_weights, attn_scores = self.attention(queries, keys, values, W_ep, attn_mask)
        else:
            out, attn_weights = self.attention(queries, keys, values, W_ep, attn_mask)

        out = out.transpose(-2, -1).contiguous().view(B, L, -1)
        out = self.W_out(out)

        # if self.res_attention:
        #     return out, attn_weights, attn_scores
        # else:
        #     return out, attn_weights
        return out
#%%

class BasicTransformerLayer(nn.Module):
    def __init__(self, num_patch, num_group, d_model, num_head, d_ff=None, attn_dropout=0.0, proj_dropout=0.0, res_attention=False, 
                 qkv_bias=False, qk_norm=nn.LayerNorm, qk_scale=None, norm_layer=nn.LayerNorm, activation=nn.GELU):
        super(BasicTransformerLayer, self).__init__()
        self.num_patch = num_patch
        self.num_group = num_group
        self.n_heads = num_head
        self.d_model = d_model
        d_ff = d_model * 4 if d_ff is None else d_ff

        self.temp_attn = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.norms = nn.ModuleList([norm_layer(d_model) for _ in range(2)])

        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        # pre-normalization
        b, d, n, d_model = x.shape

        x = rearrange(x, 'b d n d_model -> (b d) n d_model')
        x_in = self.norms[0](x)
        x_enc = x + self.temp_attn(x_in, x_in, x_in, attn_mask=mask) # temporal attention + add
        x_out = x_enc + self.ffn1(self.norms[1](x_enc)) # FFN + add
        out = rearrange(x_out, '(b d) n d_model -> b d n d_model', b=b, d=d)

        return out

class TemporalChannelTransformerLayer(nn.Module):
    def __init__(self, num_patch, num_group, d_model, num_head, d_ff=None, attn_dropout=0.0, proj_dropout=0.0, res_attention=False, 
                 qkv_bias=False, qk_norm=nn.LayerNorm, qk_scale=None, norm_layer=nn.LayerNorm, activation=nn.GELU):
        super(TemporalChannelTransformerLayer, self).__init__()
        self.num_patch = num_patch
        self.num_group = num_group
        self.n_heads = num_head
        self.d_model = d_model
        d_ff = d_model * 4 if d_ff is None else d_ff

        self.temp_attn = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.ch_attn = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.norms = nn.ModuleList([norm_layer(d_model) for _ in range(4)])

        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))
        self.ffn2 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        # pre-normalization
        b, d, n, d_model = x.shape

        x = rearrange(x, 'b d n d_model -> (b d) n d_model')
        x_in = self.norms[0](x)
        x_enc = x + self.temp_attn(x_in, x_in, x_in, attn_mask=mask) # temporal attention + add
        x_out = x_enc + self.ffn1(self.norms[1](x_enc)) # FFN + add

        x = rearrange(x_out, '(b d) n d_model -> (b n) d d_model', b=b)
        x_in = self.norms[2](x)
        x_group = self.ch_attn(x_in, x_in, x_in, attn_mask=mask)
        x_out = x + x_group
        x_out = x_out + self.ffn2(self.norms[3](x_out)) # FFN, add & norm

        out = rearrange(x_out, '(b n) d d_model -> b d n d_model', b=b, d=d)

        return out


class EEGTransformerLayer(nn.Module):
    def __init__(self, num_patch, num_group, d_model, num_head, d_ff=None, attn_dropout=0.0, proj_dropout=0.0, res_attention=False, 
                 qkv_bias=False, qk_norm=nn.LayerNorm, qk_scale=None, norm_layer=nn.LayerNorm, activation=nn.GELU):
        super(EEGTransformerLayer, self).__init__()
        self.num_patch = num_patch
        self.num_group = num_group
        self.n_heads = num_head
        self.d_model = d_model
        d_ff = d_model * 4 if d_ff is None else d_ff

        self.temp_attn = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.ch_attn_group = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.ch_attn_bwgroup = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.ch_attn_degroup = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.loc_router = nn.Parameter(torch.randn(num_patch, num_group, d_model), requires_grad=True)

        self.norms = nn.ModuleList([norm_layer(d_model) for _ in range(5)])

        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))
        self.ffn2 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        # pre-normalization
        b, d, n, d_model = x.shape

        x = rearrange(x, 'b d n d_model -> (b d) n d_model')
        x_in = self.norms[0](x)
        x_enc = x + self.temp_attn(x_in, x_in, x_in, attn_mask=mask) # temporal attention + add
        x_out = x_enc + self.ffn1(self.norms[1](x_enc)) # FFN + add

        x = rearrange(x_out, '(b d) n d_model -> (b n) d d_model', b=b)
        x_in = self.norms[2](x)
        loc_router = self.norms[3](repeat(self.loc_router[:n], 'n g d_model -> (repeat n) g d_model', repeat=b))
        x_group = self.ch_attn_group(loc_router, x_in, x_in)
        # x_group = self.ch_attn_bwgroup(x_group, x_group, x_group) # group attention
        x_group = self.ch_attn_degroup(x_in, x_group, x_group)
        x_out = x + x_group
        x_out = x_out + self.ffn2(self.norms[4](x_out)) # FFN, add & norm

        out = rearrange(x_out, '(b n) d d_model -> b d n d_model', b=b, d=d)

        return out
    
class EEGTransformerLayer2(nn.Module):
    def __init__(self, num_patch, num_group, d_model, num_head, d_ff=None, attn_dropout=0.0, proj_dropout=0.0, res_attention=False, 
                 qkv_bias=False, qk_norm=nn.LayerNorm, qk_scale=None, norm_layer=nn.LayerNorm, activation=nn.GELU):
        super(EEGTransformerLayer2, self).__init__()
        self.num_patch = num_patch
        self.num_group = num_group
        self.n_heads = num_head
        self.d_model = d_model
        d_ff = d_model * 4 if d_ff is None else d_ff

        self.temp_attn = MultiHeadAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.ch_attn = ElectrodePositionAttention(d_model, num_head, res_attention=res_attention, attn_dropout=attn_dropout, proj_dropout=proj_dropout, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale)
        self.norms = nn.ModuleList([norm_layer(d_model) for _ in range(4)])
        self.loc_router = nn.Parameter(torch.randn(num_patch, num_group, d_model), requires_grad=True)
        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))
        self.ffn2 = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        # pre-normalization
        b, d, n, d_model = x.shape

        x = rearrange(x, 'b d n d_model -> (b d) n d_model')
        x_in = self.norms[0](x)
        x_enc = x + self.temp_attn(x_in, x_in, x_in, attn_mask=mask) # temporal attention + add
        x_out = x_enc + self.ffn1(self.norms[1](x_enc)) # FFN + add

        x = rearrange(x_out, '(b d) n d_model -> (b n) d d_model', b=b)
        x_in = self.norms[2](x)
        loc_router = self.norms[3](repeat(self.loc_router[:n], 'n g d_model -> (repeat n) g d_model', repeat=b))
        x_group = self.ch_attn(x_in, x_in, x_in, loc_router)
        x_out = x + x_group
        x_out = x_out + self.ffn2(self.norms[3](x_out)) # FFN, add & norm

        out = rearrange(x_out, '(b n) d d_model -> b d n d_model', b=b, d=d)

        return out

class EEGTransformer(nn.Module):
    def __init__(self, num_patch, num_group, num_layer, d_model, num_head, d_ff=None, dropout=0.0, 
                 attn_dropout=0.0, res_attention=False, qkv_bias=True, qk_norm=None, qk_scale=None, 
                 norm_layer=nn.LayerNorm, activation=nn.GELU, encoder_type='epa'):
        super(EEGTransformer, self).__init__()
        
        if encoder_type == 't':
            layer = BasicTransformerLayer(num_patch, num_group, d_model, num_head, d_ff, attn_dropout, dropout, res_attention, 
                                    qkv_bias, qk_norm, qk_scale, norm_layer, activation)
        elif encoder_type == 'tc':
            layer = TemporalChannelTransformerLayer(num_patch, num_group, d_model, num_head, d_ff, attn_dropout, dropout, res_attention, 
                                    qkv_bias, qk_norm, qk_scale, norm_layer, activation)
        elif encoder_type == 'epa':
            layer = EEGTransformerLayer(num_patch, num_group, d_model, num_head, d_ff, attn_dropout, dropout, res_attention, 
                                    qkv_bias, qk_norm, qk_scale, norm_layer, activation)
        elif encoder_type == 'epa2':
            layer = EEGTransformerLayer2(num_patch, num_group, d_model, num_head, d_ff, attn_dropout, dropout, res_attention, 
                                    qkv_bias, qk_norm, qk_scale, norm_layer, activation)
        self.attn_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layer)])
            
    def forward(self, x, mask=None):
        for layer in self.attn_layers:
            x = layer(x, mask)
        return x