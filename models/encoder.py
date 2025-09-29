#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.attention import EEGTransformer
from utils import create_mask, KoLeo_loss, get_normalization_fn, get_activation_fn

#%%

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=mean-std, b=mean+std)

class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_channel=1, out_channel=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_channel)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        bs, num_channel, num_patch, patch_size = x.shape
        x = rearrange(x, 'B C N T -> B (C N) T') # (bs, num_total_patch (num_input_channel*num_patch), patch_size)
        x = x.unsqueeze(1) # (bs, 1, num_total_patch, patch_size)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B D (C N) T -> B C N (T D)', C=num_channel) # (bs, num_channel, num_patch, d_model)
        return x
    
class LinearPatchEmbedding(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape # (bs, channel, num_patch, d_model)
        x = self.proj(x) # (bs, embed_dim, )
        x = x.permute(0, 2, 3, 1) # (bs, num_channel, num_patch, embed_dim)
        return x

#%%

class EEGModel(nn.Module):
    def __init__(self, seq_len, patch_size, in_channel, out_channel, num_class, num_group, 
                d_model, num_head, num_layer, qkv_bias=False, qk_norm=None, qk_scale=None, dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0,
                norm=nn.LayerNorm, activation=nn.GELU, use_abs_pos_emb=True, use_ch_emb=True, use_mean_pooling=False, init_std=0.02,
                encoder_type='epa2', head_type='onelayer', sample_duration=None, target_channel=None):
        
        super().__init__()

        if (norm is not None) and (type(norm)==str):
            norm = get_normalization_fn(norm)
        if (activation is not None) and (type(activation)==str):
            activation = get_activation_fn(activation)
        if (qk_norm is not None) and (type(qk_norm)==str):
            qk_norm = get_normalization_fn(qk_norm)

        self.patch_embed = TemporalConv(out_channel=out_channel) if in_channel == 1 else LinearPatchEmbedding(patch_size=patch_size, in_chans=in_channel, embed_dim=d_model)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patch = seq_len // patch_size
        self.num_class = num_class
        self.init_std = init_std

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, d_model), requires_grad=True)
        else:
            self.pos_embed = None
            
        if use_ch_emb:
            self.channel_embed = nn.Parameter(torch.zeros(1, 128, d_model), requires_grad=True)
        else:
            self.channel_embed = None

        # self.time_embed = nn.Parameter(torch.zeros(1, 16, d_model), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout)

        self.rel_pos_bias = None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layer)]  # stochastic depth decay rule

        self.attention = EEGTransformer(num_patch=self.num_patch, num_group=num_group, num_layer=num_layer, d_model=d_model, num_head=num_head, d_ff=None, 
                                        dropout=dropout, attn_dropout=attn_dropout, res_attention=False, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, 
                                        norm_layer=norm, activation=activation, encoder_type=encoder_type,)

        self.norm = nn.Identity() if use_mean_pooling else norm(d_model)
        self.fc_norm = norm(d_model) if use_mean_pooling else None
        
        target_dim = 1 if num_class in [0, 2] else num_class

        if head_type=='identity':
            # for vector quantizer encoder
            self.head = nn.Identity()
        elif head_type=='pretrain':
            # for nova fm model pre-training
            self.head = nn.Linear(d_model, num_class)
        
        # for downstream tasks
        elif head_type=='onelayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B D C N'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(), # (B, D)
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='twolayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B D C N'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(), # (B, D)
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='norm-onelayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B D C N'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(), # (B, D)
                norm(d_model),
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='norm-twolayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B D C N'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(), # (B, D)
                norm(d_model),
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='allpatch-onelayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B (C N D)'),
                nn.Linear(d_model*sample_duration*target_channel, target_dim)
             )
        elif head_type=='allpatch-twolayer':
            self.head = nn.Sequential(
                Rearrange('B C N D -> B (C N D)'),
                nn.Linear(d_model*sample_duration*target_channel, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, target_dim)
            )

        elif head_type=='conv-onelayer':
            self.head = nn.Sequential(
                nn.Conv2d(target_channel, 1, kernel_size=(sample_duration, 1), stride=(sample_duration, 1)),
                nn.Flatten(), # (B, D)
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='conv-twolayer':
            self.head = nn.Sequential(
                nn.Conv2d(target_channel, 1, kernel_size=(sample_duration, 1), stride=(sample_duration, 1)),
                nn.Flatten(), # (B, D)
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, target_dim)
            )

        elif head_type=='conv-norm-onelayer':
            self.head = nn.Sequential(
                nn.Conv2d(target_channel, 1, kernel_size=(sample_duration, 1), stride=(sample_duration, 1)),
                nn.Flatten(), # (B, D)
                norm(d_model),
                nn.Linear(d_model, target_dim)
            )
        elif head_type=='conv-norm-twolayer':
            self.head = nn.Sequential(
                nn.Conv2d(target_channel, 1, kernel_size=(sample_duration, 1), stride=(sample_duration, 1)),
                nn.Flatten(), # (B, D)
                norm(d_model),
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, target_dim)
            )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        if self.channel_embed is not None:
            trunc_normal_(self.channel_embed, std=self.init_std)
        
        if head_type!='identity':
            if isinstance(self.head, nn.Sequential):
                for layer in self.head:
                    if isinstance(layer, nn.Linear):
                        trunc_normal_(layer.weight, std=self.init_std)
            else:
                trunc_normal_(self.head.weight, std=self.init_std)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Sequential): 
            for layer in m:
                self._init_weights(layer)   

    def forward_encoder(self, x, input_chans=None, mask=None):
        x = self.patch_embed(x) # (bs, num_channel, num_patch, d_model)
        batch_size = x.shape[0]
        num_channel = 128 if input_chans is None else len(input_chans)
        
        num_patch = x.shape[2]
        if mask is not None:
            mask_token = repeat(self.mask_token.unsqueeze(0), '1 1 1 D -> B C N D', B=batch_size, C=num_channel, N=num_patch)
            w = mask.unsqueeze(-1).type_as(mask_token)
            x = x * (1-w) + mask_token * w
        
        if self.pos_embed is not None:
            pos_embed = repeat(self.pos_embed[:, :num_patch], '1 N D -> B C N D', B=batch_size, C=num_channel)
            x = x + pos_embed

        # for encoder, not decoder
        if self.channel_embed is not None:
            channel_embed_used = self.channel_embed[:, input_chans] if input_chans is not None else self.channel_embed[: , :num_channel]
            channel_embed = repeat(channel_embed_used, '1 C D -> B C N D', B=batch_size, N=num_patch)
            x = x + channel_embed
        
        x = self.pos_drop(x)
        x = self.attention(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x)

        return x # (bs, num_channel, num_patch, d_model)

    def forward(self, x, input_chans=None, mask=None, return_emb=False):
        if x.dim()==3:
            x = rearrange(x, 'B C (N T) -> B C N T', T=self.patch_size)
        emb = self.forward_encoder(x, input_chans, mask)
        out = self.head(emb)

        if return_emb:
            return emb, out
        return out
    
class EncoderSVREEGPretrain(nn.Module):
    def __init__(self, args, encoder, vq_encoder, ):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.vq_encoder = vq_encoder
        
        for param in self.vq_encoder.parameters():
            param.requires_grad = False
  
    def forward(self, x, input_chans):
        
        x = rearrange(x, 'B C (N T) -> B C N T', T=self.args.patch_size)
        x_masked, mask = create_mask(x, mask_prob=self.args.mask_prob)
        emb, codebook_pred = self.encoder(x_masked, input_chans, mask=mask, return_emb=True)
        codebook_pred = codebook_pred.reshape(-1, self.encoder.num_class)
        with torch.no_grad():
            codebook_target = self.vq_encoder.get_codebook_indices(x, input_chans)
        
        koleo_loss = self.calculate_koleo_loss(emb)
        codebook_loss = self.calculate_rec_loss(codebook_pred, codebook_target)
        loss = self.args.alpha * koleo_loss + codebook_loss
        return loss, koleo_loss, codebook_loss
        
    def calculate_koleo_loss(self, emb):
        # emb: (bs, num_channel, num_patch, d_model)
        koleo_loss = KoLeo_loss(emb)
        return koleo_loss
    
    def calculate_rec_loss(self, x_codebook_pred, codebook_target):
        # x_codebook_pred: (bs*num_channel*num_patch, num_codebook)
        # codebook_target: (bs*num_channel*num_patch, )
        loss = F.cross_entropy(x_codebook_pred, codebook_target)
        return loss

# %%