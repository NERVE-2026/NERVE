#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from einops import rearrange, repeat
#%%
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
        # out_channel = 8 -> d_model = 200
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        # self.gelu1 = nn.GELU()
        # self.norm1 = nn.GroupNorm(4, out_channel)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        # self.gelu2 = nn.GELU()
        # self.norm2 = nn.GroupNorm(4, out_channel)
        # self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        # self.norm3 = nn.GroupNorm(4, out_channel)
        # self.gelu3 = nn.GELU()

        # out_channel = 8 -> d_model = 128
        # out_channel = 16 -> d_model = 256
        # out_channel = 32 -> d_model = 512
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 15), stride=(1, 4), padding=(0, 7))
        # self.gelu1 = nn.GELU()
        # self.norm1 = nn.GroupNorm(4, out_channel)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        # self.gelu2 = nn.GELU()
        # self.norm2 = nn.GroupNorm(4, out_channel)
        # self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), stride=(1, 3),padding=(0, 0))
        # self.norm3 = nn.GroupNorm(4, out_channel)
        # self.gelu3 = nn.GELU()


        # out_channel = 8 -> d_model = 128
        # out_channel = 16 -> d_model = 256
        # out_channel = 32 -> d_model = 512
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 15), stride=(1, 6), padding=(0, 7))
        # self.gelu1 = nn.GELU()
        # self.norm1 = nn.GroupNorm(4, out_channel)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        # self.gelu2 = nn.GELU()
        # self.norm2 = nn.GroupNorm(4, out_channel)
        # self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        # self.norm3 = nn.GroupNorm(4, out_channel)
        # self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        bs, num_channel, num_patch, patch_size = x.shape
        x = rearrange(x, 'B C N T -> B (C N) T') # (bs, num_total_patch (num_input_channel*num_patch), patch_size)
        x = x.unsqueeze(1) # (bs, 1, num_total_patch, patch_size)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B D (C N) T -> B C N (T D)', C=num_channel) # (bs, num_channel, num_patch, d_model)
        return x
# %%
x = torch.randn(2, 4, 10, 200)
conv = TemporalConv(out_channel=32)
conv(x).shape
# %%
