import os
import warnings
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
import numpy as np
from operator import mul, xor
from einops.layers.torch import Rearrange
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange, repeat


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, T, C, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, T, H, W, C) -> (N, T, C, H, W)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape)==4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape)==5:
                x = self.weight[:,None, None, None] * x + self.bias[:, None, None, None]
            return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        class Transpose_Dim12(nn.Module):
            """ Transpose Dim1 and Dim2 of a tensor."""

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat*4, kernel_size=( 3, 3), padding=( 1, 1)))
                m.append(nn.PixelShuffle(2))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv2d(num_feat, num_feat, kernel_size=( 3, 3), padding=( 1, 1)))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat,  num_feat, kernel_size=( 3, 3), padding=( 1, 1)))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv2d(num_feat, num_feat, kernel_size=(3, 3), padding=( 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, tok_dim, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(tok_dim, FeedForward(tok_dim, mlp_dim, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))]))
    def forward(self, x):

        for tok_ff, ff in self.layers:

            x_tok = x.transpose(-1,-2)
            x = tok_ff(x_tok) + x_tok
            x = x.transpose(-1,-2)
            x = ff(x) + x
        return x


class rebotnet(nn.Module):


    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 temp_dim=2,
                 img_size=[6, 64, 64],
                 window_size=[6, 8, 8],
                 depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 indep_reconsts=[11, 12],
                 embed_dims=[96, 192, 384, 768],
                 num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 pa_frames=2,
                 deformable_groups=16,
                 recal_all_flows=False,
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 mlp_dim = 1024,
                 dropout = 0.1,
                 bottle_dim = 96,
                 bottle_depth = 4,
                 patch_size=2,
                 layer_scale_init_value=1e-6, 
                 out_indices=[0, 1, 2, 3],
                 dim_head = 64
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising

        dims =embed_dims

        # conv_first
       
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans*2, dims[0], kernel_size=2, stride=2),#, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)#, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

        ### end ConvNext

        ### start bottleneck

        image_size = img_size[-1]//4
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        patch_dim = embed_dims[-1] * patch_height * patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        nn.Linear(patch_dim, embed_dims[-1]),
        )

        big_patch = 16
        patch_dim_big = 3 * big_patch * big_patch

        self.big_embedding1 = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = big_patch, p2 = big_patch),
        nn.Linear(patch_dim_big, embed_dims[-1]),
        )

        self.big_embedding2 = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = big_patch, p2 = big_patch),
        nn.Linear(patch_dim_big, embed_dims[-1]),
        )

        self.pool = nn.MaxPool1d(2, 2)

        self.bottleneck = Transformer(576,bottle_dim, bottle_depth, num_heads[-1], dim_head, mlp_dim, dropout)

        self.temporal_transformer = Transformer(576,bottle_dim, bottle_depth, num_heads[-1], dim_head, mlp_dim, dropout)

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        num_feat = embed_dims[-1]
        num_feat1 = embed_dims[-2]
        num_feat2 = embed_dims[-3]
        num_feat3 = embed_dims[-4]

        self.upsample1 = Upsample(2, num_feat)
        self.upsample2 = Upsample(2, num_feat)
        self.upsample3 = Upsample(2, num_feat1)
        self.upsample4 = Upsample(2, num_feat2)

        self.upsamplef1 = Upsample(2, num_feat3)
        self.upsamplef2 = Upsample(4, num_feat3)

        self.conv_last = nn.Conv2d(num_feat3, 3, kernel_size=( 3, 3), padding=(1, 1))

        self.chchange1 = nn.Conv2d(num_feat, num_feat1, kernel_size=( 3, 3), padding=(1, 1))
        self.chchange2 = nn.Conv2d(num_feat1, num_feat2, kernel_size=( 3, 3), padding=(1, 1))
        self.chchange3 = nn.Conv2d(num_feat2, num_feat3, kernel_size=( 3, 3), padding=(1, 1))


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # x: (N, D, C, H, W)
        
        x_org = x[:,1,...].clone()
        

        x = rearrange(x, 'b t c h w -> b (t c) h w')

        x_1 = self.big_embedding1(x[:,0:3,...])
        x_2 = self.big_embedding2(x[:,3:6,...])

        x_temp = torch.cat((x_1,x_2), dim=1)


        outs = []
        for i in range(4):

            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        
        x = x_out
        ## bottleneck
        x_size = x.size()
        x_temp = self.pool(x_temp.transpose(1,2))
        x_temp = self.temporal_transformer(x_temp.transpose(1,2))

        x = self.to_patch_embedding(x)
        x = self.bottleneck(x)

        x = x + x_temp

        x = x.transpose(1, 2).view(x_size[0], x_size[1], x_size[2], x_size[3])  # B Ph*Pw C

        ### decoder

        x = self.upsample2(x)
        x = self.chchange1(x) + outs[-2]
        x = self.upsample3(x) 
        x = self.chchange2(x)+ outs[-3]
        x = self.upsample4(x) 
        x = self.chchange3(x)+ outs[-4]

        if self.upscale ==1:
            x = self.upsamplef1(x)
            x = self.conv_last(x) 
            return x + x_org
        else:
            x = self.upsamplef1(x)
            x = self.upsamplef2(x)

            x = self.conv_last(x) 

            return x + torch.nn.functional.interpolate(x_org, size=(H, W), mode='bilinear', align_corners=False)

