"""
StyleNAT-specific PyTorch Modules

Authors: Steven Walton & Ali Hassani
Provided with StyleNAT
Contact: walton.stevenj@gmail.com
"""
import torch
from torch import nn
from torch.nn.init import trunc_normal_
import warnings
import os
from natten.functional import natten2dqkrpb, natten2dav
from natten import (
    use_fused_na,
    use_autotuner
)
use_fused_na(True)
use_autotuner(True)

class HydraNeighborhoodAttention(nn.Module):
    def __init__(self,
                 dim,
                 kernel_sizes, # Array for kernel sizes
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dilations=[1], # Array of dilations
                 ):
        super().__init__()
        if len(kernel_sizes) == 1 and len(dilations) != 1:
            kernel_sizes = [kernel_sizes[0] for _ in range(len(dilations))]
        elif len(dilations) == 1 and len(kernel_sizes) != 1:
            dilations = [dilations[0] for _ in range(len(kernel_sizes))]
        assert(len(kernel_sizes) == len(dilations)),f"Number of kernels ({(kernel_sizes)}) must be the same as number of dilations ({(dilations)})"
        self.num_splits = len(kernel_sizes)
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        asserts = []
        for i in range(len(kernel_sizes)):
            asserts.append(kernel_sizes[i] > 1 and kernel_sizes[i] % 2 == 1)
            if asserts[i] == False:
                warnings.warn(f"Kernel_size {kernel_sizes[i]} needs to be >1 and odd")
        assert(all(asserts)),f"Kernel sizes must be >1 AND odd. Got {kernel_sizes}"

        self.window_size = []
        for i in range(len(dilations)):
            self.window_size.append(self.kernel_sizes[i] * self.dilations[i])

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Needs to be fixed if we want uneven head splits. // is floored
        # division
        if num_heads % len(kernel_sizes) == 0:
            self.rpb = nn.ParameterList([nn.Parameter(torch.zeros(num_heads//self.num_splits, (2*k-1), (2*k-1))) for k in kernel_sizes])
            self.clean_partition = True
        else:
            diff = num_heads - self.num_splits * (num_heads // self.num_splits)
            rpb = [nn.Parameter(torch.zeros(num_heads//self.num_splits, (2*k-1), (2*k-1))) for k in kernel_sizes[:-diff]]
            for k in kernel_sizes[-diff:]:
                rpb.append(nn.Parameter(torch.zeros(num_heads//self.num_splits + 1, (2*k-1), (2*k-1))))
            assert(sum(r.shape[0] for r in rpb) == num_heads),f"Got {sum(r.shape[0] for r in rpb)} heads."
            self.rpb = nn.ParameterList(rpb)

            self.clean_partition = False
            self.shapes = [r.shape[0] for r in rpb]
            warnings.warn(f"Number of partitions ({self.num_splits}) do not "\
                    f"evenly divide the number of heads ({self.num_heads}). "\
                    f"We evenly divide the remainder between the last {diff} "\
                    f"heads This may cause unexpected behavior. Your head " \
                    f"partitions look like {self.shapes}")

        [trunc_normal_(rpb, std=0.02, mean=0.0, a=-2., b=2.) for rpb in self.rpb]
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)

        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0) * self.scale
        k = k.squeeze(0)
        v = v.squeeze(0)

        if self.clean_partition:
            q = q.chunk(self.num_splits, dim=1)
            k = k.chunk(self.num_splits, dim=1)
            v = v.chunk(self.num_splits, dim=1)
        else:
            i = 0
            _q = []
            _k = []
            _v = []
            for h in self.shapes:
                _q.append(q[:, i:i+h, :, :])
                _k.append(k[:, i:i+h, :, :])
                _v.append(v[:, i:i+h, :, :])
                i = i+h
            q, k, v = _q, _k, _v


        attention = [natten2dqkrpb(_q, _k, _rpb, _kernel_size, _dilation) \
                     for _q,_k,_rpb,_kernel_size,_dilation in zip(q, k, self.rpb, self.kernel_sizes, self.dilations)]
        attention = [a.softmax(dim=-1) for a in attention]
        attention = [self.attn_drop(a) for a in attention]

        x = [natten2dav(_attn, _v, _k, _d) \
             for _attn, _v, _k, _d in zip(attention, v, self.kernel_sizes, self.dilations)]

        x = torch.cat(x, dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        return self.proj_drop(self.proj(x))

class MHSARPB(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dilation=None #ignored
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02)
        coords_h = torch.arange(kernel_size)
        coords_w = torch.arange(kernel_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, L, L
        coords_flatten = torch.flatten(coords, 1)  # 2, L^2
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, L^2, L^2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # L^2, L^2, 2
        relative_coords[:, :, 0] += kernel_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += kernel_size - 1
        relative_coords[:, :, 0] *= 2 * kernel_size - 1
        relative_position_index = torch.flipud(torch.fliplr(relative_coords.sum(-1)))  # L^2, L^2
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_pb(self, attn):
        relative_position_bias = self.rpb.permute(1, 2, 0).flatten(0, 1)[self.relative_position_index.view(-1)].view(
            self.kernel_size ** 2, self.kernel_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return attn + relative_position_bias

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        if N != num_tokens:
            raise RuntimeError(f"Feature map size ({H} x {W}) is not equal to " +
                               f"expected size ({self.kernel_size} x {self.kernel_size}). " +
                               f"Consider changing sizes or padding inputs.")
        # Faster implementation -- just MHSA
        # If the feature map size is equal to the kernel size, NAT will be equivalent to self-attention.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B x heads x N x N
        attn = self.apply_pb(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        return self.proj_drop(self.proj(x))


class NeighborhoodAttentionSplitHead(nn.Module):
    '''
    ===========================================
    ============= This is Legacy ==============
    ===========================================
    Please use HydraNeighborhoodAttention above.
    The reason this exists is because it was our first version. We keep it for
    the FFHQ experiments so that we do not have to rewrite the model
    checkpoints.
    '''
    def __init__(self, dim, kernel_size_0, kernel_size_1, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dilation_0=1, dilation_1=1):
        super().__init__()
        warnings.warn(f"Using Legacy Hydra-NA, this is deprecated. Please use "\
                      "HydraNeighborhoodAttention function instead")
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        # First kernel
        assert kernel_size_0 > 1 and kernel_size_0 % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size_0}."
        assert kernel_size_0 in [3, 5, 7, 9, 11, 13, 15, 31, 45, 63], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, 13, 15, 45, and 63; got {kernel_size_0}."
        self.kernel_size_0 = kernel_size_0
        # Second kernel
        assert kernel_size_1 > 1 and kernel_size_1 % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size_1}."
        assert kernel_size_1 in [3, 5, 7, 9, 11, 13, 15, 31, 45, 63], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, 13, 15, 45, and 63; got {kernel_size_1}."
        self.kernel_size_1 = kernel_size_1
        assert dilation_0 >= 1 and dilation_1 >= 1, \
            f"Dilation values must be greater than or equal to 1, got" + \
            f"{dilation_0} and {dilation_1}."
        self.dilation_0 = dilation_0
        self.dilation_1 = dilation_1
        self.window_size_0 = self.kernel_size_0 * self.dilation_0
        self.window_size_1 = self.kernel_size_1 * self.dilation_1
        for ks in [kernel_size_0, kernel_size_1]:
            if ks not in [3, 5, 7, 9, 11, 13]:
                warnings.warn("You are not using recommended NA settings " + \
                        "(kernel_size <= 13), which will slow things down. " + \
                        f"You are running kernel_size={ks}.")
        if self.head_dim != 32:
            warnings.warn("You are not using recommended NA settings " + \
                    "(head_dim= 13), which will slow things down. " + \
                    f"You are running dim={self.head_dim}.")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb0 = nn.Parameter(torch.zeros(num_heads//2, (2 * kernel_size_0 - 1), (2 * kernel_size_0 - 1)))
        self.rpb1 = nn.Parameter(torch.zeros(num_heads//2, (2 * kernel_size_1 - 1), (2 * kernel_size_1 - 1)))

        trunc_normal_(self.rpb0, std=.02, mean=0., a=-2., b=2.)
        trunc_normal_(self.rpb1, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        # Split along heasd
        q0, q1 = q.chunk(chunks=2, dim=1)
        k0, k1 = k.chunk(chunks=2, dim=1)
        v0, v1 = v.chunk(chunks=2, dim=1)

        attn0 = natten2dqkrpb(q0, k0, self.rpb0, self.kernel_size_0, self.dilation_0)
        attn0 = attn0.softmax(dim=-1)
        attn0 = self.attn_drop(attn0)

        x0 = natten2dav(attn0, v0, self.kernel_size_0, self.dilation_0)

        attn1 = natten2dqkrpb(q1, k1, self.rpb1, self.kernel_size_1, self.dilation_1)
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        x1 = natten2dav(attn1, v1, self.kernel_size_1, self.dilation_1)

        x = torch.cat([x0, x1],dim=1)

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        return self.proj_drop(self.proj(x))

