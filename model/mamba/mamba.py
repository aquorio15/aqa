import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft
from utils import *
from mha import *
from conv_kan import *
from kan_linear import *
from cross_attention import *
from self_attention import *
import os
import sys
import numpy as np

import torchvision
import torch.nn as nn

from dmamba import MambaConfig, ResidualBlock
import torch.nn.init as init

import math
import math
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from einops import rearrange, repeat, einsum
#%%
def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _ = data.shape
    index = new_order.repeat(B, t, 1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        d_conv=4,
        conv_bias=True,
        bias=False,
        conv_type="kan",
    ):

        super().__init__()
        self.d_model = d_model  # Model dimension d_model
        self.d_state = d_state  # SSM state expansion factor
        self.d_conv = d_conv  # Local convolution width
        self.expand = expand  # Block expansion factor
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):

        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):

        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):

        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class FusionMambaBlock(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        d_conv=4,
        conv_bias=False,
        bias=False,
        conv_type="kan"
    ):

        super().__init__()
        self.d_model = d_model  # Model dimension d_model
        self.d_state = d_state  # SSM state expansion factor
        self.d_conv = d_conv  # Local convolution width
        self.expand = expand  # Block expansion factor
        self.conv_bias = conv_bias
        self.bias = bias

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.conv_type = conv_type
        if self.conv_type == "kan":
            self.in_proj = FastKANLayer(self.d_model, self.d_inner * 2)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)
        # if self.conv_type == "kan":
        #     self.conv1d = ConvKAN(
        #         in_channels=self.d_inner,
        #         out_channels=self.d_inner,
        #         bias=self.conv_bias,
        #         kernel_size=self.d_conv,
        #         groups=self.d_inner,
        #         padding=self.d_conv - 1,
        #     )
        # else:
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        if self.conv_type == "kan":
            self.x_proj = FastKANLayer(self.d_inner, self.dt_rank + self.d_state * 2)
        else:
            self.x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False
            )
        # dt_proj projects Δ from dt_rank to d_in
        if self.conv_type == "kan":
            self.dt_proj = FastKANLayer(self.dt_rank, self.d_inner)
        else:
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        if self.conv_type == "kan":
            self.out_proj = FastKANLayer(self.d_inner, self.d_model)
        else:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x, z):

        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        z_and_res = self.in_proj(z)

        (x, res1) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        (z, res2) = z_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        z = rearrange(z, "b l d_in -> b d_in l")
        z = self.conv1d(z)[:, :, :l]
        z = rearrange(z, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        z = F.silu(z)

        w = self.ssm(z)

        img = y * F.silu(res2)

        audio = w * F.silu(res1)

        output_img = self.out_proj(img)
        output_audio = self.out_proj(audio)
        # if self.dynamic_weight:
        # return output_img, output_audio
        # else: 
        return torch.cat([output_img, output_audio], dim=1)
        # return fused_output, output_img, output_audio

    def ssm(self, x):

        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


########################## NOTE: Much faster due to implementation of flash-attention ################################################


class SingleMambaBlockimage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(
            dim, expand=1, d_state=8, bimamba_type="v2", if_devide_out=True
        )

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip


class SingleMambaBlockaudio(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(
            dim, expand=1, d_state=8, bimamba_type="v2", if_devide_out=True
        )

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, conv_type="kan"):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = FusionMambaBlock(dim, conv_type=conv_type)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, input1)
        # output = self.norm2(output)
        return output


class FusionMamba(nn.Module):
    def __init__(self, dim, depth=1, conv_type="kan"):
        super().__init__()
        self.img_mamba_layers = nn.ModuleList([])
        self.aud_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.img_mamba_layers.append(SingleMambaBlockimage(dim))
            self.aud_mamba_layers.append(SingleMambaBlockaudio(dim))
        # self.img_cross_mamba = CrossMambaBlock(dim)
        # self.aud_cross_mamba = CrossMambaBlock(dim)
        self.fusion = CrossMambaBlock(dim, conv_type=conv_type)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, img, audio):
        B = img.size(0) 
        for img_layer, aud_layer in zip(self.img_mamba_layers, self.aud_mamba_layers):
            img_output = img_layer(img)
            audio_output = aud_layer(audio)

        # img_fusion = self.img_cross_mamba(img, audio)
        # aud_fusion = self.aud_cross_mamba(audio, img)
        # fusion = self.out_proj((img_fusion + aud_fusion) / 2)
        # if self.dynamic:
        fusion = self.fusion(img_output, audio_output)
        # # print(fused_img)
        # attention_pro_comb_img = torch.sum(fused_img, dim=-2) / math.sqrt(fused_img.size(1) * 1)
        # # print(attention_pro_comb_img)
        # weight_norm_img = F.softmax(attention_pro_comb_img, dim=-1)
        # # print(weight_norm_img)    
        # embs_img = [(weight_norm_img[idx].unsqueeze(0) * F.normalize(img_output[idx])).unsqueeze(0) for idx in range(B)]
        # joint_emb_img = torch.cat(embs_img, dim=0)
        # # print(joint_emb_img)
        # attention_pro_comb_aud = torch.sum(fused_audio, dim=-2) / math.sqrt(fused_audio.size(1) * 1)
        # weight_norm_aud = F.softmax(attention_pro_comb_aud, dim=-1)
        # embs_aud = [(weight_norm_aud[idx].unsqueeze(0) * F.normalize(audio_output[idx])).unsqueeze(0) for idx in range(B)]
        # joint_emb_aud = torch.cat(embs_aud, dim=0)
        # fusion = torch.cat([joint_emb_img, joint_emb_aud], dim=1)
        # print(fusion)
        return fusion
        # else:
        #     fusion = self.fusion(img_output, audio_output)
        #     return fusion



class FusionMambareordering(nn.Module):
    def __init__(self, dim, depth=1, conv_type="kan"):
        super().__init__()
        self.img_mamba_layers = nn.ModuleList([])
        self.aud_mamba_layers = nn.ModuleList([])
        self.mamba_configs = MambaConfig(d_model=dim)
        self.fusing_ratios = 1
        for _ in range(depth):
            self.img_mamba_layers.append(ResidualBlock(config = self.mamba_configs))
            self.aud_mamba_layers.append(ResidualBlock(config = self.mamba_configs))
        self.fusion = CrossMambaBlock(dim, conv_type=conv_type)
        self.out_proj = nn.Linear(dim, dim)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
    def forward(self, img, audio):
        # print(img.shape)
        # print(audio.shape)
        B = img.size(0) 
        img = reorder_data(img, self.fusing_ratios)
        audio = reorder_data(audio, self.fusing_ratios)
        for img_layer, aud_layer in zip(self.img_mamba_layers, self.aud_mamba_layers):
            img_output = img_layer(img)
            audio_output = aud_layer(audio)

        # img_fusion = self.img_cross_mamba(img, audio)
        # aud_fusion = self.aud_cross_mamba(audio, img)
        # fusion = self.out_proj((img_fusion + aud_fusion) / 2)
        # if self.dynamic:
        fusion = self.fusion(img_output, audio_output)
        # print(fused_img)
        # attention_pro_comb_img = torch.sum(fused_img, dim=-2) / math.sqrt(fused_img.size(1) * 1)
        # # print(attention_pro_comb_img)
        # weight_norm_img = F.softmax(attention_pro_comb_img, dim=-1)
        # # print(weight_norm_img)    
        # embs_img = [(weight_norm_img[idx].unsqueeze(0) * F.normalize(img_output[idx])).unsqueeze(0) for idx in range(B)]
        # joint_emb_img = torch.cat(embs_img, dim=0)
        # # print(joint_emb_img)
        # attention_pro_comb_aud = torch.sum(fused_audio, dim=-2) / math.sqrt(fused_audio.size(1) * 1)
        # weight_norm_aud = F.softmax(attention_pro_comb_aud, dim=-1)
        # embs_aud = [(weight_norm_aud[idx].unsqueeze(0) * F.normalize(audio_output[idx])).unsqueeze(0) for idx in range(B)]
        # joint_emb_aud = torch.cat(embs_aud, dim=0)
        # fusion = torch.cat([joint_emb_img, joint_emb_aud], dim=1)
        # print(fusion)
        return fusion

# a = FusionMambareordering(dim=256).cuda()
# x = torch.randn(32, 225, 256).cuda()
# y = torch.randn(32, 225, 256).cuda()
# z = a(x, y)
# print(z.shape)

# %%
