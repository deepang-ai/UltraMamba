import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class CROSS_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            modal_num=3,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.modal_num = modal_num

        self.in_proj_TC = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_VC = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_VG = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d_TC = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.conv2d_VC = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_VG = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj_fuse = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        self.x_proj_weight_fuse = nn.Parameter(torch.stack([t.weight for t in self.x_proj_fuse], dim=0))
        del self.x_proj_fuse

        self.fuse = nn.Conv2d(in_channels=self.d_inner * 3, out_channels=self.d_inner, kernel_size=1, bias=False)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)

        self.out_proj_TC = self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VC = self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VG = self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x_TC: torch.Tensor, x_VC: torch.Tensor, x_VG: torch.Tensor):

        B, C, H, W = x_TC.shape
        L = H * W
        K = 4

        x_fuse = self.fuse(torch.cat([x_TC, x_VC, x_VG], dim=1))

        x_hwwh_fuse = torch.stack(
            [x_fuse.view(B, -1, L), torch.transpose(x_fuse, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)

        x_hwwh_TC = torch.stack(
            [x_TC.view(B, -1, L), torch.transpose(x_TC, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)
        x_hwwh_VC = torch.stack(
            [x_VC.view(B, -1, L), torch.transpose(x_VC, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)
        x_hwwh_VG = torch.stack(
            [x_VG.view(B, -1, L), torch.transpose(x_VG, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1).view(B, 2, -1, L)

        xs_fuse = torch.cat([x_hwwh_fuse, torch.flip(x_hwwh_fuse, dims=[-1])], dim=1)

        xs_TC = torch.cat([x_hwwh_TC, torch.flip(x_hwwh_TC, dims=[-1])], dim=1)
        xs_VC = torch.cat([x_hwwh_VC, torch.flip(x_hwwh_VC, dims=[-1])], dim=1)
        xs_VG = torch.cat([x_hwwh_VG, torch.flip(x_hwwh_VG, dims=[-1])], dim=1)

        x_dbl_fuse = torch.einsum("b k d l, k c d -> b k c l", xs_fuse.view(B, K, -1, L), self.x_proj_weight_fuse)

        dts, Bs, Cs = torch.split(x_dbl_fuse, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs_TC = xs_TC.float().view(B, -1, L)  # (b, k * d, l)
        xs_VC = xs_VC.float().view(B, -1, L)  # (b, k * d, l)
        xs_VG = xs_VG.float().view(B, -1, L)  # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        #########################################
        out_y_TC = self.selective_scan(
            xs_TC, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y_TC.dtype == torch.float

        inv_y_TC = torch.flip(out_y_TC[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y_TC = torch.transpose(out_y_TC[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y_TC = torch.transpose(inv_y_TC[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        #########################################
        out_y_VC = self.selective_scan(
            xs_VC, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y_VC.dtype == torch.float

        inv_y_VC = torch.flip(out_y_VC[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y_VC = torch.transpose(out_y_VC[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y_VC = torch.transpose(inv_y_VC[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        #########################################
        out_y_VG = self.selective_scan(
            xs_VG, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y_VG.dtype == torch.float

        inv_y_VG = torch.flip(out_y_VG[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y_VG = torch.transpose(out_y_VG[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y_VG = torch.transpose(inv_y_VG[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y_TC[:, 0], inv_y_TC[:, 0], wh_y_TC, invwh_y_TC, out_y_VC[:, 0], inv_y_VC[:,
                                                                                    0], wh_y_VC, invwh_y_VC, out_y_VG[:,
                                                                                                             0], inv_y_VG[
                                                                                                                 :,
                                                                                                                 0], wh_y_VG, invwh_y_VG

    def forward(self, x_TC: torch.Tensor, x_VC: torch.Tensor, x_VG: torch.Tensor, **kwargs):

        # x = torch.cat((x1,x2),dim=-1)

        B, H, W, C = x_TC.shape

        # print(x_TC.shape)

        xz_TC = self.in_proj_TC(x_TC)

        # print(xz_TC.shape)

        xz_VC = self.in_proj_VC(x_VC)

        xz_VG = self.in_proj_VG(x_VG)

        x_TC, z_TC = xz_TC.chunk(2, dim=-1)  # (b, h, w, d)

        x_VC, z_VC = xz_VC.chunk(2, dim=-1)  # (b, h, w, d)

        x_VG, z_VG = xz_VG.chunk(2, dim=-1)  # (b, h, w, d)

        x_TC = x_TC.permute(0, 3, 1, 2).contiguous()
        x_TC = self.act(self.conv2d_TC(x_TC))  # (b, d, h, w)

        x_VC = x_VC.permute(0, 3, 1, 2).contiguous()
        x_VC = self.act(self.conv2d_VC(x_VC))  # (b, d, h, w)

        x_VG = x_VG.permute(0, 3, 1, 2).contiguous()
        x_VG = self.act(self.conv2d_VG(x_VG))  # (b, d, h, w)

        y1_TC, y2_TC, y3_TC, y4_TC, y1_VC, y2_VC, y3_VC, y4_VC, y1_VG, y2_VG, y3_VG, y4_VG = self.forward_core(x_TC,
                                                                                                               x_VC,
                                                                                                               x_VG)

        assert y1_TC.dtype == torch.float32
        y_TC = y1_TC + y2_TC + y3_TC + y4_TC
        y_TC = torch.transpose(y_TC, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_TC = self.out_norm(y_TC)
        y_TC = y_TC * F.silu(z_TC)
        out_TC = self.out_proj_TC(y_TC)

        assert y1_VC.dtype == torch.float32
        y_VC = y1_VC + y2_VC + y3_VC + y4_VC
        y_VC = torch.transpose(y_VC, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_VC = self.out_norm(y_VC)
        y_VC = y_VC * F.silu(z_VC)
        out_VC = self.out_proj_VC(y_VC)

        assert y1_VG.dtype == torch.float32
        y_VG = y1_VG + y2_VG + y3_VG + y4_VG
        y_VG = torch.transpose(y_VG, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_VG = self.out_norm(y_VG)
        y_VG = y_VG * F.silu(z_VG)
        out_VG = self.out_proj_VG(y_VG)

        if self.dropout is not None:
            out_TC = self.dropout(out_TC)
            out_VC = self.dropout(out_VC)
            out_VG = self.dropout(out_VG)

        return torch.stack([out_TC, out_VC, out_VG], dim=1)


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = CROSS_SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.hidden_dim = hidden_dim

    def forward(self, input: torch.Tensor):
        x_TC = self.ln_1(input[:, 0, :, :, :])
        x_VC = self.ln_1(input[:, 1, :, :, :])
        x_VG = self.ln_1(input[:, 2, :, :, :])

        x = input + self.drop_path(self.self_attention(x_TC, x_VC, x_VG))
        return x


class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSMEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, depths=[1, 1, 1, 1],
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed_TC = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                           norm_layer=norm_layer if patch_norm else None)

        self.patch_embed_VC = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                           norm_layer=norm_layer if patch_norm else None)

        self.patch_embed_VG = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                           norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples_TC = nn.ModuleList()
        self.downsamples_VC = nn.ModuleList()
        self.downsamples_VG = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples_TC.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))
                self.downsamples_VC.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))
                self.downsamples_VG.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):

        x_ret_TC = []
        x_ret_VC = []
        x_ret_VG = []

        x_TC = x[:, 0, :, :, :]
        x_VC = x[:, 1, :, :, :]
        x_VG = x[:, 2, :, :, :]

        x_ret_TC.append(x_TC)
        x_ret_VC.append(x_VC)
        x_ret_VG.append(x_VG)

        x_TC = self.patch_embed_TC(x_TC)  # Patch Embedding
        x_VC = self.patch_embed_TC(x_VC)
        x_VG = self.patch_embed_TC(x_VG)

        x_TC = self.pos_drop(x_TC)
        x_VC = self.pos_drop(x_VC)
        x_VG = self.pos_drop(x_VG)

        x = torch.stack((x_TC, x_VC, x_VG), dim=1)

        for s, layer in enumerate(self.layers):

            x_out = layer(x)

            x_out_TC = x_out[:, 0, :, :, :]
            x_out_VC = x_out[:, 0, :, :, :]
            x_out_VG = x_out[:, 0, :, :, :]

            # Vss Layer
            x_ret_TC.append(x_out_TC.permute(0, 3, 1, 2))  #####################################
            x_ret_VC.append(x_out_VC.permute(0, 3, 1, 2))
            x_ret_VG.append(x_out_VG.permute(0, 3, 1, 2))

            if s < len(self.downsamples_TC):
                x_out_TC = self.downsamples_TC[s](x_out_TC)  # Patch Merging
                x_out_VC = self.downsamples_VC[s](x_out_VC)
                x_out_VG = self.downsamples_VG[s](x_out_VG)
                x = torch.stack((x_out_TC, x_out_VC, x_out_VG), dim=1)

        return x_ret_TC, x_ret_VC, x_ret_VG


class Correction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x1, x2, return_map=False):
        q = self.q(x2)
        k = self.k(x1)
        v = self.v(x1)

        mean_x1 = q.mean(dim=1)
        mean_x1 = mean_x1.unsqueeze(dim=1)
        mean_x1 = mean_x1.repeat(1, q.size()[1], 1, 1)
        abs_x1 = q - mean_x1
        abs_pow_x1 = torch.pow(abs_x1, 2)
        abs_pow_sum_x1 = abs_pow_x1.sum(dim=1)

        mean_x2 = k.mean(dim=1)
        mean_x2 = mean_x2.unsqueeze(dim=1)
        mean_x2 = mean_x2.repeat(1, k.size()[1], 1, 1)
        abs_x2 = k - mean_x2
        abs_pow_x2 = torch.pow(abs_x2, 2)
        abs_pow_sum_x2 = abs_pow_x2.sum(dim=1)

        cov_1 = abs_x1 * abs_x2
        cov_1 = cov_1.sum(dim=1)
        cov_2 = abs_pow_sum_x1 * abs_pow_sum_x2
        cov_2 = cov_2.sqrt()
        cov = torch.relu(cov_1 / (cov_2 + 1e-4))

        correction_map = cov

        cov = cov.unsqueeze(1)
        cov = cov.repeat(1, x1.size()[1], 1, 1)
        output = v * cov


        if return_map:
            return output, correction_map
        else:
            return output


class MusoMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=1,
            img_size=352,
            feat_size=[48, 96, 192, 384, 768],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,

            norm_name="instance",
            res_block: bool = True,
            spatial_dims=2,
            deep_supervision: bool = False,
            modal_num=3,
            depths=[1, 1, 1, 1],
            use_pretrain=True,
            model_dir="./pretrained_model/vssmtiny_dp01_ckpt_epoch_292.pth"
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.modal_num = modal_num

        self.Correction_input = Correction(channels=3)
        self.Corrections = nn.ModuleList()
        for i in range(len(self.feat_size)):
            self.Corrections.append(Correction(channels=feat_size[i]))

        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_chans, in_chans, kernel_size=7, stride=2, padding=3),
        #     nn.InstanceNorm2d(in_chans, eps=1e-5, affine=True),
        # )

        self.stems = nn.ModuleList()
        for i in range(self.modal_num):
            self.stems.append(nn.Sequential(
                nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
                nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
            ))

        self.spatial_dims = spatial_dims

        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=feat_size[0], depths=depths, dims=feat_size[1:])

        # self.CROSS_SS2Ds = nn.ModuleList()
        # for i in range(len(self.feat_size)):
        #     self.CROSS_SS2Ds.append(CROSS_SS2D(d_model=feat_size[i], dropout=drop_path_rate, d_state=math.ceil(feat_size[1] / 6)))

        # self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=in_chans*self.modal_num , dims=feat_size[1:])

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans * self.modal_num,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0] * self.modal_num,
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1] * self.modal_num,
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2] * self.modal_num,
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3] * self.modal_num,
            out_channels=self.feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder6 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[4] * self.modal_num,
            out_channels=self.feat_size[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[4],
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # deep supervision support
        self.deep_supervision = deep_supervision
        self.out_layers = nn.ModuleList()
        for i in range(4):
            self.out_layers.append(UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[i],
                out_channels=self.out_chans
            ))



    def forward(self, x_TC, x_VC, x_VG, return_map=False):

        x_input = torch.stack((x_TC, x_VC, x_VG), dim=1)


        vss_ins = []
        for i in range(self.modal_num):
            x_conv = self.stems[i](x_input[:, i, :, :, :])
            vss_ins.append(x_conv)

        vss_input = torch.stack((vss_ins[0], vss_ins[1], vss_ins[2]), dim=1)
        vss_out_TC, vss_out_VC, vss_out_VG = self.vssm_encoder(vss_input)

        TC_maps, VC_maps = [], []
        for i in range(len(self.feat_size)):
            if return_map:
                spec_TC, TC_map = self.Corrections[i](vss_out_TC[i], vss_out_VG[i], return_map)
                TC_maps.append(TC_map)
                spec_VC, VC_map = self.Corrections[i](vss_out_VC[i], vss_out_VG[i], return_map)
                VC_maps.append(VC_map)
            else:
                spec_TC = self.Corrections[i](vss_out_TC[i], vss_out_VG[i], return_map)
                spec_VC = self.Corrections[i](vss_out_VC[i], vss_out_VG[i], return_map)
            vss_out_TC[i] = spec_TC

            vss_out_VC[i] = spec_VC

        cross_outs = []
        for i in range(len(self.feat_size)):
            # cross_out = self.CROSS_SS2Ds[i](vss_outs[0][i].transpose(1, 2).transpose(2, 3), vss_outs[1][i].transpose(1, 2).transpose(2, 3))
            # cross_out = torch.cat((cross_out.transpose(2, 3).transpose(1, 2), vss_outs[2][i]), dim=1)
            # cross_outs.append(cross_out)
            cross_out = torch.cat((vss_out_TC[i], vss_out_VC[i], vss_out_VG[i]), dim=1)
            cross_outs.append(cross_out)

            # elif i == len(self.feat_size) - 1:
            #     cross_out = self.CROSS_SS2Ds[i](vss_outs[0][i].transpose(1, 2).transpose(2, 3),
            #                                     vss_outs[1][i].transpose(1, 2).transpose(2, 3),
            #                                     vss_outs[2][i].transpose(1, 2).transpose(2, 3),
            #                                     ).transpose(2, 3).transpose(1, 2)
            #     cross_outs.append(cross_out)

        x_TC = self.Correction_input(x_TC, x_VG)
        x_VC = self.Correction_input(x_VC, x_VG)

        x_in = torch.cat((x_TC, x_VC, x_VG), dim=1)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(cross_outs[0])
        enc3 = self.encoder3(cross_outs[1])
        enc4 = self.encoder4(cross_outs[2])
        enc5 = self.encoder5(cross_outs[3])
        enc_hidden = cross_outs[4]

        dec4 = self.decoder6(enc_hidden, enc5)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)



        if self.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(len(feat_out)):
                pred = self.out_layers[i](feat_out[i])
                out.append(pred)
        else:
            out = self.out_layers[0](dec_out)


        if return_map:
            return out, TC_maps, VC_maps
        else:
            return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True

    def load_pretrained_ckpt(model, ckpt_path, modal_num):
        print(f"Loading weights from: {ckpt_path}")
        skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias",
                       "patch_embed.proj.weight", "patch_embed.proj.bias",
                       "patch_embed.norm.weight", "patch_embed.norm.weight"]

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_dict = model.state_dict()
        # print(model_dict.keys())
        if modal_num == 1:
            for k, v in ckpt['model'].items():
                if k in skip_params:
                    print(f"Skipping weights: {k}")
                    continue
                kr = f"vssm_encoder.{k}"
                if "downsample" in kr:
                    i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                    kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
                    assert kr in model_dict.keys()
                if kr in model_dict.keys():
                    assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                    model_dict[kr] = v
                else:
                    print(f"Passing weights: {k}")

            model.load_state_dict(model_dict)
        elif modal_num == 3:
            for k, v in ckpt['model'].items():
                if k in skip_params:
                    print(f"Skipping weights: {k}")
                    continue
                for i in range(modal_num):
                    kr = f"vssm_encoders.{i}.{k}"
                    if "downsample" in kr:
                        i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                        kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
                        assert kr in model_dict.keys()
                    if kr in model_dict.keys():
                        assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                        model_dict[kr] = v
                    else:
                        print(f"Passing weights: {k}")

            model.load_state_dict(model_dict)
        else:
            print("modal_num error!")

        return model


if __name__ == '__main__':
    device = 'cuda:0'

    x = torch.randn(size=(1, 3, 352, 352))

    model = MusoMamba(
        in_chans=3,
        out_chans=1,
        feat_size=[48, 96, 192, 384, 768],
        deep_supervision=True,

    )
    print("--------")
    print(model(x).size())
    # print(module(test_x).size())