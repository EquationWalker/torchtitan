'''
based on v6
scale up
'''
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from .common import *
from einops import rearrange
from torchtitan.protocols import ModelProtocol

from .args import EABNetArgs
from torchtitan.experiments.qwen3.model.args import Qwen3ModelArgs
import torch.nn.functional as F
from torchtitan.models.attention import build_attention
from torchtitan.protocols.train_spec import ModelProtocol


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5

        # RMSNorm added here to the here to include the q-k norm
        # This is one of the main differences between Llama3 and Qwen3
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Adding the q_norm and k_norm here
        # Last layer of adding q-k norm
        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv, scale=self.scaling)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Hidden dimension is directly added from the model argsS
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)

        self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        x = x + self.attention(self.attention_norm(x), rope_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)

  
      
class EaBNet(nn.Module, ModelProtocol):
    def __init__(self,
                 args: EABNetArgs
                 ):
        """
        :param k1: kernel size in the 2-D GLU, (2, 3) by default
        :param k2: kernel size in the UNet-blok, (1, 3) by defauly
        :param c: channel number in the 2-D Convs, 64 by default
        :param M: mic number, 9 by default
        :param embed_dim: embedded dimension, 64 by default
        :param kd1: kernel size in the Squeezed-TCM (dilation-part), 5 by default
        :param cd1: channel number in the Squeezed-TCM (dilation-part), 64 by default
        :param d_feat: channel number in the Squeezed-TCM(pointwise-part), 256 by default
        :param p: the number of Squeezed-TCMs within a group, 6 by default
        :param q: group numbers, 3 by default
        :param is_causal: causal flag, True by default
        :param is_u2: whether U^{2} is set, True by default
        :param bf_type: beamformer type, "lstm" by default
        :param topo_type: topology type, "mimo" and "miso", "mimo" by default
        :param intra_connect: intra connection type, "cat" by default
        :param norm_type: "IN" by default.

        Note: as IN will not accumulate mean and var statistics in both training and inference phase, it can not
        guarantee strict causality. If you wanner use IN, an optional method is to calculate the accumulated statistics
        in both training and inference stages. Besides, you can also choose other norms like BN, LN, cLN.
        """
        super().__init__()

        SCALE = 17
        M, F = args.M, args.F

        self.k1 = (2, 3)
        self.k2 = (1, 3)
        # self.c = 64
        self.c = SCALE*64
        self.M = M
        # self.embed_dim = 64
        self.embed_dim = SCALE*64
        self.kd1 = 5 #5
        # self.cd1 = 64
        self.cd1 = SCALE*64
        # self.d_feat = 256
        self.d_feat = SCALE*256
        self.p = 6
        self.q = 6# 3 # N stcn
        self.is_causal = True
        self.is_u2 = True
        self.intra_connect = "cat"
        self.norm_type = "IN" # IN

        self.en = U2Net_Encoder(M*2, self.k1, self.k2, self.c, self.intra_connect, self.norm_type)
        self.de = U2Net_Decoder(self.embed_dim, self.c, self.k1, self.k2, self.intra_connect, self.norm_type)

        self.bf_map = BF(self.embed_dim, M, F, 10)

        stcn_list = []
        for _ in range(self.q):
            stcn_list.append(SqueezedTCNGroup(self.kd1, self.cd1, self.d_feat, self.p, self.is_causal, self.norm_type))
        self.stcns = nn.ModuleList(stcn_list)
        
    def init_weights(self, buffer_device=None):
        self.bf_map.init_weights(buffer_device)
        

    def forward(self, inpt: Tensor) -> Tensor:
        """
        :param inpt: (B, M, F, T, 2) -> (batchsize, n_mics, 2, n_freq, n_time)
        :return: beamformed estimation: (B, F, T, 2)
        """
        #print('EABNet', inpt.dtype, inpt.shape)
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=1)
        inpt = rearrange(inpt, 'b m f t c -> b t f m c')
        b_size, seq_len, freq_len, M, _ = inpt.shape
        x = inpt.transpose(-2, -1).contiguous()
        x = x.reshape(b_size, seq_len, freq_len, -1).permute(0,3,1,2)
        x, en_list = self.en(x) 
        #print(f"After encoder {x.shape}")
        c = x.shape[1]
        x = x.transpose(-2, -1).contiguous().reshape(b_size, -1, seq_len)
        x_acc = torch.zeros(x.size(), requires_grad=True,device=x.device)
        for i in range(len(self.stcns)):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x = x_acc
        x = x.reshape(b_size, c, -1, seq_len).transpose(-2, -1).contiguous()
        x = self.de(x, en_list)
        #print(f"After decoder {x.shape}")
        
        bf_w = self.bf_map(x)  # (B, T, F, M, 2)
        bf_w_r, bf_w_i = bf_w[...,0], bf_w[...,-1]
        esti_x_r, esti_x_i = (bf_w_r*inpt[...,0]-bf_w_i*inpt[...,-1]).sum(dim=-1), \
                                (bf_w_r*inpt[...,-1]+bf_w_i*inpt[...,0]).sum(dim=-1)
        esti_stft = torch.stack((esti_x_r, esti_x_i), dim=1)
        esti_stft = rearrange(esti_stft, 'b c t f -> b f t c')
        return esti_stft
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [M]'.format(self._get_name(),
                                                      num_parameters / 10 ** 6)
        
class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = c
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=4, is_deconv=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_end, k1, (1,2)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end)
        )
    def forward(self, x: Tensor):
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class U2Net_Decoder(nn.Module):
    def __init__(self, embed_dim, c, k1, k2, intra_connect, norm_type):
        super(U2Net_Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_beg = c
        k_end = (2, 5)

        meta_unet = []
        meta_unet.append(
            En_unet_module(c_beg*2, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c*2, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c*2, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c*2, c, k1, k2, intra_connect, norm_type, scale=4, is_deconv=True)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2d(c*2, embed_dim, k_end, (1,2)),
            NormSwitch(norm_type, "2D", embed_dim),
            nn.PReLU(embed_dim)
        )

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        for i in range(len(self.meta_unet_list)):
            tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
            x = self.meta_unet_list[i](tmp)
        x = torch.cat((x, en_list[0]), dim=1)
        x = self.last_conv(x)
        return x


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 is_deconv: bool,
                 ):
        super(En_unet_module, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.intra_connect = intra_connect
        self.scale = scale
        self.is_deconv = is_deconv

        in_conv_list = []
        if not is_deconv:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1,2)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1,2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi

class SqueezedTCNGroup(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(SqueezedTCNGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.is_causal = is_causal
        self.norm_type = norm_type

        # Components
        self.tcm_list = nn.ModuleList([SqueezedTCM(kd1, cd1, 2**i, d_feat, is_causal, norm_type) for i in range(p)])

    def forward(self, x):
        for i in range(self.p):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 dilation: int,
                 d_feat: int,
                 is_causal: bool,
                 norm_type: str,
    ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            pad = ((kd1-1)*dilation, 0)
        else:
            pad = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kd1, dilation=dilation, bias=False)
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )
    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class BF(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 M: int,
                 F: int,
                 n_layers: int = 2,
                 num_heads: int = 8):
        super(BF, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.F = F
        self.model_args = Qwen3ModelArgs(
        max_seq_len=601,
        head_dim=embed_dim//num_heads,
        dim=embed_dim,
        n_layers=n_layers,
        n_heads=num_heads,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=embed_dim*2,
        rope_theta=1000000,
    )
        # Components
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(TransformerBlock(i, self.model_args))
        self.w_dnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2*M)
        )
        self.norm = nn.RMSNorm(embed_dim, eps=self.model_args.norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        # norm
        B, _, T, F = x.shape
        #self.norm(embed_x.permute(0,3,2,1).contiguous())
        # print(f"BF-1 {x.shape}")
        x = x.reshape(B*F, T, -1)
        #print(f"BF-2 {x.shape}")
        for rnn in self.layers:
            x = rnn(x, self.rope_cache)
        x = x.reshape(B, F, T, -1).transpose(1, 2).contiguous()
        x = self.norm(x)
        bf_w = self.w_dnn(x).reshape(B, T, F, self.M, 2)
        return bf_w
    
    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):

        buffer_device = buffer_device or 'cuda'
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        
        for layer in self.layers:
            if layer is not None:
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3


        nn.init.trunc_normal_(
            self.w_dnn[0].weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )
        nn.init.trunc_normal_(
                self.w_dnn[2].weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )


def calculate_memory_usage(model, input_size, batch_size):
    # 1. 参数显存占用
    num_params = sum(p.numel() for p in model.parameters())
    param_memory = num_params * 4  # float32，占用4字节

    # 2. 中间激活显存占用
    # 创建一个示例输入
    example_input = torch.randn(*input_size)
    
    # 前向传播计算激活的显存占用
    with torch.no_grad():
        activations = model(example_input)

    # 假设最后一层输出的形状
    activation_memory = activations.numel() * 4  # 每个元素占用4字节

    # 3. 梯度显存占用
    gradient_memory = param_memory  # 与参数显存相同

    # 4. 优化器状态显存占用（假设使用 Adam 优化器）
    optimizer_memory = param_memory  # 与参数显存相同

    # 5. 总显存占用
    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory

    # 转换为 MB
    return total_memory / (1024 ** 2)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNet
    x = torch.randn((1, 8,161, 751, 2))  #.cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    eabnet = EaBNet()
    # from packaging.version import Version
    # if Version(torch.__version__) >= Version('2.0.0'):
    #     SSFNet_small = torch.compile(SSFNet_small)
    # torch.cuda.synchronize(7)
    import time
    ts = time.time()
    y = eabnet(x)
    # torch.cuda.synchronize(7)
    te = time.time()
    print(eabnet)
    print(y.shape)
    print(te - ts)
    
    memory_usage = calculate_memory_usage(eabnet, x.shape, 1)
    print(f"Estimated memory usage: {memory_usage:.2f} MB")

    eabnet = eabnet.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(eabnet, display=False) as fcm:
        y = eabnet(x)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in eabnet.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")

