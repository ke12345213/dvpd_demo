import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNNAttention(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_heads=4,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.Wq = nn.Linear(hidden_dim * 2, emb_dim)
            self.Wk = nn.Linear(hidden_dim * 2, emb_dim)
            self.Wv = nn.Linear(hidden_dim * 2, emb_dim)
        else:
            self.Wq = nn.Linear(hidden_dim, emb_dim)
            self.Wk = nn.Linear(hidden_dim, emb_dim)
            self.Wv = nn.Linear(hidden_dim, emb_dim)
        self.Wo = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x:(b,t,d)
        B, T, _ = x.size()
        x, _ = self.rnn(x)  # (b,t,2*h)
        q = self.Wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,t,d/h)
        k = self.Wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,t,d/h)
        v = self.Wv(x).reshape(B, T, self.n_heads, -1).transpose(1, 2)  # (b,h,t,d/h)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.mul(attn, self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (b,h,t,d/h)

        out = out.transpose(1, 2).reshape(B, T, -1)  # (b,t,d)
        out = self.Wo(out)

        return out


class DualPathRNNAttention(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            n_heads=4,
            dropout_p=0.1,
            temb_dim=None,
    ):
        super().__init__()
        self.intra_norm = nn.LayerNorm([n_freqs, emb_dim])
        self.intra_rnn_attn = RNNAttention(emb_dim, hidden_dim, n_heads, dropout_p, bidirectional=True)


        self.inter_norm = nn.LayerNorm([n_freqs, emb_dim])
        self.inter_rnn_attn = RNNAttention(emb_dim, hidden_dim, n_heads, dropout_p, bidirectional=True)

        if temb_dim is not None:
            self.intra_t_proj = nn.Linear(temb_dim, emb_dim)
            self.inter_t_proj = nn.Linear(temb_dim, emb_dim)

    def forward(self, x, temb=None):
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        if temb is not None:
            x = x + self.intra_t_proj(temb)[:, None, None, :]
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        if temb is not None:
            x = x + self.inter_t_proj(temb)[:, None, None, :]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3)  # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class Former(nn.Module):
    def __init__(
            self,
            emb_dim=64,
            hidden_dim=128,
            n_freqs=32,
            n_heads=4,
            dropout_p=0.1,
            temb_dim=None,
    ):
        super().__init__()
        self.dp_rnn_attn = DualPathRNNAttention(emb_dim, hidden_dim, n_freqs, n_heads, dropout_p, temb_dim)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=4, dropout_p=dropout_p)

    def forward(self, x, temb=None):
        x = self.dp_rnn_attn(x, temb)
        x = self.conv_glu(x)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as FF
import math
from torch.nn import init
from torch.nn.parameter import Parameter
import difflib
from torch_complex import functional as FC
from typing import *
from torch.nn import Parameter, init
import math 
import difflib
from torch.nn import *
from .tetst import MamberBlock


class lisa(nn.Module): 
    def __init__(self, dim, kernel=3, dilation=1, group=4, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))  
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)  
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)  
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True) 
        gap_kernel = (None,1) if H else (1, None) 
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)  # [B, C, 1, 1]
        filter = self.conv(filter) 
        n, c, h, w = x.shape
        x = FF.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)  # [B, G, 1, k, p*q] 
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w) 

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type = "cuda", enabled = False):
            if x.ndim == 4:
                _, C, _, _ = x.shape
                stat_dim = (1,)
            else:
                raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
            mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
            std_ = torch.sqrt(torch.clamp(x.var(dim=stat_dim, unbiased=False, keepdim=True), self.eps))  # [B,1,T,F]
            x_hat = (x - mu_) / (std_ )
                
            x_hat = x_hat * self.gamma + self.beta

            return x_hat


class MultiRangeGridNetBlock_Att(nn.Module):
    def __init__(self,  
                emb_dim,
                emb_ks,
                emb_hs,
                n_freqs,
                n_head=4,
                approx_qk_dim=512,
                activation="prelu",
                eps=1e-5,):
        super().__init__()
        self.emb_dim = emb_dim
        self.intra_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch1_att = lisa(dim=emb_dim, kernel=3, dilation=3, group=4, H=True)  # kernel=1, 

        self.intra_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch2_att = lisa(dim=emb_dim, kernel=3, dilation=5, group=4, H=True)  # kernel=4, 

        self.intra_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch3_att = lisa(dim=emb_dim, kernel=3, dilation=7, group=4,  H=True)  # kernel=8, 


        self.inter_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch1_att = lisa(dim=emb_dim, kernel=3, dilation=3, group=4,  H=False)  # kernel=1, 

        self.inter_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch2_att = lisa(dim=emb_dim, kernel=3, dilation=5, group=4,  H=False)  # kernel=4, 

        self.inter_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch3_att = lisa(dim=emb_dim, kernel=3, dilation=7, group=4, H=False)  # kernel=8, 
        self.gamma_1 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_1 = nn.Parameter(torch.ones(emb_dim,1,1))
        self.gamma_2 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_2 = nn.Parameter(torch.ones(emb_dim,1,1))
        self.gamma_3 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_3 = nn.Parameter(torch.ones(emb_dim,1,1))

        # self.conv_b = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.conv_b = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1),  # 48
            nn.GroupNorm(1, emb_dim, eps=eps),
            nn.PReLU()  # 48
        )

        self.globle_frond = MamberBlock(dim = emb_dim, num_heads = 4, ffn_expansion_factor=2.66, bias = False, LayerNorm_type = 'WithBias')
        self.globle_end = MamberBlock(dim = emb_dim, num_heads = 4, ffn_expansion_factor=2.66, bias = False, LayerNorm_type = 'WithBias')

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, Q = x.shape

        x = self.globle_frond(x)

        # -------------------- intra --------------------
        intra_input = x
        b1 = self.intra_branch1_att(self.intra_branch1_norm(intra_input))
        b2 = self.intra_branch2_att(self.intra_branch2_norm(intra_input))
        b3 = self.intra_branch3_att(self.intra_branch3_norm(intra_input))
    
        # -------------------- inter --------------------
        # inter_input = intra
        b1 = self.inter_branch1_att(self.inter_branch1_norm(b1))
        b2 = self.inter_branch2_att(self.inter_branch2_norm(b2))
        b3 = self.inter_branch3_att(self.inter_branch3_norm(b3))

        b1_all = self.gamma_1 * b1 + intra_input * self.beta_1
        b2_all = self.gamma_2 * b2 + intra_input * self.beta_2
        b3_all = self.gamma_3 * b3 + intra_input * self.beta_3
        b = b1_all + b2_all + b3_all
        output = self.conv_b(b)
        out = self.globle_end(output)
        return out



def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler

class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat



class GridNetBlock_Att(nn.Module):
    def __init__(self,  
                emb_dim,
                emb_ks,
                emb_hs,
                n_freqs,
                n_head=4,
                approx_qk_dim=512,
                activation="prelu",
                eps=1e-5,):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        # -------------------- intra --------------------
        self.intra_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch1_att = lisa(dim=emb_dim, kernel=3, dilation=3, group=4, H=True)  # kernel=1, freq

        self.intra_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch2_att = lisa(dim=emb_dim, kernel=3, dilation=5, group=4, H=True)  # kernel=4, freq

        self.intra_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch3_att = lisa(dim=emb_dim, kernel=3, dilation=7, group=4,  H=True)  # kernel=8, freq

        # self.intra_fusion_conv = nn.Conv2d(emb_dim*3, emb_dim, kernel_size=1)
        # self.intra_fusion_norm = LayerNormalization4D(emb_dim)

        # -------------------- inter --------------------
        self.inter_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch1_att = lisa(dim=emb_dim, kernel=3, dilation=3, group=4,  H=False)  # kernel=1, time

        self.inter_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch2_att = lisa(dim=emb_dim, kernel=3, dilation=5, group=4,  H=False)  # kernel=4, time

        self.inter_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch3_att = lisa(dim=emb_dim, kernel=3, dilation=7, group=4, H=False)  # kernel=8, time


        self.gamma_1 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_1 = nn.Parameter(torch.ones(emb_dim,1,1))
        self.gamma_2 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_2 = nn.Parameter(torch.ones(emb_dim,1,1))
        self.gamma_3 = nn.Parameter(torch.zeros(emb_dim,1,1))
        self.beta_3 = nn.Parameter(torch.ones(emb_dim,1,1))

        # self.conv_b = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.conv_b = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1),  # 48
            nn.GroupNorm(1, emb_dim, eps=eps),
            nn.PReLU()  # 48
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        self.attn_conv_Q = nn.ModuleList()
        self.attn_conv_K = nn.ModuleList()
        self.attn_conv_V = nn.ModuleList()

        for ii in range(n_head):
            self.attn_conv_Q.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                )
            )
            self.attn_conv_K.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                )
            )
            self.attn_conv_V.append(
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                )
            )
        self.attn_concat_proj = nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            )
    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, Q = x.shape


        # -------------------- intra --------------------
        intra_input = x
        b1 = self.intra_branch1_att(self.intra_branch1_norm(intra_input))
        b2 = self.intra_branch2_att(self.intra_branch2_norm(intra_input))
        b3 = self.intra_branch3_att(self.intra_branch3_norm(intra_input))
    
        # -------------------- inter --------------------
        # inter_input = intra
        b1 = self.inter_branch1_att(self.inter_branch1_norm(b1))
        b2 = self.inter_branch2_att(self.inter_branch2_norm(b2))
        b3 = self.inter_branch3_att(self.inter_branch3_norm(b3))

        b1_all = self.gamma_1 * b1 + intra_input * self.beta_1
        b2_all = self.gamma_2 * b2 + intra_input * self.beta_2
        b3_all = self.gamma_3 * b3 + intra_input * self.beta_3
        b = b1_all + b2_all + b3_all
        output = self.conv_b(b)
        batch = output

        # attention
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self.attn_conv_Q[ii](batch))
            all_K.append(self.attn_conv_K[ii](batch))
            all_V.append(self.attn_conv_V[ii](batch))

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = FF.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, T, -1]
        )  # [B, C, T, Q])
        batch = self.attn_concat_proj(batch)  # [B, C, T, Q])

        out = batch + output
        return out

if __name__ == '__main__':
    x = torch.randn(4, 64, 256, 128)
    # m = Former(emb_dim=32, hidden_dim=64)
    N = GridNetBlock_Att(emb_dim=64,emb_ks=4,emb_hs=1,n_freqs=128)
    print(sum([p.numel() for p in N.parameters()]))
    y = N(x)
    print(y.shape)
