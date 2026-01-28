import logging

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["IBN", "get_norm"]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class SyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class FrozenBatchNorm(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    # return nn.BatchNorm2d(out_channels)

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm,
            "syncBN": SyncBatchNorm,
            "GhostBN": GhostBatchNorm,
            "FrozenBN": FrozenBatchNorm,
            "GN": lambda channels, **args: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels, **kwargs)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# registry.py

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self):
        def _register(cls):
            name = cls.__name__
            if name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self._name}")
            self._module_dict[name] = cls
            return cls
        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._module_dict[name]



import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable           

import sys


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output
    
# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """
    def __init__(self, rnn_type, input_size, hidden_size, output_size, 
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer    
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                   )
            
    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
            
        output = self.output(output)
            
        return output
    

# dual-path RNN with transform-average-concatenate (TAC)
class DPRNN_TAC(nn.Module):
    """
    Deep duaL-path RNN with transform-average-concatenate (TAC) applied to each layer/block.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """
    def __init__(self, rnn_type, input_size, hidden_size, output_size, 
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN_TAC, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # DPRNN + TAC for 3D input (ch, N, T)
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])
        
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.ch_norm = nn.ModuleList([])
        
        
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.ch_transform.append(nn.Sequential(nn.Linear(input_size, hidden_size*3),
                                                   nn.PReLU()
                                                  )
                                    )
            self.ch_average.append(nn.Sequential(nn.Linear(hidden_size*3, hidden_size*3),
                                                 nn.PReLU()
                                                )
                                  )
            self.ch_concat.append(nn.Sequential(nn.Linear(hidden_size*6, input_size),
                                                nn.PReLU()
                                               )
                                 )
                
            
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN and TAC modules. For causal setting change them to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.ch_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                   ) 
        
    def forward(self, input, num_mic):
        # input shape: batch, ch, N, dim1, dim2
        # num_mic shape: batch, 
        # apply RNN on dim1 first, then dim2, then ch
        
        batch_size, ch, N, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            # intra-segment RNN
            output = output.view(batch_size*ch, N, dim1, dim2)  # B*ch, N, dim1, dim2
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*ch*dim2, dim1, -1)  # B*ch*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*ch*dim2, dim1, N
            row_output = row_output.view(batch_size*ch, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B*ch, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output  # B*ch, N, dim1, dim2
            
            # inter-segment RNN
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*ch*dim1, dim2, -1)  # B*ch*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, N
            col_output = col_output.view(batch_size*ch, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B*ch, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output  # B*ch, N, dim1, dim2
            
            # TAC for cross-channel communication
            ch_input = output.view(input.shape)  # B, ch, N, dim1, dim2
            ch_input = ch_input.permute(0,3,4,1,2).contiguous().view(-1, N)  # B*dim1*dim2*ch, N
            ch_output = self.ch_transform[i](ch_input).view(batch_size, dim1*dim2, ch, -1)  # B, dim1*dim2, ch, H
            # mean pooling across channels
            if num_mic.max() == 0:
                # fixed geometry array
                ch_mean = ch_output.mean(2).view(batch_size*dim1*dim2, -1)  # B*dim1*dim2, H
            else:
                # only consider valid channels
                ch_mean = [ch_output[b,:,:num_mic[b]].mean(1).unsqueeze(0) for b in range(batch_size)]  # 1, dim1*dim2, H
                ch_mean = torch.cat(ch_mean, 0).view(batch_size*dim1*dim2, -1)  # B*dim1*dim2, H
            ch_output = ch_output.view(batch_size*dim1*dim2, ch, -1)  # B*dim1*dim2, ch, H
            ch_mean = self.ch_average[i](ch_mean).unsqueeze(1).expand_as(ch_output).contiguous()  # B*dim1*dim2, ch, H
            ch_output = torch.cat([ch_output, ch_mean], 2)  # B*dim1*dim2, ch, 2H
            ch_output = self.ch_concat[i](ch_output.view(-1, ch_output.shape[-1]))  # B*dim1*dim2*ch, N
            ch_output = ch_output.view(batch_size, dim1, dim2, ch, -1).permute(0,3,4,1,2).contiguous()  # B, ch, N, dim1, dim2
            ch_output = self.ch_norm[i](ch_output.view(batch_size*ch, N, dim1, dim2))  # B*ch, N, dim1, dim2
            output = output + ch_output
            
        output = self.output(output)  # B*ch, N, dim1, dim2
            
        return output
    
# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2, 
                 layer=4, segment_size=100, bidirectional=True, model_type='DPRNN',
                 rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()

        assert model_type in ['DPRNN', 'DPRNN_TAC'], "model_type can only be 'DPRNN' or 'DPRNN_TAC'."
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.model_type = model_type
        
        self.eps = 1e-8
        
        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        
        # DPRNN model
        self.DPRNN = getattr(sys.modules[__name__], model_type)(rnn_type, self.feature_dim, self.hidden_dim, self.feature_dim*self.num_spk, 
                                         num_layers=layer, bidirectional=bidirectional)
        
    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
    
    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)
        
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        
        segments1 = input[:,:,:-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:,:,segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
        
        return segments.contiguous(), rest
        
    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)
        
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size*2)  # B, N, K, L
        
        input1 = input[:,:,:,:segment_size].contiguous().view(batch_size, dim, -1)[:,:,segment_stride:]
        input2 = input[:,:,:,segment_size:].contiguous().view(batch_size, dim, -1)[:,:,:-segment_stride]
        
        output = input1 + input2
        if rest > 0:
            output = output[:,:,:-rest]
        
        return output.contiguous()  # B, N, T
    
    def forward(self, input):
        pass