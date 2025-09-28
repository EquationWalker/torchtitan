from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable


def numParams(net):
    import numpy as np
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


class NormSwitch(nn.Module):
    """
    Currently, BN, IN, and cLN are considered
    """
    def __init__(self,
                 norm_type: str,
                 dim_size: str,
                 c: int,
                 ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.dim_size = dim_size
        self.c = c

        assert norm_type in ["BN", "IN", "cLN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            if dim_size == "1D":
                self.norm = nn.BatchNorm1d(c)
            else:
                self.norm = nn.BatchNorm2d(c)
        elif norm_type == "IN":
            if dim_size == "1D":
                self.norm = nn.InstanceNorm1d(c, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(c, affine=True)
        elif norm_type == "cLN":
            if dim_size == "1D":
                self.norm = CumulativeLayerNorm1d(dim_size, affine=True)
            else:
                self.norm = CumulativeLayerNorm2d(dim_size, affine=True)

    def forward(self, x):
        return self.norm(x)
    

class CumulativeLayerNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1), requires_grad=True)
        else:
            self.gain = torch.ones(1, num_features, 1,requires_grad=False)
            self.bias = torch.zeros(1, num_features, 1,requires_grad=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel*(seq_len+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(inpt)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.gain = torch.ones(1,num_features,1,1, requires_grad=False)
            self.bias = torch.zeros(1,num_features,1,1, requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1,3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1,3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class Skip_connect(nn.Module):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == 'add':
            x = x_main + x_aux
        elif self.connect == 'cat':
            x = torch.cat((x_main, x_aux), dim=1)
        return x
    

class Chomp_T(nn.Module):
    def __init__(self,
                 t):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, :-self.t, :]



class Conv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, k, (1, 2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        )
    def forward(self, x):
        return self.conv(x)


class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t-1, 0), value=0.),   # for causal-setting
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                  stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        deconv_list = []
        if self.intra_connect == 'add':
            deconv_list.append(nn.ConvTranspose2d(c, c, k, (1, 2)))
        elif self.intra_connect == 'cat':
            deconv_list.append(nn.ConvTranspose2d(2*c, c, k, (1, 2)))
        deconv_list.append(NormSwitch(norm_type, "2D", c)),
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class GateConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                Chomp_T(k_t-1))
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()