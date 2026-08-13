import torch
import torch.nn as nn
import torch.nn.functional as F

"""CFDPN: Cross-scale Feature Focused-Diffusion Pyramid Network (paper).

Components: SAG, ChannelAlign, LightFusion.
"""

from ultralytics.train_door_detect.modules.SPPSA import PSABlock
from ultralytics.nn.modules import Conv


######################################## SAG (CFDPN) ########################################


class GlobalExtraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cv = Conv(dim, dim, 1, 1)
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x1 = self.cv(x)
        x = x * self.avgpool(x1)
        x2 = self.maxpool(x)

        return x2


class ContextExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.cv = Conv(2, 1, 1, 1)

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x3 = torch.cat([x1, x2], 1)
        x4 = self.cv(x3)

        return x4


class SAGFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.global_ = GlobalExtraction(dim // 2)
        self.local_ = ContextExtraction()
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.cv = Conv(dim, dim, 1, 1)

    def forward(self, x):
        a, b = torch.chunk(x, 2, 1)
        g = self.global_(a)
        l = self.local_(b)
        x1 = (self.act1(a) * l) + a
        x2 = (self.act2(b) * g) + b
        x3 = torch.cat([x1, x2], 1)
        x4 = self.cv(x3)

        return x4


class SAG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = SAGFusion(dim)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            y = self.multi(inputs)
            return y
        else:
            y = []
            for i in range(len(inputs)):
                yi = self.multi(inputs[i])
                y.append(yi)

            return y


######################################## Fusion ########################################
class LightFusion(nn.Module):
    """Lightweight fusion module in CFDPN (paper). Same logic as original Fusion_L."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        in_channels = sum(in_channels_list)
        self.csp_rep = CSPRepLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.csp_rep(x)
        return x


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


######################################## GPSA ########################################
class GPSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.psa = PSABlock(dim, attn_ratio=0.5, num_heads=dim // 64)

        self.gate = self.globalavgchannelpool

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def forward(self, x):
        # PSA gate on x
        psa = self.psa(x)
        x = x * self.gate(psa)

        return x


######################################## ChannelAlign (CFDPN) ########################################

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ChannelAlign(nn.Module):
    """Channel-alignment convolution in CFDPN (paper). Same logic as original Conv_L."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, order=0):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.order = order

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x[self.order])))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x[self.order]))
