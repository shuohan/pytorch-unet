# -*- coding: utf-8 -*-

import torch
from pytorch_layers import create_activ, create_dropout
from pytorch_layers import create_two_upsample, create_norm
from pytorch_layers import create_k3_conv, create_k1_conv


class _ConvBlock(torch.nn.Sequential):
    """Abstract class of the conv block.
    
    Attributes:
        in_channels (int): The number of channels of the input.
        out_channels (int): The number of channels of the output.
    
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        in_channels = torch.tensor(in_channels, dtype=torch.int)
        out_channels = torch.tensor(out_channels, dtype=torch.int)
        self.register_buffer('in_channels', in_channels)
        self.register_buffer('out_channels', out_channels)
        self.add_module('conv', self._create_conv())
        self.add_module('norm', self._create_norm())
        self.add_module('activ', create_activ())
        self.add_module('dropout', create_dropout())

    def _create_conv(self):
        raise NotImplementedError

    def _create_norm(self):
        raise NotImplementedError


class ConvBlock(_ConvBlock):
    """Convolution block used in UNet."""
    def _create_conv(self):
        return create_k3_conv(self.in_channels, self.out_channels,bias=False)
    def _create_norm(self):
        return create_norm(self.out_channels)


class ProjBlock(_ConvBlock):
    """Convolution block with kernel size 1."""
    def _create_conv(self):
        return create_k1_conv(self.in_channels, self.out_channels,bias=False)
    def _create_norm(self):
        return create_norm(self.out_channels)


class _ContractingBlock(torch.nn.Sequential):
    """Abstract class of the input block.

    Attributes:
        in_channels (int): The number of channels of the input.
        out_channels (int): The number of channels of the output.
        inter_channels (int): The number of channels of the intermediate tensor.
    
    """
    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        in_channels = torch.tensor(in_channels, dtype=torch.int)
        out_channels = torch.tensor(out_channels, dtype=torch.int)
        inter_channels = torch.tensor(inter_channels, dtype=torch.int)
        self.register_buffer('in_channels', in_channels)
        self.register_buffer('out_channels', out_channels)
        self.register_buffer('inter_channels', inter_channels)
        self.add_module('conv1', self._create_conv1())
        self.add_module('conv2', self._create_conv2())

    def _create_conv1(self):
        raise NotImplementedError

    def _create_conv2(self):
        raise NotImplementedError


class InputBlock(_ContractingBlock):
    """The input block of the UNet."""
    def _create_conv1(self):
        return ConvBlock(self.in_channels, self.inter_channels)
    def _create_conv2(self):
        return ConvBlock(self.inter_channels, self.out_channels)


class ContractingBlock(_ContractingBlock):
    """The contracting block of the UNet."""
    def _create_conv1(self):
        return ConvBlock(self.in_channels, self.inter_channels)
    def _create_conv2(self):
        return ConvBlock(self.inter_channels, self.out_channels)


class _ExpandingBlock(torch.nn.Module):
    """The abstract class of the expanding block.

    Attributes:
        in_channels (int): The number of the channels of the input.
        shortcut_channels (int): The number of the channels of the shortchut.
        out_channels (int): The number of channels of the output.
        conv1 (torch.nn.Module): The first convolution block.
        conv2 (torch.nn.Module): The second convolution block.
    
    """
    def __init__(self, in_channels, shortcut_channels, out_channels):
        super().__init__()
        in_channels = torch.tensor(in_channels, dtype=torch.int)
        out_channels = torch.tensor(out_channels, dtype=torch.int)
        shortcut_channels = torch.tensor(shortcut_channels, dtype=torch.int)
        self.register_buffer('in_channels', in_channels)
        self.register_buffer('out_channels', out_channels)
        self.register_buffer('shortcut_channels', shortcut_channels)
        self.conv1 = self._create_conv1()
        self.conv2 = self._create_conv2()

    def forward(self, input, shortcut):
        output = torch.cat((input, shortcut), dim=1) # concat channels
        output = self.conv1(output)
        output = self.conv2(output)
        return output

    def _create_conv1(self):
        raise NotImplementedError

    def _create_conv2(self):
        raise NotImplementedError


class ExpandingBlock(_ExpandingBlock):
    """The expanding block of the UNet."""
    def _create_conv1(self):
        in_channels = self.in_channels + self.shortcut_channels
        return ConvBlock(in_channels, self.out_channels)
    def _create_conv2(self):
        return ConvBlock(self.out_channels, self.out_channels)


class _TransUpBlock(torch.nn.Sequential):
    """The abstract class of the transition up block.
    
    Attributes:
        in_channels (int): The number of the channels of the input.
        out_channels (int): The number of the channels of the output.
    
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = torch.tensor(in_channels, dtype=torch.int)
        out_channels = torch.tensor(out_channels, dtype=torch.int)
        self.register_buffer('in_channels', in_channels)
        self.register_buffer('out_channels', out_channels)
        self.add_module('conv', self._create_conv())
        self.add_module('up', create_two_upsample())

    def _create_conv(self):
        raise NotImplementedError


class TransUpBlock(_TransUpBlock):
    """The transition block of the UNet."""
    def _create_conv(self):
        return ProjBlock(self.in_channels, self.out_channels, bias=False)
