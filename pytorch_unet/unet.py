# -*- coding: utf-8 -*-
  
import torch
import torch.nn.functional as F
import warnings
from collections import Iterable
from pytorch_layers import create_avg_pool, create_k1_conv

from .blocks import InputBlock, ContractingBlock, ExpandingBlock, TransUpBlock


class _UNet(torch.nn.Module):
    """The Abstract of a U-shaped network.

    Attributes:
        in_channels (int): The number of the channels of the input.
        out_channels (int): The number of the channels of the output.
        num_trans_down (int): The number of transition down. This number
            controls the "depth" of the network.
        first_channels (int): The number of output channels of the input block.
            This number controls the "width" of the networ.
        max_channels (int): The maximum number of tensor channels.

    """
    def __init__(self, in_channels, out_channels, num_trans_down,
                 first_channels, max_channels=1024, output_levels=0):
        super().__init__()

        in_channels = torch.tensor(in_channels, dtype=torch.int)
        self.register_buffer('in_channels', in_channels)
        out_channels = torch.tensor(out_channels, dtype=torch.int)
        self.register_buffer('out_channels', out_channels)
        num_trans_down = torch.tensor(num_trans_down, dtype=torch.int)
        self.register_buffer('num_trans_down', num_trans_down)
        max_channels = torch.tensor(max_channels, dtype=torch.int)
        self.register_buffer('max_channels', max_channels)
        first_channels = torch.tensor(first_channels, dtype=torch.int)
        self.register_buffer('first_channels', first_channels)
        output_levels = self._convert_output_levels(output_levels)
        self.register_buffer('output_levels', output_levels)

        # encoding/contracting
        inter_channels = (in_channels + first_channels) // 2
        self.cb0 = self._create_ib(in_channels, first_channels, inter_channels)
        in_channels = first_channels
        for i in range(self.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            inter_channels = (in_channels + out_channels) // 2
            td = create_avg_pool(2)
            setattr(self, 'td%d'%(i), td)
            cb = self._create_cb(in_channels, out_channels, inter_channels)
            setattr(self, 'cb%d'%(i+1), cb)
            in_channels = out_channels

        if self.num_trans_down in self.output_levels:
            out = self._create_out(out_channels)
            setattr(self, 'out%d'%self.num_trans_down, out)

        # decoding/expanding and output
        for i in reversed(range(self.output_levels[0], self.num_trans_down)):
            out_channels = getattr(self, 'cb%d'%i).out_channels
            tu = self._create_tu(in_channels, out_channels)
            setattr(self, 'tu%d'%i, tu)
            eb = self._create_eb(out_channels, out_channels, out_channels)
            setattr(self, 'eb%d'%i, eb)
            if i in self.output_levels:
                out = self._create_out(out_channels)
                setattr(self, 'out%d'%i, out)
            in_channels = out_channels

    def _convert_output_levels(self, levels):
        levels = levels if isinstance(levels, Iterable) else [levels]
        levels = torch.sort(torch.tensor(levels, dtype=torch.int))[0]
        if levels[-1] > self.num_trans_down:
            message = ('Output levels should be less or equal to the number '
                       'of transition down.')
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        levels = levels[levels<=self.num_trans_down]
        print(levels)
        return levels

    def _calc_out_channels(self, in_channels):
        """Calculate the number of output chennals of a block."""
        out_channels = min(in_channels * 2, self.max_channels)
        return out_channels

    def _create_ib(self, in_channels, out_channels, inter_channels):
        """Returns an input block"""
        raise NotImplementedError

    def _create_cb(self, in_channels, out_channels, inter_channels):
        """Returns a contracting block"""
        raise NotImplementedError

    def _create_tu(self, in_channels, out_channels):
        """Returns a trans up block"""
        raise NotImplementedError

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        """Returns a contracting block"""
        raise NotImplementedError

    def _create_out(self, in_channels):
        """Returns an output layler"""
        raise NotImplementedError

    def forward(self, input):
        outputs = dict()

        # encoding/contracting
        output = input
        shortcuts = list()
        for i in range(self.num_trans_down):
            output = getattr(self, 'cb%d'%i)(output)
            shortcuts.append(output)
            output = getattr(self, 'td%d'%i)(output)

        # bridge
        i = self.num_trans_down
        output = getattr(self, 'cb%d'%i)(output)
        if i in self.output_levels:
            outputs[i] = getattr(self, 'out%d' % i)(output)

        # decoding/expanding
        for i in reversed(range(self.output_levels[0], self.num_trans_down)):
            output = getattr(self, 'tu%d'%i)(output)
            output = getattr(self, 'eb%d'%i)(output, shortcuts[i])
            if i in self.output_levels:
                outputs[i] = getattr(self, 'out%d' % i)(output)

        if len(outputs) == 1:
            outputs = outputs[list(outputs.keys())[0]]

        return outputs


class UNet(_UNet):
    """The UNet.

    """
    def _create_ib(self, in_channels, out_channels, inter_channels):
        return InputBlock(in_channels, out_channels, inter_channels)

    def _create_cb(self, in_channels, out_channels, inter_channels):
        return ContractingBlock(in_channels, out_channels, inter_channels)

    def _create_tu(self, in_channels, out_channels):
        return TransUpBlock(in_channels, out_channels)

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return ExpandingBlock(in_channels, shortcut_channels, out_channels)

    def _create_out(self, in_channels):
        return create_k1_conv(in_channels, self.out_channels)
