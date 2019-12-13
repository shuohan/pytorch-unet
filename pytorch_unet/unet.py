# -*- coding: utf-8 -*-
  
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_engine.layers import create_avg_pool, create_1k_conv

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
                 first_channels, max_channels=1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.num_trans_down = num_trans_down
        self.max_channels = max_channels

        # encoding/contracting
        inter_channels = (in_channels + first_channels) // 2
        self.cb0 = self._create_ib(in_channels, first_channels, inter_channels)
        in_channels = first_channels
        for i in range(self.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            inter_channels = (in_channels + out_channels) // 2
            setattr(self, 'td%d'%(i), create_avg_pool(2))
            cb = self._create_cb(in_channels, out_channels, inter_channels)
            setattr(self, 'cb%d'%(i+1), cb)
            in_channels = out_channels

        # decoding/expanding
        for i in range(self.num_trans_down):
            shortcut_ind = self.num_trans_down - i - 1
            out_channels = getattr(self, 'cb%d'%shortcut_ind).out_channels
            setattr(self, 'tu%d'%i, self._create_tu(in_channels, out_channels))
            eb = self._create_eb(out_channels, out_channels, out_channels)
            setattr(self, 'eb%d'%i, eb)
            in_channels = out_channels

        # output
        self.out = self._create_out(out_channels)

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
        # encoding/contracting
        output = input
        shortcuts = list()
        for i in range(self.num_trans_down+1):
            output = getattr(self, 'cb%d'%i)(output)
            if i < self.num_trans_down:
                shortcuts.insert(0, output)
                output = getattr(self, 'td%d'%(i))(output)

        # decoding/expanding
        for i, shortcut in enumerate(shortcuts):
            output = getattr(self, 'tu%d'%i)(output)
            output = getattr(self, 'eb%d'%i)(output, shortcut)

        # output
        self.out(output)

        return output


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
        return create_1k_conv(in_channels, self.out_classes)
