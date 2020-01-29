#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
import torch
from pytorch_unet import UNet
from pytorch_layers import Config, Dim
from torchviz import make_dot


Config.dim = Dim.THREE
Config.show()

unet = UNet(1, 4, 3, 8, output_levels=0)
print(unet)

x = torch.rand(1, 1, 16, 16, 16)
y = unet(x)
if isinstance(y, dict):
    for k, v in y.items():
        print(k, v.shape)
    outputs = list()
    for v in y.values():
        outputs.append(torch.nn.functional.interpolate(v, size=x.shape[2:]))
    output = sum(outputs)
else:
    print(y.shape)
    output = y
dot = make_dot(output, params=dict(unet.named_parameters()))
dot.format = 'svg'
dot.render('unet')
