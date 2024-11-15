import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys
import os
path_now = sys.path[0]
sys.path.insert(0, os.path.join(path_now, '../'))
from models.builder import EncoderDecoder as segmodel
from configs.config_UrbanLF_Syn import config
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table


network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)

network.eval()

# no difference found.
n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")
flops = network.flops()
print(f"number of GFLOPs: {flops / 1e9}")