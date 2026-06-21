# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 04:23:51 2024

@author: PRIA
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model
# =============================================================================

#from timm.layers import trunc_normal_, DropPath

from timm.models.layers import trunc_normal_, DropPath
from timm.models import register_model

from .convnext import Block, LayerNorm





class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        

        self.output_layer1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2, 32, kernel_size=3, stride=1, padding=1),#, output_padding=1), #256. 
            nn.ReLU(), # added on 31 March 2026
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),#, output_padding=1), #256, 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),#, output_padding=1), # 128, 64
            nn.ReLU(),
            nn.Conv2d(32, 6, kernel_size=1),
            )




    def forward(self, x):
        
        #print('\n x:', x.shape)
        
        self.units_out = 3
        
        #y1 =  self.cd1(x, nn.Sequential(self.output_layer1, self.relu)) # concrete dropout
        
        y1 = self.output_layer1(x)
        
# =============================================================================
#         print('\n y1:', y1.shape)
# =============================================================================
 
        a, b = torch.split(y1, 3, dim=1) # torch.split(y1, self.units_out, dim=1)#dim=-1)


        #out = torch.cat(tensors=(a, b), dim=1)

        return a,b








    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




    #------------------------------------------------------
    def evidence(self, x):
            softplus = nn.Softplus(beta=1)
            return softplus(x)
                           
                    
    def evidence2(self, x):
            softplus = nn.Softplus(beta=1, threshold=1)#5)
            return softplus(x)
    

        

################################   


           
##############################################################################

class Conv2DNormalGamma(nn.Module):
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv2DNormalGamma, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 4 , kernel_size=3)

    def forward(self, x):
        return self.conv(x)



    ##############################################################################





class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depth=18, dim=384, drop_path_rate=0., 
                 layer_scale_init_value=0, head_init_scale=1.,
                 ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1) # orig kernel_size=16, orig stride=16
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])

        self.norm = LayerNorm(dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)





# =============================================================================
        self.csrnet = CSRNet()
        self.apply(self._init_weights)
        

# =============================================================================





    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        
        #print('\n x_input: ', x.shape)
        x = self.stem(x)
        #print('\n x_stem: ', x.shape)
        x = self.blocks(x)
        #print('\n x1: ',x.shape)
        
        #return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return x#.mean(dim=1) #self.norm(x.mean(dim=1)) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        #print('\n x1: ',x.shape)
        
# =============================================================================
#         if x.shape[0]>1024:
#             x = x.resize
# =============================================================================
        
        x = self.forward_features(x)
        #print('\n x2: ',x.shape)
        #x = self.head(x)
        #x = x.sum() # added
        
        
        a, b = self.csrnet(x)
        
        #a, b = torch.split(x, 3, dim=1) 
        #print('\n ---------------\n', a.shape)

        #out = self.ConvGamma(x)
        #x=torch.unsqueeze(x, 0)
        #x = torch.nn.functional.interpolate(x, scale_factor=16)
        
        return a, b
    
    
    
 

 

@register_model
def convnext_isotropic_base(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=1, dim=2, **kwargs) #depth 4 previously

    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],  strict=False)
    return model

