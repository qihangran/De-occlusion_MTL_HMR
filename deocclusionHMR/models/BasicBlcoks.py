###############
# resnet block#
# resnet，注意恒等映射x和残差映射f(x),y=f(x)+x;
# 恒等映射存在通道数不一致的情况，因此恒等映射就有两种连接方式；
# 注意有building block和bottleneck block两种残差映射方式，后者主要是用来减少参数量的
################

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as sn
import torch.nn.functional as F
from math import sqrt
from torch.nn.parameter import Parameter
import numpy as np
#---------------------------------------------------------------------------
# 转移部分参数到另一个模型
#---------------------------------------------------------------------------
def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("missing keys in state_dict:{}".format(k))
    return state_dict
def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def transfer_model_frommodel(pretrained_model, model):
    #pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    pretrained_dict = transfer_state_dict(pretrained_model, model_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2/(in_ch * kernel_size[0] * kernel_size[1]))#ex:kernel size(3*3)

        self.weight = Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(x, self.weight*self.scale, self.bias, self.stride, self.padding)
#---------------------------------------------------------------------------
# init weight
#---------------------------------------------------------------------------
def init_weight(net, init_type='xavier', init_gain=0.02):
    """初始化网络权重"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

#---------------------------------------------------------------------------
# resnetblocks
#---------------------------------------------------------------------------

class resnetblock_sn(nn.Module):
    """ 残差块,BasicBlock结构,with spectral_norm--for discriminator"""
    def __init__(self, in_dim, out_dim, use_bias, downsample):
        super(resnetblock_sn, self).__init__()
        model = [sn(nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1, bias=use_bias)), nn.ReLU()]
        model += [sn(nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=use_bias))]

        self.F = nn.Sequential(*model)
        self.e = sn(nn.Conv2d(in_dim, out_dim, 1, stride=1, bias=use_bias))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.in_dim = in_dim
        self.out_dim = out_dim
        init_weight(self.F)
        init_weight(self.e)

    def forward(self, x):
        # --------shortcut----------------
        if self.in_dim == self.out_dim:
            if self.downsample:
                h_s = _downsample(x)
            else:
                h_s = x
        else:
            if self.downsample:
                h = _downsample(x)
                h_s = self.e(h)
            else:
                h_s = self.e(x)
        # --------------res--------------
        if self.downsample:
            h = _downsample(x)
            h_r = self.F(h)
        else:
            h_r = self.F(x)
        return self.relu(h_r + h_s)


class resnetblock(nn.Module):
    """ 残差块,BasicBlock结构--for d and g"""
    def __init__(self, in_dim, out_dim, use_bias, norm_layer):
        super(resnetblock, self).__init__()
        if in_dim == out_dim:# input dim = output dim and no downsapling
            dim = in_dim
            model = [nn.Conv2d(dim, dim, 3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU()]
            model += [nn.Conv2d(dim, dim, 3, padding=1, bias=use_bias), norm_layer(dim)]
        else:# more channels and downsampling
            model = [nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=use_bias), norm_layer(out_dim), nn.ReLU()]
            model += [nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=use_bias), norm_layer(out_dim)]
        self.F = nn.Sequential(*model)
        self.e = nn.Sequential(*[nn.Conv2d(in_dim, out_dim, 1, stride=1, bias=use_bias), norm_layer(out_dim)])
        self.relu = nn.ReLU(inplace=True)
        init_weight(self.F)
        init_weight(self.e)

    def forward(self, x):
        f = self.F(x)
        if f.size(1) == x.size(1):
            y = f + x
        else:
            y = f + self.e(x)
        return self.relu(y)

def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=False)

def _upsample_nearest(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h*2, w*2), mode='nearest')

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    #return nn.AvgPool2d(kernel_size=2)(x)
    h, w = x.size()[2:]
    return F.interpolate(x, scale_factor=0.5, mode='bilinear')

class resnetblock_upsample(nn.Module):
    """ may be upsampling--for generator"""
    def __init__(self, in_dim, out_dim, norm_layer, upsample=False):
        super(resnetblock_upsample, self).__init__()
        self.upsample = upsample
        self.in_dim = in_dim
        self.out_dim = out_dim

        model = [nn.Conv2d(in_dim, out_dim, 3, 1, padding=1), norm_layer(out_dim), nn.ReLU()]
        model += [nn.Conv2d(out_dim, out_dim, 3, 1, padding=1), norm_layer(out_dim)]
        self.model = nn.Sequential(*model)

        self.e = nn.Sequential(*[nn.Conv2d(in_dim, out_dim, 1, stride=1), norm_layer(out_dim)])
        self.relu = nn.ReLU(inplace=True)
        init_weight(self.model)
        init_weight(self.e)

    def forward(self, x):
        #--------shortcut----------------
        if self.in_dim == self.out_dim:
            if self.upsample:
                h_s = _upsample(x)
            else:
                h_s = x
        else:
            if self.upsample:
                h = _upsample(x)
                h_s = self.e(h)
            else:
                h_s = self.e(x)
        #--------------res--------------
        if self.upsample:
            h = _upsample(x)
            h_r = self.model(h)
        else:
            h_r = self.model(x)
        return self.relu(h_r + h_s)








