from .resnet_occ import resnet
from .fcresnet import fcresnet
from .hrnet_mannual import PoseHighResolutionNet
from .resnet_multiinput import resnet50, resnet18
from .basic_modules import BasicBlock

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'fcresnet':
        return fcresnet(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'hrnet':
        return PoseHighResolutionNet()
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet2':
        return resnet18(15)
    else:
        raise NotImplementedError('Backbone type is not implemented')

def create_backbone1(name):
    if name == 'resnet':
        return resnet50(in_channels=3, pretrained=True)
    else:
        raise NotImplementedError('Backbone type is not implemented')
