"""
ResNet backbone definition. Code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
#dropout not random. use attention to select channel weight
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:
            out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 添加遮挡相关处理，输出监督某些特征--正交约束layer3的卷积核+约束layer3响应处于不同位置
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, last_relu=False)
        #dropout再layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.atten = nn.Sequential(
        #     nn.Conv2d(1024, 32, kernel_size=1, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(32*16*16, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.Sigmoid()
        # )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 1024 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024 // 16, 1024, kernel_size=1),
            nn.Sigmoid()
        )


        self.atten1 = nn.Conv2d(1024, 32, kernel_size=1, bias=False)
        self.atten1_relu = nn.ReLU()
        self.atten2 = nn.Linear(32*16*16, 2048)
        self.atten2_relu = nn.ReLU(inplace=True)
        self.atten3 = nn.Linear(2048, 1024)
        self.atten4 = nn.Sigmoid()


    def _make_layer(self, block, planes, blocks, stride=1, last_relu = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes,last_relu=last_relu)) #最后一层不用relu，扩大空间，得到更多的正交向量
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def cosine_similarity_multi(self, a, b, rep="real"):
        """
        Compute the cosine similarity between two vectors

        Parameters:
        ----------
        a:  Tensor(N_a,D)
        b:  Tensor(N_b,D)
        rep: str
            Representation to compute cosine similarity: real | bipolar | tanh
        Return
        ------
        similarity: Tensor(N_a,N_b)
        """
        # sim_act = SIM_ACT[rep]
        a_normalized = torch.nn.functional.normalize(a, dim=1)

        a_norm = torch.norm(a, dim=1).unsqueeze(1) #nx1
        b_norm = torch.norm(b, dim=1).unsqueeze(0) #1xn
        norm_matrix = torch.einsum('ij,jm->im', [a_norm, b_norm])
        norm_matrix = 1. / (norm_matrix + 0.000001)

        b_T = torch.einsum('ki->ik', [b])
        corr = torch.einsum('ij,jm->im', [a, b_T])

        # Set non diagonal elements to 0
        corr *= torch.eye(a.shape[0], a.shape[0], device=a.device)
        norm_matrix *= torch.eye(a.shape[0], a.shape[0], device=a.device)
        similiarity = corr * norm_matrix

        return similiarity

    def softabs(self, x, steepness=10):
        return torch.sigmoid(steepness * (x - 0.5)) + torch.sigmoid(steepness * (-x - 0.5))


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # define response loss
        loss_diver2 = torch.tensor(0., device=x.device).float()
        response = x3.view(x.shape[0], x3.shape[1], -1)  # N*1024*256

        norm = torch.norm(response, dim=2).unsqueeze(2)
        norm_T = torch.einsum('kij->kji', [norm])
        norm_matrix = torch.einsum('kij,kjm->kim', [norm, norm_T])

        # Calculate autocorrelation matrix
        response_T = torch.einsum('kij->kji', [response])
        corr = torch.einsum('kij,kjm->kim', [response, response_T])
        norm_matrix = 1. / (norm_matrix+ 1e-6)

        # Set diagonal elements to 0
        corr *= (1 - torch.eye(1024, 1024, device=x.device))
        norm_matrix *= (1 - torch.eye(1024, 1024, device=x.device))

        # response loss
        loss_diver12 = corr * norm_matrix
        loss_diver22 = torch.pow(loss_diver12, 2)
        loss_diver32 = torch.nansum(loss_diver22, dim=(1,2))
        loss_diver2 = loss_diver32 / (x3.shape[1] * x3.shape[1])

        # attention dropout
        x3 = response.view(x.shape[0], x3.shape[1], 16, 16)
        x3a = self.se(x3)
        x3_human = x3 * x3a
        x3_occluder = x3 - x3_human

        # 注意力机制后，需要对特征再进行正交约束
        x3_human_sum = torch.nansum(x3_human, dim=1).view(x.shape[0], -1)
        x3_occluder_sum = torch.nansum(x3_occluder, dim=1).view(x.shape[0], -1)

        #for batch_i in range()
        similarities = self.cosine_similarity_multi(x3_human_sum, x3_occluder_sum)
        loss_sim = similarities


        x4_human = self.layer4(x3_human)
        x4_occluder = self.layer4(x3_occluder)

        return x4_human, x4_occluder, loss_diver2.mean(), loss_sim.mean()

def resnet(cfg, pretrained=True, **kwargs):
    num_layers = cfg.MODEL.BACKBONE.NUM_LAYERS
    # if num_layers == 18:
    #     return resnet18(pretrained=pretrained, **kwargs)
    # elif num_layers == 34:
    #     return resnet34(pretrained=pretrained, **kwargs)
    if num_layers == 50:
        return resnet50(pretrained=pretrained, **kwargs)
    # elif num_layers == 101:
    #     return resnet101(pretrained=pretrained, **kwargs)
    # elif num_layers == 152:
    #     return resnet152(pretrained=pretrained, **kwargs)





def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model

