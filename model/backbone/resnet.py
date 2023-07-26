import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Mlp(nn.Module):
    def __init__(self, n_features, hidden_size=512, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
        return self.layers(x)
    

class LDA(nn.Module):
    def __init__(self, in_chn, middle_chn, out_chn, conv=True):
        super(LDA, self).__init__()
        self.in_chn = in_chn
        self.middle_chn = middle_chn
        self.out_chn = out_chn
        self.conv = conv
        
        if conv:
            self.conv = nn.Conv2d(in_chn, middle_chn, kernel_size=1, stride=1, padding=0, bias=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(middle_chn, out_chn)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_chn, out_chn)
        
    def forward(self, x):
        if self.conv:
            x = self.conv(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, hyper=False):
        super(ResNetBackbone, self).__init__()
        
        self.inplanes = 64
        self.hyper = hyper
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        if not self.hyper:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.scene_head = nn.Sequential(
                Mlp(256, hidden_size=512, output_size=4),
                nn.Softmax(dim=1),
            )
            self.quality_head = Mlp(256, hidden_size=512, output_size=1)
        else:
            self.n_features = 224
            self.semantic_dim = 2048
            # local distortion aware module
            self.lda1 = LDA(256, 16, 16, conv=True)
            self.lda2 = LDA(512, 32, 32, conv=True)
            self.lda3 = LDA(1024, 64, 64, conv=True)
            self.lda4 = LDA(2048, 2048, self.n_features - 16 - 32 - 64, conv=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        
        if not self.hyper:
            feat = self.avgpool(x).view(x.size(0), -1)
            out['quality'] = self.quality_head(feat)
            out['scene'] = self.scene_head(feat)
        else:
            lda_1 = self.lda1(x)
            x = self.layer2(x)
            lda_2 = self.lda2(x)
            x = self.layer3(x)
            lda_3 = self.lda3(x)
            x = self.layer4(x)
            lda_4 = self.lda4(x)
            
            vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
            out['multiscale_feat'] = vec
            out['semantic_feat'] = x

        return out

# def resnet18_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    

def resnet50_backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
if __name__ == '__main__':
    model = resnet50_backbone(hyper=True)
    # print(model)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out['multiscale_feat'].shape)
    print(out['semantic_feat'].shape)