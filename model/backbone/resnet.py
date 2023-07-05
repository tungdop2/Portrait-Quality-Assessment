import torch
from torch import nn

from timm import create_model

class ResNetfromtimm(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.num_features
        self.model.fc = nn.Identity()

    def forward(self, x,
                return_features_only=True,
                return_all_stages=False,
    ):
        if return_features_only:
            return self.model(x)
        if return_all_stages:
            out = []
            
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            out.append(x)
            x = self.model.layer2(x)
            out.append(x)
            x = self.model.layer3(x)
            out.append(x)
            x = self.model.layer4(x)
            out.append(x)
            x = self.model.global_pool(x)
            out.append(x)
            return out
            

class ResNet18(ResNetfromtimm):
    def __init__(self):
        super().__init__(model_name='resnet18', pretrained=True)
        
class ResNet50(ResNetfromtimm):
    def __init__(self):
        super().__init__(model_name='resnet50', pretrained=True)
        
if __name__ == '__main__':
    model = ResNet18()
    print(model)
    x = torch.randn(1, 3, 1024, 1024)
    print(model(x).shape)