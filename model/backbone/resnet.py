import torch
from torch import nn

from timm import create_model

class ResNetfromtimm(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

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