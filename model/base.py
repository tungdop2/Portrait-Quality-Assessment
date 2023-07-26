import torch
from torch import nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from timm.loss import LabelSmoothingCrossEntropy

from torchmetrics import Accuracy
        

class BasePIQModel(LightningModule):
    def __init__(self, backbone, freeze_backbone=True, lr=1e-4, label_smoothing=0.1, alpha=0.5):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.quality_loss = nn.L1Loss()
        if label_smoothing > 0:
            self.scene_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.scene_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=4)
        self.val_acc = Accuracy(task='multiclass', num_classes=4)

        self.lr = lr
        self.alpha = alpha

    def forward(self, batch):
        x, target_quality, target_scene = self.get_inputs(batch)
        out = self.foward_features(x)
        quality = out['quality']
        scene = out['scene']
        return self.get_loss(quality, target_quality, scene, target_scene)
    
    def get_inputs(self, batch):
        x = batch['image']
        target_quality = batch['quality_score'].unsqueeze(1)
        target_scene = batch['scene_label']
        return x, target_quality, target_scene

    def foward_features(self, x):
        out = self.backbone(x)
        quality = out['quality']
        scene = out['scene']
        return {
            'quality': quality,
            'scene': scene
        }
    
    def get_loss(self, quality, target_quality, scene, target_scene):
        quality_loss = self.quality_loss(quality, target_quality)
        scene_loss = self.scene_loss(scene, target_scene)
        total_loss = self.alpha * quality_loss + (1 - self.alpha) * scene_loss
        
        if self.training:
            self.train_acc(scene, target_scene)
        else:
            self.val_acc(scene, target_scene)
            
        return total_loss, quality_loss, scene_loss
    
    def training_step(self, batch, batch_idx):
        loss, quality_loss, scene_loss = self.forward(batch)
        self.log('train_loss', loss)
        self.log('train_quality_loss', quality_loss)
        self.log('train_scene_loss', scene_loss)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute())
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, quality_loss, scene_loss = self.forward(batch)
        self.log('val_loss', loss)
        self.log('val_quality_loss', quality_loss)
        self.log('val_scene_loss', scene_loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.val_acc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    
class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])
    
    
class HyperPIQModel(BasePIQModel):
    def __init__(self, backbone, freeze_backbone=True, lr=1e-4, label_smoothing=0.1, alpha=0.5):
        super().__init__(backbone, freeze_backbone=True, lr=1e-4, label_smoothing=0.1, alpha=0.5)
        
        self.n_features = self.backbone.n_features
        
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(self.backbone.semantic_dim, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.n_features, 1, padding=(0, 0)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.scene_head = nn.Sequential(
            nn.Linear(self.n_features, 4),
            nn.Softmax(dim=1),
        )
        
        n_features = self.n_features
        self.fc1w = nn.Linear(self.n_features, n_features ** 2 // 2)
        self.fc1b = nn.Linear(self.n_features, n_features // 2)

        n_features = n_features // 2
        self.fc2w = nn.Linear(self.n_features, n_features ** 2 // 2)
        self.fc2b = nn.Linear(self.n_features, n_features // 2)

        n_features = n_features // 2
        self.fc3w = nn.Linear(self.n_features, n_features ** 2 // 2)
        self.fc3b = nn.Linear(self.n_features, n_features // 2)

        n_features = n_features // 2
        self.fc4w = nn.Linear(self.n_features, n_features ** 2 // 2)
        self.fc4b = nn.Linear(self.n_features, n_features // 2)
        
        n_features = n_features // 2
        self.fc5w = nn.Linear(self.n_features, n_features)
        self.fc5b = nn.Linear(self.n_features, 1)
        
    def foward_features(self, x):
        out = self.backbone(x)
        multiscale_feat = out['multiscale_feat']
        semantic_feat = out['semantic_feat']
        # print(multiscale_feat.shape)
        # print(semantic_feat.shape)
        
        out = self.semantic_conv(semantic_feat)
        out_flat = out.view(out.shape[0], -1)
        scene = self.scene_head(out_flat)
        
        n_features = self.n_features
        w1 = self.fc1w(out_flat)
        w1 = w1.view(-1, n_features // 2, n_features, 1, 1)
        b1 = self.fc1b(out_flat)
        target_net1 = TargetFC(w1, b1)
        target_net1.requires_grad_(False)
        
        n_features = n_features // 2
        w2 = self.fc2w(out_flat)
        w2 = w2.view(-1, n_features // 2, n_features, 1, 1)
        b2 = self.fc2b(out_flat)
        target_net2 = TargetFC(w2, b2)
        target_net2.requires_grad_(False)
        
        n_features = n_features // 2
        w3 = self.fc3w(out_flat)
        w3 = w3.view(-1, n_features // 2, n_features, 1, 1)
        b3 = self.fc3b(out_flat)
        target_net3 = TargetFC(w3, b3)
        target_net3.requires_grad_(False)
        
        n_features = n_features // 2
        w4 = self.fc4w(out_flat)
        w4 = w4.view(-1, n_features // 2, n_features, 1, 1)
        b4 = self.fc4b(out_flat)
        target_net4 = TargetFC(w4, b4)
        target_net4.requires_grad_(False)
        
        n_features = n_features // 2
        w5 = self.fc5w(out_flat)
        w5 = w5.view(-1, 1, n_features, 1, 1)
        b5 = self.fc5b(out_flat)
        target_net5 = TargetFC(w5, b5)
        target_net5.requires_grad_(False)
        
        multiscale_feat = multiscale_feat.view(multiscale_feat.shape[0], multiscale_feat.shape[1], 1, 1)
        out = target_net1(multiscale_feat)
        out = F.relu(out)
        out = target_net2(out)
        out = F.relu(out)
        out = target_net3(out)
        out = F.relu(out)
        out = target_net4(out)
        out = F.relu(out)
        out = target_net5(out).squeeze()
        
        return {
            'quality': out,
            'scene': scene
        }

    