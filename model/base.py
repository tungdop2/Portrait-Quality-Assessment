import torch
from torch import nn
from pytorch_lightning import LightningModule
from timm.loss import LabelSmoothingCrossEntropy
from torchmetrics import Accuracy
from torchmetrics.regression import SpearmanCorrCoef

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
        

class BasePIQModel(LightningModule):
    def __init__(self, backbone, freeze_backbone=True, lr=1e-4, label_smoothing=0.1, alpha=0.5):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Quality head - regression
        self.quality_head = Mlp(self.backbone.n_features, output_size=1)
        # Scene head - classification
        self.scene_head = nn.Sequential(
            Mlp(self.backbone.n_features, output_size=4),
            nn.Softmax(dim=1)
        )
        
        self.quality_loss = nn.L1Loss()
        if label_smoothing > 0:
            self.scene_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.scene_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=4)
        self.val_acc = Accuracy(task='multiclass', num_classes=4)
        self.train_corr = SpearmanCorrCoef()
        self.val_corr = SpearmanCorrCoef()

        self.lr = lr
        self.alpha = alpha

    def forward(self, batch):
        x, target_quality, target_scene = self.get_inputs(batch)
        out = self.foward_features(x)
        quality = self.quality_head(out['quality_feat'])
        scene = self.scene_head(out['scene_feat'])
        return self.get_loss(quality, target_quality, scene, target_scene)
    
    def get_inputs(self, batch):
        x = batch['image']
        target_quality = batch['quality_score'].unsqueeze(1)
        target_scene = batch['scene_label']
        return x, target_quality, target_scene

    def foward_features(self, x):
        out = self.backbone(x, return_multiscale=False)
        quality_feat = out['multiscale_feat']
        scene_feat = out['semantic_feat']
        return {
            'quality_feat': quality_feat,
            'scene_feat': scene_feat
        }
    
    def get_loss(self, quality, target_quality, scene, target_scene):
        quality_loss = self.quality_loss(quality, target_quality)
        scene_loss = self.scene_loss(scene, target_scene)
        total_loss = self.alpha * quality_loss + (1 - self.alpha) * scene_loss
        
        if self.training:
            self.train_acc(scene, target_scene)
            self.train_corr(quality[:, 0], target_quality[:, 0])
        else:
            self.val_acc(scene, target_scene)
            self.val_corr(quality[:, 0], target_quality[:, 0])
            
        return total_loss, quality_loss, scene_loss
    
    def training_step(self, batch, batch_idx):
        loss, quality_loss, scene_loss = self.forward(batch)
        self.log('train_loss', loss)
        self.log('train_quality_loss', quality_loss)
        self.log('train_scene_loss', scene_loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_corr', self.train_corr, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, quality_loss, scene_loss = self.forward(batch)
        self.log('val_loss', loss)
        self.log('val_quality_loss', quality_loss)
        self.log('val_scene_loss', scene_loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]