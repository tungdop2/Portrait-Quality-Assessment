import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

def build_head(n_features, type='linear', hidden_size=512, num_layers=1):
    # build head for regression task
    assert type in ['linear', 'mlp']
    if type == 'linear':
        head = nn.Linear(n_features, 1)
    elif type == 'mlp':
        head = []
        for i in range(num_layers):
            if i == 0:
                head.append(nn.Linear(n_features, hidden_size))
                head.append(nn.ReLU())
            else:
                head.append(nn.Linear(hidden_size, hidden_size))
                head.append(nn.ReLU())
                
        head.append(nn.Linear(hidden_size, 1))
        head = nn.Sequential(*head)
    # elif type == 'transformer':
    #     pass
    else:
        raise NotImplementedError
    
    return head

class PIQModel(LightningModule):
    def __init__(self, backbone, freeze_backbone=True, lr=1e-4, head='linear'):
        assert head in ['linear', 'mlp']
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
                

        self.head = build_head(n_features=self.backbone.n_features, type=head)

        self.lr = lr
        self.loss = self.get_loss()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def get_loss(self):
        return nn.L1Loss()
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['score'].unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['score'].unsqueeze(1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]