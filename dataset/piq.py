import os
import numpy as np
import pandas as pd
from typing import Optional

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from prefetch_generator import BackgroundGenerator

ROOT = 'PIQ2023'
IMAGE_DIR = 'PIQ2023/Images'
SPLIT_METHODS = ['Device', 'Scene']
HEADS = ['Details', 'Exposure', 'Overall']

CSV_PATH = 'PIQ2023/Test split/{} Split/{}Split_{}_Scores_{}.csv'

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class PIQDataset(Dataset):
    def __init__(self, transform, split_method, head, split):
        self.transform = transform

        csv_path = CSV_PATH.format(split_method, split_method, split, head)
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = Image.open(os.path.join(IMAGE_DIR, item['IMAGE PATH'].replace('\\', '/')))
        if self.transform:
            image = self.transform(image)
            
        sample = {
            'image': image,
        }
            
        return sample


class PIQDataModule(LightningDataModule):
    def __init__(self, size=512, batch_size: int = 4, num_workers: int = 2, transform=None,
            split_method='Device', head='Details', **kwargs):
        
        assert split_method in SPLIT_METHODS
        assert head in HEADS
        
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
            
        self.split_method = split_method
        self.head = head
        self.kwargs = kwargs

    def prepare_data(self): 
        self.train_dataset = PIQDataset(self.transform, self.split_method, self.head, split='Train')
        self.val_dataset = PIQDataset(self.transform, self.split_method, self.head, split='Test')

    def train_dataloader(self):
        return DataLoaderX(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, **self.kwargs)
    
    def val_dataloader(self):
        return DataLoaderX(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, **self.kwargs)
    
if __name__ == '__main__':
    dm = PIQDataModule()
    dm.prepare_data()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    for batch in train_loader:
        print(batch['image'].shape)
        break
    for batch in val_loader:
        print(batch['image'].shape)
        break
