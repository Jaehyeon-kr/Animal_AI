

import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torchvision.io import read_image
import matplotlib.pyplot as plt


import pandas 
import os 
import pandas as pd 

class AnimnalDataset(Dataset):

    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        
        self.img_labels = pd.read_csv(annotations, names=["path","label"])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self,):
        return len(self.img_labels) # 1 x N (img_numbers)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path) # tensor로 변환
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image" : image, "label" : label}

        return image, label 
    
    
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform= ToTensor() 
)


test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform= ToTensor() 
)

train_loader = DataLoader(training_data, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle=True)


next(iter(train_loader))