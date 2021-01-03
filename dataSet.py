from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch import nn
import numpy as np
from colorizers import *

class MD(nn.Module):
    def __init__(self, root, transform=None, train=None):
        
        self.root = root
        self.train = train
        self.dataIndex = os.listdir(os.path.join(root, train, "1"))
        if transform == True:
            self.transform = transform

    def __len__(self):
        return len(self.dataIndex)

    def load(self, x):
        input = np.asarray(Image.open(os.path.join(self.root, self.train, "1", x)))
        label = np.asarray(Image.open(os.path.join(self.root, self.train, "2", x)))
        if(input.ndim==2):
            input = np.tile(input[:,:,None],3)
        
        return input,label
    
    def __getitem__(self, idx):
        temp = self.dataIndex[idx]
        input, label = self.load(temp)
        label = transforms.ToTensor()(label) 
        return input, label
