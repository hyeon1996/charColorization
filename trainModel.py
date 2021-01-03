# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:48:28 2020

@author: 조용현
"""

import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from dataSet import *
import copy
from colorizers import *
import torch.utils.model_zoo as model_zoo

def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

criterion = nn.CrossEntropyLoss()

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")

device_num = "0"
batch_size = 1
num_epochs = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = device_num
device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print("GPU_number : ", device_num, '\tGPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == '__main__':
    
    dataset = {'train': MD(root="/home/whdydgus9/Desktop/images", train = "train"),
                'val': MD(root="/home/whdydgus9/Desktop/images", train="valid")}
    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=batch_size),
        'val': DataLoader(dataset['val'], batch_size=batch_size)
    }

    save_path = './model'
    os.makedirs(save_path, exist_ok=True)
    
    model=eccv16(pretrained=True)
    model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location=device,check_hash=True))
    model.to(device)
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    
    
    for epoch in range(num_epochs):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------------------' * 10)
        now = time.time()

        if (epoch + 1) % valtest == 0:
            uu = ['train', 'val']
        else:
            uu = ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            num_ = 0

            
            for inputs, labels in dataloaders[phase]:                              
                inputs, labels = torch.squeeze(inputs), torch.squeeze(labels)
                npinputs = inputs.numpy()
                labels = labels.to(device)
                input_og, input_rs = preprocess_img(npinputs, HW=(256,256))
                input_rs = input_rs.to(device)
                out_img = postprocess_tens(input_og, model(input_rs).cpu())

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    out_img = torch.from_numpy(out_img)
                    out_img = out_img.permute(2,0,1)
                    out_img = out_img.to(device)
                    
                    loss = L2_loss(out_img, labels)
                    loss.requires_grad = True
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = metrics['loss'] / epoch_samples

            print("Loss :", epoch_loss)

            # deep copy the model

            savepath = save_path + '/new_{}_L1_{}_E_{}.pth'
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_loss, epoch))

            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_loss, epoch))

        print(time.time() - now)
