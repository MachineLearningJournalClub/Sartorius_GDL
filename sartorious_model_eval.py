from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

from pytorch_unet import UNet
from sartorious_dataset import SartoriousDataset
import torch.optim as optim
import os
import PIL

img_preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = UNet(num_input_ch = 1, num_output_ch = 2)

trained_models_path = 'trained_models'
for modelname in sorted(os.listdir(trained_models_path)):
    if '99' not in modelname:
        continue
    model.load_state_dict(torch.load(f'{trained_models_path}/{modelname}'))
    model.eval()
    for i,testid in enumerate(['7ae19de7bc2a','d8bfd1dafdc4','d48ec7815252']):
        img = PIL.Image.open(f'data/test/{testid}.png')
        img = img_preprocess(img).squeeze()
        x = img.unsqueeze(0).unsqueeze(0)
        yhat = model(x)
        plt.figure(1)
        plt.subplot(2,3,1+i)
        plt.imshow(x[0,:,:].cpu().squeeze().detach().numpy())
        plt.subplot(2,3,4+i)
        plt.imshow(F.softmax(yhat,dim=1)[0,1,:,:].cpu().squeeze().detach().numpy())
    plt.show()
