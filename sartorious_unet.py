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

# preprocess = transforms.Compose([
#     transforms.Resize((224,224)),
# ])
preprocess = transforms.Compose([
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])
train_dataset = SartoriousDataset('./data', preprocess=preprocess)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=6)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = UNet(num_input_ch = 1, num_output_ch = 2).to(device)
# weight = torch.FloatTensor([1,8]).to(device)
# criterion = nn.CrossEntropyLoss(weight = weight)

model = UNet(num_input_ch = 1, num_output_ch = 1).to(device)
weight = torch.FloatTensor([1,8]).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters())

for epoch in range(10):    
    total_epoch_loss = 0.0
    model.train()
    for i, data in enumerate(dataloader, 0):
        data = data
        x, y = data[:,0,:,:], data[:,1,:,:]
        x = x.unsqueeze(1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x).squeeze()
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()
    print(f'[mean loss epoch {epoch + 1}]: {total_epoch_loss / train_dataset.N:.3f}')
    #if epoch % 5 == 4:
    #    torch.save(model.state_dict(), f'sartorious_UNET_weights_epoch_{epoch}.pth')

print('Finished Training')

# import pickle
# with open('cached_pairs_raw_dataset.pk', 'wb') as handle:
#     pickle.dump(train_dataset.cached_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#torch.save(model.state_dict(), 'sartorious_UNET_weights.pth')

model.eval()
yhat = model(x).squeeze()
plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(x[0,:,:].cpu().squeeze().detach().numpy())
plt.subplot(1,3,2)
plt.imshow(y[0,:,:].cpu().squeeze().detach().numpy())
plt.subplot(1,3,3)
plt.imshow(torch.sigmoid(yhat)[0,:,:].cpu().squeeze().detach().numpy())
plt.show()

