from torchvision import transforms
import torch
import torch.utils.data as data
import PIL
import pandas as pd
from scipy import ndimage as ndi
import numpy as np
import os
import pickle

class SartoriousDataset(data.Dataset):
    def __init__(self, rootpath, preprocess = None):
        super(SartoriousDataset, self).__init__()
        
        self.rootpath = rootpath
        self.preprocess = preprocess

        self.img_preprocess = transforms.ToTensor()

        self.df = pd.read_csv(f'{self.rootpath}/train.csv')
        self.ids = self.df['id'].unique()
        self.N = len(self.ids)

        self.height = self.df['height'].unique()[0]
        self.width = self.df['width'].unique()[0]
        if 'cached_pairs_raw_dataset.pk' in os.listdir('./'):
            with open('cached_pairs_raw_dataset.pk', 'rb') as handle:
                self.cached_pairs = pickle.load(handle)
        else:
            self.cached_pairs = {}

    def get_pair(self, idfile):
        print(f'get_pair called with {idfile}')
        img = PIL.Image.open(f'{self.rootpath}/train/{idfile}.png')
        img = self.img_preprocess(img).squeeze()

        mask = torch.zeros((self.height,self.width))
        sub_df = self.df[self.df['id'] == idfile]
        for _, row in sub_df.iterrows():
            single_mask = np.zeros((self.height*self.width))
            annotation = np.asarray(list(map(int,row['annotation'].split()))).reshape((-1,2))
            for start, length in annotation:
                start = start-1
                single_mask[start:start+length] = 1
            single_mask = single_mask.reshape((self.height,self.width))
            single_mask = ndi.binary_fill_holes(single_mask)
            r_nz,c_nz = single_mask.nonzero()
            mask[r_nz,c_nz] = 1.0
        
        return img, mask

    def __getitem__(self, index):
        idfile = self.ids[index]

        if idfile in self.cached_pairs:
            img, mask = self.cached_pairs[idfile]
        else:
            img, mask = self.get_pair(idfile)
            self.cached_pairs[idfile] = (img, mask)

        comb = self.preprocess(torch.stack((img,mask)))
        return comb

    def __len__(self):
        return self.N