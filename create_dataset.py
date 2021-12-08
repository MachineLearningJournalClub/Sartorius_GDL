import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import h5py


df = pd.read_csv('train.csv')

output_dataset_file = h5py.File('mydataset.hdf5','w')

height = 520
width = 704

output_dataset = output_dataset_file.create_dataset('delta_map',
(len(df.groupby('id')),height,width,3), dtype = 'f'
)

input_dataset = output_dataset_file.create_dataset('train',
(len(df.groupby('id')),height,width), dtype = 'f'
)

print('letto df')

for i, (idfile, group) in enumerate(df.groupby('id')):
    print(f'inizio file {i}')
    tot_mask = np.zeros((height,width,3))
    for _, row in group.iterrows():
        single_mask = np.zeros((height*width),dtype=np.uint8)
        annotation = np.asarray(list(map(int,row['annotation'].split()))).reshape((-1,2))
        for start, length in annotation:
            start = start-1
            single_mask[start:start+length] = 1
        single_mask = single_mask.reshape((height,width))
        single_mask = ndi.binary_fill_holes(single_mask)
        x,y = single_mask.nonzero()
        tot_mask[x,y,0] = y.mean() - y
        tot_mask[x,y,1] = x.mean() - x
        tot_mask[x,y,2] = 1
    output_dataset[i,...] = tot_mask
    input_dataset[i,...] = plt.imread(f'train/{idfile}.png')

output_dataset_file.close()