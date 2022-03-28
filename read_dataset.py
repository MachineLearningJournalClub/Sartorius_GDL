import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np

with h5py.File('mydataset.hdf5','r') as f:
    outputs = torch.Tensor(np.asarray(f['delta_map']))
    inputs = torch.Tensor(np.asarray(f['train']))
    yolo = outputs[:,:,:,-1]
    print((yolo > 0).sum())
    print((yolo <= 0).sum())