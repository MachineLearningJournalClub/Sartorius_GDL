import h5py
import matplotlib.pyplot as plt

with h5py.File('mydataset.hdf5','r') as f:
    outputs = f['delta_map']
    inputs = f['train']
    for i in range(5):
       i = 1
       plt.figure(figsize = (12,12))
       #plt.subplot(1,2,1)
       #plt.imshow(outputs[i,:,:,0])
       #plt.subplot(1,2,2)
       plt.imshow(inputs[i,:,:])
       plt.show()