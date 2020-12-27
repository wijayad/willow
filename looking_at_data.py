# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:39:28 2020

@author: Daniel
"""
import h5py
import numpy as np
#accessing file
filename = 'C:\\Users\\Daniel Wijaya\\Downloads\\experiment_C20200403-113421.h5'

data = h5py.File(filename,"r")
#prints the keys of the dataset
print("Keys- %s" % data.keys())

#aux_data = data["aux_data"]
#print(np.shape(aux_data))
channel_data = data["channel_data"]
print(np.shape(channel_data))
#chip_live = data["chip_live"]
#print(np.shape(chip_live))
#sample_index = data["sample_index"]
#print(np.shape(sample_index))

#actual sensor data only from first 256 columns of channel_data
channel_reading = channel_data[:,0:256]
#normalize and center
channel_reading = channel_reading - channel_reading.mean(axis=0)
channel_reading = channel_reading / channel_reading.std(axis=0)

del data, channel_data

from sklearn.decomposition import FastICA
ica = FastICA(n_components=256, whiten = True, random_state = 1, max_iter = 10000, tol=0.01)
#try with algorithm = 'deflation'
#try seeding FastICA with an integer so that can check consistency between running it for different intervals
#channel_reading_ica = ica.fit_transform(channel_reading)
#need to split into two since unable to allocate memory, splitting unfortunately doesnt work as algorithm does not converge
channel_reading_ica = ica.fit_transform(channel_reading[0:1000000,:])
channel_reading_ica2 = ica.fit_transform(channel_data[1000000:-1,:])
channel_reading_ica_full = np.concatenate((channel_reading_ica, channel_reading_ica2),axis=0)


from matplotlib import pyplot as plt
plt.figure()
plt.plot(channel_reading[:,0])
#plt.plot(channel_reading[:,:]) #overlays all channels