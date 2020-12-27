# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:51:44 2020
This code is used to create a visualization of the map between all of the 
channel readings in the granular region and one (for now) channel reading in 
the molecular region

requires file outputs from the SpikeDetection.py script
@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

#%% open file
path = 'C:\\Users\\Daniel Wijaya\\Downloads\\'
spikesDF = pd.read_csv(path+'experiment_C20200330-162815_spikes_in_s2.csv')
spikesDict = spikesDF.to_dict(orient='list')
#for now consider only shank 0, simple spikes for now (S0, SS & GC in string)

#this will serve as source of outputs(y), gives time stamps for positive examples
spikesDictS0_SS = {key:[element for element in value if np.isnan(element)==False] 
                   for key,value in spikesDict.items() if 
                   key.find('S0')>=0 and key.find('SS')>=0}

#this serves as souce of inputs (x), for both positive and negative examples
spikesDict_GC = {key:[element for element in value if np.isnan(element)==False]
                 for key,value in spikesDict.items() if
                 key.find('GC')>=0}
for key, values in spikesDictS0_SS.items():
    print(str(key) + ' ' + str(len(values)))
for key, values in spikesDict_GC.items():
    print(str(key) + ' ' + str(len(values)))

#%% functions
    
#data creation
# each example involves an input array and output label
# input array is formed by 1 second of all granular channel readings
# output label is 1 if spike occurs in chosen channel in molecular layer
# or 0 if spike did not occur
    
#creation of data when spike occurs (label 1)
def createDataPos(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, 
                  first = True):
    #creates positive examples (ie. for when spike occurs in molecular channel)
    sensorList = []
    if subdiv == 1000:
        m = len(Y_times)
        noSensors = len(inputSensorSpikes)
        n = noSensors*subdiv
        X = np.zeros(shape = (m,n))
        numSpikes = np.zeros(shape = (m,n))
        for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
            sensorList+=[sensor]
            for yIter, y_t in enumerate(Y_times):
                tempArr = np.array([round((element-y_t+1)*subdiv) for 
                                element in inputSensorSpikes[sensor] if 
                                element > y_t-1 and 
                                element <= y_t])
                if len(tempArr)!=0:
                    X[yIter, sensorIter*subdiv + tempArr-1] = 1
    else:
        m = len(Y_times)
        noSensors = len(inputSensorSpikes)
        n = noSensors*subdiv
        X = np.zeros(shape = (m,n))
        numSpikes = np.zeros(shape = (m,n))
        for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
            sensorList+=[sensor]
            for yIter, y_t in enumerate(Y_times):
                for lv1 in range(subdiv):
                    t_i = round(y_t-t_window*(subdiv-lv1)/subdiv,3)
                    t_f = round(y_t-t_window*(subdiv-lv1-1)/subdiv,3)
                    tempArr = np.divide(np.array([element for element in 
                                        inputSensorSpikes[sensor] if
                                        element >=t_i and element <=t_f]) - t_i,
                                        t_f - t_i) 
                    numSpikes[yIter, subdiv*sensorIter+lv1] = len(tempArr)
                    if numSpikes[yIter, subdiv*sensorIter+lv1] != 0:
                        if first == True:
                            X[yIter, lv1 + subdiv*sensorIter] = tempArr[0]
                        else: #last value
                            X[yIter, lv1 + subdiv*sensorIter] = tempArr[-1]
                            #consider first spike in time range
    return X, numSpikes, sensorList

#creation of non-spike data (label 0)
def createDataNeg(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, 
                  first = True):
    # creates negative examples for chosen molecular channel (ie. no activation 
    # in chosen moelcular channel)
    # to avoid unbalanced data, setup problem to have same number of negative 
    # and positive examples
    sensorList = []
    Rand_times = np.sort(
        np.round(
            np.random.rand(len(Y_times))*(Y_times[-1]-t_window) + t_window, 3
            )
        )
    for lv1, t in enumerate(Rand_times):
        while t in Y_times:
            Rand_times[lv1] = np.random.rand()
            t = Rand_times[lv1]
    if subdiv == 1000:
        m = len(Y_times)
        noSensors = len(inputSensorSpikes)
        n = noSensors*subdiv
        X = np.zeros(shape = (m,n))
        numSpikes = np.zeros(shape = (m,n))
        for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
            sensorList+=[sensor]
            for yIter, y_t in enumerate(Rand_times):
                tempArr = np.array([round((element-y_t+1)*subdiv) for 
                                element in inputSensorSpikes[sensor] if 
                                element > y_t-1 and 
                                element <= y_t]).astype(int)
                if len(tempArr)!=0:
                    X[yIter, sensorIter*subdiv + tempArr-1] = 1
    else:
        m = len(Y_times)
        noSensors = len(inputSensorSpikes)
        n = noSensors*subdiv
        X = np.zeros(shape = (m,n))
        numSpikes = np.zeros(shape = (m,n))
        for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
            #for each time the output activated
            sensorList += [sensor]
            for yIter, y_t in enumerate(Rand_times):
                for lv1 in range(subdiv):
                    t_i = round(y_t-t_window*(subdiv-lv1)/subdiv,3)
                    t_f = round(y_t-t_window*(subdiv-lv1-1)/subdiv,3)
                    tempArr = np.divide(np.array([element for element in 
                                        inputSensorSpikes[sensor] if
                                        element >=t_i and element <=t_f]) - t_i,
                                        t_f - t_i) 
                    numSpikes[yIter,lv1+subdiv*sensorIter]=len(tempArr)
                    if numSpikes[yIter,lv1+subdiv*sensorIter] != 0:
                        if first == True:
                            X[yIter, lv1+subdiv*sensorIter] = tempArr[0]
                        else:
                            X[yIter, lv1+subdiv*sensorIter] = tempArr[-1]
    return X, numSpikes, sensorList, Rand_times
#%% data creation
# As channel 112 has the most neural activity, use this to test hypothesis
# look at 1s time range prior to activation time, so do not include first 1s
chosen_channel = 112
Y_times = [round(element,3) for element in \
           spikesDictS0_SS['S0_C{}_SS'.format(chosen_channel)] if element > 1]

n = 1000
X_pos, numSpikes_pos, sensorList_pos = createDataPos(Y_times, 
                                                     spikesDict_GC,
                                                     subdiv = n)
X_neg, numSpikes_neg, sensorList_neg, Neg_times = createDataNeg(Y_times,
                                                                spikesDict_GC,
                                                                subdiv = n)
Y_pos = np.ones(shape=(len(Y_times),1))
Y_neg = np.zeros(shape=(len(Y_times),1))

X = np.concatenate((X_pos,X_neg),axis = 0)
Y = np.concatenate((Y_pos,Y_neg),axis = 0)

Data = np.concatenate((X,Y),axis=1)
np.random.shuffle(Data)
X = Data[:,:-1]
Y = np.reshape(Data[:,-1], newshape = (np.shape(Y)))

# =============================================================================
# #split into train and test data, unnecessary
# X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size = 0.2)
# =============================================================================
noInputs = len(spikesDict_GC)
#%% model parameters and model creation
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 5e-4,
    decay_steps=10000,
    decay_rate=0.9,
)

#create model
# note that performance has very little performance difference regardless of
# dropout value. This being said, highest validation accuracies occur with high
# dropout values (~0.8). Neurological hypothesis for this is the
# overcompleteness of neural structures
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dropout(0.8))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule, epsilon = 1e-6),
    loss = "binary_crossentropy", #use categorical_crossentropy if using one-hot encodings
    metrics = ["accuracy"])

#%%training
early_stop = tf.keras.callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 10,
    restore_best_weights = True,
)

history = model.fit(X, Y, batch_size = 128, epochs = 50, 
                    callbacks = [early_stop],
                    validation_split = 0.2)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['acc', 'val_acc']].plot()
#%% visualzing weights of relationship and average activity profile 
# shows approximate relationship between activations at certain times in the
# granular region. Be careful to note this does linearly represent relationship
# between regions due to the sigmoid non-linearity.

weights = model.get_weights()
plt.figure(figsize=(7, 10))
plt.imshow(np.reshape(weights[0], (noInputs,n)), aspect = 'auto')
plt.colorbar()
plt.title('weight matrix')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.xticks(list(range(1,1000,200)), list(range(-1000,0,200)))
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

plt.figure(figsize=(7, 10))
plt.imshow(np.reshape(abs(weights[0]), (noInputs,n)), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix)')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

plt.figure(figsize=(7, 10))
pos_weights = np.reshape(weights[0], (noInputs,n)).copy()
pos_weights[pos_weights<0]=0
plt.imshow(pos_weights, aspect = 'auto')
plt.colorbar()
plt.title('weight matrix, positives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

plt.figure(figsize=(7, 10))
neg_weights = np.reshape(weights[0], (noInputs,n)).copy()
neg_weights[pos_weights>0]=0
plt.imshow(abs(neg_weights), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix), negatives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

#plot average activation
plt.imshow(np.reshape(np.sum(X_pos, axis = 0)/np.shape(X_pos)[0], 
                      (noInputs,n)), aspect = 'auto')
plt.colorbar()
plt.title('normalized average activation')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_pos)), sensorList_pos)
