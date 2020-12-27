# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:44:07 2020

@author: Daniel
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout
import keras


spikesDF = pd.read_csv('C:\\Users\\Daniel Wijaya\\Downloads\\experiment_C20200330-162815_spikes_in_s2.csv')
spikesDict = spikesDF.to_dict(orient='list')
#for now consider only shank 0, simple spikes for now (S0, SS & GC in string)

#this will serve as SOURCE of outputs(y), gives time stamps for positive examples
spikesDictS0_SS = {key:[element for element in value if np.isnan(element)==False] 
                   for key,value in spikesDict.items() if 
                   key.find('S0')>=0 and key.find('SS')>=0}

#this serves as SOURCE of inputs (x), for both positive and negative examples
spikesDict_GC = {key:[element for element in value if np.isnan(element)==False]
                 for key,value in spikesDict.items() if
                 key.find('GC')>=0}
for key, values in spikesDictS0_SS.items():
    print(str(key) + ' ' + str(len(values)))
for key, values in spikesDict_GC.items():
    print(str(key) + ' ' + str(len(values)))
#notice that channel 112 has most, so try with channel 112 for now
#look at 1s time range prior to activation time, therefore dont include first 1s
Y_times = [round(element,3) for element in spikesDictS0_SS['S0_C112_SS'] if element > 1]

def createDataPos(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, first = True):
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
def createDataNeg(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, first = True):
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
#shuffle, not sure if necessary

Data = np.concatenate((X,Y),axis=1)
np.random.shuffle(Data)
X = Data[:,:-1]
Y = np.reshape(Data[:,-1], newshape = (np.shape(Y)))

#split into train and test data
#do a 80-20% split
X_train = X[:round(len(X)*0.8),:]
X_test = X[round(len(X)*0.8):,:]
Y_train = Y[:round(len(X)*0.8),:]
Y_test = Y[round(len(X)*0.8):,:]

noInputs = len(spikesDict_GC)

#model 1
inputs_1_1 = Input(shape=(noInputs*n,))

x_1_1 = Dropout(0.3)(inputs_1_1)
outputs_1_1 = Dense(1, activation = 'sigmoid')(x_1_1)

model_1_1 = Model(inputs = inputs_1_1, outputs = outputs_1_1)
model_1_1.summary()

model_1_1.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_1_1 = model_1_1.fit(X_train, Y_train, batch_size = 128, epochs = 30)
test_scores_1_1 = model_1_1.evaluate(X_test,Y_test, verbose = 2)
print(test_scores_1_1)

weights_1_1 = model_1_1.get_weights()
plt.imshow(np.reshape(weights_1_1[0], (noInputs,n)), aspect = 'auto')
plt.colorbar()
plt.title('weight matrix')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

plt.imshow(np.reshape(abs(weights_1_1[0]), (noInputs,n)), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix)')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

pos_weights = np.reshape(weights_1_1[0], (noInputs,n)).copy()
pos_weights[pos_weights<0]=0
plt.imshow(pos_weights, aspect = 'auto')
plt.colorbar()
plt.title('weight matrix, positives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

neg_weights = np.reshape(weights_1_1[0], (noInputs,n)).copy()
neg_weights[pos_weights>0]=0
plt.imshow(abs(neg_weights), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix), negatives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg)), sensorList_neg)
plt.show()

print(np.sum(np.reshape(abs(weights_1_1[0]), (noInputs,n)), axis=1))

print(np.sum(np.reshape(weights_1_1[0], (noInputs,n)), axis=1))
print('without abs(), mostly negative')

print(np.sum(pos_weights, axis=1))
