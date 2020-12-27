# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:39:14 2020

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

spikesDF = pd.read_csv('C:\\Users\\Daniel Wijaya\\Downloads\\experiment_C20200330-162815_spikes_in_s2.csv')
spikesDict = spikesDF.to_dict(orient='list')
#for now consider only shank 0, simple spikes for now (S0, SS & GC in string)

#this will serve as SOURCE of outputs(y), gives time stamps for positive examples
spikesDictS0_SS = {key:[element for element in value if np.isnan(element)==False] 
                   for key,value in spikesDict.items() if 
                   key.find('S0')>=0 and key.find('SS')>=0}

#this serves as SOURCE of inputs (x), for both positive and negative examples
spikesDictS0_GC = {key:[element for element in value if np.isnan(element)==False] 
                   for key,value in spikesDict.items() if 
                   key.find('S0')>=0 and key.find('GC')>=0}
for key, values in spikesDictS0_SS.items():
    print(str(key) + ' ' + str(len(values)))
for key, values in spikesDictS0_GC.items():
    print(str(key) + ' ' + str(len(values)))
#notice that channel 112 has most, so try with channel 112 for now
#look at 1s time range prior to activation time, therefore dont include first 1s
Y_times = [round(element,3) for element in spikesDictS0_SS['S0_C112_SS'] if element > 1]

#creating training dataset, for now do for channel 112 ^
#method 1
#when was most recent spike in other sensors
#create for positive examples first
def createDataPos(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, first = True):
    
    #for test case, inputSensorSpikes = spikesDictS0_GC, ie. signals from the granular layer of that shank
    #sensorI = {}
    sensorList = []#so know which sensor each col refers to
    
    
    #numPosEx = len(Y_times)
    #for sensor in inputSensorSpikes.keys():
    #    sensorI[sensor] = 0
    #for each sensor...
# =============================================================================
##removed style from input, for default, put subdiv = 1
#     if style == 1:
#         X = np.zeros(shape = (len(Y_times),len(inputSensorSpikes)))
#         #shape is m x n
#         numSpikes = np.zeros(shape = (len(Y_times),len(inputSensorSpikes)))
#         for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
#             #for each time the output activated
#             sensorList += [sensor]
#             for yIter, y_t in enumerate(Y_times):
#                 t_i = y_t-t_window #tail end of time to consider, 1 second before
#                 #do a list comprehension for the timeslot to consider for the inputs (GC layer)
#                 #only do ^ for now, should probably change to do list slice in future so more efficient
#                 #as already know that the list is ordered
#                 tempArr = np.array([element for element in inputSensorSpikes[sensor] 
#                             if element >=t_i and element <=y_t]) - t_i
#                 #tempList contains time from start of considered time (ie. t_i), 
#                 #if activated exactly at same time as y_t, has value 1
#                 #if no activation, has value 0
#                 #UP TO THIS POINT, GENERAL FOR ALL SENSORS
#                 #print(len(tempArr))
#                 #method 1, just consider most recent t
#                 #default value is just 0, so just need to do non-zero stuff
#                 #THIS IS THE PART THAT CHANGES TO CHANGE WHAT DATA IS INCLUDED
#                 numSpikes[yIter, sensorIter] = len(tempArr)
#                 if len(tempArr) != 0: #no activation, therefore put as 0
#                     X[yIter, sensorIter] = tempArr[-1]
# =============================================================================
                    
#    elif style == 2: #subdivide each time range into n-categories
    #print(subdiv)
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
            #numSpikes is redundant in this case
            # for yIter, y_t in enumerate(Y_times):
# =============================================================================
#                 X[yIter, y_t*1000] =
#                 for lv1 in range(subdiv):
#                     t_i = round(y_t-1+lv1*0.001,3)
#                     t_f = round(t_i+0.001)
#                     np.any(inputSensorSpikes[sensor]) if
#                                         element >=t_i and element <=t_f]) - t_i,
#                                         t_f - t_i) 
#                     #subtract by t_i & divide by t_f - t_i to normalize on each 
#                     #time bucket
#                     #print(subdiv*sensorIter+lv1)
#                     numSpikes[yIter, subdiv*sensorIter+lv1] = len(tempArr)
#                     if numSpikes[yIter, subdiv*sensorIter+lv1] != 0:
#                         if first == True:
#                             X[yIter, lv1 + subdiv*sensorIter] = tempArr[0]
#                         else: #last value
#                             X[yIter, lv1 + subdiv*sensorIter] = tempArr[-1]
# =============================================================================
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
                    #subtract by t_i & divide by t_f - t_i to normalize on each 
                    #time bucket
                    #print(subdiv*sensorIter+lv1)
                    numSpikes[yIter, subdiv*sensorIter+lv1] = len(tempArr)
                    if numSpikes[yIter, subdiv*sensorIter+lv1] != 0:
                        if first == True:
                            X[yIter, lv1 + subdiv*sensorIter] = tempArr[0]
                        else: #last value
                            X[yIter, lv1 + subdiv*sensorIter] = tempArr[-1]
                            #consider first spike in time range
            #print(t_i,t_f)
    return X, numSpikes, sensorList
#create positive examples
X_pos, numSpikes_pos, sensorList_pos = createDataPos(Y_times, spikesDictS0_GC)
Y_pos = np.ones(shape=(len(Y_times),1))


def createDataNeg(Y_times, inputSensorSpikes, t_window = 1, subdiv = 1, first = True):
    sensorList = []
    
    #print(type(t_window))
    Rand_times = np.sort(
        np.round(
            np.random.rand(len(Y_times))*(Y_times[-1]-t_window) + t_window, 3
            )
        )
    for lv1, t in enumerate(Rand_times):
        while t in Y_times:
            Rand_times[lv1] = np.random.rand()
            t = Rand_times[lv1]
# =============================================================================
#     #this is the OLD version
#     X = np.zeros(shape = (len(Y_times), len(inputSensorSpikes))) #avoid unbalanced data for now
#     numSpikes = np.zeros(shape = (len(Y_times),len(inputSensorSpikes)))
#     for sensorIter, sensor in enumerate(inputSensorSpikes.keys()):
#         #for each time the output activated
#         sensorList += [sensor]
#        # for yIter, y_t in enumerate(Y_times):
#         for yIter, y_t in enumerate(Rand_times):
#             t_i = y_t-t_window
#             tempArr = np.array([element for element in inputSensorSpikes[sensor] 
#                         if element >=t_i and element <=y_t]) - t_i
#             numSpikes[yIter, sensorIter] = len(tempArr)
#             if len(tempArr) != 0: #no activation, therefore put as 0
#                 X[yIter, sensorIter] = tempArr[-1]
#     #end old version
# =============================================================================
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
X_neg, numSpikes_neg, sensorList_neg, Neg_times = createDataNeg(Y_times, spikesDictS0_GC)
Y_neg = np.zeros(shape=(len(Y_times),1))

#method 2
#create data (1 second chunks for each dataset)
#for each input
#1 second of each sensor split into 10 buckets
#how long ago were previous 10 activations in each sensor [normalized]
#if no activation, then just be 0 (ie. padded inputs)
#each will have custom # of sensors, trying to build up a
n = 10  
X_pos2, numSpikes_pos2, sensorList_pos2 = createDataPos(Y_times, 
                                                        spikesDictS0_GC,
                                                        subdiv = n,
                                                        first = False)
X_neg2, numSpikes_neg2, sensorList_neg2, Neg_times2 = createDataNeg(Y_times,
                                                                    spikesDictS0_GC,
                                                                    subdiv = n,
                                                                    first = False)
X2 = np.concatenate((X_pos2,X_neg2),axis = 0)
Y2 = np.concatenate((Y_pos,Y_neg),axis = 0)

#combine positive and negative examples
X = np.concatenate((X_pos, X_neg), axis = 0)
Y = np.concatenate((Y_pos, Y_neg), axis = 0)

X2 = np.concatenate((X_pos2,X_neg2),axis = 0)
Y2 = np.concatenate((Y_pos,Y_neg),axis = 0)
#shuffle, not sure if necessary
Data = np.concatenate((X,Y), axis = 1)
np.random.shuffle(Data)
#v2 of data
X = Data[:,:-1]
Y = np.reshape(Data[:,-1], newshape = (np.shape(Y)))

Data2 = np.concatenate((X2,Y2),axis=1)
np.random.shuffle(Data2)
X2 = Data2[:,:-1]
Y2 = np.reshape(Data2[:,-1], newshape = (np.shape(Y2)))

#split into train and test data
#do a 80-20% split
X_train = X[:round(len(X)*0.8),:]
X_test = X[round(len(X)*0.8):,:]
Y_train = Y[:round(len(X)*0.8),:]
Y_test = Y[round(len(X)*0.8):,:]

X_train2 = X2[:round(len(X)*0.8),:]
X_test2 = X2[round(len(X)*0.8):,:]
Y_train2 = Y2[:round(len(X)*0.8),:]
Y_test2 = Y2[round(len(X)*0.8):,:]



#now to the ML part
##THIS IS ASSUMING THE FIRST DATA FORMAT
#dont need to normalize since just do 1s 
from keras.layers import Input
#from keras.utils import plot_model#need to install pydot
from keras.models import Model
from keras.layers import Dense, Dropout

#model 1

inputs = Input(shape=(9,))
x = Dense(10, activation = 'relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(20, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation = 'relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation = 'sigmoid')(x)

test_model = Model(inputs = inputs, outputs = outputs)
test_model.summary()

test_model.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])

history = test_model.fit(X_train, Y_train, epochs = 100)
test_scores = test_model.evaluate(X_test,Y_test, verbose = 2)


#model 2, more layers

inputs2 = Input(shape=(9,))
x2 = Dense(50, activation = 'relu')(inputs2)
x2 = Dropout(0.2)(x2)
x2 = Dense(40, activation = 'relu')(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(30, activation = 'relu')(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(20, activation = 'relu')(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(10, activation = 'relu')(x2)
x2 = Dropout(0.2)(x2)
outputs2 = Dense(1, activation = 'sigmoid')(x2)

test_model2 = Model(inputs = inputs2, outputs = outputs2)
test_model2.summary()

test_model2.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),
    metrics = ['accuracy'])

history2 = test_model2.fit(X_train, Y_train, epochs = 100)
test_scores2 = test_model2.evaluate(X_test,Y_test, verbose = 2)

#model 3, fewer layers
inputs3 = Input(shape=(9,))
x3 = Dense(10, activation = 'relu')(inputs3)
x3 = Dropout(0.2)(x3)
x3 = Dense(20, activation = 'relu')(x3)
x3 = Dropout(0.2)(x3)
outputs3 = Dense(1, activation = 'sigmoid')(x3)

test_model3 = Model(inputs = inputs3, outputs = outputs3)
test_model3.summary()

test_model3.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),
    metrics = ['accuracy'])

history3 = test_model3.fit(X_train, Y_train, epochs = 100)
test_scores3 = test_model3.evaluate(X_test,Y_test, verbose = 2)


#for second type of input
#second type of positive examples (10 bins, 100 bins, etc.)
n = 100
X_pos2, numSpikes_pos2, sensorList_pos2 = createDataPos(Y_times, 
                                                        spikesDictS0_GC,
                                                        subdiv = n)
X_neg2, numSpikes_neg2, sensorList_neg2, Neg_times2 = createDataNeg(Y_times,
                                                                    spikesDictS0_GC,
                                                                    subdiv = n)
X2 = np.concatenate((X_pos2,X_neg2),axis = 0)
Y2 = np.concatenate((Y_pos,Y_neg),axis = 0)
#shuffle, not sure if necessary

Data2_2 = np.concatenate((X2,Y2),axis=1)
np.random.shuffle(Data2)
X2 = Data2[:,:-1]
Y2 = np.reshape(Data2[:,-1], newshape = (np.shape(Y2)))

#split into train and test data
#do a 80-20% split

X_train2 = X2[:round(len(X)*0.8),:]
X_test2 = X2[round(len(X)*0.8):,:]
Y_train2 = Y2[:round(len(X)*0.8),:]
Y_test2 = Y2[round(len(X)*0.8):,:]

#model v (for visualization)
inputs_2_v = Input(shape=(9*n,))
x_2_v = Dropout(0.92)(inputs_2_v)
outputs_2_v = Dense(1, activation = 'sigmoid')(x_2_v)

model_2_v = Model(inputs = inputs_2_v, outputs = outputs_2_v)
model_2_v.summary()

model_2_v.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_2_v = model_2_v.fit(X_train2, Y_train2, batch_size = 128, epochs = 200)
test_scores_2_v = model_2_v.evaluate(X_test2,Y_test2, verbose = 2)
print(test_scores_2_v)

weights_2_v = model_2_v.get_weights()#returns list of BOTH weights and biases
#index even gives weights, odd indicies give bias weights

plt.matshow(np.reshape(weights_2_v[0],(9,n)))
plt.colorbar()
plt.ylabel('sensor')
plt.xlabel('time (1s duration in total)')

#note for specific layers
#model_2_1.layers[1].get_weights()#returns list of both weights and bias weights
#model_2_1.layers[1].get_weights()[0]#gives weights
#model_2_1.layers[1].get_weights()[1]#gives biases weights


#model 0
inputs_2_0 = Input(shape=(9*n,))
x_2_0 = Dense(128, activation = 'relu')(inputs_2_0)
x_2_0 = Dropout(0.97)(x_2_0)

outputs_2_0 = Dense(1, activation = 'sigmoid')(x_2_0)

model_2_0 = Model(inputs = inputs_2_0, outputs = outputs_2_0)
model_2_0.summary()

model_2_0.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_2_0 = model_2_0.fit(X_train2, Y_train2, batch_size = 128, epochs = 150)
test_scores_2_0 = model_2_0.evaluate(X_test2,Y_test2, verbose = 2)
print(test_scores_2_0)

# =============================================================================
# #model 0
# inputs_2_0 = Input(shape=(9*n,))
# x_2_0 = Dense(256, activation = 'relu')(inputs_2_0)
# x_2_0 = Dropout(0.915)(x_2_0)
# x_2_0 = Dense(32, activation = 'relu')(x_2_0)
# x_2_0 = Dropout(0.915)(x_2_0)
# 
# outputs_2_0 = Dense(1, activation = 'sigmoid')(x_2_0)
# 
# model_2_0 = Model(inputs = inputs_2_0, outputs = outputs_2_0)
# model_2_0.summary()
# 
# model_2_0.compile( 
#     loss = 'binary_crossentropy',
#     optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
#     metrics = ['accuracy'])
# #note that fit does not re-initialize values (done in compile)
# history_2_0 = model_2_0.fit(X_train2, Y_train2, batch_size = 128, epochs = 100)
# test_scores_2_0 = model_2_0.evaluate(X_test2,Y_test2, verbose = 2)
# print(test_scores_2_0)
# 
# =============================================================================
#model 1
inputs_2_1 = Input(shape=(9*n,))
x_2_1 = Dense(256, activation = 'relu')(inputs_2_1)
x_2_1 = Dropout(0.8)(x_2_1)
x_2_1 = Dense(64, activation = 'relu')(x_2_1)
x_2_1 = Dropout(0.8)(x_2_1)
x_2_1 = Dense(16, activation = 'relu')(x_2_1)
x_2_1 = Dropout(0.8)(x_2_1)

outputs_2_1 = Dense(1, activation = 'sigmoid')(x_2_1)

model_2_1 = Model(inputs = inputs_2_1, outputs = outputs_2_1)
model_2_1.summary()

model_2_1.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_2_1 = model_2_1.fit(X_train2, Y_train2, batch_size = 128, epochs = 50)
test_scores_2_1 = model_2_1.evaluate(X_test2,Y_test2, verbose = 2)
print(test_scores_2_1)

#model 2
inputs_2_2 = Input(shape=(9*n,))
x_2_2 = Dense(1028, activation = 'relu')(inputs_2_2)
x_2_2 = Dropout(0.79)(x_2_2)
x_2_2 = Dense(256, activation = 'relu')(x_2_2)
x_2_2 = Dropout(0.79)(x_2_2)
x_2_2 = Dense(64, activation = 'relu')(x_2_2)
x_2_2 = Dropout(0.79)(x_2_2)
x_2_2 = Dense(16, activation = 'relu')(x_2_2)
x_2_2 = Dropout(0.79)(x_2_2)

outputs_2_2 = Dense(1, activation = 'sigmoid')(x_2_2)

model_2_2 = Model(inputs = inputs_2_2, outputs = outputs_2_2)
model_2_2.summary()

model_2_2.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_2_2 = model_2_2.fit(X_train2, Y_train2, batch_size = 128, epochs = 100)
test_scores_2_2 = model_2_2.evaluate(X_test2,Y_test2, verbose = 2)
print(test_scores_2_2)

#new test n
n = 1000
X_pos3, numSpikes_pos3, sensorList_pos3 = createDataPos(Y_times, 
                                                        spikesDictS0_GC,
                                                        subdiv = n)
X_neg3, numSpikes_neg3, sensorList_neg3, Neg_times3 = createDataNeg(Y_times,
                                                                    spikesDictS0_GC,
                                                                    subdiv = n)
X3 = np.concatenate((X_pos3,X_neg3),axis = 0)
Y3 = np.concatenate((Y_pos,Y_neg),axis = 0)
#shuffle, not sure if necessary

Data3 = np.concatenate((X3,Y3),axis=1)
np.random.shuffle(Data3)
X3 = Data3[:,:-1]
Y3 = np.reshape(Data3[:,-1], newshape = (np.shape(Y3)))

#split into train and test data
#do a 80-20% split
X_train3 = X3[:round(len(X)*0.8),:]
X_test3 = X3[round(len(X)*0.8):,:]
Y_train3 = Y3[:round(len(X)*0.8),:]
Y_test3 = Y3[round(len(X)*0.8):,:]

#model 1
inputs_3_1 = Input(shape=(9*n,))
#0.9 works best so far, 4096, < div 8, div 8, accuracy = 0.84125
#0.92 works even better, 2048, 258, accuracy = 0.84250
#0.95 works same as ^, 1024, accuracy = 0.84250
#0.96, works same as ^, 1024, accuracy = 0.84250
#suggests perhaps that this is the upper limit of accuracy
#x_3_1 = Dense(1024, activation = 'relu')(inputs_3_1)
#x_3_1 = Dropout(0.95)(x_3_1)
# x_3_1 = Dense(258, activation = 'relu')(x_3_1)
# x_3_1 = Dropout(0.92)(x_3_1)
#x_3_1 = Dense(64, activation = 'relu')(x_3_1)
#x_3_1 = Dropout(0.9)(x_3_1)


#no hidden layers, [0.404820476770401, 0.8262500166893005], 0.5
#[0.4080710780620575, 0.8287500143051147], 0.7
#[0.4250020503997803, 0.8412500023841858], 0.9 #also looks like could impove with more training
#[0.4326312530040741, 0.8387500047683716], 0.95
#using 0.8 bc very similar training accuracy and test accuracy
#higher dropout values tank training accuracy, so probably not too good
x_3_1 = Dropout(0.8)(inputs_3_1)
outputs_3_1 = Dense(1, activation = 'sigmoid')(x_3_1)

model_3_1 = Model(inputs = inputs_3_1, outputs = outputs_3_1)
model_3_1.summary()

model_3_1.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_3_1 = model_3_1.fit(X_train3, Y_train3, batch_size = 128, epochs = 30)
test_scores_3_1 = model_3_1.evaluate(X_test3,Y_test3, verbose = 2)
print(test_scores_3_1)

weights_3_1 = model_3_1.get_weights()
plt.imshow(np.reshape(weights_3_1[0], (9,n)), aspect = 'auto')
plt.colorbar()
plt.title('weight matrix')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg3)), sensorList_neg3)
plt.show()

plt.imshow(np.reshape(abs(weights_3_1[0]), (9,n)), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix)')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg3)), sensorList_neg3)

pos_weights = np.reshape(weights_3_1[0], (9,n)).copy()
pos_weights[pos_weights<0]=0
plt.imshow(pos_weights, aspect = 'auto')
plt.colorbar()
plt.title('weight matrix, positives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg3)), sensorList_neg3)

neg_weights = np.reshape(weights_3_1[0], (9,n)).copy()
neg_weights[pos_weights>0]=0
plt.imshow(abs(neg_weights), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix), negatives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg3)), sensorList_neg3)



print(np.sum(np.reshape(abs(weights_3_1[0]), (9,n)),axis=1))

print(np.sum(np.reshape(weights_3_1[0], (9,n)),axis=1))
print('without abs(), mostly negative')




X_pos4, numSpikes_pos4, sensorList_pos4 = createDataPos(Y_times, 
                                                        spikesDictS0_GC,
                                                        subdiv = n)
X_neg4, numSpikes_neg4, sensorList_neg4, Neg_times4 = createDataNeg(Y_times,
                                                                    spikesDictS0_GC,
                                                                    subdiv = n)
# =============================================================================
# 
# X_pos3_ = np.around(X_pos3)
# X_pos4_ = np.around(X_pos4)
# indX = []
# indY = []
# for lv1 in range(np.shape(X_pos3)[0]):
#     for lv2 in range(np.shape(X_pos3)[1]):
#         if X_pos3_[lv1,lv2]!=X_pos3_[lv1,lv2]:
#             print(lv1, lv2)
#             indX += [lv1]
#             indY += [lv2]
#             
# indX, indY = np.where(X_pos3_ != X_pos4_)
# compareM = X_pos3_==X_pos4_
# np.where(compareM==False)
# =============================================================================
X4 = np.concatenate((X_pos4,X_neg4),axis = 0)
Y4 = np.concatenate((Y_pos,Y_neg),axis = 0)
#shuffle, not sure if necessary

Data4 = np.concatenate((X4,Y4),axis=1)
np.random.shuffle(Data4)
X4 = Data4[:,:-1]
Y4 = np.reshape(Data4[:,-1], newshape = (np.shape(Y4)))

#split into train and test data
#do a 80-20% split
X_train4 = X4[:round(len(X)*0.8),:]
X_test4 = X4[round(len(X)*0.8):,:]
Y_train4 = Y4[:round(len(X)*0.8),:]
Y_test4 = Y4[round(len(X)*0.8):,:]

inputs_4_1 = Input(shape=(9*n,))
#0.9 works best so far, 4096, < div 8, div 8, accuracy = 0.84125
#0.92 works even better, 2048, 258, accuracy = 0.84250
#0.95 works same as ^, 1024, accuracy = 0.84250
#0.96, works same as ^, 1024, accuracy = 0.84250
#suggests perhaps that this is the upper limit of accuracy
x_4_1 = Dense(1024, activation = 'relu')(inputs_4_1)
#x_4_1 = Dropout(0.95)(x_4_1)
#x_4_1 = Dense(258, activation = 'relu')(x_4_1)
#x_4_1 = Dropout(0.92)(x_4_1)
#x_4_1 = Dense(64, activation = 'relu')(x_4_1)
#x_4_1 = Dropout(0.9)(x_4_1)

#no hidden layers, [0.404820476770401, 0.8262500166893005], 0.5
#[0.4080710780620575, 0.8287500143051147], 0.7
#[0.4250020503997803, 0.8412500023841858], 0.9 #also looks like could impove with more training
#[0.4326312530040741, 0.8387500047683716], 0.95
#using 0.8 bc very similar training accuracy and test accuracy
#higher dropout values tank training accuracy, so probably not too good
x_4_1 = Dropout(0.8)(inputs_4_1)
outputs_4_1 = Dense(1, activation = 'sigmoid')(x_4_1)

model_4_1 = Model(inputs = inputs_4_1, outputs = outputs_4_1)
model_4_1.summary()

model_4_1.compile( 
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(),#learning_rate = 0.0001
    metrics = ['accuracy'])
#note that fit does not re-initialize values (done in compile)
history_4_1 = model_4_1.fit(X_train4, Y_train4, batch_size = 128, epochs = 30)
test_scores_4_1 = model_4_1.evaluate(X_test4,Y_test4, verbose = 2)
print(test_scores_4_1)

weights_4_1 = model_4_1.get_weights()
plt.imshow(np.reshape(weights_4_1[0], (9,n)), aspect = 'auto')
plt.colorbar()
plt.title('weight matrix')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg4)), sensorList_neg4)
plt.show()

plt.imshow(np.reshape(abs(weights_4_1[0]), (9,n)), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix)')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg4)), sensorList_neg4)
plt.show()

pos_weights = np.reshape(weights_4_1[0], (9,n)).copy()
pos_weights[pos_weights<0]=0
plt.imshow(pos_weights, aspect = 'auto')
plt.colorbar()
plt.title('weight matrix, positives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg4)), sensorList_neg4)
plt.show()

neg_weights = np.reshape(weights_4_1[0], (9,n)).copy()
neg_weights[pos_weights>0]=0
plt.imshow(abs(neg_weights), aspect = 'auto')
plt.colorbar()
plt.title('abs(weight matrix), negatives only')
plt.ylabel('channel')
plt.xlabel('time [ms]')
plt.yticks(np.arange(len(sensorList_neg4)), sensorList_neg4)
plt.show()








#also because previous models (only using most recent signal) has like
#60% accuracy, this is very good, since this implies that there is correlation
#with spikes in previous 1s time
weights_2_1 = model_2_1.get_weights()#returns list of BOTH weights and biases
#index even gives weights, odd indicies give bias weights

plt.matshow(weights_2_1[0])
plt.colorbar()
#note for specific layers
model_2_1.layers[1].get_weights()#returns list of both weights and bias weights
model_2_1.layers[1].get_weights()[0]#gives weights
model_2_1.layers[1].get_weights()[1]#gives biases weights

predictions_2_1 = model_2_1.predict(X2)
#plot 10 predictions with highest certainty of being a spike
top10 = np.zeros(shape=(3,10))#first row are values, second row are indicies, 3rd row is correct class
for lv1,prediction in enumerate(predictions_2_1):
    for lv2, val in enumerate(top10[0]):
        if prediction > val and lv2 == 9:
            top10[0,lv2] = prediction
            top10[1,lv2] = lv1
            top10[2,lv2] = Y2[lv1]
        elif prediction > val:
            continue
        elif prediction < val and lv2 != 0 and prediction > top10[0,lv2-1]:
            top10[0,lv2-1] = prediction
            top10[1,lv2-1] = lv1
            top10[2,lv2-1] = Y2[lv1]
            break
least10 = np.ones(shape=(3,10))#first row are values, second row are indicies, 3rd row is correct class
for lv1,prediction in enumerate(predictions_2_1):
    for lv2, val in enumerate(least10[0]):
        if prediction < val and lv2 == 9:
            least10[0,lv2] = prediction
            least10[1,lv2] = lv1
            least10[2,lv2] = Y2[lv1]
        elif prediction < val:
            continue
        elif prediction > val and lv2 != 0 and prediction < least10[0,lv2-1]:
            least10[0,lv2-1] = prediction
            least10[1,lv2-1] = lv1
            least10[2,lv2-1] = Y2[lv1]
            break

channelGeometry = np.array([
    [109,254,253,111,231,247,246,229,228,255,236,235,234,238,239,232,227,241,242,225,224],
    [-1,110,248,252,99,230,114,245,96,108,250,237,123,233,126,240,120,226,119,243,101],
    [249,105,104,112,113,98,97,115,244,106,107,251,127,122,121,125,124,103,102,118,117],
    [44,191,190,46,47,188,183,166,165,172,173,170,169,175,176,160,161,178,179,163,100],
    [-1,45,48,189,34,167,51,182,164,171,185,174,43,168,63,177,57,162,60,116,38],
    [184,40,35,49,50,33,32,180,181,41,42,186,187,59,58,62,61,56,39,55,54],
    [135,23,28,14,13,30,31,139,138,22,21,133,132,4,5,1,2,7,24,8,9],
    [-1,15,18,29,130,12,152,155,137,134,148,20,145,0,151,6,142,3,158,25,75],
    [19,128,129,17,16,131,136,153,154,147,146,149,150,144,143,156,157,141,140,159,91],
    [198,86,87,79,78,93,94,76,203,85,84,196,64,69,70,66,67,88,89,73,74],
    [-1,199,81,92,195,77,217,95,202,197,83,68,210,65,214,71,207,72,222,90,204],
    [82,193,194,80,216,200,201,218,219,192,211,212,213,209,208,215,223,206,205,221,220]
    ])
#not sure how to plot this in a way that makes sense...
# =============================================================================
# def plotSpikes(activations, channelList = sensorList_pos, shankNo = 0, channelGeometry = channelGeometry):
#     #activations would be X2
#     n = int(len(activations)/len(channelList))
#     shankChannels = channelGeometry[shankNo*3:(shankNo+1)*3,:].T
#     shankSpikeAssign = np.ones(shape = (21,3*n))*-1
#     #print(shankSpikeAssign)
#     for lv1, channel in enumerate(channelList):
#         channel = int(sensorList_pos[0][4:sensorList_pos[0].find('GC')-1])
#         #print(channel)
#         #print(shankChannels)
#         #print(np.where(shankChannels == channel))
#         row, col = np.where(shankChannels == channel)
#         activation = activations[lv1*n:(lv1+1)*n]
#         #print(activation)
#         print(shankSpikeAssign[int(row), int(col*n):int((col+1)*n)])
#         
#         shankSpikeAssign[row,col*n:(col+1)*n] = activation
#     plt.matshow(shankSpikeAssign)
#     plt.colorbar()
# =============================================================================

n = 10
for lv1 in range(np.shape(top10)[1]):
#    activations = np.divide(X2[int(top10[1,lv1]),:] + np.array(9*list(range(n))), n)
    activations = X2[int(top10[1,lv1]),:]
    #for lv2 in range(10):
    #    activations[]
    #    *np.array(range(10))
    plt.matshow(activations.reshape(9,10))
    plt.title('positive examples')
    plt.show()
    #plotSpikes(activations)
    activations = X2[int(least10[1,lv1]),:]
    plt.matshow(activations.reshape(9,10))
    plt.title('negative examples')

#look at errors... no errors...
for lv1, val in enumerate(np.around(predictions_2_1)-Y2):
    if val != 0:
        print(lv1)
        print(predictions_2_1[lv1])
        print(Y2[lv1])
#check more values
        
X_neg2_, _,_,_ = createDataNeg(Y_times, spikesDictS0_GC, subdiv = 10, first = False)
predictions_2_1_ = model_2_1.predict(X_neg2_)
sum(predictions_2_1_)
sum(np.around(predictions_2_1_))#yea... only 1 mistake....

