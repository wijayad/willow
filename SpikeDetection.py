# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:54:21 2020
Note that elements of this script were pulled from Willow scripts written by Haley Speed
https://github.com/haleyspeed/willow
@author: Daniel
"""

import h5py
import numpy as np
#import os
import matplotlib.pyplot as plt
from IPython.display import HTML
from scipy.signal import butter, sosfiltfilt, sosfreqz
#import time
#import pandas as pd
import scipy 
#import csv
#accessing file
filename = 'C:\\Users\\Daniel Wijaya\\Downloads\\experiment_C20200330-162815.h5'
# change filepath as required.

fs = 30000 #sample rate
order = 6 #order of filter
lowcut = 400
highcut = 5000

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

#%%

def plotStackedGraph3(graphData,col1,col2,col3,start=0):
    #lv1 = 1
    numObs = 15000
    time = np.array(range(start,start+numObs))
    time = time / fs
    plt.figure(figsize=(300,100))
    
    for sensorIndex in range(np.shape(col1)[0]):
        #CHANGE FORMAT TO CYCLE THRU SENSORCOL
        ax1 = plt.subplot(21,3,sensorIndex*3+1)
        ax1.plot(time, graphData[start:start+numObs,col1[sensorIndex]])#channel_reading_orig
        if sensorIndex != 0:
            #plot the center one too
            ax1 = plt.subplot(21,3,sensorIndex*3+2)
            ax1.plot(time, graphData[start:start+numObs,col2[sensorIndex]])
        #note subplot index increases going right (goes along rows first)
        ax1 = plt.subplot(21,3,sensorIndex*3+3)
        ax1.plot(time, graphData[start:start+numObs,col3[sensorIndex]])
        #lv1+=1

def butter_bandpass(lowcut, highcut, fs, order=order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

data = h5py.File(filename,"r")
#print("Keys- %s" % data.keys())
channel_data = data["channel_data"]
print(np.shape(channel_data))

#actual sensor data only from first 256 columns of channel_data
channel_reading = channel_data[:,:256]*0.195 # scale to microvolts as well

data.close()

channel_reading_f = np.zeros(shape = np.shape(channel_reading[:1000000,:]))#just take a million to speed calculations up
print(HTML("<h4>Analyzing channel: "))
for sensor in range(np.shape(channel_reading)[1]):
    print(sensor, end = ' ')
    x = np.arange(len(channel_reading[:,sensor]))
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = sosfreqz(sos, worN=2000)
    channel_reading_f[:,sensor] = butter_bandpass_filter(channel_reading[:1000000,sensor], lowcut, highcut, fs, order=order)

#h5_spikedetector section
#Savitzky-Golay Smoothing
def sgolay2d ( z, window_size, order, derivative=None):
    """
    z is the input data
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')
channel_reading_smoothed = sgolay2d(channel_reading_f, window_size=41, order=4)

start_search = 0    # Time to start looking for events (in seconds)
end_search = 1      # Time to stop looking for events (in seconds)
threshold = 0.5       # Multiplier of the std of the smoothed mean for every 10ms with 10% overlap

def event_search(raw, smooth, start, end, thresh): 
    start = int(start * fs)
    end = int(end*fs)
    mn = fs-150 # minimum length of analysis window (0.995 s)
    peaks_x = {} # dict to catch peak times 
    peaks_y = {} # dict to catch peak amps
    peaks_b = {} # dict to catch peak baseline amps
    events_x = {} # dict to catch event trace x (3 ms before and after peak)
    events_y = {} # dict to catch event trace y (3 ms before and after peak)
    events_b = {} # dict to catch event trace baseline (3 ms before and after peak)
    print('Analyzing channel: ', end = ' ')
    for j in range(np.shape(raw)[1]):
        print(j, end = ' ')
        i = start
    #    epoch = 0
        peak_x = []
        peak_y = []
        peak_b = []
        rep = 0
        while i < end-11: # search all the rows 
            if i % mn == 0 and i < end - mn: # moving window every 10ms, % returns remainder so moving window every 995ms i think
                #unless for the if statement and the +mn should be changed s.t. mn would all be replaced by 10 (10ms)
                #this formulation also requires start = 0 (or some multiple of mn)
                savg = np.median(abs(smooth[i:i+mn,j]))
                ssd = np.std(smooth[i:i+mn,j])*thresh
            if raw[i,j] > savg *ssd  and raw[i,j] > max(raw[i+1:i+10,j]) and raw[i,j] > raw[i-1,j] and i-10 >= 0:
                peak_x.append(i)
                peak_y.append(raw[i,j])
                peak_b.append(savg)
                events_x['xCh'+str(j)+'_'+str(rep)] = list(range(i-10,i+10))
                events_y['yCh'+str(j)+'_'+str(rep)] = raw[i-10:i+10,j]
                rep = rep + 1
            i = i + 1
        peaks_x[str(j)] = peak_x
        peaks_y[str(j)] = peak_y
        peaks_b[str(j)] = peak_b
    peaks = [peaks_x,peaks_y,peaks_b]
    events = [events_x, events_y, events_b]
    return peaks, events

def event_search2(raw, start, end, thresh):
    #altered function
    # different ssd definition (prev definition was overly specific)
    # ssd unused, and peak baseline amps not recorded
    start = int(start * fs)
    end = int(end*fs)
    peaks_x = {} # dict to catch peak times 
    peaks_y = {} # dict to catch peak amps
    events_x = {} # dict to catch event trace x (3 ms before and after peak)
    events_y = {} # dict to catch event trace y (3 ms before and after peak)
    print('Analyzing channel: ', end = ' ')
    for j in range(np.shape(raw)[1]):
        print(j, end = ' ')
        i = start
        peak_x = []
        peak_y = []
        rep = 0
        ssd = np.std(raw[:,j])*thresh
        while i < end-11: # search all the rows 
            #if i % mn == 0 and i < end - mn: # moving window every 10ms, % returns remainder so moving window every 995ms i think
            #    #unless for the if statement and the +mn should be changed s.t. mn would all be replaced by 10 (10ms)
            #    #this formulation also requires start = 0 (or some multiple of mn)
            #    savg = np.median(abs(raw[i:i+mn,j]))
            #    ssd = np.std(raw[i:i+mn,j])*thresh
            if raw[i,j] > ssd  and raw[i,j] > max(raw[i+1:i+10,j]) and raw[i,j] > raw[i-1,j] and i-10 >= 0: 
                peak_x.append(i)
                peak_y.append(raw[i,j])
                events_x['xCh'+str(j)+'_'+str(rep)] = list(range(i-10,i+10))
                events_y['yCh'+str(j)+'_'+str(rep)] = raw[i-10:i+10,j]
                rep = rep + 1
            i = i + 1
        peaks_x[str(j)] = peak_x
        peaks_y[str(j)] = peak_y
    peaks = [peaks_x,peaks_y]
    events = [events_x, events_y]
    return peaks, events
            
peaks, events = event_search(channel_reading_f[:30000,:], channel_reading_smoothed, start_search, end_search, threshold)
#peaks2, events2 = event_search(channel_reading_f[:30000,:], channel_reading_smoothed2, start_search, end_search, 0.4)
peaks3, events3 = event_search2(channel_reading_f[:30000,:], start_search, end_search, 5)
#for data smoothed via scipy savitz-golay filter, need to use thresh = 0.4 to get similar results (smoothing function is less smoothed than Haleys)

#%% data visualization

start = 0 #change as required (to view different data ranges)
end = 30000  #change as required (to view different data ranges)
chan = 0#choose channel to view data, set to channel 112 as default
figsize = (50,10)
xs = list(range(np.shape(channel_reading_f)[0]))

#visualize channel readings and smoothed channel readings for a channel
fig, ax = plt.subplots(figsize = (50,10))
ax.plot(channel_reading_f[start:end,chan])
ax.plot(channel_reading_smoothed[start:end,chan])
ax.legend()
plt.show()

#visualize channel readings, smoothed channel readings, and peaks for a channel
#peaks defined by event_search function (from Haleys script)
fig, ax = plt.subplots(figsize = figsize)
ax.plot(xs[start:end],channel_reading_f[start:end,0])
ax.plot(xs[start:end],channel_reading_smoothed[start:end,0])
ax.plot(peaks[0][str(0)],
    peaks[1][str(0)], 
    ls = 'none', marker = '*', markersize = 6, color = 'black')
plt.show()

# =============================================================================
# fig, ax = plt.subplots(figsize = figsize)
# ax.plot(xs[start:end],channel_reading_f[start:end,0])
# ax.plot(xs[start:end],channel_reading_smoothed2[start:end,0])
# ax.plot(peaks2[0][str(0)],
#     peaks2[1][str(0)], 
#     ls = 'none', marker = '*', markersize = 6, color = 'black')
# plt.show()
# =============================================================================

#visualize channel readings, smoothed channel readings, and peaks for a channel
#peaks defined by event_search2 function (altered parameters)
fig, ax = plt.subplots(figsize = figsize)
ax.plot(xs[start:end],channel_reading_f[start:end,0])
ax.plot(peaks3[0][str(0)],
    peaks3[1][str(0)], 
    ls = 'none', marker = '*', markersize = 6, color = 'black')
plt.show()