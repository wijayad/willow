# -*- coding: utf-8 -*-
"""
conversion script
contains a litany of utility functions for processing and visualizing the 
neural data
Created on Thu Jul 21 18:28:31 2020
@author: Daniel
"""
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython.play import HTML
from scipy.signal import butter, sosfiltfilt, sosfreqz
import pandas as pd
import pickle

#accessing file
path = 'C:\\Users\\Daniel Wijaya\\Downloads\\'
filename = path+'experiment_C20200330-162815.h5'
fs = 30000 #sample rate 
order = 6 #order of filter
lowcut = 400
highcut = 5000
noShanks = 4
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
#note that channelGeometry has -1 as channel # where there are no channels
missingChannels = [10,11,26,27,36,37,52,53]
#missingChannels is a list of channel #s that are not included in the probes

#%% Various functions
# visualization function
def plotStackedGraph3(graphData,peaks,col1,col2,col3,start=0):
    #lv1 = 1
    numObs = 30000
    #xs = np.array(range(start,start+numObs))
    #xs = xs / fs
    plt.figure(figsize=(300,100))
    
    for sensorIndex in range(np.shape(col1)[0]):
        #CHANGE FORMAT TO CYCLE THRU SENSORCOL
        ax1 = plt.subplot(21,3,sensorIndex*3+1)
        ax1.plot(graphData[start:start+numObs,col1[sensorIndex]])#channel_reading_orig
        ax1.plot(peaks[0][str(col1[sensorIndex])],
            peaks[1][str(col1[sensorIndex])],
            ls = 'none', marker = '*', markersize = 6, color = 'black')
        if sensorIndex != 0:
            #plot the center one too
            ax1 = plt.subplot(21,3,sensorIndex*3+2)
            ax1.plot(graphData[start:start+numObs,col2[sensorIndex]])
            ax1.plot(peaks[0][str(col2[sensorIndex])],
                peaks[1][str(col2[sensorIndex])],
                ls = 'none', marker = '*', markersize = 6, color = 'black')
        #note subplot index increases going right (goes along rows first)
        ax1 = plt.subplot(21,3,sensorIndex*3+3)
        ax1.plot(graphData[start:start+numObs,col3[sensorIndex]])
        ax1.plot(peaks[0][str(col3[sensorIndex])],
            peaks[1][str(col3[sensorIndex])],
            ls = 'none', marker = '*', markersize = 6, color = 'black')
    plt.show()

# butterpass filters for data (initial preprocessing step)
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

# spike detection function
def event_search(raw, start, end, thresh): 
    #new event_search function
    #for data smoothed via scipy savitz-golay filter (ie. code used by Haley)
    #need to use thresh = 0.4 to get similar results
    #note this event_search has the advantage of cataloging both positive
    # and negative spikes
    #moreover, definition of ssd is over much broader time range
    start = int(start * fs)
    end = min(int(end*fs), np.shape(raw)[0])
    peaks_x = {} # dict to catch peak times 
    peaks_y = {} # dict to catch peak amps
    print('Analyzing channel: ', end = ' ')
    for j in range(np.shape(raw)[1]):
        # skip over missing channels
        if j in missingChannels:
            peaks_x[str(j)] = []
            peaks_y[str(j)] = []
            continue
        print(j, end = ' ')
        i = start
        peak_x = []
        peak_y = []
        rep = 0
        # initialize in case does not start on every 5th second
        ssd = np.std(raw[0:fs*5,j])*thresh
        while i < end-11: # search all the rows (time)
            if i%(fs*5) == 0:
                ssd = np.std(raw[i:i+fs*5,j])*thresh
            if abs(raw[i,j]) > ssd  and abs(raw[i,j]) > max(abs(raw[i+1:i+10,j])) \
                and abs(raw[i,j]) > abs(raw[i-1,j]) and i-10 >= 0: 
                # assume peaks can last 10 ms
                peak_x.append(i)
                peak_y.append(raw[i,j])
                rep = rep + 1
            i = i + 1
        peaks_x[str(j)] = peak_x
        peaks_y[str(j)] = peak_y
    peaks = [peaks_x,peaks_y]
    return(peaks)

#cross correlation function
def channelCrossCorrelate(peaks):
    # determine the cross-correlation between channels
    # can be used to determine different neural regions
    noChannels = len(peaks[0])
    crossCorr = np.zeros(shape=(noChannels,noChannels))
    print('Analyzing channel: ', end = ' ')
    for lv1 in range(noChannels): 
        print(lv1 , end = ' ')
        for lv2 in range(lv1,noChannels):
            #reset variables for unique channel combination
            countSame = 0
            lv3 = 0
            lv4 = 0
            numPeaks1 = len(peaks[0][str(lv1)])
            numPeaks2 = len(peaks[0][str(lv2)])
            while lv3 < numPeaks1 and lv4 < numPeaks2:
                #compare first peak locations
                if abs(peaks[0][str(lv1)][lv3]-peaks[0][str(lv2)][lv4])<=30:
                    #peaks are in same 1 ms (30 timepoints)
                    countSame += 2 #to account that one peak in each counts
                    lv3+=1
                    lv4+=1
                elif peaks[0][str(lv1)][lv3] < peaks[0][str(lv2)][lv4]:
                    lv3 += 1
                elif peaks[0][str(lv1)][lv3] > peaks[0][str(lv2)][lv4]:
                    lv4+=1
            totalPeaks = numPeaks1 + numPeaks2
            if totalPeaks == 0:
                crossCorr[lv1,lv2] = 0
            else:
                crossCorr[lv1,lv2]=countSame/totalPeaks
            #now reflect upper triangle to lower triangle
            # since we cre about combinations of sensors
            crossCorr[lv2,lv1]=crossCorr[lv1,lv2]
    return crossCorr
# plot cross correlation
def plotCrossCorr(crossCorr, shankMap, extraTitle = 'Unfiltered', path = path):
    for lv1 in range(noShanks):
        #plt.figure(figsize = (8,10))
        plt.matshow(crossCorr[shankMap[str(lv1)], :][:, shankMap[str(lv1)]])
        plt.colorbar()
        plt.title('Shank {} Cross-Correlation, {}'.format(lv1, extraTitle), 
                  y=1.15)
        plt.savefig(path+'crossCorrShank{}_{}.png'.format(lv1, extraTitle),
                    dpi = 300)
        
# visualizing single peaks
def peekAtPeaks(peaks,data,channel,index=0):
    #plots the specified peak (index determines which peak in the peaks list
    # to plot). Note this plots a 7ms timerange, with the spike occuring @1.5ms
    peakI = peaks[0][str(channel)][index]
    plt.plot(data[peakI-0.0015*fs:peakI+0.0055*fs,channel])

#function to eliminate spikes that look like noise by comparing with adjacent channels
def spikesOrNoise(peaks, channelMap, thresh=0.5):
    #eliminates spikes that are not seen in certain number (8*thresh) adjacent
    # channels to the channel being investigated
    #note that dimensions of channelMap needs to be rows*cols
    peaks_fx = {}
    peaks_fy = {}
    print('Analyzing Channel:', end = ' ')
    noRows = np.shape(channelMap)[0]
    noCols = np.shape(channelMap)[1]
    for row in range(noRows):
        for col in range(noCols):
            channelA = channelMap[row][col]
            if channelA == -1:
                continue
            peak_x = []
            peak_y = []
            print(channelA, end = ' ')
            #initialize lv2start dictionary here
            # use this to determine starting point for search
            # to avoid searching spikes that have already been analyzed
            lv2start = {}
            for row2 in range(row-1,row+2):
                for col2 in range(col-1,col+2):
                    if row2 not in range(noRows) or col2 not in range(noCols) or\
                        channelMap[row2][col2]==-1 or (row2==row and col2==col):
                            continue
                    lv2start[str(row2)+str(col2)]=0
            for lv1 in range(len(peaks[0][str(channelA)])):
                countTrue = 0
                countTotal = 0
                peakA = peaks[0][str(channelA)][lv1]
                peakA_y = peaks[1][str(channelA)][lv1]
                for row2 in range(row-1,row+2):
                    for col2 in range(col-1,col+2):
                        if row2 not in range(noRows) or col2 not in range(noCols) or\
                            channelMap[row2][col2] == -1 or (row2==row and col2==col): 
                            continue
                        channelB = channelMap[row2][col2]
                        countTotal+=1
                        for lv2 in range(lv2start[str(row2)+str(col2)], 
                                         len(peaks[0][str(channelB)])):
                            peakB = peaks[0][str(channelB)][lv2]
                            #check if peak exists in other channel within 1 ms
                            if abs(peakA-peakB)<=0.001*fs:
                                countTrue+=1
                                lv2start[str(row2)+str(col2)] = lv2+1
                                break
                            elif peakB - peakA > 0.001*fs:
                                break
                if countTrue/countTotal >= thresh:
                    peak_x.append(peakA)
                    peak_y.append(peakA_y)
            peaks_fx[str(channelA)] = peak_x
            peaks_fy[str(channelA)] = peak_y
    peaks_f = [peaks_fx, peaks_fy]
    return peaks_f

# utility function to return subset of spikes
def disectPeaks(peaks,startT = 0, endT = 1):
    #take section of peaks based on index & start and end indicies
    #useful for plotting the waveform for x-axis precision
    start = startT*fs
    end = endT*fs
    subsetPeaks_x = {}
    subsetPeaks_y = {}
    for lv1 in range(len(peaks[0])):
        for lv2 in range(len(peaks[0][str(lv1)])):
            if start>=peaks[0][str(lv1)][lv2]:
                startI = lv2
        for lv2 in range(len(peaks[0][str(lv1)])):
            if end<peaks[0][str(lv1)][lv2]: 
                endI = lv2
        subsetPeaks_x[str(lv1)] = peaks[0][str(lv1)][startI:endI]
        subsetPeaks_y[str(lv1)] = peaks[1][str(lv1)][startI:endI]
    subsetPeaks = [subsetPeaks_x, subsetPeaks_y]
    return subsetPeaks

#following 2 functions are used for computing signal-to-noise ratio of 
#different channels
def getSNR(peak, data, channel):
    #returns SNR of peak (noise calculated as the std deviation of 1s of data)
    peak_x = peak["peak_x"]
    peak_y = peak["peak_y"]
    if peak_x + fs > np.shape(data)[0]:
        std = np.std(data[peak_x-fs-1:peak_x+1, channel])
    else:
        std = np.std(data[peak_x:peak_x+fs, channel])
    return(peak_y / std)
def getT(aPeak):
    return(aPeak["peak_x"])

#due to oversampling, multiple channel pick up signals from a single source,
#therefore assign the spike to the channel with the highest SNR
def assignPeaks(peaks_f, data, region):
    #assign peak to the channel with the highest SNR in specified region
    #region based on visual determination of layer boundary 
    # (molecular vs granular)
    region = region.flatten()
    allPeaks = []
    print('Compiling Peaks from Channel:', end = ' ')
    for channel in region:
        if channel == -1:
            continue
        print(channel, end = ' ')
        #combine all peak times
        for lv1 in range(len(peaks[0][str(channel)])):
            allPeaks.append({"channel": channel, 
                             "peak_x": peaks[0][str(channel)][lv1],
                             "peak_y":peaks[1][str(channel)][lv1]})
    #sort list by time
    allPeaks.sort(key=getT)
    assignedPeaks = []
    print('assigning peaks')
    #cycle through elements in allPeaks
    #initialize timeA and maxSNR
    timeA = -999
    maxSNR = 0
    for peakData in allPeaks:
        #peakData contains channel, peak_x(timestep), peak_y (amplitude)
        SNR = getSNR(peakData, data, peakData["channel"])
        if abs(getT(peakData)-timeA)<=30:
            if SNR > maxSNR:
                peakA = peakData
        else:
            #append peakA to assignedPeaks
            if timeA != -999:
                assignedPeaks.append(peakA)
            #re-initialize to start new spike comparison
            peakA = peakData
            timeA = getT(peakA)
            maxSNR = getSNR(peakA, data, peakA["channel"])
    return(assignedPeaks)

#count spike occurances in each channel
def numChannelsPeaks(assignedPeaks, numT, numChannels = 256):
    #create dictionary where key is channel # and value is # of spike occurances
    #assigned to that channel
    #take output of assignPeaks as input
    channelCountPeaks = {}
    for channels in range(numChannels):
        channelCountPeaks[str(channels)] = 0
    for peakSet in assignedPeaks.values():
        for peaks in peakSet:
            channel = peaks["channel"]
            channelCountPeaks[str(channel)] += 1
    #change to spikes per second
    duration = numT / fs
    channelFreqPeaks = {key: value/duration for key,value in channelCountPeaks.items()}
    return(channelFreqPeaks, channelCountPeaks)

# =============================================================================
# #determine spikes that are complex, other functions were used in favour of 
# #this function in the end (below)
# def complexSpikeSearch(peaks):
#     #complex spikes defined as:
#     # spikes with relativeY >= 0.25 (relative spike magnitude compared to max
#     # spike magnitude in channel)
#     # spikes that show secondary spikes within 4 ms 
#     # (2 and 3 ms also tested, 4ms determined to be best)
#     complexSpikes = []
#     for channel in peaks[0].keys():
#         maxPeak = max(abs(np.array(peaks[1][channel])))
#         for lv1 in range(len(peaks[0][channel])-1):
#             peak_x = peaks[0][channel][lv1]
#             peak_y = peaks[1][channel][lv1]
#             #data[peak_x, channel]
#             if len(complexSpikes) !=0 and \
#                 abs(peak_x-complexSpikes[-1]["peak_x"]) <= 0.01*30000:
#                     #only one complex spike per 10 ms (avoids categorizing
#                     #secondary spikes as complex if there are tertiary spikes)
#                     continue
#             if peak_y / maxPeak >= 0.3 and \
#                 abs(peak_x-peaks[0][channel][lv1+1]) <= 0.004*30000:
#                     cSpike = {"peak_x":peak_x,"peak_y":peak_y,"channel":channel}
#                     complexSpikes.append(cSpike)
#     return(complexSpikes)
# =============================================================================
# =============================================================================
# #function to categorize spikes into complex and simple spikes, unused as
# # granular region should only have simple spikes
# def complexOrSimpleAssign(assignedPeaks, peaks):
#     complexSpikes = []
#     simpleSpikes = []
#     maxPeaks = {}
#     for channel in peaks[0]:
#         if int(channel) in missingChannels:
#             continue
#         maxPeaks[str(channel)] = max(abs(np.array(peaks[1][channel])))
#     for regionPeaks in assignedPeaks.values():        
#         for peak in regionPeaks:
#             #peak is dict containing channel, peak_x, peak_y
#             peak_x = peak["peak_x"]
#             peak_y = peak["peak_y"]
#             channel = str(peak["channel"])
#             maxPeak = maxPeaks[channel]
#             if len(complexSpikes) !=0 and \
#                 abs(peak_x-complexSpikes[-1]["peak_x"]) <= 0.004*30000:
#                     #only one complex spike per 10 ms
#                     continue
#             peaksI = peaks[0][channel].index(peak_x)
#             cSpike = {"peak_x":peak_x,"peak_y":peak_y,"channel":channel}#,"":}
#             if peak_y / maxPeak >= 0.3 and \
#                 peaksI+1 < len(peaks[0][channel]) and \
#                     abs(peak_x-peaks[0][channel][peaksI+1]) <= 0.004*30000:
#                         complexSpikes.append(cSpike)
#             else:
#                 simpleSpikes.append(cSpike)
#     return(complexSpikes, simpleSpikes)
# =============================================================================
def molecularComplexOrSimpleAssign(assignedPeaks, peaks):
    #similar to above code, with exception that it assumes no complex spikes in
    #granular region
    complexSpikes = []
    simpleSpikes = []
    maxPeaks = {}
    for channel in peaks[0]:
        if int(channel) in missingChannels:
            continue
        maxPeaks[str(channel)] = max(abs(np.array(peaks[1][channel])))
    for region, regionPeaks in assignedPeaks.items():        
        for peak in regionPeaks:
            #peak is dict containing channel, peak_x, peak_y
            peak_x = peak["peak_x"]
            peak_y = peak["peak_y"]
            channel = str(peak["channel"])
            maxPeak = maxPeaks[channel]
            if len(complexSpikes) !=0 and \
                abs(peak_x-complexSpikes[-1]["peak_x"]) <= 0.004*30000:
                    #only one complex spike per 10 ms
                    continue
            peaksI = peaks[0][channel].index(peak_x)
            cSpike = {"peak_x":peak_x,"peak_y":peak_y,"channel":channel}#,"":}
            if peak_y / maxPeak >= 0.3 and \
                peaksI+1 < len(peaks[0][channel]) and \
                    abs(peak_x-peaks[0][channel][peaksI+1]) <= 0.004*30000 and \
                        "R1" in region:
                        complexSpikes.append(cSpike)
            else:
                simpleSpikes.append(cSpike)
    return(complexSpikes, simpleSpikes)

#plotting function to visualize frequency of spikes in each channel
def plotSpikeFreq(channelFreq, spikeType, allShanks = True):
    if allShanks == True:
        fig, axes = plt.subplots(1,4,figsize=(11,7.5))
        fig.suptitle("{} Spike Frequency Plot by Channel (Hz)".format(spikeType), fontsize = 16)#use fontsize argument
        fig.text(0.55, 0.04, 'x (µm)', ha='center', fontsize = 12)
        fig.text(0.12, 0.5, 'y (µm)', va='center', rotation='vertical', fontsize = 12)
        plt.subplots_adjust(top = 0.85,bottom = 0.1, wspace = 0.1)
        for lv1 in range(4): #loop thru all functions
            shankChannels = channelGeometry[lv1*3:(lv1+1)*3,:].T
            shankSpikeAssign = np.zeros(shape = (21,3))
            #print(shankSpikeAssign)
            for rows in range(21):
                for cols in range(3):
                    if shankChannels[rows,cols] == -1:
                        continue
                    channel = shankChannels[rows,cols]
                    shankSpikeAssign[rows,cols] = channelFreq[str(channel)]
            axes[lv1].plot((np.ones(shape=(1,3))*boundaryPt[lv1]-0.5).flatten().tolist(),
                           color = "white")
            pcm = axes[lv1].matshow(shankSpikeAssign)
            axes[lv1].set_xticklabels(
                ["",0+lv1*250,15+lv1*250,30+lv1*250,45+lv1*250], fontsize = 10)
            #confirm the shank separation
            axes[lv1].set_yticklabels(["",0,-75,-150,-225,-300], fontsize = 10) 
            axes[lv1].set_title("Shank {}".format(lv1), fontsize = 12)
            cbar = plt.colorbar(pcm, ax = axes[lv1])
            cbar.ax.tick_params(labelsize = 10)
    else:
        for lv1 in range(4):
            shankChannels = channelGeometry[lv1*3:(lv1+1)*3,:].T
            shankSpikeAssign = np.zeros(shape = (21,3))
            for rows in range(21):
                for cols in range(3):
                    if shankChannels[rows,cols] == -1:
                        continue
                    channel = shankChannels[rows,cols]
                    shankSpikeAssign[rows,cols] = channelFreq[str(channel)]
            fig = plt.figure()
            plt.xlabel('x (µm)')
            plt.ylabel('y (µm)')
            plt.subplots_adjust(top = 1,bottom = 0.1, wspace = 0.1)
            ax1 = fig.add_subplot(111)
            ax1.plot(
                (np.ones(shape=(1,3))*boundaryPt[lv1]-0.5).flatten().tolist(),
                color = "white")
            pcm = ax1.matshow(shankSpikeAssign)
            ax1.set_xticklabels(["",0,15,30,45], fontsize = 10)
            ax1.set_yticklabels(["",0,-75,-150,-225,-300], fontsize = 10)
            ax1.set_title("Shank {}".format(lv1))
            cbar = plt.colorbar(pcm, ax = ax1)
            cbar.ax.tick_params(labelsize = 10)
            
# count number of spikes in channel
def channelSpikeCount(separatedSpikes, numT, numChannels = 256):
    #create dictionary where key is channel # and value is # of spikes
    #assigned to that channel
    #take output of assignPeaks as input
    #needed bc different data structure, can combine both functions that count 
    #number of spikes
    channelCountPeaks = {}
    for channels in range(numChannels):
        channelCountPeaks[str(channels)] = 0
    for peak in separatedSpikes:
        channel = peak["channel"]
        channelCountPeaks[str(channel)] += 1
    #change to spikes per second
    duration = numT / fs
    channelFreqPeaks = {key: value/duration for key,value in \
                        channelCountPeaks.items()}
    return(channelFreqPeaks, channelCountPeaks)

#reformat spike data to make it easier to work with
def spikesReformat(spikeList, region, shank, spikeTypeOrRegion):
    #first create an object kinda like peaks[0], ie. contains list of peaks_x
    #with dictionary keys being the channel
    spikeDict = {}
    for spike in spikeList:
        channel = spike['channel']
        if int(channel) not in region:
            continue
        keyStr = 'S'+shank+'_'+'C'+channel+'_'+spikeTypeOrRegion
        if keyStr in spikeDict.keys():
            spikeDict[keyStr].append(spike['peak_x'])
        else:
            spikeDict[keyStr] = [spike['peak_x']]
        #need to delete dictionary stuff with less than 10% of max in the given region
    maxSpikes = max([len(value) for value in spikeDict.values()])
    spikeDict = {key:value for key, value in spikeDict.items() if \
                 len(value) / maxSpikes >=0.1}
    return(spikeDict)

# plot average waveforms of each channel
def plotAveSpike(spikes, data, channel, spikeType, window = 0.005, pretime = -0.001):
    #window and pretime in seconds
    peaks_x = [spike['peak_x'] for spike in spikes 
                     if spike['channel']==str(channel)]
    arrayPeaks = np.zeros(shape = (len(peaks_x), int(fs*window)))
    #each row represents values from a spike
    for lv1, peak_x in enumerate(peaks_x):
        start = int(peak_x + fs*pretime)
        end = int(peak_x + (window+pretime)*fs)
        arrayPeaks[lv1, :] = data[start:end, int(channel)]
    avePeak = np.average(arrayPeaks, axis = 0)
    xs = np.divide(np.array(range(int(fs*window))), fs)
    
    plt.title('Channel {} Average {} Waveform'.format(str(channel), spikeType))
    plt.plot(xs, avePeak)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.show()
    return(arrayPeaks)

#utility function, and eliminate channels that show very few spikes
def listChannels(spikeList, region):
    channelList = []
    spikeDict = {}
    for spike in spikeList:
        channel = spike['channel']
        if int(channel) not in region:
            continue
        if channel in spikeDict.keys():
            spikeDict[channel].append(spike['peak_x'])
        else:
            spikeDict[channel] = [spike['peak_x']]
    maxSpikes = max([len(value) for value in spikeDict.values()])
    channelList = [key for key, value in spikeDict.items() if \
                 len(value) / maxSpikes >=0.1]
        #do not include channels that have less than 10% of the spikes seen in 
        # the channel with the most spikes
    return(channelList)

#return average spike waveform for all channels
def aveSpike(spikes, data, channel, window = 0.005, pretime = -0.001):
    #window and pretime in seconds
    #ax1 = plt.subplot(21,3,sensorIndex*3+3)
    peaks_x = [spike['peak_x'] for spike in spikes 
                     if spike['channel']==str(channel)]
    arrayPeaks = np.zeros(shape = (len(peaks_x), int(fs*window)))
    for lv1, peak_x in enumerate(peaks_x):
        start = int(peak_x + fs*pretime)
        end = int(peak_x + (window+pretime)*fs)
        arrayPeaks[lv1, :] = data[start:end, int(channel)]
    avePeak = np.average(arrayPeaks, axis = 0)
    return(avePeak)
#%% initial data processing - opening and filtering
# open file
data = h5py.File(filename,"r")
channel_data = data["channel_data"]
print(np.shape(channel_data))

# actual sensor data only from first 256 columns of channel_data
# other columns are currently blank (capacity for further probes)
# scale data to microvolts
channel_reading = channel_data[:,:256]*0.195 
data.close()

channel_reading_f = np.zeros(shape = np.shape(channel_reading[:,:]))
print(HTML("<h4>Analyzing channel: "))
for sensor in range(np.shape(channel_reading)[1]):
    if sensor in missingChannels:
        #missing channels do not contain any valid data
        continue
    print(sensor, end = ' ')
    x = np.arange(len(channel_reading[:,sensor]))
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = sosfreqz(sos, worN=2000)
    channel_reading_f[:,sensor] = butter_bandpass_filter(
        channel_reading[:,sensor], lowcut, highcut, fs, order=order)
del channel_data

# save filtered data
hf_bandpassed = h5py.File(path+'experiment_C20200330-162815_f.h5','w')
hf_bandpassed.create_dataset('channel_reading_f', data = channel_reading_f)
hf_bandpassed.close()

# filtered data can be reopened via the following code
hf_bandpassed = h5py.File(path+'experiment_C20200330-162815_f.h5','r')
channel_reading_f = hf_bandpassed.get('channel_reading_f')
# need to convert type from h5 dataset to numpy array
# note that it is not explicitly necessary - conversion is slow
# one can keep as is and continue analysis, given file remains open
channel_reading_f = np.array(channel_reading_f)
hf_bandpassed.close()

##do for a subset of channel_reading_f so that can iterate to determine hyperparameters
#channel_reading_f = channel_reading_f[:100000,:]

#%% given filtered data, determine spikes of neural data 

# Time to start looking for events (in seconds)
start_search = 0    
# Time to stop looking for events (in seconds), used more to limit time range
# in the event one is viewing a subset of data
end_search = 999    

peaks = event_search(channel_reading_f[:,:], start_search, end_search, 4)
# threshold value of 4 used, this was determined to be the most appropriate 
# value by Prof. John Welsh, this is a multiplier of the std of the smoothed 
# mean for every 10ms with 10% overlap

#check to make sure there are an appropriate number of spikes identified
# in each time range (previously used in part to determine appropriate params)
print('number of spikes in each sensor over time period')
for i in range(256):
    print('channel: ' + str(i) + ', spike array shape: '+ str(np.shape(peaks[0][str(i)])))#, np.shape(peaks2[0][str(i)]))
plotStackedGraph3(channel_reading_f[:30000,:],peaks,channelGeometry[0],
                  channelGeometry[1],channelGeometry[2])

#save peak data
pickle_out = open(path+'\experiment_C20200330-162815_peaks_x.pkl',"wb")
pickle.dump(peaks[0], pickle_out)
pickle_out.close()
pickle_out = open(path+'experiment_C20200330-162815_peaks_y.pkl',"wb")
pickle.dump(peaks[1], pickle_out)
pickle_out.close()

# filtered data can be reopened via the following code
pickle_in = open(path+'experiment_C20200330-162815_peaks_x.pkl',"rb")
pickle_in2 = open(path+'experiment_C20200330-162815_peaks_y.pkl', "rb")
peaks = [pickle.load(pickle_in), pickle.load(pickle_in2)]
pickle_in.close()
pickle_in2.close()
#%% visualize data

#plots channel readings and spikes for one channel (channel 0) for 1 second
#this can be easily altered to involve different time ranges, channels, etc.
fig, ax = plt.subplots(figsize = (50,10))
ax.plot(channel_reading_f[:fs,0])
ax.plot(peaks[0][str(0)],
    peaks[1][str(0)], 
    ls = 'none', marker = '*', markersize = 6, color = 'black')

#close up look @ peaks
print("individual peaks can be seen using the peekAtPeaks function.")
peekAtPeaks(peaks,channel_reading_f,channel = 0, index = 0)
#channel and index can be changed depending on which spike you want to view
#%%cross correlation
#determine cross correlation, to ensure location of channels plays a role
# in the cross correlation as expected
crossCorr = channelCrossCorrelate(peaks)
#plot crossCorr on a colormap, need to make sure geometry plays a role

#create a new channel mapping such that closest sensors are beside each other
#starting from one corner, arbitrarily choose say rows closer than columns when 
#channels are equidistant (or some way to create grid lines based on distance maybe)
shankMap = {}
for lv1 in range(noShanks):
    #do the shank(_)map as a function
    shankMap[str(lv1)] = [channelGeometry[lv1*3][0], channelGeometry[lv1*3+2][0]]
    for lv2 in range(1,np.shape(channelGeometry)[1]):
        for lv3 in range(3):
            shankMap[str(lv1)] += [channelGeometry[3*lv1 + lv3][lv2]]
plotCrossCorr(crossCorr, shankMap)

#%%post-processing of identified spikes

#run identified spieks through spikesOrNoise to eliminate any spikes determined
# to be noise
peaks_f_byShank = {}
for lv1 in range(noShanks):
    sensorGeometry = channelGeometry[3*lv1:3*(lv1+1),:]
    peaksShank_x = {key:value for key, value in peaks[0].items() if (int(key) in sensorGeometry)}
    peaksShank_y = {key:value for key, value in peaks[1].items() if (int(key) in sensorGeometry)}
    peaksShank = [peaksShank_x, peaksShank_y]
    peaks_f_byShank[str(lv1)] = spikesOrNoise([peaksShank_x, peaksShank_y], 
                                              sensorGeometry.T, 0.5)
peaks_f = [{**peaks_f_byShank[str(0)][0], **peaks_f_byShank[str(1)][0], 
            **peaks_f_byShank[str(2)][0], **peaks_f_byShank[str(3)][0]},
           {**peaks_f_byShank[str(0)][1], **peaks_f_byShank[str(1)][1], 
            **peaks_f_byShank[str(2)][1], **peaks_f_byShank[str(3)][1]}]
#fill missing channels with list = [0]
missingChannels = [10,11,26,27,36,37,52,53]
for chan in missingChannels:
    peaks_f[0][str(chan)]=[0]
    peaks_f[1][str(chan)]=[0]

#verify that spikesOrNoise has labelled some spikes as noise (as expected)
#print out # of spikes before spikesOrNoise
numSpikes = sum([len(value) for key, value in peaks[0].items() if (int(key) in channelGeometry)])
#print out # of spikes after spikesOrNoise
numSpikes_spikesOrNoise = sum([len(value) for value in peaks_f[0].values()])

#check to make sure that peaks_f have fewer spikes (to ensure correct)
for col in channelGeometry:
    for chan in col:
        if chan == -1:
            continue
        print(len(peaks_f[0][str(chan)]), end = ' ')
        print(len(peaks[0][str(chan)]), end = ' ')
        print('filtered has less: '+str(len(peaks_f[0][str(chan)])<=len(peaks[0][str(chan)])))
#visually verify that it looks like function has worked
plotStackedGraph3(channel_reading_f[:30000,:],peaks_f,channelGeometry[0],
                  channelGeometry[1],channelGeometry[2])

#check and replot cross correlation
#this more clearly shows neural regions than the initial cross correlation plots
crossCorr_f = channelCrossCorrelate(peaks_f)
plotCrossCorr(crossCorr_f, shankMap, 'Filtered')

#%% investigate SNR
SNR = []
depth = []
for channel in range(np.shape(channel_reading_f)[1]):
    if channel in missingChannels:
        continue
    noiseStd = np.std(channel_reading_f[:1000000, channel])
    peakStd = np.std(peaks_f[1][str(channel)])
    SNR.append(peakStd / noiseStd) #have this as x-axis
    #find depth (row) of channel
    #find index of channel in channelGeometry
    depth.append(21-int(np.where(channelGeometry == channel)[1]))#have this as y-axis
plt.plot(SNR, depth, 'o')
plt.xlabel('SNR')
plt.ylabel('depth (21 is at surface, 0 is deepest)')

#%% visually determined boundary between molecular layer and granular layer
boundaryPt = [4,18,13,10] #describes which row marks the transition point
# visually determined for now based on cross correlation
# possible to write a function or use a support vector machine to separate
# into the two regions

#%% make spikes unique in each region (molecular and granular)
#assign peaks to highest SNR channel in region
#recall unique spikes within a 1ms time range determined to be from same source
assignedPeaks = {}
for lv1 in range(noShanks):
    region1 = channelGeometry[lv1*3:(lv1+1)*3, :boundaryPt[lv1]]
    region2 = channelGeometry[lv1*3:(lv1+1)*3, boundaryPt[lv1]:]
    assignedPeaks["S{}R1".format(lv1)] = assignPeaks(peaks_f, 
                                                     channel_reading_f, 
                                                     region1)
    assignedPeaks["S{}R2".format(lv1)] = assignPeaks(peaks_f, 
                                                     channel_reading_f, 
                                                     region2)

#save results for assigned peaks
pickle_out = open(path+'experiment_C20200330-162815_assignedPeaks.pkl',"wb")
pickle.dump(assignedPeaks, pickle_out)
pickle_out.close()
#can re-load assigned peak data using the following
pickle_in = open(path+'experiment_C20200330-162815_assignedPeaks.pkl', "rb")
assignedPeaks = pickle.load(pickle_in)
pickle_in.close()

#%% check to make sure spike frequency makes sense and visualize data
#count number of spikes in each channel
channelFreqPeaks, channelCountPeaks = numChannelsPeaks(assignedPeaks, np.shape(channel_reading_f)[0])

numSpikes_spikesOrNoise_assigned = sum(channelCountPeaks.values())
print("Initial number of spikes = {}.".format(numSpikes))
print("Number of spikes after eliminating spikes not showing in some of adjacent channels = {} with Yield = {}".format(
    numSpikes_spikesOrNoise, round(numSpikes_spikesOrNoise/numSpikes,3)))
print("Number of unique spikes = {}. Yield is {} and {} comparing to initial data and adjacency filtered data respectively.".format(
    numSpikes_spikesOrNoise_assigned, round(numSpikes_spikesOrNoise_assigned/numSpikes,3), 
    round(numSpikes_spikesOrNoise_assigned/numSpikes_spikesOrNoise,3)))

#plot frequency of spikes onto visual chart
fig, axes = plt.subplots(1,4)
for lv1 in range(4): #loop thru all functions
    shankChannels = channelGeometry[lv1*3:(lv1+1)*3,:].T
    shankSpikeAssign = np.zeros(shape = (21,3))
    for rows in range(21):
        for cols in range(3):
            if shankChannels[rows,cols] == -1:
                continue
            channel = shankChannels[rows,cols]
            shankSpikeAssign[rows,cols] = channelFreqPeaks[str(channel)]
    axes[lv1].plot((np.ones(shape=(1,3))*boundaryPt[lv1]-0.5).flatten().tolist(),
                   color = "white")
    pcm = axes[lv1].matshow(shankSpikeAssign)
    plt.colorbar(pcm, ax = axes[lv1])

#%% categorize spikes into simple and complex spikes
# simple spikes are regions with only single spike
# complex spikes show spike activity following shortly after the initial spike
# and generally have larger amplitudes

# =============================================================================
# #based on first function
# complexSpikes = complexSpikeSearch(peaks)
# #note that complexSpikes is a list where each entry is a dictionary
# # {"peak_x":peak_x,"peak_y":peak_y,"channel":channel}
# #peek at results to make sure results make sense (expect much less complex
# # spikes than simple spikes)
# len(complexSpikes)
# =============================================================================

# =============================================================================
# complexSpikes, simpleSpikes = complexOrSimpleAssign(assignedPeaks, peaks)
# channelFreqPeaksC, channelCountPeaksC = channelSpikeCount(complexSpikes, np.shape(channel_reading_f)[0])
# channelFreqPeaks, channelCountPeaks = channelSpikeCount(simpleSpikes, np.shape(channel_reading_f)[0])
# plotSpikeFreq(channelFreqPeaksC) #plot complex spike frew
# plotSpikeFreq(channelFreqPeaks) #plot simple spike freq
# =============================================================================
    
#assume complex spikes only found in the molecular layer
complexSpikes2, simpleSpikes2 = molecularComplexOrSimpleAssign(assignedPeaks, 
                                                               peaks)

channelFreqPeaksC_g, channelCountPeaksC_g = channelSpikeCount(complexSpikes2, 
                                                              np.shape(channel_reading_f)[0])
channelFreqPeaks_g, channelCountPeaks_g = channelSpikeCount(simpleSpikes2, 
                                                            np.shape(channel_reading_f)[0])

plotSpikeFreq(channelFreqPeaksC_g, "Complex")
plotSpikeFreq(channelFreqPeaks_g, "Simple")
#if individual plots for each shank is desired, specify allShanks = False

#%% code to create csv file (one for complex spikes and one for simple spikes)

#create subsets based on shank and granular or molecular layer
#recall boundaryPt = [4,18,13,10]
spikeDict = {}
for lv1 in range(noShanks):
    region1 = channelGeometry[lv1*3:(lv1+1)*3, :boundaryPt[lv1]]
    region2 = channelGeometry[lv1*3:(lv1+1)*3, boundaryPt[lv1]:]
    spikeDict = {**spikeDict,
                 **spikesReformat(complexSpikes2, region1, str(lv1), 'CS'),
                 **spikesReformat(simpleSpikes2, region1, str(lv1), 'SS'),
                 **spikesReformat(simpleSpikes2, region2, str(lv1), 'GC')}

#change time units to seconds
spikeDict = {key:list(np.round(np.divide(value,fs),3))
             for key, value in spikeDict.items()}
#convert each list (value in dictionary) into a dictionary where each value is
#changed into a pd.Series object, then convert that new dict into a dataframe
spikeDF = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in spikeDict.items()]))
#save output
spikeDF.to_csv(path + 'experiment_C20200330-162815_spikes_in_s2.csv',
               index = False)

#%%data visualization, average spike waveform

#choose channel 200 to plot average spike wavform, but can change as desired
arrayPeaks = plotAveSpike(simpleSpikes2, channel_reading_f, 200, 'Simple')
arrayPeaks = plotAveSpike(complexSpikes2, channel_reading_f, 200, 'Complex')

#can also visualize all channels
#data visualization for all channels (instead of just a single channel)
xs = np.divide(np.array(range(int(fs*0.005))), fs)
for lv1 in range(noShanks):
    region1 = channelGeometry[lv1*3:(lv1+1)*3, :boundaryPt[lv1]]
    region2 = channelGeometry[lv1*3:(lv1+1)*3, boundaryPt[lv1]:]
    channelList = {}
    channelList['CS'] = listChannels(complexSpikes2, region1)
    channelList['SS'] = listChannels(simpleSpikes2, region1)
    channelList['GR'] = listChannels(simpleSpikes2, region2)
    for key, channels in channelList.items():
        if key == 'CS':
            spikes = complexSpikes2
        else:
            spikes = simpleSpikes2
        organized = [channel for channel in channels if int(channel)
                     in channelGeometry[3*lv1:3*(lv1+1),:].T.flatten()]
        fig, axes = plt.subplots(len(channels),1,figsize = (2,1.5*len(channels)))
        plt.subplots_adjust(hspace = 0.7)
        for lv2, channel in enumerate(organized):
            plotSpike = aveSpike(spikes,channel_reading_f,channel)
            axes[lv2].plot(xs, plotSpike)
            pltName = "C {} Ave. Waveform {} ".format(channel, key)
            axes[lv2].set_title(pltName, fontsize = 12)
        plt.savefig(path+"S{}_{}.png".format(lv1, key), bbox_inches = "tight")
        plt.show()

# =============================================================================
# UNUSED CODE, previous methodologies that were determined to not be as precise
# as required
#
# #not sure if below is used
# def peekAtPeaks(peaks,data,channel,index=0):
#     #will likely want to put increased functionality, make it optional to use 
#     #either index or approximate location of spike to determine which spike to plot
#     peakI = peaks[0][str(channel)][index]
#     #note, 45 timepoints in 1.5 ms
#     #change ^ slightly...
#     plt.plot(data[peakI-30:peakI+120,channel])
#     #y axis mv
#     #x axis s
#     return None
# #example of using peekAtPeaks()
# peekAtPeaks(peaks,channel_reading_f,200,6)
# 
# def spikeCharacter(data, assignedPeaks, peaks):
#     #for now just add spike width (define when reach zero)
#     #note can also define spike width as width that crosses threshold
#     #spike height (already should be in assignSpikes)
#     #whether spikes exist in quick succession (will have to define timeframe)
#     #slope maybe (though do not do this for now)
#     spikeCharacteristics = []
#     for regionPeaks in assignedPeaks.values():
#         #cycle through and characterize every peak
#         for peak in regionPeaks:
#             #peak is a dictionary that contains channel, peak_x, peak_y
#             peakChar = peak.copy()
#             #find peak width here...
#             #may want to replace comparing with 0 to comparing with thresh
#             peakSign = np.sign(peak["peak_y"])
#             for lv1 in range(150):# look up to _ spaces to the right and left
#                 #data is a #t x #chan array
#                 if peakSign * data[peak["peak_x"]-lv1,peak["channel"]] <= 0: #error bc never exactly zero
#                     #do not need to record, can just assume from lv1
#                     break
#             for lv2 in range(150):
#                 if peakSign * data[peak["peak_x"]-lv1,peak["channel"]] <= 0:
#                     #do not need to record, can just assume from lv2
#                     break
#             peakChar["width_0"] = lv2+lv1
#             #put STD
#             peakChar["STD1s"] = getSNR(peak, data, peak["channel"])
#             peakChar["relative_y"] = peak["peak_y"] / max(abs(np.array(peaks[1][str(peak["channel"])])))
#             #has other spikes close by
#             #use peaks for this
#             peaksI = peaks[0][str(peak["channel"])].index(peak["peak_x"])
#             #use 30 bc search for spikes beside in 2ms interval
#             if abs(peak["peak_x"] - peaks[0][str(peak["channel"])][peaksI-1]) < 60 and\
#                 np.sign(peaks[0][str(peak["channel"])][peaksI-1]) * np.sign(peak["peak_y"])\
#                     or abs(peak["peak_x"] - peaks[0][str(peak["channel"])][peaksI+1]) < 60 and\
#                         np.sign(peaks[0][str(peak["channel"])][peaksI-1]) * np.sign(peak["peak_y"]):
#                             peakChar["closeby"] = 1
#             else:
#                 peakChar["closeby"]=0
#                 #should just change closeby to show how close the closest adjacent peak is
#             spikeCharacteristics.append(peakChar)
#            # print(len(spikeCharacteristics))
#    # print(peakChar.items())
#     return(spikeCharacteristics)
# spikeCharacteristics = spikeCharacter(channel_reading_f, assignedPeaks, peaks)
# amps = np.array([peak["peak_y"] for peak in spikeCharacteristics])
# #amps /= max(amps)
# widths = np.array([peak["width_0"] for peak in spikeCharacteristics])
# widths = widths / max(widths)
# STD1s = np.array([peak["STD1s"] for peak in spikeCharacteristics])
# #STD1s = STD1s / max(STD1s)
# times = np.array([peak["peak_x"] for peak in spikeCharacteristics]) / fs
# closeBy = np.array([peak["closeby"] for peak in spikeCharacteristics])
# relativeY = np.array([peak["relative_y"] for peak in spikeCharacteristics])
# SNR = amps / STD1s
# SNR = SNR / max(SNR)
# #instead of SNR maybe should look at it as size of peaks relative to other peaks
# #in that given channel (provided by object spikes)
# plt.plot(times[:1000],widths[:1000], "g*")
# plt.plot(times[:1000],SNR[:1000], "b*")
# plt.plot(closeBy, relativeY, "o")
# plt.xlabel("widths")
# plt.ylabel("SNR")
# plt.show()
# 
# for lv1, spike in enumerate(spikeCharacteristics):
#     if widths[lv1] >= 0.7:
#         #print(lv1, spike)
#         plt.plot(channel_reading_f[spike["peak_x"]-50:spike["peak_x"]+50,spike["channel"]])
#         plt.show()
#         #plt.axis('off')
# 
# choose = 109 #chosen sensor for manual peak comb
# for lv1 in range(40):
#     plt.figure(figsize=(10,10))
#     for lv2 in range(5): 
#         ax1 = plt.subplot(5,1,lv2+1)
#         ax1.plot(channel_reading_f[(lv1*5+lv2)*5000:(lv1*5+lv2+1)*5000,choose])
#         temppeaks, tempevents = event_search(
#             channel_reading_f[:,[choose]], 
#             (lv1*5+lv2)*5000/30000, (lv1*5+lv2+1)*5000/30000, 4)
#         
#         ax1.plot(np.array(temppeaks[0][str(0)])-(lv1*5+lv2)*5000,
#             temppeaks[1][str(0)], #key is 0 here bc event_search doesnt consider channel #
#             ls = 'none', marker = '*', markersize = 6, color = 'black')
#     plt.savefig("C:\\Users\\Daniel Wijaya\\Documents\\EEG data\\t{}".format(lv1))
#     plt.close()
# 
# nPeaks = []
# for lv1 in range(40):
#     for lv2 in range(5): 
#         temppeaks, tempevents = event_search(
#             channel_reading_f[:,[choose]], 
#             (lv1*5+lv2)*5000/30000, (lv1*5+lv2+1)*5000/30000, 4)
#         nPeaks.append(len(temppeaks[0][str(0)]))
# 
# probableComplex = [0, 58, 70, 104, 106, 131, 132, 149, 195, 231, 266, 279, 298, 303, 329, 364, 464, 527, 550, 559, 627, 665, 717, 746, 767, 801, 831, 860, 864, 890, 936, 969, 1001, 1008, 1036, 1065]
# def spikeCharacter2(data, probableComplex, peaks):
#     spikeCharacteristics = []
#     for peakI in probableComplex:
#         peakChar = {'channel' : 109, 'peak_x':peaks[0][str(109)][peakI], 
#                     'peak_y':peaks[1][str(109)][peakI]}
#         peakChar["relative_y"] = peakChar['peak_y'] / max(abs(np.array(peaks[1][str(109)])))
#         closestPeak = min(abs(peakChar['peak_x'] - peaks[0][str(109)][peakI-1]), abs(peakChar['peak_x'] - peaks[0][str(109)][peakI+1]))
#         peakChar['closeby'] = closestPeak
#         spikeCharacteristics.append(peakChar)
#     return(spikeCharacteristics)
# spikeCharacteristics2 = spikeCharacter2(channel_reading_f, probableComplex, peaks)
# closeBy2 = np.array([peak["closeby"] for peak in spikeCharacteristics2])
# relativeY2 = np.array([peak["relative_y"] for peak in spikeCharacteristics2])
# plt.plot(closeBy2, relativeY2, "o")
# plt.show()
# 
# 
# 
# for i in probableComplex:
#     peekAtPeaks(peaks,channel_reading_f,109,i)
#     plt.savefig("C:\\Users\\Daniel Wijaya\\Documents\\EEG data\\spikeAt{}".format(i))
#     plt.close()
# 
# besideSpikes = []
# for lv1, close in enumerate(closeBy):
#     if close == 1:
#         besideSpikes.append(spikeCharacteristics[lv1]['peak_x'])
#         #print(spikeCharacteristics[lv1]['peak_x'])
# besideSpikes.sort()
# #besideSpikes = [160, 161, 161, 161, 162, 253, 254, 260, 260, 260, 260, 267, 267, 1355, 1491, 8446, 10496, 12270, 12540, 22758, 27771, 27824, 28288, 29492, 30072, 30107, 33357, 37067, 37682, 37734, 38500, 42079, 42082, 42082, 42193, 42194, 42194, 42194, 42194, 42194, 42194, 42205, 42213, 42214, 42214, 42218, 42218, 42219, 48758, 49011, 51436, 54350, 55322, 57690, 58102, 62522, 65744, 66440, 68310, 70340, 70381, 72787, 72822, 73170, 74849, 74966, 74966, 74966, 74966, 74966, 74966, 74966, 74967, 74972, 75050, 75098, 77543, 77544, 77546, 77546, 81029, 84613, 86294, 89940, 89940, 89994, 93726, 94245, 94490, 94545, 94546, 95041, 95041, 95041, 95041, 95042, 95042, 95071, 95072, 95072, 95277, 96292, 96305, 97149, 97361, 97393, 98303, 98347, 98600, 98610, 99686, 99809, 99849, 101435, 103824, 106215, 107046, 107734, 107735, 107735, 107735, 107735, 107735, 107741, 107742, 107756, 107757, 108206, 108950, 112390, 115140, 115615, 120774, 122955, 123951, 125618, 140162, 140507, 140507, 140507, 140507, 140507, 140507, 140507, 140507, 140535, 147157, 148791, 155562, 158271, 159476, 161530, 161531, 161532, 163576, 164806, 165159, 171489, 172108, 172154, 172194, 173269, 173269, 173269, 173269, 173269, 173269, 173270, 173276, 173303, 173303, 173304, 173304, 173304, 173304, 173305, 173305, 183291, 183348, 184192, 184192, 184193, 184802, 184936, 185860, 185860, 185919, 187778, 188627, 188642, 190682, 190682, 190683, 190683, 190683, 191330, 191331, 194037, 195576, 197859, 199387, 202834, 205918, 206034, 206034, 206035, 206035, 206036, 206036, 209536, 210913, 211070, 214055, 214146, 216872, 217925, 219693, 226059, 226097, 228492, 230628, 230840, 230891, 231552, 231785, 231816, 234228, 234228, 234228, 234228, 234229, 235303, 235870, 235904, 235943, 236146, 236301, 237315, 238804, 238805, 238805, 238805, 238806, 254034, 254162, 254162, 254162, 257793, 257847, 260098, 260734, 260808, 270633, 271271, 271568, 271568, 271568, 271568, 271569, 271605, 284033, 284033, 290427, 295135, 295135, 304339, 304339, 304339, 304340, 304340, 321159, 334931, 341594, 343770, 343770, 347394, 348515, 348516, 356561, 369835, 369835, 369835, 369835, 369835, 369872, 371853, 371853, 373475, 373475, 373475, 373475, 373476, 375412, 379356, 379388, 380290, 387410, 390784, 390784, 390784, 390784, 393938, 399488, 399488, 399488, 399488, 402604, 402604, 402605, 402605, 402648, 402648, 426378, 435373, 435374, 435374, 435374, 440024, 454993, 468141, 468141, 468141, 468177, 468177, 472491, 472523, 494289, 494315, 500908, 500908, 503920, 507010, 512615, 512715, 512715, 512775, 519020, 521331, 533674, 533675, 533714, 566443, 599210, 631980]
# besideSpikes_f = []
# for lv1, spike_x in enumerate(besideSpikes):
#     if abs(spike_x - besideSpikes[lv1-1]) > 150:
#         besideSpikes_f.append(spike_x)
#         
# 
# probableIndex = []
# for i in probableComplex:
#     probableIndex.append(peaks[0][str(109)][i])
#     
# def complexSpikeSearch(data, assignSpikes, peaks, peaks_f):#or use assignedPeaks):
#     #unsure at what point to analyze data... start by looking at fully
#     #processed data but this might be wrong (may have to look at original data)
#     #and run processing function on it afterwards
#     return()
# =============================================================================
