# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:24:28 2020
SNR plotting project
Requires that initial bandpass filtering already occurs
See SpikeDetection_.py files to generate these 
@author: Daniel
"""
# -*- coding: utf-8 -*-
#imports and constants
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random as rand

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
missingChannels = [10,11,26,27,36,37,52,53]
numChannels = 256
#%% functions

# create list of neural spike activity and spikes that are assumed to be noise
def findPeaks(raw, start, end, thresh): 
    start = int(start * fs)
    end = min(int(end*fs), np.shape(raw)[0])
    peaks_x = {} # dict to catch peak times 
    peaks_y = {} # dict to catch peak amps
    noisePeaks_y = {} # dict to catch noise peak amps
    print('Analyzing channel: ', end = ' ')
    for j in range(np.shape(raw)[1]):
        #skip over missing channels
        if j in missingChannels:
            peaks_x[str(j)] = []
            peaks_y[str(j)] = []
            noisePeaks_y[str(j)] = []
            continue
        print(j, end = ' ')
        i = start
        peak_x = []
        peak_y = []
        noise_y = []
        rep = 0
        ssd = np.std(raw[0:fs*5,j])*thresh
        while i < end-11: # search all the rows (time)
            if i%(fs*5) == 0:
                ssd = np.std(raw[i:i+fs*5,j])*thresh #defined over 5s window
            if abs(raw[i,j]) > ssd  and abs(raw[i,j]) > max(abs(raw[i+1:i+10,j])) \
                and abs(raw[i,j]) > abs(raw[i-1,j]) and i-10 >= 0: 
                #assume peaks can last 10 ms
                #capture peak data
                peak_x.append(i)
                peak_y.append(raw[i,j])
                rep = rep + 1
            elif abs(raw[i,j]) > abs(raw[i-1,j]):
                #capture noise peak data
                noise_y.append(raw[i,j])
            i = i + 1
        peaks_x[str(j)] = peak_x
        peaks_y[str(j)] = peak_y
        noisePeaks_y[str(j)] = noise_y
    peaks = [peaks_x,peaks_y]
    return(peaks, noisePeaks_y)

#find SNR, defined as largest spike amplitude / ave noise peak
def SNRmod(peaks, noisePeaks_y):
    #peaks is a list, idx 0 is x, idx 1 is y
    #need to convert peaks_y and noisePeaks_y elements to numpy array to use
    #abs() function
    peaks_x, peaks_y = peaks
    maxAmp = {channel:max(abs(np.array(peakList))) for channel, peakList in 
              peaks_y.items() if int(channel) not in missingChannels}
    indexMax = {}
    for key, value in maxAmp.items():
        #print(key, value)
        indexMax[key] = peaks_x[key][
            list(abs(np.array(peaks_y[key]))).index(value)
            ]
    #calculate ave noisePeak
    aveNoise = {channel:np.mean(np.abs(np.array(peakList))) for 
                channel, peakList in noisePeaks_y.items() if int(channel) not in
                missingChannels}
    SNR = {}
    for i in range(numChannels):
        if i not in missingChannels:
            SNR[str(i)] = maxAmp[str(i)]/aveNoise[str(i)]
    return(SNR, aveNoise, indexMax)

#function to concatenate all noise together (without the spike activity)
def concatenateNoise2(peaks_x, channel_reading_f):
    #concatenates noise (eliminates all signals determined to be neural activity)
    spikeW = int(0.001*fs)#fs = 30000, so spikeW = 30
    noise = {}
    print('Analyzing channel - ', end = ' ')
    for channel, peakList in peaks_x.items():
        print(' {}'.format(channel), end = ' ')
        if len(peakList) > 0:
            #initialize first noise segment (up til first spike)
            noise[channel] = channel_reading_f[:peakList[0]-spikeW, int(channel)]
            val2P = noise[channel][-1]
        for lv1 in range(len(peakList)-1):
            T1 = peakList[lv1]
            T2 = peakList[lv1+1]
            #check if enough noise between neural spikes to include
            # do not want very small fragments of noise
            if T2 - T1 > 150:
                #loop through first x values of range to try to get close to val2
                for lv2 in range(T1 + spikeW, T1 + 2*spikeW):
                    #check for crossover point
                    if np.prod(channel_reading_f[lv2:lv2+2, int(channel)]):
                        T1 = lv2+1
                if T1 == peakList[lv1]:
                    #if crossover point not found, just choose timepoint where 
                    #reading is closest to val2P
                    checkWindow = list(abs(
                        channel_reading_f[T1 + spikeW:T1+2*spikeW,
                                          int(channel)]-val2P))
                    T1 = checkWindow.index(min(checkWindow))
                #concatenate based on new chosen T1
                noise[channel] = np.concatenate(
                    (noise[channel], channel_reading_f[T1:T2, int(channel)])
                    )
                #reset val2P to new endpt
                val2P = noise[channel][-1]
    return noise
#%%
#to open the file
path = 'C:\\Users\\Daniel Wijaya\\Downloads\\'
hf_bandpassed = h5py.File(path + 'experiment_C20200330-162815_f.h5','r')
channel_reading_f = hf_bandpassed.get('channel_reading_f')[()]
#[()] converts this into a numpy array
hf_bandpassed.close()

#characterize spike and noise peak data
start_search = 0    # Time to start looking for events (in seconds)
end_search = 999      # Time to stop looking for events (in seconds)

peaks, noisePeaks_y = findPeaks(channel_reading_f[:,:], start_search, end_search, 4)

SNR, aveNoise, indexMax = SNRmod(peaks, noisePeaks_y)

#plot noise on histogram
plt.figure()
plt.hist(aveNoise.values(), bins = 256)
plt.title('Noise Histogram, all channels')
plt.xlabel('Mean Noise Spike Amplitude [µV]')
plt.ylabel('Number of Channels')
#plot SNR as histogram of values for all channels
plt.figure()
plt.hist(SNR.values(),bins = 256)
plt.title('SNR Histogram, all channels')
plt.xlabel('SNR (max spike amplitude / mean noise spike amplitude)')
plt.ylabel('Number of Channels')
#plot samples of channels with highest SNR
#say good SNR for channels with SNR > 80
SNRgood = {key:value for key,value in SNR.items() if value > 80}
noiseGood = {key:value for key,value in aveNoise.items() if key in SNRgood.keys()}
plt.figure()
plt.hist(noiseGood.values(), bins = len(noiseGood))
plt.title('Noise Histogram, channels with SNR > 80')
plt.xlabel('Mean Noise Spike Amplitude [µV]')
plt.ylabel('Number of Channels')
plt.figure()
plt.hist(SNRgood.values(), bins = len(SNRgood))
plt.title('SNR Histogram, channels with SNR > 80')
plt.xlabel('SNR (max spike amplitude / mean noise spike amplitude)')
plt.ylabel('Number of Channels')

#plot records of good SNR channels
halfWindow = 750
randomSpike = True
for key in SNRgood.keys():
    #for random time point in channel, uncomment line below
    if randomSpike:
        startT = rand.choice(peaks[0][key])
        figName = 'Channel Random Peak {} (At Time {}).png'.format(key, round(startT/fs,2))
    else:
        startT = indexMax[key]
        figName = 'Channel Max Peak {} (At Time {}).png'.format(key, round(startT/fs,2))
    plt.figure()
    xs = np.array(range(max(0,startT-halfWindow), startT+halfWindow))/fs#x values for plot
    plt.plot(xs, channel_reading_f[max(0, startT-halfWindow):startT+halfWindow,int(key)])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('Channel Max Peak {} (At Time {})'.format(key, round(startT/fs,2)))
    plt.savefig(path+'SNR plots\\' + figName)

# compile all noise into one object (for easier analysis)
noiseReadings = concatenateNoise2(peaks[0],channel_reading_f[:,:])

#std dev of noise
noiseSTD = {}
SNR2 = {}
#calculate new SNR based on ^ definition of noise
for channel, channelNoise in noiseReadings.items():
    noiseSTD[channel] = np.std(channelNoise)
    SNR2[channel] = abs(channel_reading_f[indexMax[channel],
                                          int(channel)]) / noiseSTD[channel]

#plot on histogram
plt.figure()
plt.hist(noiseSTD.values(), bins = 256)
plt.title('Noise Histogram, all channels')
plt.xlabel('Noise Standard Deviation [µV]')
plt.ylabel('Number of Channels')
#plot SNR as histogram of values for all channels
plt.figure()
plt.hist(SNR2.values(),bins = 256)
plt.title('SNR Histogram, all channels')
plt.xlabel('SNR (max spike amplitude / noise standard deviation)')
plt.ylabel('Number of Channels')
#plot samples of channels with highest SNR
#say good SNR for channels with SNR > 80
SNRgood2 = {key:value for key,value in SNR2.items() if value > 60}
noiseGood2 = {key:value for key,value in noiseSTD.items() if key in SNRgood.keys()}
plt.figure()
plt.hist(noiseGood2.values(), bins = len(noiseGood))
plt.title('Noise Histogram, channels with SNR > 60')
plt.xlabel('Mean Noise Spike Amplitude [µV]')
plt.ylabel('Number of Channels')
plt.figure()
plt.hist(SNRgood2.values(), bins = len(SNRgood))
plt.title('SNR Histogram, channels with SNR > 60')
plt.xlabel('SNR (max spike amplitude / noise standard deviation)')
plt.ylabel('Number of Channels')

#for specific peak, can choose based on spike index in peaks list
randomSpike = False
for key in SNRgood2.keys():
    if randomSpike:
        startT = rand.choice(peaks[0][key])
        figName = 'Channel Random Peak {} (At Time {})(RMS).png'.format(key, round(startT/fs,2))
    else:
        startT = indexMax[key]
        figName = 'Channel Max Peak {} (At Time {})(RMS).png'.format(key, round(startT/fs,2))
    plt.figure()
    xs = np.array(range(max(0,startT-halfWindow), startT+halfWindow))/fs#x values for plot
    plt.plot(xs, channel_reading_f[max(0, startT-halfWindow):startT+halfWindow,int(key)])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('Channel Max Peak {} (At Time {})'.format(key, round(startT/fs,2)))
    plt.savefig(path+'SNR plots\\' + figName)

plt.psd(noiseReadings[str(0)], Fs = fs)
plt.title('Noise Power Spectral Density Channel {}'.format(0))