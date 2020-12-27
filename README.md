# Willow

## Overview

Analysis of ultra-high density neural recordings using the Willow system. These recordings must be scaled and filtered before analysis using spike sorting scripts such as Mountainsort, Kilosort, etc. I also wrote a spike sorting algorithm in `SpikeDetection.py` as I do not have access to a CUDA compatible device (required for the other sorting scripts).

 `looking_at_data.py`: this python script that can be used to plot the channel readings on a probe on subplots positioned to match the channel positions found on the probe.
 
 `SpikeDetection.py`: this python script that can process and visualize the neural data. Initial processing involves scaling data, applying a butterpass filter, and a spike detection algorithm written with the advice of Prof. John Welsh. Spikes are detected by taking peaks that exceed the standard deviation of a 5 second window of neural readings multiplied by a chosen constant. More details can be seen by reading the `event_search` function in this script. Additional processing steps available include cross-correlating (and plotting) the channels activity, an additional step to filter out noise, spike assignment to a single channel, spike sorting, and plotting of spike frequency.
 
 `SNR_plot_project.py`: this python script analyzes the signal-to-noise ratio of the channels and creates visual plots for them.
 
 `granular_molecular_relationship_MLmapping.py`: this python script uses the neral spike data to generate training data in order to create a relationship between granular channels and a molecular channel with logistic regression. The weights of this function can then be used to visualize said relationship.
 
## Recommended Workflow



