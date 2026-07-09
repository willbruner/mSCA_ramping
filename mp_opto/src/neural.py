import numpy as np
from tqdm import tqdm

def bin_spiketimes(spike_times,dt,wdw_start,wdw_end):
    """
    Function that puts spikes into bins
	Stolen from Josh's Neural_Decoding.preprocessing_funcs, modified to work on list of arrays

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=len(spike_times) #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in tqdm(range(num_neurons)):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data

def bin_spiketimes_bounds(spike_times, dt, bounds):
    #bounds must be Nx2 array of start/stop times

    return [bin_spiketimes(spike_times, dt, bound[0], bound[1]+1) for bound in
            bounds]

def bin_timeseries(timeseries, binsize):
    n = timeseries.size
    remainder = n%binsize
    if remainder!=0:
        timeseries = timeseries[:-remainder]

    return timeseries.reshape(n//binsize, binsize) @ np.ones(binsize)

def bin_neural(neural, binsize):
    #samples x neurons
    binned = np.empty((neural.shape[0]//binsize, neural.shape[1]))
    for i in range(neural.shape[1]):
        binned[:,i] = bin_timeseries(neural[:, i], binsize)
    return binned

def bin_neural_bounds(neural, binsize, bounds):
    output = []
    for bound in bounds:
        trial_neural = neural[bound[0]:bound[1]+1]
        output.append(bin_neural(trial_neural, binsize))

    return output

def bin_neural_trials(neural_list, binsize):
    #list of neural (samples x neurons)
    return [bin_neural(neural, binsize) for neural in neural_list]




