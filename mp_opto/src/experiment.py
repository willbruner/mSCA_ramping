import numpy as np
import pandas as pd
import copy
import pdb

def climbing_bounds_from_logical(isclimbing):
    isclimbing = np.squeeze(isclimbing)
    climb_change=isclimbing[1:].astype(int)-isclimbing[:-1].astype(int)
    climb_start= climb_change==1
    climb_end= climb_change==-1

    climb_start_idx= np.where(climb_start)[0]
    climb_end_idx= np.where(climb_end)[0]

    num_total_climbs=len(climb_start_idx)

    climbing_bounds = np.column_stack((climb_start_idx, climb_end_idx))

    return climbing_bounds

def getLaserBounds(laser, laser_thresh=0.1, climbing_logical=None,
        constant_laser = True):
    #this only works for CONSTANT LASER TIMES
    #

    laser_idxs=laser>laser_thresh #When laser is above thresh

    #Find start and end of laser
    laser_change=laser_idxs[1:].astype(int)-laser_idxs[:-1].astype(int)
    laser_start= laser_change==1
    laser_end= laser_change==-1

    #Get start/end indices for bins
    laser_start_idx= np.where(laser_start)[0]
    laser_end_idx = np.where(laser_end)[0]

    if laser_end_idx[0] == 0:
        print('co12 handfix lol')
        laser_end_idx=laser_end_idx[1:]
    

    if constant_laser:
        post = np.median(laser_end_idx - laser_start_idx)
        laser_end_idx = laser_start_idx + post

    num_lasers=len(laser_start_idx)

    laser_bounds = np.zeros((num_lasers, 2))
    laser_bounds[:,0] = laser_start_idx
    laser_bounds[:,1] = laser_end_idx

    laser_bounds = laser_bounds.astype(int) 
    #probably not necessary but something weird is happening

    if climbing_logical is not None:
        keeps = goodLasers(laser_bounds, climbing_logical)
        num_discards = len(keeps) - len(laser_bounds)
        laser_bounds = laser_bounds[keeps, :]
        if num_discards > 0: 
            print(f'discarding {num_discards} lasers that \
                    dont occur during climbing')
    

    laser_trial_data = arr2trials(laser, bounds=laser_bounds, 
            duration = laser.shape[0], concat=False)
    laser_powers = [np.max(laser_trial) for laser_trial in laser_trial_data]
    return laser_bounds, laser_powers

def concatTrials(arr):
    # MAYBE NOT USED ANYMOR

    # trial length x neurons x trials -> time series  x neurons
    # not vectorized but this is easier for me
    # use flatten_list (from wiener_filter) for a list
    print('gentle warning: must be trials_length x neurons x trial  for concat')
    assert arr.ndim == 3, 'something weird about dims for concattrials'

    num_trials = arr.shape[2]
    num_neurons = arr.shape[1]
    trial_length = arr.shape[0]
    
    output = np.zeros((num_trials * trial_length, num_neurons))

    #some truly terrible code, np reshape is impossible
    for i in range(num_neurons):
        temp = []
        for j in range(num_trials):
            temp.append(arr[:,i,j])
        output[:, i] = np.array(temp).flatten()
    return output

def bounds2Logical(bounds, duration=None):
    #bounds is a Nx2 array
    if duration is None:
        duration = bounds[-1,1]
    logical = np.zeros(duration)
    for bound in bounds:
        logical[bound[0]:bound[1]] = True
    return logical

def logical2Bounds(logical):
    #messy messy messy
    temp = np.insert(logical, 0, 0)
    new_logical = np.append(temp, 0)
    change = np.diff(new_logical)
    start= change==1
    end= change==-1
    start_idx = np.where(start)[0]
    end_idx = np.where(end)[0]

    
    return (np.vstack((start_idx, end_idx)).T + 1)


def arr2trials(arr, bounds, duration, concat=True):
    #time series of (samples x neurons), divide into trials based on bounds
    #might not work for single time series data
    index = bounds2Logical(bounds, duration).astype(bool)
    temp = arr[index, ...]
    if concat:
        return temp
    else:
        seams = getSeamsFromBounds(bounds)
        return unstitchSeams(temp, seams)

def getSeamsFromList(mylist):
    return np.cumsum([trial.shape[0] for trial in mylist])

def getSeamsFromBounds(bounds, binsize=1):
    #binsize is if you want to downsample
    return np.cumsum(np.diff(bounds, axis=1)//binsize)
    
def unstitchSeams(arr, seams):
    #maybe only for diyas brain, but if we have concatenated based on Bounds
    #and then want to break em apart again, use this
    #useful for stuff like PCA or smoothing where you want to apply to 
    #everything, and then re-obtain tria
    #assert arr.shape[0] == bounds[-1,-1]

    #either directly pass in seams, or get it from bounds
    
    return np.split(arr, seams[:-1].astype(int), axis=0)
    #if evenBoundsBool(seams):
    #    return np.moveaxis(output, 0, -1)
    #else:
    #    return output

def evenBoundsBool(bounds):
    #returns true if bounds divide into same length
    #bounds is nx2 array

    diffy = np.diff(bounds)
    return np.all(diffy == diffy[0])

def list2array(mylist):
    #trials of timeseries x neurons, in list form
    #convert to array
    #TRIALS MUST BE EVEN!!
    
    return np.moveaxis(np.array(mylist), 0, 2)

def omitBoundInBounds(bigBound, smallBound):
    #we omit anything where smallbound is active (and not in bigbound)
    bigLogical = bounds2Logical(bigBound, bigBound[-1,1])
    smallLogical = bounds2Logical(smallBound, bigBound[-1,1])

    newLogical = ((bigLogical-smallLogical)==1)*1
    return logical2Bounds(newLogical)

def reBoundInBounds(bigBound, smallBound):
    #if we were to be excluding anything not in bigbound
    #then we redo smallBound
    #duration will be if you want to turn back into a logical

    bigBoundLogical = bounds2Logical(bigBound, duration=bigBound[-1,1]).astype(bool)
    smallBoundLogical = bounds2Logical(smallBound,duration=bigBound[-1,1])
    smallBoundLogicalCut =smallBoundLogical[bigBoundLogical]
    bigBoundLogicalCut  = (bigBoundLogical*1)[bigBoundLogical]

    duration = np.size(bigBoundLogicalCut)
    return logical2Bounds(smallBoundLogicalCut), duration



def goodLasers(laser_bounds, climbing_logical): 
    
    return [np.any(climbing_logical[bound[0]:bound[1]]==1) for bound in laser_bounds]

def minimumBoundSize(bounds, min_size=100):
    newbound = [bound for bound in bounds if (bound[1]-bound[0]) > min_size]
    return np.array(newbound)

def adjustBounds(bounds, pre, post):
    new_bounds = copy.deepcopy(bounds)
    if new_bounds[0,0] - pre < 0:
        print('removing first laser')
        new_bounds = new_bounds[1:,:]
    new_bounds[:,0] = new_bounds[:,0] - pre
    new_bounds[:,1] = new_bounds[:,1] + post
    return new_bounds

def antiBounds(bounds, duration):
    logical = bounds2Logical(bounds, duration=duration)
    print(logical.shape)
    temp = 1-logical
    return logical2Bounds(temp)

