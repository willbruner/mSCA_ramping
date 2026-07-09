import numpy as np
from src.utils.gen_utils import *
from src.wiener_filter import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

def get_top_models(ratio_vector, r2_vector, edges=None, num_bins=15):
    if edges is None:
        _, edges = np.histogram(ratio_vector, bins=num_bins)
    if type(ratio_vector) is list:
        ratio_vector = np.array(ratio_vector)
    if type(r2_vector) is list:
        r2_vector = np.array(r2_vector)

    idxs = np.arange(len(ratio_vector))

    ratio_keep = []
    r2_keep = []
    idxs_keep = [] 

    
    for i in range(edges.size-1):
        condition1 = ratio_vector > edges[i]
        condition2 = ratio_vector < edges[i+1]
        indices = np.argwhere(np.logical_and(condition1,condition2))
        if indices.size < 1:
            #r2_keep.append([np.nan])
            #ratio_keep.append([np.nan])
            continue
        
        keep = np.argmax(r2_vector[indices])
        temp1 = r2_vector[indices]
        temp2 = temp1[keep]
        r2_keep.append(temp2)

        temp1 = ratio_vector[indices]
        temp2 = temp1[keep]
        ratio_keep.append(temp2)

        temp1 = idxs[indices]
        temp2 = temp1[keep]
        idxs_keep.append(temp2)

    ratio_keep, r2_keep, idxs_keep = squeezer(ratio_keep, r2_keep,
            idxs_keep)
    print(ratio_keep)

    sort_idxs = ratio_keep.argsort()
    sort_ratio = ratio_keep[sort_idxs]
    sort_r2 = r2_keep[sort_idxs]
    sort_idxs = idxs_keep[sort_idxs]
    return sort_ratio, sort_r2, sort_idxs, edges 

def trial_sem(neural, fs=1, pre_baseline=None):
    num_trials = neural.shape[0]
    neural_avgavg = np.average(neural, axis=(0,2)) / (fs/1000)
    temp = np.average(neural, axis=0)
    neural_avgsem = np.std(temp, axis=1) / np.sqrt(num_trials) / (fs/1000)

    if pre_baseline is not None:
        neural_bs = np.average(neural[:,:pre_baseline,:]) / (fs/1000)
        print(neural_bs)
        neural_avgavg = neural_avgavg-neural_bs


    return neural_avgavg, neural_avgsem

def trial_sem_multisession(neural_list, fs=1, pre_baseline=None):
    new_list = []
    num_trials = len(neural_list)
    curr_neurons = 0
    num_neurons=0
    for trial in neural_list:
        if trial.shape[1] != curr_neurons:
            curr_neurons = trial.shape[1]
            num_neurons+= curr_neurons
        new_list.append(np.average(trial, axis=1))
    avg = np.average(new_list, axis=0) / (fs/1000)
    sem = np.std(new_list, axis=0) / np.sqrt(num_trials) / (fs/1000)

    if pre_baseline is not None:
        temp = np.array(new_list)
        bs = np.average(temp[:, :pre_baseline]) / (fs/1000)
        avg = avg - bs


    return avg, sem



def plot_trial_sem(avg, sem, pre=50, colors=None, labels='None'):
    fig, ax = plt.subplots()
    for idx in range(len(avg)):
        y = avg[idx]
        yhat = sem[idx]
        x = np.arange(len(y)) - pre
        if labels is not None:
            label = labels[idx]
        else:
            label=None
        if colors is not None:
            color = colors[idx]
        else:
            color=None
        ax.plot(x, y, color=color, label=label)
        ax.fill_between(x, y - yhat, y+yhat, color=color, alpha=0.2)

    ax.legend()

    return fig, ax

def collapse_h(h, nlags, var=None, bias=True):
        features = h.shape[0] - 1 
        repeats = features / nlags
        assert features % repeats ==0, 'something wrong'
        new_h = np.abs(h[1:,:])
        temp = np.arange(features) % repeats
        collapse_list = []

        for neuron in np.arange(repeats):
            collapse_list.append(np.average(np.sum(new_h[temp==neuron,:],
                axis=0), weights=var))


        return np.array(collapse_list)

def collapse_h_lags(h, nlags, var=None, bias=True):
        features = h.shape[0] - 1 
        num_neurons = features / nlags
        assert features % nlags ==0, 'something wrong'
        assert features % num_neurons == 0, 'something wrong'
        new_h = np.abs(h[1:,:])
        collapse_list = []

        iterator = np.arange(nlags+1) * num_neurons
        iterator = iterator.astype(int)

        for idx in range(len(iterator)-1):
            start = iterator[idx]
            end = iterator[idx+1]
            collapse_list.append(np.average(np.sum(new_h[start:end,:],
                axis=0), weights=var))


        return np.array(collapse_list)

def mask_input_region(X_format, nlags, num_region1):
        features = X_format.shape[1]
        repeats = features/nlags
        assert features % repeats == 0, 'something wrong'

        temp =  np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1 

        return region1_mask, region2_mask

def mask_h_region(h, nlags, num_region1, keep_bias=False):
        features = h.shape[0]
        mod_test = features % nlags
        if mod_test == 1:
            print('this h has bias')
            features=features-1
            has_bias=True
        elif mod_test ==0:
            print('this h has no bias term already')
            has_bias=False
        else:
            print('i think an error has happened')
            return 0

        repeats = features / nlags
        assert features % repeats ==0, 'something wrong'

        temp = np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1

        if has_bias:

            region1_mask = np.hstack([keep_bias, temp < num_region1])
            region2_mask = np.hstack([keep_bias, temp >= num_region1])

        return region1_mask, region2_mask



def split_h_predics(X, h, nlags, num_region1):
    h1_mask, h2_mask = mask_h_region(h, nlags, num_region1, keep_bias=False)
    r1_mask, r2_mask = mask_input_region(X, nlags, num_region1)
    bias = h[0,:] 

    X1 = X[:,r1_mask]
    X2 = X[:,r2_mask]
    h1 = h[h1_mask,:]
    h2 = h[h2_mask,:]

    yhat1 = X1@h1
    yhat2 = X2@h2

    return yhat1, yhat2, bias

def divergence(obs1, obs2, PCAObject=None):
    if PCAObject==None:
        PCAObject = PCA(n_components=obs1.shape[1]).fit(obs1)
    PCs1 = PCAObject.transform(obs1)
    PCs2 = PCAObject.transform(obs2)

    return _calc_divergence(PCs1, PCs2)

def _calc_divergence(PCs1, PCs2):

        #helper function when calculating have two subspaces


    difference = np.abs(np.var(PCs2, axis=0) - np.var(PCs1, axis=0))
    summy = np.var(PCs2, axis=0) + np.var(PCs1, axis=0)
    weights = np.var(PCs1, axis=0)

    return np.average(difference / summy, weights=weights)









